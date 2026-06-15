from dataclasses import dataclass
from typing import List
import torch
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from torchvision import datasets, transforms

@dataclass
class TaskData:
    task_id: int
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    classes: list
    num_classes: int
    class_to_local: dict

class LabelMappedSubset(Dataset):
    def __init__(self, base, indices, class_to_local, transform_override=None, perm=None):
        self.base, self.indices = base, list(indices)
        self.class_to_local = class_to_local
        self.transform_override = transform_override
        self.perm = perm
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        x, y = self.base[self.indices[idx]]
        if self.transform_override is not None:
            x = self.transform_override(x)
        if self.perm is not None:
            flat = x.view(-1)[self.perm]
            x = flat.view_as(x)
        return x, self.class_to_local[int(y)]

def _indices_by_classes(targets, classes):
    targets = torch.as_tensor(targets)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for c in classes:
        mask |= (targets == c)
    return mask.nonzero(as_tuple=False).view(-1).tolist()

def _split_train_val(ds, val_ratio=0.1, seed=42):
    n_val = max(1, int(len(ds) * val_ratio))
    n_train = len(ds) - n_val
    return random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

def build_split_mnist(root='./datasets', batch_size=128, num_workers=2, seed=42):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root, train=False, download=True, transform=tfm)
    class_pairs = [(0,1),(2,3),(4,5),(6,7),(8,9)]
    tasks=[]
    for tid, classes in enumerate(class_pairs):
        mapper={c:i for i,c in enumerate(classes)}
        tr_ds = LabelMappedSubset(train, _indices_by_classes(train.targets, classes), mapper)
        te_ds = LabelMappedSubset(test, _indices_by_classes(test.targets, classes), mapper)
        tr, va = _split_train_val(tr_ds, seed=seed+tid)
        tasks.append(TaskData(tid, DataLoader(tr,batch_size,shuffle=True,num_workers=num_workers),
                              DataLoader(va,batch_size,shuffle=False,num_workers=num_workers),
                              DataLoader(te_ds,batch_size,shuffle=False,num_workers=num_workers), list(classes), 2, mapper))
    return tasks, (1,28,28)

def build_permuted_mnist(root='./datasets', num_tasks=20, batch_size=128, num_workers=2, seed=42):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root, train=False, download=True, transform=tfm)
    tasks=[]
    for tid in range(num_tasks):
        perm = torch.randperm(28*28, generator=torch.Generator().manual_seed(seed+tid))
        mapper={c:c for c in range(10)}
        tr_ds = LabelMappedSubset(train, range(len(train)), mapper, perm=perm)
        te_ds = LabelMappedSubset(test, range(len(test)), mapper, perm=perm)
        tr, va = _split_train_val(tr_ds, seed=seed+tid)
        tasks.append(TaskData(tid, DataLoader(tr,batch_size,shuffle=True,num_workers=num_workers),
                              DataLoader(va,batch_size,shuffle=False,num_workers=num_workers),
                              DataLoader(te_ds,batch_size,shuffle=False,num_workers=num_workers), list(range(10)), 10, mapper))
    return tasks, (1,28,28)

def build_split_cifar100(root='./datasets', batch_size=128, num_workers=2, seed=42):
    tfm_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))])
    tfm_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))])
    train = datasets.CIFAR100(root, train=True, download=True, transform=tfm_train)
    test = datasets.CIFAR100(root, train=False, download=True, transform=tfm_test)
    targets_train, targets_test = train.targets, test.targets
    tasks=[]
    for tid in range(20):
        classes = list(range(tid*5, tid*5+5)); mapper={c:i for i,c in enumerate(classes)}
        tr_ds = LabelMappedSubset(train, _indices_by_classes(targets_train, classes), mapper)
        te_ds = LabelMappedSubset(test, _indices_by_classes(targets_test, classes), mapper)
        tr, va = _split_train_val(tr_ds, seed=seed+tid)
        tasks.append(TaskData(tid, DataLoader(tr,batch_size,shuffle=True,num_workers=num_workers),
                              DataLoader(va,batch_size,shuffle=False,num_workers=num_workers),
                              DataLoader(te_ds,batch_size,shuffle=False,num_workers=num_workers), classes, 5, mapper))
    return tasks, (3,32,32)

def build_tasks(name, **kwargs):
    if name == 'split_mnist': return build_split_mnist(**kwargs)
    if name == 'permuted_mnist': return build_permuted_mnist(**kwargs)
    if name == 'split_cifar100': return build_split_cifar100(**kwargs)
    raise ValueError(f'Unknown dataset: {name}')
