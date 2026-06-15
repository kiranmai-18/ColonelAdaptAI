import torch
import torch.nn.functional as F

class PrototypeBuffer:
    def __init__(self):
        self.prototypes = {}  # (task_id, class_id) -> tensor
    @torch.no_grad()
    def update_from_loader(self, model, loader, task_id, device):
        model.eval(); sums={}; counts={}
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            feats = model.forward_features(x, task_id, use_perturb=True)
            for c in y.unique():
                m = y==c; key=(int(task_id), int(c.item()))
                sums[key]=sums.get(key, torch.zeros_like(feats[0].detach().cpu())) + feats[m].detach().cpu().sum(0)
                counts[key]=counts.get(key,0)+int(m.sum().item())
        for k in sums: self.prototypes[k] = sums[k] / max(1, counts[k])
    def get_task_matrix(self, task_id, device):
        keys=sorted([k for k in self.prototypes if k[0]==int(task_id)], key=lambda x:x[1])
        if not keys: return None, None
        return torch.stack([self.prototypes[k] for k in keys]).to(device), torch.tensor([k[1] for k in keys], device=device)
    def all(self): return self.prototypes
