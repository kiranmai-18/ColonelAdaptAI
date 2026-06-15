import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTCNN(nn.Module):
    feature_dims = [32, 64, 128]
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64*7*7, 128)
    def forward_layers(self, x):
        feats=[]
        x = F.relu(self.conv1(x)); x = F.max_pool2d(x,2); feats.append(x)
        x = F.relu(self.conv2(x)); x = F.max_pool2d(x,2); feats.append(x)
        x = torch.flatten(x,1)
        x = F.relu(self.fc(x)); feats.append(x)
        return x, feats
    def forward(self,x): return self.forward_layers(x)[0]

class CIFARCNN(nn.Module):
    feature_dims = [64, 128, 256]
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.block3 = nn.Sequential(nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)))
    def forward_layers(self,x):
        feats=[]
        x = self.block1(x); x = F.max_pool2d(x,2); feats.append(x)
        x = self.block2(x); x = F.max_pool2d(x,2); feats.append(x)
        x = self.block3(x); x = torch.flatten(x,1); feats.append(x)
        return x, feats
    def forward(self,x): return self.forward_layers(x)[0]

def build_backbone(name, input_shape=None):
    if name == 'auto':
        name = 'mnist_cnn' if input_shape and input_shape[0] == 1 else 'cifar_cnn'
    if name == 'mnist_cnn': return MNISTCNN(), MNISTCNN.feature_dims, 128
    if name == 'cifar_cnn': return CIFARCNN(), CIFARCNN.feature_dims, 256
    raise ValueError(name)
