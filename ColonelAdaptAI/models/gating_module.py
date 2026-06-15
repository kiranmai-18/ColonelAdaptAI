import torch.nn as nn
class TaskAwareGating(nn.Module):
    def __init__(self, emb_dim, num_layers, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, num_layers), nn.Sigmoid())
    def forward(self, z): return self.net(z)
