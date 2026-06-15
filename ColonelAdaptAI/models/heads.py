import torch.nn as nn
class MultiHeadClassifier(nn.Module):
    def __init__(self, feature_dim):
        super().__init__(); self.feature_dim=feature_dim; self.heads=nn.ModuleDict()
    def add_head(self, task_id, num_classes):
        key=str(task_id)
        if key not in self.heads: self.heads[key]=nn.Linear(self.feature_dim, num_classes)
    def forward(self, features, task_id):
        return self.heads[str(task_id)](features)
