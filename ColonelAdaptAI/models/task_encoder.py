import torch.nn as nn
class TaskEncoder(nn.Module):
    def __init__(self, max_tasks=50, emb_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(max_tasks, emb_dim)
    def forward(self, task_ids):
        return self.embedding(task_ids)
