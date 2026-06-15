import torch
import torch.nn as nn
class LayerWisePerturbations(nn.Module):
    def __init__(self, max_tasks, feature_dims, init_std=0.01):
        super().__init__()
        self.max_tasks=max_tasks; self.feature_dims=feature_dims
        self.params = nn.ParameterDict()
        for t in range(max_tasks):
            for i,d in enumerate(feature_dims):
                self.params[f'{t}_{i}'] = nn.Parameter(torch.randn(d)*init_std)
    def get(self, task_id, layer_id): return self.params[f'{int(task_id)}_{int(layer_id)}']
    def layer_params(self, task_id): return [self.get(task_id,i) for i in range(len(self.feature_dims))]
