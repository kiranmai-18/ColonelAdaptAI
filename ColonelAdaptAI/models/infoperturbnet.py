import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import build_backbone
from .task_encoder import TaskEncoder
from .gating_module import TaskAwareGating
from .perturbation_units import LayerWisePerturbations
from .heads import MultiHeadClassifier

class InfoPerturbNet(nn.Module):
    def __init__(self, input_shape, backbone_name='auto', max_tasks=20, task_embedding_dim=32, perturb_init_std=0.01):
        super().__init__()
        self.backbone, self.feature_dims, self.output_dim = build_backbone(backbone_name, input_shape)
        self.task_encoder = TaskEncoder(max_tasks, task_embedding_dim)
        self.gating = TaskAwareGating(task_embedding_dim, len(self.feature_dims))
        self.perturbations = LayerWisePerturbations(max_tasks, self.feature_dims, perturb_init_std)
        self.heads = MultiHeadClassifier(self.output_dim)
    def add_task_head(self, task_id, num_classes): self.heads.add_head(task_id, num_classes)
    def _inject(self, activation, perturb, alpha):
        if activation.dim() == 4:
            p = perturb.view(1, -1, 1, 1)
        else:
            p = perturb.view(1, -1)
        return activation + alpha.view(-1,1,1,1)*p if activation.dim()==4 else activation + alpha.view(-1,1)*p
    def forward_features(self, x, task_id, return_intermediate=False, use_perturb=True):
        tid_tensor = torch.full((x.size(0),), int(task_id), dtype=torch.long, device=x.device)
        z = self.task_encoder(tid_tensor)
        gates = self.gating(z)
        feats=[]
        # manually mirror supported backbones
        b = self.backbone
        if hasattr(b, 'conv1'):
            x = F.relu(b.conv1(x)); x = F.max_pool2d(x,2)
            if use_perturb: x = self._inject(x, self.perturbations.get(task_id,0), gates[:,0])
            feats.append(x)
            x = F.relu(b.conv2(x)); x = F.max_pool2d(x,2)
            if use_perturb: x = self._inject(x, self.perturbations.get(task_id,1), gates[:,1])
            feats.append(x)
            x = torch.flatten(x,1); x = F.relu(b.fc(x))
            if use_perturb: x = self._inject(x, self.perturbations.get(task_id,2), gates[:,2])
            feats.append(x)
        else:
            x = b.block1(x); x = F.max_pool2d(x,2)
            if use_perturb: x = self._inject(x, self.perturbations.get(task_id,0), gates[:,0])
            feats.append(x)
            x = b.block2(x); x = F.max_pool2d(x,2)
            if use_perturb: x = self._inject(x, self.perturbations.get(task_id,1), gates[:,1])
            feats.append(x)
            x = b.block3(x); x = torch.flatten(x,1)
            if use_perturb: x = self._inject(x, self.perturbations.get(task_id,2), gates[:,2])
            feats.append(x)
        return (x, feats, gates) if return_intermediate else x
    def forward(self, x, task_id):
        f = self.forward_features(x, task_id)
        return self.heads(f, task_id)
