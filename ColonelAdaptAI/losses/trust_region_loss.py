import torch
import torch.nn.functional as F

def trust_region_loss(model, task_id, delta=1.0):
    loss = torch.tensor(0., device=next(model.parameters()).device)
    for p in model.perturbations.layer_params(task_id):
        loss += F.relu(torch.norm(p, p=2) - delta).pow(2)
    return loss
