import torch
import torch.nn.functional as F

def prototype_refinement_loss(features, labels, proto_matrix, temperature=0.07):
    if proto_matrix is None:
        return torch.tensor(0., device=features.device)
    f = F.normalize(features, dim=1)
    p = F.normalize(proto_matrix, dim=1)
    logits = f @ p.t() / temperature
    return F.cross_entropy(logits, labels.clamp(max=proto_matrix.size(0)-1))
