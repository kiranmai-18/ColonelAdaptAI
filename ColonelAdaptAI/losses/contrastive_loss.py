import torch
import torch.nn.functional as F

def supervised_contrastive_loss(features, labels, temperature=0.07):
    if features.size(0) < 2: return torch.tensor(0., device=features.device)
    f = F.normalize(features, dim=1)
    sim = f @ f.t() / temperature
    labels = labels.view(-1,1)
    mask = torch.eq(labels, labels.t()).float().to(features.device)
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=features.device)
    mask = mask * logits_mask
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1e-8)
    return -mean_log_prob_pos.mean()
