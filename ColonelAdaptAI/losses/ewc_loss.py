import torch

def ewc_loss(model, fisher_memory):
    if not fisher_memory.fisher or not fisher_memory.params:
        return torch.tensor(0., device=next(model.parameters()).device)
    loss = torch.tensor(0., device=next(model.parameters()).device)
    for n,p in model.named_parameters():
        if n in fisher_memory.fisher and n in fisher_memory.params:
            loss += (fisher_memory.fisher[n].to(p.device) * (p - fisher_memory.params[n].to(p.device)).pow(2)).sum()
    return 0.5 * loss
