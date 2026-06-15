import torch
import torch.nn.functional as F

class FisherMemory:
    def __init__(self):
        self.fisher = {}
        self.params = {}
    @torch.no_grad()
    def consolidate_params(self, model):
        self.params = {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    def estimate(self, model, loader, task_id, device, max_samples=512):
        model.eval(); fisher={n:torch.zeros_like(p, device=device) for n,p in model.named_parameters() if p.requires_grad}
        seen=0
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            model.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x, task_id), y)
            loss.backward()
            bs=x.size(0); seen += bs
            for n,p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += (p.grad.detach()**2) * bs
            if seen >= max_samples: break
        for n in fisher: fisher[n] = fisher[n] / max(1, seen)
        self.fisher = {n:v.detach().clone() for n,v in fisher.items()}
        self.consolidate_params(model)
