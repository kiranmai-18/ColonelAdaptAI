import torch

@torch.no_grad()
def evaluate(model, loader, task_id, device, use_perturb=True):
    model.eval(); correct=0; total=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        if use_perturb:
            logits = model(x, task_id)
        else:
            features = model.forward_features(x, task_id, use_perturb=False)
            logits = model.heads(features, task_id)
        pred = logits.argmax(1)
        correct += int((pred==y).sum().item()); total += y.numel()
    return correct / max(1,total)

@torch.no_grad()
def evaluate_seen_tasks(model, tasks, upto_task, device, use_perturb=True):
    return [evaluate(model, tasks[j].test_loader, j, device, use_perturb=use_perturb) for j in range(upto_task+1)]
