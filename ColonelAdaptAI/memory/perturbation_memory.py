class PerturbationMemory:
    def __init__(self): self.store={}
    def add_perturbation(self, task_id, layer_id, tensor): self.store[(int(task_id), int(layer_id))]=tensor.detach().cpu().clone()
    def get_perturbation(self, task_id, layer_id): return self.store.get((int(task_id), int(layer_id)))
    def update_perturbation(self, task_id, layer_id, tensor): self.add_perturbation(task_id, layer_id, tensor)
