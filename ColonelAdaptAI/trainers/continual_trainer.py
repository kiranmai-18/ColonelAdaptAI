import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from memory.prototype_buffer import PrototypeBuffer
from memory.fisher_memory import FisherMemory
from losses.ewc_loss import ewc_loss
from losses.prototype_loss import prototype_refinement_loss
from losses.trust_region_loss import trust_region_loss
from losses.contrastive_loss import supervised_contrastive_loss
from trainers.evaluator import evaluate_seen_tasks
from utils.metrics import average_accuracy, forgetting_measure, backward_transfer
from utils.visualization import plot_accuracy_matrix, plot_curve

class ContinualTrainer:
    def __init__(self, model, tasks, cfg, device, logger):
        self.model=model.to(device); self.tasks=tasks; self.cfg=cfg; self.device=device; self.logger=logger
        self.proto_buffer=PrototypeBuffer(); self.fisher_memory=FisherMemory()
        self.out_dir=Path(cfg['experiment']['output_dir']); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.acc_matrix=np.full((len(tasks), len(tasks)), np.nan, dtype=float)
        self.summary=[]
    def _optimizer(self, task_id):
        lr = self.cfg['training']['lr_first_task'] if task_id==0 else self.cfg['training']['lr_next_tasks']
        return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.cfg['training'].get('weight_decay',0.0))
    def train_task(self, task):
        tid=task.task_id; self.model.add_task_head(tid, task.num_classes)
        opt=self._optimizer(tid); epochs=self.cfg['training']['epochs_per_task']
        loss_cfg=self.cfg['loss']; start=time.time()
        for ep in range(epochs):
            self.model.train(); total=0.0; n=0
            for x,y in tqdm(task.train_loader, desc=f'Task {tid} Epoch {ep+1}/{epochs}', leave=False):
                x,y=x.to(self.device),y.to(self.device)
                opt.zero_grad(set_to_none=True)
                feats, _, _ = self.model.forward_features(x, tid, return_intermediate=True)
                logits = self.model.heads(feats, tid)
                ce = F.cross_entropy(logits,y)
                proto_matrix,_ = self.proto_buffer.get_task_matrix(tid, self.device)
                proto_loss = prototype_refinement_loss(feats, y, proto_matrix, loss_cfg['proto_temperature'])
                tr_loss = trust_region_loss(self.model, tid, loss_cfg['trust_delta'])
                ewcl = ewc_loss(self.model, self.fisher_memory)
                cont = supervised_contrastive_loss(feats, y, loss_cfg['proto_temperature']) if loss_cfg.get('contrastive_lambda',0)>0 else torch.tensor(0., device=self.device)
                loss = ce + loss_cfg['ewc_lambda']*ewcl + loss_cfg['proto_lambda']*proto_loss + loss_cfg['trust_lambda']*tr_loss + loss_cfg.get('contrastive_lambda',0)*cont
                loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0); opt.step()
                total += float(loss.item())*x.size(0); n += x.size(0)
            self.logger.log(f'Task {tid} epoch {ep+1}: loss={total/max(1,n):.4f}')
        self.proto_buffer.update_from_loader(self.model, task.train_loader, tid, self.device)
        self.fisher_memory.estimate(self.model, task.train_loader, tid, self.device, max_samples=self.cfg['memory']['fisher_samples'])
        elapsed=time.time()-start
        accs=evaluate_seen_tasks(self.model, self.tasks, tid, self.device)
        self.acc_matrix[tid,:tid+1]=accs
        aa=average_accuracy(self.acc_matrix, tid); fm=forgetting_measure(self.acc_matrix, tid); bwt=backward_transfer(self.acc_matrix, tid)
        self.summary.append({'task':tid,'avg_accuracy':aa,'forgetting':fm,'bwt':bwt,'time_sec':elapsed})
        self.logger.log(f'After task {tid}: ACC={aa:.4f}, FM={fm:.4f}, BWT={bwt:.4f}, time={elapsed:.1f}s')
    def fit(self):
        for task in self.tasks: self.train_task(task)
        self.save_results(); return self.acc_matrix, self.summary
    def save_results(self):
        pd.DataFrame(self.acc_matrix).to_csv(self.out_dir/'accuracy_matrix.csv', index=False)
        pd.DataFrame(self.summary).to_csv(self.out_dir/'summary_results.csv', index=False)
        plot_accuracy_matrix(self.acc_matrix, self.out_dir/'accuracy_matrix.png')
        plot_curve([s['avg_accuracy'] for s in self.summary], 'Average Accuracy', 'Average Accuracy Across Tasks', self.out_dir/'avg_accuracy.png')
        plot_curve([s['forgetting'] for s in self.summary], 'Forgetting', 'Forgetting Across Tasks', self.out_dir/'forgetting.png')
        torch.save({'model':self.model.state_dict(),'acc_matrix':self.acc_matrix,'summary':self.summary}, self.out_dir/'checkpoint.pt')
