"""Functional rehearsal-based continual-learning baselines.

Implemented methods:
- finetune: current task only
- replay: supervised episodic replay
- agem: Averaged Gradient Episodic Memory projection
- gem: task-balanced GEM-style projection against stored task gradients
- der: Dark Experience Replay logit distillation
- derpp: DER + supervised replay cross-entropy
- mir: Maximally Interfered Retrieval sample selection

The implementation is intentionally compact and shares the same model, task stream,
metrics, and output protocol as ColonelAdaptAI.
"""
from __future__ import annotations
import copy, random, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from trainers.evaluator import evaluate_seen_tasks
from utils.metrics import average_accuracy, forgetting_measure, backward_transfer
from utils.visualization import plot_accuracy_matrix, plot_curve


class EpisodicMemory:
    """Class/task-balanced episodic memory with optional stored logits for DER."""
    def __init__(self, capacity: int = 2000):
        self.capacity = int(capacity)
        self.items: List[Tuple[torch.Tensor, int, int, Optional[torch.Tensor]]] = []

    def __len__(self):
        return len(self.items)

    def add_batch(self, x, y, task_id: int, logits: Optional[torch.Tensor] = None):
        x = x.detach().cpu(); y = y.detach().cpu()
        logits = None if logits is None else logits.detach().cpu()
        for i in range(x.size(0)):
            li = None if logits is None else logits[i]
            self.items.append((x[i].clone(), int(y[i]), int(task_id), li.clone() if li is not None else None))
        self._trim()

    def _trim(self):
        if len(self.items) <= self.capacity:
            return
        # Reservoir-like uniform trimming keeps the implementation simple and unbiased.
        keep = set(random.sample(range(len(self.items)), self.capacity))
        self.items = [it for k, it in enumerate(self.items) if k in keep]

    def sample(self, batch_size: int, device, task_id: Optional[int] = None):
        pool = self.items if task_id is None else [it for it in self.items if it[2] == task_id]
        if not pool:
            return None
        batch = random.sample(pool, min(batch_size, len(pool)))
        xs = torch.stack([b[0] for b in batch]).to(device)
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        tids = torch.tensor([b[2] for b in batch], dtype=torch.long, device=device)
        if all(b[3] is not None for b in batch):
            logs = torch.stack([b[3] for b in batch]).to(device)
        else:
            logs = None
        return xs, ys, tids, logs

    def task_ids(self):
        return sorted(set(t for _, _, t, _ in self.items))


class RehearsalBaselineTrainer:
    def __init__(self, model: nn.Module, tasks, cfg: dict, device, logger, method: str):
        self.model = model.to(device)
        self.tasks = tasks
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.method = method.lower()
        self.out_dir = Path(cfg['experiment']['output_dir']) / f'baseline_{self.method}'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.acc_matrix = np.full((len(tasks), len(tasks)), np.nan, dtype=float)
        self.summary = []
        bcfg = cfg.get('baselines', {})
        self.memory = EpisodicMemory(capacity=bcfg.get('memory_size', 2000))
        self.replay_batch_size = int(bcfg.get('replay_batch_size', cfg['training']['batch_size']))
        self.der_alpha = float(bcfg.get('der_alpha', 0.5))
        self.der_beta = float(bcfg.get('der_beta', 0.5))
        self.mir_candidates = int(bcfg.get('mir_candidates', 64))
        self.mir_k = int(bcfg.get('mir_k', self.replay_batch_size))

    def _optimizer(self, task_id):
        lr = self.cfg['training']['lr_first_task'] if task_id == 0 else self.cfg['training']['lr_next_tasks']
        return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.cfg['training'].get('weight_decay', 0.0))

    def _flat_grad(self):
        vec = []
        for p in self.model.parameters():
            if p.requires_grad:
                vec.append(torch.zeros_like(p).flatten() if p.grad is None else p.grad.detach().flatten())
        return torch.cat(vec) if vec else torch.tensor([], device=self.device)

    def _set_flat_grad(self, vec):
        offset = 0
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.copy_(vec[offset:offset+n].view_as(p))
            offset += n

    def _forward_no_perturb(self, x, tid: int):
        features = self.model.forward_features(x, tid, use_perturb=False)
        return self.model.heads(features, tid)

    def _multi_task_ce(self, x, y, tids):
        losses = []
        for tid in tids.unique().tolist():
            mask = tids == tid
            logits = self._forward_no_perturb(x[mask], int(tid))
            losses.append(F.cross_entropy(logits, y[mask]))
        return torch.stack(losses).mean() if losses else torch.tensor(0., device=self.device)

    def _multi_task_mse_logits(self, x, tids, stored_logits):
        losses = []
        if stored_logits is None:
            return torch.tensor(0., device=self.device)
        for tid in tids.unique().tolist():
            mask = tids == tid
            pred = self._forward_no_perturb(x[mask], int(tid))
            target = stored_logits[mask]
            if pred.shape == target.shape:
                losses.append(F.mse_loss(pred, target))
        return torch.stack(losses).mean() if losses else torch.tensor(0., device=self.device)

    def _replay_loss(self):
        batch = self.memory.sample(self.replay_batch_size, self.device)
        if batch is None:
            return torch.tensor(0., device=self.device)
        rx, ry, rt, rlogits = batch
        if self.method in {'der', 'derpp'}:
            loss = self.der_alpha * self._multi_task_mse_logits(rx, rt, rlogits)
            if self.method == 'derpp':
                loss = loss + self.der_beta * self._multi_task_ce(rx, ry, rt)
            return loss
        return self._multi_task_ce(rx, ry, rt)

    def _apply_agem_projection(self, current_loss):
        if len(self.memory) == 0:
            current_loss.backward(); return
        current_loss.backward()
        g = self._flat_grad()
        self.model.zero_grad(set_to_none=True)
        ref_batch = self.memory.sample(self.replay_batch_size, self.device)
        if ref_batch is None:
            self._set_flat_grad(g); return
        rx, ry, rt, _ = ref_batch
        ref_loss = self._multi_task_ce(rx, ry, rt)
        ref_loss.backward()
        g_ref = self._flat_grad()
        dot = torch.dot(g, g_ref)
        if dot < 0:
            g = g - (dot / (torch.dot(g_ref, g_ref) + 1e-12)) * g_ref
        self.model.zero_grad(set_to_none=True)
        self._set_flat_grad(g)

    def _apply_gem_projection(self, current_loss):
        if len(self.memory) == 0:
            current_loss.backward(); return
        current_loss.backward()
        g = self._flat_grad()
        self.model.zero_grad(set_to_none=True)
        for old_tid in self.memory.task_ids():
            ref_batch = self.memory.sample(max(1, self.replay_batch_size // max(1, len(self.memory.task_ids()))), self.device, task_id=old_tid)
            if ref_batch is None:
                continue
            rx, ry, rt, _ = ref_batch
            ref_loss = self._multi_task_ce(rx, ry, rt)
            ref_loss.backward()
            g_ref = self._flat_grad()
            dot = torch.dot(g, g_ref)
            if dot < 0:
                g = g - (dot / (torch.dot(g_ref, g_ref) + 1e-12)) * g_ref
            self.model.zero_grad(set_to_none=True)
        self._set_flat_grad(g)

    def _mir_batch(self, opt):
        if len(self.memory) == 0:
            return None
        cand = self.memory.sample(self.mir_candidates, self.device)
        if cand is None:
            return None
        rx, ry, rt, rlog = cand
        with torch.no_grad():
            before = []
            for i in range(rx.size(0)):
                before.append(F.cross_entropy(self._forward_no_perturb(rx[i:i+1], int(rt[i])), ry[i:i+1], reduction='none'))
            before = torch.cat(before)
        # Create a small virtual SGD step from current gradients, score loss increase, then restore.
        state = copy.deepcopy(self.model.state_dict())
        lr = opt.param_groups[0]['lr']
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-lr)
        with torch.no_grad():
            after = []
            for i in range(rx.size(0)):
                after.append(F.cross_entropy(self._forward_no_perturb(rx[i:i+1], int(rt[i])), ry[i:i+1], reduction='none'))
            after = torch.cat(after)
        self.model.load_state_dict(state)
        k = min(self.mir_k, rx.size(0))
        idx = torch.topk(after - before, k=k).indices
        return rx[idx], ry[idx], rt[idx], None if rlog is None else rlog[idx]

    def train_task(self, task):
        tid = task.task_id
        self.model.add_task_head(tid, task.num_classes)
        opt = self._optimizer(tid)
        epochs = self.cfg['training']['epochs_per_task']
        start = time.time()
        for ep in range(epochs):
            self.model.train(); total = 0.0; n = 0
            for x, y in tqdm(task.train_loader, desc=f'{self.method.upper()} Task {tid} Epoch {ep+1}/{epochs}', leave=False):
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad(set_to_none=True)
                logits = self._forward_no_perturb(x, tid)
                loss = F.cross_entropy(logits, y)
                if self.method in {'replay', 'der', 'derpp'}:
                    loss = loss + self._replay_loss()
                    loss.backward()
                elif self.method == 'agem':
                    self._apply_agem_projection(loss)
                elif self.method == 'gem':
                    self._apply_gem_projection(loss)
                elif self.method == 'mir':
                    loss.backward(retain_graph=True)
                    mir = self._mir_batch(opt)
                    self.model.zero_grad(set_to_none=True)
                    logits = self._forward_no_perturb(x, tid)
                    loss = F.cross_entropy(logits, y)
                    if mir is not None:
                        rx, ry, rt, _ = mir
                        loss = loss + self._multi_task_ce(rx, ry, rt)
                    loss.backward()
                else:  # finetune
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                with torch.no_grad():
                    store_logits = self._forward_no_perturb(x, tid) if self.method in {'der', 'derpp'} else None
                    self.memory.add_batch(x, y, tid, store_logits)
                total += float(loss.item()) * x.size(0); n += x.size(0)
            self.logger.log(f'{self.method.upper()} task {tid} epoch {ep+1}: loss={total/max(1,n):.4f}')
        elapsed = time.time() - start
        accs = evaluate_seen_tasks(self.model, self.tasks, tid, self.device, use_perturb=False)
        self.acc_matrix[tid, :tid+1] = accs
        aa = average_accuracy(self.acc_matrix, tid); fm = forgetting_measure(self.acc_matrix, tid); bwt = backward_transfer(self.acc_matrix, tid)
        self.summary.append({'method': self.method, 'task': tid, 'avg_accuracy': aa, 'forgetting': fm, 'bwt': bwt, 'time_sec': elapsed, 'memory_items': len(self.memory)})
        self.logger.log(f'{self.method.upper()} after task {tid}: ACC={aa:.4f}, FM={fm:.4f}, BWT={bwt:.4f}, time={elapsed:.1f}s')

    def fit(self):
        for task in self.tasks:
            self.train_task(task)
        self.save_results()
        return self.acc_matrix, self.summary

    def save_results(self):
        pd.DataFrame(self.acc_matrix).to_csv(self.out_dir / 'accuracy_matrix.csv', index=False)
        pd.DataFrame(self.summary).to_csv(self.out_dir / 'summary_results.csv', index=False)
        plot_accuracy_matrix(self.acc_matrix, self.out_dir / 'accuracy_matrix.png')
        plot_curve([s['avg_accuracy'] for s in self.summary], 'Average Accuracy', f'{self.method.upper()} Average Accuracy', self.out_dir / 'avg_accuracy.png')
        plot_curve([s['forgetting'] for s in self.summary], 'Forgetting', f'{self.method.upper()} Forgetting', self.out_dir / 'forgetting.png')
        torch.save({'model': self.model.state_dict(), 'acc_matrix': self.acc_matrix, 'summary': self.summary}, self.out_dir / 'checkpoint.pt')
