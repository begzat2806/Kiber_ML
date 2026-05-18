"""
train.py — Training loop, Trainer, optimizer/scheduler factories.

ЭКСПОРТИРУЕТ (всё что импортирует main.py):
    Trainer
    build_optimizer
    build_scheduler
    build_criterion
    evaluate_on_test
"""

import logging
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TRAIN_CFG, DATA_CFG, PATHS
from utils import (
    MetricsTracker,
    EarlyStopping,
    CheckpointManager,
    CSVMetricsLogger,
    format_metrics,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 1. OPTIMIZER
# ══════════════════════════════════════════════════════════════

def build_optimizer(model: nn.Module, cfg=None) -> torch.optim.Optimizer:
    """
    AdamW — Adam с корректным decoupled weight decay.
    Loshchilov & Hutter, 2017.
    """
    cfg = cfg or TRAIN_CFG
    return torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.learning_rate,
        weight_decay = cfg.weight_decay,
        betas        = (0.9, 0.999),
        eps          = 1e-8,
    )


# ══════════════════════════════════════════════════════════════
# 2. SCHEDULER
# ══════════════════════════════════════════════════════════════

def build_scheduler(optimizer, cfg=None):
    cfg = cfg or TRAIN_CFG

    # Фаза 1: Linear warmup (эпохи 1-10)
    # LR растёт от 3e-5 до 3e-4
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 0.1,
        end_factor   = 1.0,
        total_iters  = 10,
    )
    # Фаза 2: Cosine annealing (эпохи 11-200)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = cfg.cosine_t_max - 10,
        eta_min = cfg.min_lr,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers = [warmup, cosine],
        milestones = [10],
    )

# ══════════════════════════════════════════════════════════════
# 3. CRITERION
# ══════════════════════════════════════════════════════════════

def build_criterion(graph_data: Dict, device: torch.device, cfg=None) -> nn.Module:
    """
    CrossEntropyLoss + label smoothing + class weights.
    """
    cfg = cfg or TRAIN_CFG

    weights = None
    if cfg.use_class_weights:
        from dataset import ThreatIncidentDataset
        tmp_ds  = ThreatIncidentDataset(
            graph_data["incident_X"],
            graph_data["incident_y"],
        )
        weights = tmp_ds.get_class_weights().to(device)
        logger.info(
            "Class weights: [%s]",
            ", ".join(f"{w:.2f}" for w in weights.cpu().tolist())
        )

    return nn.CrossEntropyLoss(
        weight         = weights,
        label_smoothing= cfg.label_smoothing,
    )


# ══════════════════════════════════════════════════════════════
# 4. TRAINER
# ══════════════════════════════════════════════════════════════

class Trainer:
    """
    Полный цикл обучения и валидации.

    Использование:
        trainer = Trainer(model, optimizer, scheduler, criterion,
                          train_loader, val_loader, graph_data, device)
        history = trainer.fit()
    """

    def __init__(self,
                 model:        nn.Module,
                 optimizer:    torch.optim.Optimizer,
                 scheduler,
                 criterion:    nn.Module,
                 train_loader: DataLoader,
                 val_loader:   DataLoader,
                 graph_data:   Dict,
                 device:       torch.device,
                 cfg=None):

        self.model        = model
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.criterion    = criterion
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.cfg          = cfg or TRAIN_CFG

        # Граф — статические данные на device
        self.node_features = graph_data["node_features"].to(device)
        self.edge_index    = graph_data["edge_index"].to(device)
        self.class_names   = graph_data["class_names"]
        self.n_classes     = graph_data["n_classes"]

        self.ckpt_mgr = CheckpointManager()
        self.csv_log  = CSVMetricsLogger()
        self.stopper  = EarlyStopping(
            patience = self.cfg.early_stopping_patience,
            delta    = self.cfg.early_stopping_delta,
        )

        self.best_val_loss = float("inf")
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss":  [],
            "train_acc":  [], "val_acc":   [],
            "val_f1":     [],
        }

    def fit(self, start_epoch: int = 0) -> Dict:
        logger.info("=" * 60)
        logger.info("Training: epochs %d -> %d | lr=%.2e | device=%s",
                    start_epoch, self.cfg.max_epochs,
                    self.cfg.learning_rate, self.device)
        logger.info("=" * 60)

        for epoch in range(start_epoch, self.cfg.max_epochs):
            t0 = time.time()

            train_metrics = self._train_one_epoch(epoch)
            val_metrics   = self._validate_one_epoch(epoch)

            self.scheduler.step()
            lr      = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            logger.info(
                "[Epoch %3d/%d | %4.1fs] "
                "train loss=%.4f acc=%.4f | "
                "val loss=%.4f acc=%.4f f1=%.4f | lr=%.2e",
                epoch + 1, self.cfg.max_epochs, elapsed,
                train_metrics["loss"], train_metrics["accuracy"],
                val_metrics["loss"],   val_metrics["accuracy"],
                val_metrics["f1_macro"], lr,
            )

            self.csv_log.log(epoch + 1, train_metrics, val_metrics, lr)

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1_macro"])

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.ckpt_mgr.save_best(
                    self.model, self.optimizer, self.scheduler,
                    epoch + 1, val_metrics,
                )

            if (epoch + 1) % self.cfg.checkpoint_every == 0:
                self.ckpt_mgr.save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch + 1, val_metrics,
                )

            if self.stopper(val_metrics["loss"]):
                logger.info(
                    "Early stopping at epoch %d (best=%.4f)",
                    epoch + 1, self.best_val_loss,
                )
                break

        logger.info("Done. Best val_loss: %.4f", self.best_val_loss)
        return self.history

    def _train_one_epoch(self, epoch: int) -> Dict:
        self.model.train()
        tracker = MetricsTracker(self.n_classes)

        for step, (X, y) in enumerate(self.train_loader):
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            out  = self.model(X, self.node_features, self.edge_index)
            loss = self.criterion(out["logits"], y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.cfg.grad_clip,
            )

            self.optimizer.step()

            with torch.no_grad():
                preds = out["probs"].argmax(dim=-1)
            tracker.update(loss.item(), preds, y, len(y))

            if step % self.cfg.log_every_n_steps == 0:
                logger.debug("  epoch=%d step=%d loss=%.4f",
                             epoch + 1, step, loss.item())

        return tracker.compute()

    def _validate_one_epoch(self, epoch: int) -> Dict:
        self.model.eval()
        tracker = MetricsTracker(self.n_classes)

        with torch.no_grad():
            for X, y in self.val_loader:
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                out   = self.model(X, self.node_features, self.edge_index)
                loss  = self.criterion(out["logits"], y)
                preds = out["probs"].argmax(dim=-1)

                tracker.update(loss.item(), preds, y, len(y))

        return tracker.compute()


# ══════════════════════════════════════════════════════════════
# 5. TEST EVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate_on_test(model:         nn.Module,
                     test_loader:   DataLoader,
                     criterion:     nn.Module,
                     node_features: torch.Tensor,
                     edge_index:    torch.Tensor,
                     device:        torch.device,
                     n_classes:     int = None) -> Tuple[Dict, torch.Tensor]:
    """
    Финальный тест на hold-out выборке.
    Вызывается один раз из main.py после обучения.

    Возвращает: (metrics_dict, confusion_matrix_tensor)
    """
    n_classes = n_classes or DATA_CFG.num_classes

    model.eval()
    tracker = MetricsTracker(n_classes)

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out   = model(X, node_features, edge_index)
            loss  = criterion(out["logits"], y)
            preds = out["probs"].argmax(dim=-1)

            tracker.update(loss.item(), preds, y, len(y))

    metrics = tracker.compute()
    cm      = tracker.get_confusion_matrix()

    logger.info("=" * 60)
    logger.info("TEST RESULTS:")
    for k, v in metrics.items():
        logger.info("  %-20s %.4f", k, v)
    logger.info("=" * 60)

    return metrics, cm