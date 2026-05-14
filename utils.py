"""
utils.py
─────────────────────────────────────────────────────────
РОЛЬ В ПРОЕКТЕ:
    Полностью независимый модуль утилит.
    Не импортирует model.py или dataset.py — только config.py.

ЗАВИСИМОСТИ:
    config.py → PATHS, DATA_CFG

ЭКСПОРТИРУЕТ:
    setup_logging()      → main.py (первым делом)
    MetricsTracker       → train.py (накапливает метрики эпохи)
    EarlyStopping        → train.py (решает когда остановить)
    CheckpointManager    → train.py (сохраняет/загружает модель)
    plot_training_curves() → main.py / train.py (после обучения)
    plot_confusion_matrix() → main.py (после теста)
    plot_semantic_heatmap() → main.py (Knowledge Graph)

ДАННЫЕ МЕЖДУ МОДУЛЯМИ:
    train.py → MetricsTracker.update(loss, preds, targets)
    train.py → CheckpointManager.save(model, optimizer, epoch, metrics)
    train.py → EarlyStopping(val_loss) → bool (остановить?)
    main.py  → plot_*() принимают простые списки/тензоры

ЧТО ОСТАВЛЕНО ТЕБЕ:
    - Подключить TensorBoard / W&B вместо CSV-логгера
    - Реализовать plot_attention_heatmap() для XAI
    - Добавить экспорт метрик в JSON для CI/CD pipelines
─────────────────────────────────────────────────────────
"""

import os
import csv
import json
import logging
import logging.handlers
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np

from config import PATHS, DATA_CFG, TRAIN_CFG

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 1. ЛОГИРОВАНИЕ
# ══════════════════════════════════════════════════════════════

def setup_logging(log_level: str = "INFO",
                  log_to_file: bool = True) -> logging.Logger:
    """
    Настраивает логирование: консоль + ротируемый файл.

    Вызов в main.py — самая первая строка:
        logger = setup_logging()

    Формат: [2025-01-15 14:32:01] INFO    train.py:87 — Epoch 5/150 ...
    """
    os.makedirs(PATHS["logs_dir"], exist_ok=True)

    fmt     = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(filename)s:%(lineno)d — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root    = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Консоль
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # Файл с ротацией (макс 10 MB, 3 backup-файла)
    if log_to_file:
        fh = logging.handlers.RotatingFileHandler(
            PATHS["train_log"],
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return root


# ══════════════════════════════════════════════════════════════
# 2. МЕТРИКИ
# ══════════════════════════════════════════════════════════════

class MetricsTracker:
    """
    Накапливает и вычисляет метрики за эпоху.

    Использование в train.py (в train_one_epoch):

        tracker = MetricsTracker(n_classes=10)

        for X, y in loader:
            out    = model(X, node_feat, edge_idx)
            loss   = criterion(out["logits"], y)
            preds  = out["probs"].argmax(dim=-1)

            tracker.update(
                loss   = loss.item(),
                preds  = preds,
                targets= y,
                n      = len(y),
            )

        metrics = tracker.compute()
        # → {"loss": 0.42, "accuracy": 0.87, "f1_macro": 0.85, ...}

    Метрики вычисляются вручную (без sklearn) чтобы избежать
    зависимости и работать на GPU-тензорах напрямую.
    """

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.reset()

    def reset(self):
        self._loss_sum  = 0.0
        self._n_samples = 0
        self._n_steps   = 0
        # Confusion matrix: CM[pred, true] или CM[true, pred]
        # Используем CM[true, pred] — строка = истинный класс
        self._cm = torch.zeros(self.n_classes, self.n_classes, dtype=torch.long)

    def update(self,
               loss:    float,
               preds:   torch.Tensor,
               targets: torch.Tensor,
               n:       int) -> None:
        """
        loss:    скалярное значение loss текущего шага
        preds:   (B,) — предсказанные классы (argmax)
        targets: (B,) — истинные классы
        n:       размер батча
        """
        self._loss_sum  += loss * n
        self._n_samples += n
        self._n_steps   += 1

        # Обновляем confusion matrix
        preds_cpu   = preds.detach().cpu()
        targets_cpu = targets.detach().cpu()
        for t, p in zip(targets_cpu, preds_cpu):
            t_i = t.item()
            p_i = p.item()
            if 0 <= t_i < self.n_classes and 0 <= p_i < self.n_classes:
                self._cm[t_i, p_i] += 1

    def compute(self) -> Dict[str, float]:
        """
        Вычисляет все метрики из накопленной confusion matrix.

        Возвращает dict:
            loss        — среднее за эпоху
            accuracy    — глобальная точность
            f1_macro    — macro F1 (среднее по классам)
            f1_weighted — weighted F1 (по частоте классов)
            precision_macro
            recall_macro

        Математика (per-class):
            precision_c = TP_c / (TP_c + FP_c)
            recall_c    = TP_c / (TP_c + FN_c)
            f1_c        = 2 * P_c * R_c / (P_c + R_c)

        Macro: простое среднее по классам
        Weighted: среднее взвешенное по поддержке (n_c / N)
        """
        cm     = self._cm.float()
        n_cls  = self.n_classes
        eps    = 1e-8

        # TP[c] = cm[c, c]
        TP = cm.diag()                               # (n_cls,)
        # FP[c] = сумма столбца c - TP[c]
        FP = cm.sum(dim=0) - TP                      # (n_cls,)
        # FN[c] = сумма строки c - TP[c]
        FN = cm.sum(dim=1) - TP                      # (n_cls,)
        # Поддержка класса (истинных примеров)
        support = cm.sum(dim=1)                      # (n_cls,)

        precision = TP / (TP + FP + eps)             # (n_cls,)
        recall    = TP / (TP + FN + eps)             # (n_cls,)
        f1        = 2 * precision * recall / (precision + recall + eps)

        # Macro: среднее по классам
        valid     = support > 0
        p_macro   = precision[valid].mean().item()
        r_macro   = recall[valid].mean().item()
        f1_macro  = f1[valid].mean().item()

        # Weighted: взвешенное по поддержке
        total     = support.sum().clamp(min=1)
        f1_w      = (f1 * support / total).sum().item()

        # Accuracy: диагональ / всего
        accuracy  = (TP.sum() / total).item()
        avg_loss  = self._loss_sum / max(self._n_samples, 1)

        return {
            "loss":             avg_loss,
            "accuracy":         accuracy,
            "f1_macro":         f1_macro,
            "f1_weighted":      f1_w,
            "precision_macro":  p_macro,
            "recall_macro":     r_macro,
        }

    def get_confusion_matrix(self) -> torch.Tensor:
        """Возвращает CM (n_classes, n_classes) для визуализации."""
        return self._cm.clone()


def compute_top_k_accuracy(probs:   torch.Tensor,
                            targets: torch.Tensor,
                            k:       int = 3) -> float:
    """
    Top-K accuracy: считается попаданием если истинный класс
    находится среди топ-K предсказанных.

    Использование в validate_one_epoch:
        top3 = compute_top_k_accuracy(out["probs"], y_batch, k=3)
    """
    top_k   = probs.topk(k, dim=-1).indices   # (B, K)
    correct = top_k.eq(targets.unsqueeze(1).expand_as(top_k))
    return correct.any(dim=-1).float().mean().item()


# ══════════════════════════════════════════════════════════════
# 3. EARLY STOPPING
# ══════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Останавливает обучение если val_loss не улучшается.

    Использование в train.py (в конце каждой эпохи):

        stopper = EarlyStopping(patience=20, delta=1e-4)

        for epoch in range(max_epochs):
            val_metrics = validate(...)

            if stopper(val_metrics["loss"]):
                logger.info("Early stopping triggered")
                break

    Алгоритм:
        Если val_loss не улучшился на delta за patience эпох →
        устанавливает stopper.should_stop = True.
    """

    def __init__(self,
                 patience: int   = None,
                 delta:    float = None):
        self.patience   = patience or TRAIN_CFG.early_stopping_patience
        self.delta      = delta    or TRAIN_CFG.early_stopping_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Возвращает True если нужно остановить обучение.
        Вызывать каждую эпоху с текущим val_loss.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            logger.debug(
                "EarlyStopping: no improvement %d/%d",
                self.counter, self.patience
            )

        self.should_stop = self.counter >= self.patience
        return self.should_stop

    @property
    def best(self) -> float:
        return self.best_loss


# ══════════════════════════════════════════════════════════════
# 4. CHECKPOINT MANAGER
# ══════════════════════════════════════════════════════════════

class CheckpointManager:
    """
    Сохраняет и восстанавливает состояние обучения.

    Использование в train.py:

        ckpt_mgr = CheckpointManager()

        # Сохранить лучшую модель:
        if val_loss < best_val_loss:
            ckpt_mgr.save_best(model, optimizer, scheduler, epoch, metrics)

        # Сохранить периодический чекпоинт:
        if epoch % checkpoint_every == 0:
            ckpt_mgr.save_checkpoint(model, optimizer, scheduler, epoch, metrics)

        # Загрузить для продолжения обучения:
        start_epoch, metrics = ckpt_mgr.load(model, optimizer, scheduler)

    Структура файла чекпоинта:
        {
            "epoch":            int,
            "model_state":      OrderedDict,
            "optimizer_state":  dict,
            "scheduler_state":  dict,    # если есть
            "metrics":          dict,    # val_loss, f1 и т.д.
            "config":           dict,    # конфиг на момент сохранения
        }
    """

    def __init__(self,
                 best_path: str = None,
                 ckpt_dir:  str = None):
        self.best_path = best_path or PATHS["best_model"]
        self.ckpt_dir  = ckpt_dir  or PATHS["checkpoints_dir"]
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def save_best(self,
                  model:     nn.Module,
                  optimizer,
                  scheduler,
                  epoch:     int,
                  metrics:   Dict) -> None:
        """Сохраняет лучшую модель (по val_loss / val_f1)."""
        state = self._build_state(model, optimizer, scheduler, epoch, metrics)
        torch.save(state, self.best_path)
        logger.info("✓ Best model saved (epoch %d, val_loss=%.4f)",
                    epoch, metrics.get("loss", float("nan")))

    def save_checkpoint(self,
                        model:     nn.Module,
                        optimizer,
                        scheduler,
                        epoch:     int,
                        metrics:   Dict) -> None:
        """Периодическое сохранение (каждые N эпох)."""
        path  = os.path.join(self.ckpt_dir, f"checkpoint_epoch_{epoch:04d}.pt")
        state = self._build_state(model, optimizer, scheduler, epoch, metrics)
        torch.save(state, path)
        logger.info("Checkpoint saved → %s", path)

    def load(self,
             model:     nn.Module,
             optimizer  = None,
             scheduler  = None,
             path:      str = None) -> Tuple[int, Dict]:
        """
        Загружает чекпоинт и восстанавливает состояние.

        Возвращает:
            start_epoch: int  — с какой эпохи продолжать
            metrics:     dict — метрики на момент сохранения

        Использование:
            start_epoch, last_metrics = ckpt_mgr.load(model, optimizer, scheduler)
            for epoch in range(start_epoch, max_epochs):
                ...
        """
        path  = path or self.best_path
        state = torch.load(path, map_location="cpu", weights_only=False)

        model.load_state_dict(state["model_state"])
        if optimizer and "optimizer_state" in state:
            optimizer.load_state_dict(state["optimizer_state"])
        if scheduler and "scheduler_state" in state:
            scheduler.load_state_dict(state["scheduler_state"])

        epoch   = state.get("epoch", 0)
        metrics = state.get("metrics", {})
        logger.info("Checkpoint loaded ← %s (epoch %d)", path, epoch)
        return epoch + 1, metrics

    @staticmethod
    def _build_state(model, optimizer, scheduler, epoch, metrics) -> Dict:
        from config import get_config
        return {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "metrics":         metrics,
            "config":          get_config(),
        }


# ══════════════════════════════════════════════════════════════
# 5. CSV METRICS LOGGER
#    Записывает метрики каждой эпохи в CSV для анализа
# ══════════════════════════════════════════════════════════════

class CSVMetricsLogger:
    """
    Логирует train/val метрики каждой эпохи в CSV.

    Использование в train.py:
        csv_log = CSVMetricsLogger()
        ...
        csv_log.log(epoch, train_metrics, val_metrics, lr)

    Файл: logs/metrics.csv
    Заголовок: epoch, lr, train_loss, train_acc, val_loss, val_acc, val_f1_macro

    TODO: заменить на W&B / TensorBoard (см. комментарий ниже)
    """

    def __init__(self, path: str = None):
        self.path    = path or PATHS["metrics_csv"]
        self._header_written = False
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def log(self,
            epoch:        int,
            train_metrics: Dict,
            val_metrics:   Dict,
            lr:            float) -> None:

        row = {
            "epoch":           epoch,
            "lr":              f"{lr:.6f}",
            "train_loss":      f"{train_metrics.get('loss', 0):.4f}",
            "train_acc":       f"{train_metrics.get('accuracy', 0):.4f}",
            "val_loss":        f"{val_metrics.get('loss', 0):.4f}",
            "val_acc":         f"{val_metrics.get('accuracy', 0):.4f}",
            "val_f1_macro":    f"{val_metrics.get('f1_macro', 0):.4f}",
        }

        mode = "a" if self._header_written else "w"
        with open(self.path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


# ══════════════════════════════════════════════════════════════
# 6. ВИЗУАЛИЗАЦИЯ
# ══════════════════════════════════════════════════════════════

def plot_training_curves(metrics_csv: str = None,
                         save_dir:    str = None) -> None:
    """
    Строит графики loss и accuracy по данным CSV.

    Использование после обучения в main.py:
        plot_training_curves()

    Сохраняет:
        plots/training_curves.png
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # без GUI
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        logger.warning("matplotlib/pandas не установлены — пропускаем графики")
        return

    path     = metrics_csv or PATHS["metrics_csv"]
    save_dir = save_dir    or PATHS["plots_dir"]
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(path):
        logger.warning("metrics.csv не найден: %s", path)
        return

    df = pd.read_csv(path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("CyberThreat AI — Training Curves", fontsize=14)

    # Loss
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", color="#2196F3")
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val Loss",   color="#F44336")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(df["epoch"], df["train_acc"], label="Train Acc", color="#2196F3")
    axes[1].plot(df["epoch"], df["val_acc"],   label="Val Acc",   color="#F44336")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # F1 + LR
    ax_f1 = axes[2]
    ax_lr = ax_f1.twinx()
    ax_f1.plot(df["epoch"], df["val_f1_macro"], label="Val F1 Macro",
               color="#4CAF50", linewidth=2)
    ax_lr.plot(df["epoch"], df["lr"].astype(float), label="LR",
               color="#FF9800", linestyle="--", alpha=0.6)
    ax_f1.set_title("Val F1 & Learning Rate")
    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("F1")
    ax_lr.set_ylabel("LR")
    ax_f1.grid(alpha=0.3)
    ax_f1.legend(loc="upper left")
    ax_lr.legend(loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Training curves → %s", out_path)


def plot_confusion_matrix(cm:          torch.Tensor,
                          class_names: List[str] = None,
                          save_dir:    str = None) -> None:
    """
    Рисует confusion matrix.

    Использование после теста в main.py:
        cm = test_tracker.get_confusion_matrix()
        plot_confusion_matrix(cm, graph_data["class_names"])

    Аргументы:
        cm:          Tensor (n_classes, n_classes)
        class_names: список имён классов
        save_dir:    куда сохранить PNG
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn не установлены")
        return

    save_dir    = save_dir    or PATHS["plots_dir"]
    class_names = class_names or DATA_CFG.threat_classes
    os.makedirs(save_dir, exist_ok=True)

    cm_np = cm.numpy()
    # Нормализуем по строкам (по истинным классам)
    row_sums = cm_np.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm  = cm_np / row_sums

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("CyberThreat AI — Confusion Matrix", fontsize=14)

    # Абсолютные значения
    sns.heatmap(cm_np, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0])
    axes[0].set_title("Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", fontsize=8)

    # Нормализованные (recall per class)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1])
    axes[1].set_title("Normalized (recall)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix → %s", out_path)


def plot_semantic_heatmap(sim_matrix:  torch.Tensor,
                          class_names: List[str] = None,
                          save_dir:    str = None) -> None:
    """
    Визуализирует матрицу семантического сходства из Knowledge Graph.

    Показывает насколько семантически близки классы угроз —
    это помогает понять какие рёбра граф будет строить автоматически.

    Использование в main.py:
        plot_semantic_heatmap(graph_data["sim_matrix"],
                              graph_data["class_names"])
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn не установлены")
        return

    save_dir    = save_dir    or PATHS["plots_dir"]
    class_names = class_names or DATA_CFG.threat_classes
    os.makedirs(save_dir, exist_ok=True)

    sim_np = sim_matrix.numpy()

    plt.figure(figsize=(10, 8))
    mask = np.eye(len(class_names), dtype=bool)   # скрываем диагональ (=1)
    sns.heatmap(
        sim_np,
        annot=True, fmt=".2f",
        cmap="RdYlGn", center=0,
        xticklabels=class_names,
        yticklabels=class_names,
        mask=mask,
        vmin=-1, vmax=1,
    )
    plt.title("Semantic Similarity Matrix — Knowledge Graph Nodes", fontsize=13)
    plt.xlabel("Threat Class")
    plt.ylabel("Threat Class")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    out_path = os.path.join(save_dir, "semantic_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Semantic heatmap → %s", out_path)


# ══════════════════════════════════════════════════════════════
# 7. ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ
# ══════════════════════════════════════════════════════════════

def set_seed(seed: int = None) -> None:
    """
    Фиксирует все источники случайности для воспроизводимости.
    Вызывать в main.py до создания любых объектов.

    Использование:
        from utils import set_seed
        set_seed(42)
    """
    import random
    seed = seed or TRAIN_CFG.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Детерминированные операции CUDA (может замедлить обучение)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def get_device() -> torch.device:
    """
    Автоматически выбирает лучшее доступное устройство.

    Порядок приоритета: CUDA → MPS (Apple Silicon) → CPU

    Использование:
        device = get_device()
        model  = model.to(device)
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info("Device: CUDA (%s)", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        logger.info("Device: MPS (Apple Silicon)")
    else:
        dev = torch.device("cpu")
        logger.info("Device: CPU")
    return dev


def format_metrics(metrics: Dict, prefix: str = "") -> str:
    """
    Форматирует словарь метрик в читаемую строку для логов.

    Использование:
        logger.info(format_metrics(val_metrics, prefix="Val"))
        # → "Val | loss=0.2341 | acc=0.8712 | f1=0.8650"
    """
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    line = " | ".join(parts)
    return f"{prefix} | {line}" if prefix else line


def count_model_params(model: nn.Module) -> Dict[str, int]:
    """
    Считает параметры модели.

    Возвращает:
        {"total": N, "trainable": M, "frozen": N-M}
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total":     total,
        "trainable": trainable,
        "frozen":    total - trainable,
    }


# ══════════════════════════════════════════════════════════════
# TODO — ЧТО ОСТАВЛЕНО ТЕБЕ
# ══════════════════════════════════════════════════════════════
#
# 1. WEIGHTS & BIASES:
#    import wandb
#    wandb.init(project="cyberthreat-ai", config=get_config())
#    # В конце каждой эпохи:
#    wandb.log({"val_loss": val_loss, "val_f1": val_f1, "epoch": epoch})
#
# 2. TENSORBOARD:
#    from torch.utils.tensorboard import SummaryWriter
#    writer = SummaryWriter(log_dir=PATHS["logs_dir"])
#    writer.add_scalars("loss", {"train": t_loss, "val": v_loss}, epoch)
#    writer.add_figure("confusion_matrix", fig, epoch)
#
# 3. ATTENTION HEATMAP (XAI):
#    def plot_attention_heatmap(attn, class_names, incident_idx):
#        # attn: (B, n_classes) — из model.get_attention_weights()
#        # Показывает на какие классы граф "смотрит" для каждого инцидента
#        ...
#
# 4. EMBEDDING VISUALIZATION (t-SNE / UMAP):
#    from sklearn.manifold import TSNE
#    tsne   = TSNE(n_components=2)
#    emb_2d = tsne.fit_transform(incident_embs.numpy())
#    # scatter по цветам классов
#
# ══════════════════════════════════════════════════════════════