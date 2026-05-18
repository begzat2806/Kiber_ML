"""
config.py
─────────────────────────────────────────────────────────
РОЛЬ В ПРОЕКТЕ:
    Единственный источник правды (Single Source of Truth).
    Все модули импортируют конфиги отсюда — никаких
    магических чисел в других файлах.

ЗАВИСИМОСТИ: нет (независимый модуль)

ЭКСПОРТИРУЕТ:
    DATA_CFG      → dataset.py
    MODEL_CFG     → model.py
    TRAIN_CFG     → train.py
    INFERENCE_CFG → inference.py
    PATHS         → все модули
    get_config()  → main.py (для логирования)

КАК ПОДКЛЮЧИТЬ:
    from config import DATA_CFG, MODEL_CFG, TRAIN_CFG, PATHS

ЧТО ОСТАВЛЕНО ТЕБЕ:
    - Добавить YAML-парсер (см. TODO ниже)
    - Настроить augmentation-параметры
    - Добавить distributed training конфиг
─────────────────────────────────────────────────────────
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict


# ══════════════════════════════════════════════════════
# ПУТИ ПРОЕКТА
# ══════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATHS: Dict[str, str] = {
    "data_dir":        os.path.join(BASE_DIR, "data"),
    "checkpoints_dir": os.path.join(BASE_DIR, "checkpoints"),
    "logs_dir":        os.path.join(BASE_DIR, "logs"),
    "plots_dir":       os.path.join(BASE_DIR, "plots"),
    # ── конкретные файлы ──
    "graph_data":      os.path.join(BASE_DIR, "data",        "threat_graph.pt"),
    "best_model":      os.path.join(BASE_DIR, "checkpoints", "best_model.pt"),
    "last_checkpoint": os.path.join(BASE_DIR, "checkpoints", "last_checkpoint.pt"),
    "train_log":       os.path.join(BASE_DIR, "logs",        "train.log"),
    "metrics_csv":     os.path.join(BASE_DIR, "logs",        "metrics.csv"),
}


# ══════════════════════════════════════════════════════
# КОНФИГ ДАННЫХ
# ══════════════════════════════════════════════════════

@dataclass
class DataConfig:
    """
    Параметры датасета и Knowledge Graph.

    ПЕРЕДАЁТСЯ В:
        dataset.py → ThreatDataGenerator
        dataset.py → create_dataloaders()
    """

    # ── классы угроз (MITRE ATT&CK inspired) ──────────
    threat_classes: List[str] = field(default_factory=lambda: [
        "Ransomware",
        "SQL_Injection",
        "Phishing",
        "DDoS",
        "Zero_Day_Exploit",
        "APT",
        "Botnet",
        "Man_in_Middle",
        "Insider_Threat",
        "Supply_Chain",
    ])

    # ── размерности ───────────────────────────────────
    num_classes:      int = 10     # len(threat_classes)
    node_feature_dim: int = 128    # размерность вектора узла графа
    num_samples:      int = 5000   # всего инцидентов в датасете

    # ── сплит ─────────────────────────────────────────
    train_ratio: float = 0.70
    val_ratio:   float = 0.15
    test_ratio:  float = 0.15      # остаток; train+val+test == 1.0

    # ── типы рёбер Knowledge Graph ────────────────────
    edge_types: List[str] = field(default_factory=lambda: [
        "exploits",      # A использует уязвимость B
        "precedes",      # A предшествует B в kill chain
        "targets",       # A и B нацелены на один актив
        "mitigated_by",  # A и B снижаются одной мерой
        "similar_to",    # семантическое сходство (auto)
    ])

    # ── порог семантического сходства (cosine sim) ────
    # выше порога → автоматически добавляется ребро similar_to
    semantic_threshold: float = 0.45

    # ── аугментация обучающей выборки ─────────────────
    aug_noise_std:    float = 0.03   # гауссов шум
    aug_dropout_prob: float = 0.05   # feature dropout


# ══════════════════════════════════════════════════════
# КОНФИГ МОДЕЛИ
# ══════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """
    Гиперпараметры архитектуры Graph Attention Network.

    ПЕРЕДАЁТСЯ В:
        model.py → build_model()
        model.py → CyberThreatGAT.__init__()

    МАТЕМАТИКА:
        hidden_dim должен делиться на num_heads без остатка:
            head_dim = hidden_dim // num_heads
        Финальный размер после конкатенации:
            concat_dim = head_dim * num_heads = hidden_dim
    """

    # ── GAT ───────────────────────────────────────────
    num_gat_layers: int   = 3      # глубина GAT (рецептивное поле = 3-hop)
    hidden_dim:     int   = 256    # размерность скрытых состояний
    num_heads:      int   = 8      # количество голов внимания (multi-head)
    dropout:        float = 0.40   # dropout в GAT и MLP
    edge_dropout:   float = 0.15   # DropEdge (рандомное удаление рёбер)

    # ── нормализация ──────────────────────────────────
    norm_type:    str  = "layer"   # "layer" | "batch"
    use_residual: bool = True      # residual connections между GAT-слоями

    # ── MLP-классификатор (после агрегации графа) ─────
    # вход MLP = 2 * hidden_dim (инцидент + граф-контекст)
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])

    # ── инициализация весов ───────────────────────────
    weight_init: str = "xavier"    # "xavier" | "kaiming" | "orthogonal"


# ══════════════════════════════════════════════════════
# КОНФИГ ОБУЧЕНИЯ
# ══════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """
    Параметры training loop.

    ПЕРЕДАЁТСЯ В:
        train.py → Trainer.__init__()
        train.py → build_optimizer()
        train.py → build_scheduler()
    """

    # ── основные ──────────────────────────────────────
    max_epochs:  int   = 150
    batch_size:  int   = 64
    seed:        int   = 42

    # ── оптимизатор ───────────────────────────────────
    optimizer:     str   = "adamw"   # "adam" | "adamw" | "sgd"
    learning_rate: float = 2e-4
    weight_decay:  float = 3e-4      # L2 регуляризация
    grad_clip:     float = 1.0       # gradient clipping (max norm)

    # ── scheduler ─────────────────────────────────────
    scheduler:   str   = "cosine"   # "cosine" | "step" | "plateau"
    cosine_t_max: int  = 150        # T_max для CosineAnnealingLR
    min_lr:      float = 1e-6       # нижний порог LR

    # ── mixed precision (AMP) ─────────────────────────
    use_amp: bool = True            # torch.cuda.amp.GradScaler

    # ── early stopping ────────────────────────────────
    early_stopping_patience: int   = 40
    early_stopping_delta:    float = 1e-4   # мин. улучшение val_loss

    # ── loss function ─────────────────────────────────
    label_smoothing:  float = 0.15   # сглаживание меток (anti-overfit)
    use_class_weights: bool = True   # взвешенный CE для дисбаланса

    # ── логирование и чекпоинты ───────────────────────
    checkpoint_every:  int = 10     # сохранять каждые N эпох
    log_every_n_steps: int = 5      # логировать каждые N шагов


# ══════════════════════════════════════════════════════
# КОНФИГ ИНФЕРЕНСА
# ══════════════════════════════════════════════════════

@dataclass
class InferenceConfig:
    """
    Параметры inference pipeline.

    ПЕРЕДАЁТСЯ В:
        inference.py → ThreatAnalyzer.__init__()
    """

    # минимальная уверенность модели для вывода предупреждения
    confidence_threshold: float = 0.70

    # топ-K классов в выводе
    top_k: int = 3

    # включить объяснимость через attention weights
    explain_predictions: bool = True

    # пороги тяжести угрозы по уверенности модели
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "critical": 0.85,
        "high":     0.65,
        "medium":   0.40,
        "low":      0.00,
    })


# ══════════════════════════════════════════════════════
# СИНГЛТОНЫ — импортируй эти объекты в других модулях
# ══════════════════════════════════════════════════════

DATA_CFG      = DataConfig()
MODEL_CFG     = ModelConfig()
TRAIN_CFG     = TrainConfig()
INFERENCE_CFG = InferenceConfig()


# ══════════════════════════════════════════════════════
# УТИЛИТЫ КОНФИГА
# ══════════════════════════════════════════════════════

def get_config() -> Dict:
    """
    Возвращает все конфиги как словарь.
    Используется в main.py для логирования эксперимента.

    Пример:
        cfg = get_config()
        logger.info(json.dumps(cfg, indent=2, default=str))
    """
    return {
        "data":      DATA_CFG.__dict__.copy(),
        "model":     MODEL_CFG.__dict__.copy(),
        "train":     TRAIN_CFG.__dict__.copy(),
        "inference": INFERENCE_CFG.__dict__.copy(),
        "paths":     PATHS,
    }


def validate_config() -> None:
    """
    Проверяет консистентность конфига.
    Вызови в main.py перед стартом обучения.

    Пример:
        from config import validate_config
        validate_config()  # бросит AssertionError при проблеме
    """
    assert DATA_CFG.num_classes == len(DATA_CFG.threat_classes), \
        "num_classes не совпадает с len(threat_classes)"

    assert abs(DATA_CFG.train_ratio + DATA_CFG.val_ratio + DATA_CFG.test_ratio - 1.0) < 1e-6, \
        "Сумма train/val/test ratio должна быть 1.0"

    assert MODEL_CFG.hidden_dim % MODEL_CFG.num_heads == 0, \
        f"hidden_dim ({MODEL_CFG.hidden_dim}) должен делиться на num_heads ({MODEL_CFG.num_heads})"

    assert MODEL_CFG.num_gat_layers >= 1, \
        "num_gat_layers должен быть >= 1"

    print("[config] Validation passed ✓")


# ══════════════════════════════════════════════════════
# TODO — ЧТО ОСТАВЛЕНО ТЕБЕ
# ══════════════════════════════════════════════════════
#
# 1. YAML-конфиг (опционально, рекомендуется для production):
#
#    import yaml
#    def load_from_yaml(path: str) -> None:
#        with open(path) as f:
#            raw = yaml.safe_load(f)
#        # перезаписать поля DATA_CFG, MODEL_CFG и т.д.
#        for k, v in raw.get("model", {}).items():
#            setattr(MODEL_CFG, k, v)
#
# 2. CLI override (argparse поверх dataclass):
#
#    parser.add_argument("--lr", type=float)
#    args = parser.parse_args()
#    if args.lr:
#        TRAIN_CFG.learning_rate = args.lr
#
# 3. Distributed training config (если нужен multi-GPU):
#
#    @dataclass
#    class DistConfig:
#        backend: str = "nccl"
#        world_size: int = 1
#        local_rank: int = 0
#
# ══════════════════════════════════════════════════════