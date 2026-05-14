"""
dataset.py
─────────────────────────────────────────────────────────
РОЛЬ В ПРОЕКТЕ:
    1. Генерирует / загружает данные об инцидентах
    2. Строит Knowledge Graph (семантическая сеть угроз)
    3. Реализует PyTorch Dataset и DataLoader

ЗАВИСИМОСТИ:
    config.py → DATA_CFG, TRAIN_CFG, PATHS

ЭКСПОРТИРУЕТ:
    ThreatDataGenerator   → для первичной генерации графа
    ThreatIncidentDataset → PyTorch Dataset (используется в DataLoader)
    create_dataloaders()  → train.py
    get_or_generate_graph_data() → main.py / train.py

ДАННЫЕ МЕЖДУ МОДУЛЯМИ:
    ┌─────────────────────────────────────────────────────┐
    │  graph_data (dict) — центральная структура данных   │
    │                                                     │
    │  "node_features"  Tensor (n_classes, 128)           │
    │       └─→ model.py :: GATEncoder.forward()          │
    │                                                     │
    │  "edge_index"     Tensor (2, E)                     │
    │       └─→ model.py :: GATEncoder.forward()          │
    │                                                     │
    │  "edge_type"      Tensor (E,)                       │
    │       └─→ опционально в model.py (RGAT)             │
    │                                                     │
    │  "incident_X"     Tensor (N_samples, 128)           │
    │  "incident_y"     Tensor (N_samples,)               │
    │       └─→ ThreatIncidentDataset → DataLoader        │
    │                                                     │
    │  "sim_matrix"     Tensor (n_classes, n_classes)     │
    │       └─→ utils.py :: plot_semantic_graph()         │
    └─────────────────────────────────────────────────────┘

СЕМАНТИЧЕСКАЯ СЕТЬ:
    Knowledge Graph строится из двух источников:
    1. Экспертные правила (MITRE ATT&CK kill chain)
    2. Автоматические рёбра по косинусному сходству:
           sim(i,j) = (x_i · x_j) / (‖x_i‖ · ‖x_j‖)
       если sim > threshold → добавляем ребро "similar_to"

ЧТО ОСТАВЛЕНО ТЕБЕ:
    - Заменить ThreatDataGenerator реальным ETL (см. TODO)
    - Реализовать стратифицированный сплит
    - Добавить Mixup аугментацию
    - Подключить torch_geometric.data.Data вместо dict
─────────────────────────────────────────────────────────
"""

import os
import logging
import random
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from config import DATA_CFG, TRAIN_CFG, PATHS

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 1. ГЕНЕРАТОР ДАННЫХ
#    В реальном проекте этот класс заменяется на ETL-пайплайн.
#    Интерфейс сохраняется: метод build_full_graph_data() → dict
# ══════════════════════════════════════════════════════════════

class ThreatDataGenerator:
    """
    Генерирует синтетические данные об угрозах + Knowledge Graph.

    В production замени тело методов на реальные источники:
        - SIEM API (Splunk / QRadar)
        - CVE/NVD REST API
        - MITRE ATT&CK STIX/TAXII feed
        - VirusTotal / AlienVault OTX

    Интерфейс остаётся тем же: build_full_graph_data() → dict.
    Это позволяет подменить генератор без изменения train.py / model.py.

    Семантика узлов:
        Каждый из 10 классов угроз — это узел графа.
        Его признаковый вектор x ∈ R^128 — "образ" этой угрозы
        в семантическом пространстве (имитация BERT-эмбеддинга).
    """

    # ── Экспертные семантические связи (MITRE kill chain) ──
    # Источник 1: заданы вручную на основе знаний аналитиков
    # Формат: (src_class, dst_class, relation_type)
    EXPERT_RELATIONS: List[Tuple[str, str, str]] = [
        ("Phishing",         "APT",            "precedes"),
        ("Phishing",         "Ransomware",      "precedes"),
        ("Phishing",         "Insider_Threat",  "precedes"),
        ("Botnet",           "DDoS",            "exploits"),
        ("Botnet",           "Ransomware",      "exploits"),
        ("Zero_Day_Exploit", "APT",             "precedes"),
        ("Zero_Day_Exploit", "Ransomware",      "precedes"),
        ("SQL_Injection",    "APT",             "targets"),
        ("Supply_Chain",     "APT",             "precedes"),
        ("Supply_Chain",     "Ransomware",      "precedes"),
        ("APT",              "Insider_Threat",  "targets"),
        ("Man_in_Middle",    "SQL_Injection",   "similar_to"),
        ("Ransomware",       "APT",             "similar_to"),
        ("DDoS",             "Botnet",          "mitigated_by"),
    ]

    def __init__(self, n_samples: int = None, seed: int = None):
        self.n_samples = n_samples or DATA_CFG.num_samples
        self.seed      = seed      or TRAIN_CFG.seed
        self.classes   = DATA_CFG.threat_classes
        self.n_classes = DATA_CFG.num_classes

        random.seed(self.seed)
        torch.manual_seed(self.seed)

    # ──────────────────────────────────────────────────────
    # Шаг A: признаковые векторы узлов (node features)
    # ──────────────────────────────────────────────────────

    def _generate_node_features(self) -> torch.Tensor:
        """
        Создаёт матрицу признаков X ∈ R^(n_classes × dim).

        Имитирует BERT-эмбеддинги описаний угроз из MITRE ATT&CK.
        Семантически близкие угрозы → близкие векторы.

        Возвращает: Tensor (n_classes, node_feature_dim), L2-нормализован.

        TODO (real project):
            Замени на реальные BERT-эмбеддинги:

            from transformers import AutoTokenizer, AutoModel

            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            bert      = AutoModel.from_pretrained("bert-base-uncased")

            descriptions = [THREAT_DESCRIPTIONS[c] for c in self.classes]
            tokens = tokenizer(descriptions, return_tensors="pt",
                               padding=True, truncation=True)
            with torch.no_grad():
                out = bert(**tokens)
            # CLS-токен как эмбеддинг
            embeddings = out.last_hidden_state[:, 0, :]  # (n_classes, 768)
            # Проецируй в нужную размерность через nn.Linear(768, 128)
        """
        dim = DATA_CFG.node_feature_dim
        n   = self.n_classes

        proto = torch.randn(n, dim)

        # Семантические группы: близкие угрозы получают похожие векторы
        groups = [
            # сетевые атаки
            [self.classes.index(c) for c in ["DDoS", "Botnet", "Man_in_Middle"]],
            # целевые / продвинутые
            [self.classes.index(c) for c in ["APT", "Insider_Threat", "Supply_Chain"]],
            # вредоносный код
            [self.classes.index(c) for c in ["Ransomware", "Zero_Day_Exploit"]],
        ]

        for group in groups:
            base = torch.randn(dim)
            for idx in group:
                proto[idx] = base + 0.35 * torch.randn(dim)

        # L2-нормализация → все векторы на единичной сфере
        # Это делает косинусное сходство = скалярному произведению
        return F.normalize(proto, dim=1)   # (n_classes, 128)

    # ──────────────────────────────────────────────────────
    # Шаг B: матрица семантического сходства
    # ──────────────────────────────────────────────────────

    def _cosine_similarity_matrix(self,
                                  node_features: torch.Tensor) -> torch.Tensor:
        """
        sim(i,j) = x_i · x_j^T   (т.к. векторы нормализованы)

        Возвращает: Tensor (n_classes, n_classes) ∈ [-1, 1]

        Используется:
            1. Для построения автоматических рёбер графа
            2. В utils.py для визуализации семантической матрицы
        """
        return torch.mm(node_features, node_features.t())   # (C, C)

    # ──────────────────────────────────────────────────────
    # Шаг C: рёбра Knowledge Graph
    # ──────────────────────────────────────────────────────

    def _build_edges(self,
                     node_features: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Строит edge_index (COO-формат) и edge_type.

        Источник 1 — EXPERT_RELATIONS:
            Жёстко заданные смысловые связи.
            Приоритет над автоматическими рёбрами.

        Источник 2 — cosine similarity > threshold:
            Автоматически находим семантически близкие пары.
            Отношение: "similar_to".

        Возвращает:
            edge_index: Tensor (2, E)  — [src_nodes; dst_nodes]
            edge_type:  Tensor (E,)    — числовой тип ребра

        КАК ИСПОЛЬЗОВАТЬ В model.py:
            # стандартный GAT — edge_type не нужен
            out = gat_encoder(node_features, edge_index)

            # RGAT (relation-aware) — передаёт edge_type
            out = rgat_encoder(node_features, edge_index, edge_type)
        """
        type_map = {t: i for i, t in enumerate(DATA_CFG.edge_types)}
        edge_set  = {}   # (src, dst) → type_id — дедупликация

        # ── Источник 1: экспертные ──────────────────────
        for (src_name, dst_name, rel) in self.EXPERT_RELATIONS:
            if src_name in self.classes and dst_name in self.classes:
                s = self.classes.index(src_name)
                d = self.classes.index(dst_name)
                edge_set[(s, d)] = type_map.get(rel, 0)

        # ── Источник 2: автоматические по сходству ──────
        sim   = self._cosine_similarity_matrix(node_features)
        thr   = DATA_CFG.semantic_threshold
        n     = self.n_classes
        auto_type = type_map["similar_to"]

        for i in range(n):
            for j in range(n):
                if i != j and sim[i, j].item() > thr:
                    if (i, j) not in edge_set:
                        edge_set[(i, j)] = auto_type

        if not edge_set:
            # Граф не может быть пустым
            raise RuntimeError("Knowledge Graph пуст — снизь semantic_threshold")

        edges = list(edge_set.keys())
        types = [edge_set[e] for e in edges]

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, E)
        edge_type  = torch.tensor(types, dtype=torch.long)                   # (E,)

        logger.info(
            "Knowledge Graph: %d nodes, %d edges (%d expert + %d semantic)",
            n,
            len(edges),
            len(self.EXPERT_RELATIONS),
            len(edges) - len(self.EXPERT_RELATIONS),
        )
        return edge_index, edge_type

    # ──────────────────────────────────────────────────────
    # Шаг D: синтетические инциденты
    # ──────────────────────────────────────────────────────

    def _generate_incidents(self,
                            class_prototypes: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Генерирует обучающие примеры инцидентов.

        Каждый инцидент — шум вокруг прототипа своего класса:
            x_i = proto[c] + ε,   ε ~ N(0, σ²I)

        Дисбаланс классов намеренно введён (как в реальных SIEM-данных):
            Phishing / SQL_Injection — самые частые
            APT / Zero_Day_Exploit  — редкие

        TODO (real project):
            Замени на реальный ETL:
            df = pd.read_csv("siem_logs.csv")
            X  = feature_extractor.transform(df)
            y  = label_encoder.transform(df["label"])
        """
        # Реалистичный дисбаланс классов
        class_frequencies = {
            "Ransomware":       0.15,
            "SQL_Injection":    0.18,
            "Phishing":         0.20,
            "DDoS":             0.12,
            "Zero_Day_Exploit": 0.05,
            "APT":              0.04,
            "Botnet":           0.10,
            "Man_in_Middle":    0.06,
            "Insider_Threat":   0.05,
            "Supply_Chain":     0.05,
        }

        samples: List[torch.Tensor] = []
        labels:  List[int]          = []

        for cls_idx, cls_name in enumerate(self.classes):
            freq     = class_frequencies.get(cls_name, 0.1)
            n_cls    = max(1, int(self.n_samples * freq))
            proto    = class_prototypes[cls_idx]

            for _ in range(n_cls):
                # Разный уровень шума — некоторые атаки "чище" других
                sigma  = 0.15 + 0.15 * random.random()
                sample = proto + sigma * torch.randn_like(proto)
                samples.append(sample)
                labels.append(cls_idx)

        # Перемешиваем перед возвратом
        indices = list(range(len(samples)))
        random.shuffle(indices)

        X = torch.stack([samples[i] for i in indices])          # (N, 128)
        y = torch.tensor([labels[i]  for i in indices],
                         dtype=torch.long)                       # (N,)

        logger.info("Generated %d incident samples", len(X))
        return X, y

    # ──────────────────────────────────────────────────────
    # Главный метод: собирает всё вместе
    # ──────────────────────────────────────────────────────

    def build_full_graph_data(self) -> Dict:
        """
        Точка входа: строит полный граф-датасет.

        Возвращает dict graph_data — его ключи используются в:
            model.py   → "node_features", "edge_index"
            dataset.py → "incident_X", "incident_y"
            utils.py   → "sim_matrix", "class_names"
            train.py   → передаётся в Trainer

        Структура:
            graph_data = {
                "node_features": Tensor (n_classes, dim),
                "edge_index":    Tensor (2, E),
                "edge_type":     Tensor (E,),
                "incident_X":    Tensor (N, dim),
                "incident_y":    Tensor (N,),
                "sim_matrix":    Tensor (n_classes, n_classes),
                "class_names":   List[str],
                "n_classes":     int,
            }
        """
        node_features           = self._generate_node_features()
        edge_index, edge_type   = self._build_edges(node_features)
        incident_X, incident_y  = self._generate_incidents(node_features)
        sim_matrix              = self._cosine_similarity_matrix(node_features)

        return {
            "node_features": node_features,
            "edge_index":    edge_index,
            "edge_type":     edge_type,
            "incident_X":    incident_X,
            "incident_y":    incident_y,
            "sim_matrix":    sim_matrix,
            "class_names":   self.classes,
            "n_classes":     self.n_classes,
        }


# ══════════════════════════════════════════════════════════════
# 2. PYTORCH DATASET
# ══════════════════════════════════════════════════════════════

class ThreatIncidentDataset(Dataset):
    """
    PyTorch Dataset для инцидентов кибербезопасности.

    Используется в:
        create_dataloaders() → возвращает DataLoader'ы для train.py

    __getitem__ возвращает: (x: FloatTensor, y: LongTensor)
        x — признаки инцидента (dim,)
        y — класс угрозы (скаляр)

    ВАЖНО:
        Модель дополнительно принимает node_features и edge_index
        (статические, не входят в батч DataLoader'а).
        В train.py они передаются отдельно из graph_data.
    """

    def __init__(self,
                 X:         torch.Tensor,
                 y:         torch.Tensor,
                 transform: Optional[callable] = None):
        assert len(X) == len(y), "X и y должны иметь одинаковый первый размер"
        self.X         = X.float()
        self.y         = y.long()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()
        y = self.y[idx].clone()
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def get_class_weights(self) -> torch.Tensor:
        """
        Веса классов для взвешенного CrossEntropyLoss.

        Формула: w_c = N_total / (C * N_c)
            N_total — всего примеров
            C       — число классов
            N_c     — примеров класса c

        Используется в train.py:
            weights = dataset.get_class_weights().to(device)
            criterion = nn.CrossEntropyLoss(weight=weights, ...)
        """
        n_classes = DATA_CFG.num_classes
        counts    = torch.zeros(n_classes)
        for label in self.y:
            counts[label.item()] += 1
        counts  = counts.clamp(min=1)
        total   = float(len(self.y))
        weights = total / (n_classes * counts)
        return weights    # Tensor (n_classes,)


# ══════════════════════════════════════════════════════════════
# 3. АУГМЕНТАЦИЯ
#    Применяется только к train_loader через _AugmentedSubset
# ══════════════════════════════════════════════════════════════

class ThreatAugmentation:
    """
    Аугментации признакового пространства для train-выборки.

    Две техники (применяются вместе):
        1. Gaussian noise:   x' = x + ε,  ε ~ N(0, σ²)
           Имитирует шум в сетевых логах / сенсорах.

        2. Feature dropout:  случайно обнуляем p% признаков
           Имитирует пропущенные данные (partial observability).

    TODO — добавить Mixup (более сильная регуляризация):
        λ ~ Beta(α, α)
        x_mix = λ * x_i + (1-λ) * x_j
        y_mix = λ * y_i + (1-λ) * y_j   # soft labels
    """

    def __init__(self,
                 noise_std:    float = None,
                 dropout_prob: float = None):
        self.noise_std    = noise_std    or DATA_CFG.aug_noise_std
        self.dropout_prob = dropout_prob or DATA_CFG.aug_dropout_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Gaussian noise
        if self.noise_std > 0:
            x = x + self.noise_std * torch.randn_like(x)

        # 2. Feature dropout (Bernoulli mask)
        if self.dropout_prob > 0:
            mask = torch.bernoulli(
                torch.full_like(x, 1.0 - self.dropout_prob)
            )
            x = x * mask

        return x


class _AugmentedSubset(Dataset):
    """
    Обёртка: применяет аугментацию к подмножеству (Subset) датасета.
    Нужна потому что random_split возвращает Subset, а не Dataset —
    нельзя назначить transform напрямую.
    """

    def __init__(self, subset, transform: callable):
        self.subset    = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.subset[idx]
        return self.transform(x), y


# ══════════════════════════════════════════════════════════════
# 4. ФАБРИКА DATALOADERS
#    Главная функция, которую вызывает train.py
# ══════════════════════════════════════════════════════════════

def create_dataloaders(
        graph_data:    Dict,
        batch_size:    int  = None,
        num_workers:   int  = 0,
        augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Создаёт train / val / test DataLoader'ы.

    Аргументы:
        graph_data:    dict из ThreatDataGenerator.build_full_graph_data()
        batch_size:    из TRAIN_CFG.batch_size (если не указан явно)
        num_workers:   параллельные воркеры (0 = в основном процессе)
        augment_train: применять аугментацию к train

    Возвращает:
        train_loader, val_loader, test_loader

    Использование в train.py:
        train_loader, val_loader, test_loader = create_dataloaders(graph_data)

    ВАЖНО — что НЕ входит в батч DataLoader'а:
        node_features и edge_index — статические данные графа.
        Их нужно передавать в model.forward() отдельно каждый шаг:

        for X_batch, y_batch in train_loader:
            out = model(
                incident_x    = X_batch,
                node_features = graph_data["node_features"].to(device),
                edge_index    = graph_data["edge_index"].to(device),
            )

    TODO:
        Заменить random_split на стратифицированный сплит:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15)
        train_idx, test_idx = next(sss.split(X, y))
    """
    batch_size = batch_size or TRAIN_CFG.batch_size

    X = graph_data["incident_X"]   # (N, 128)
    y = graph_data["incident_y"]   # (N,)

    N      = len(X)
    n_train = int(N * DATA_CFG.train_ratio)
    n_val   = int(N * DATA_CFG.val_ratio)
    n_test  = N - n_train - n_val

    full_ds = ThreatIncidentDataset(X, y)

    # Воспроизводимый сплит
    generator = torch.Generator().manual_seed(TRAIN_CFG.seed)
    train_sub, val_sub, test_sub = random_split(
        full_ds, [n_train, n_val, n_test], generator=generator
    )

    # Аугментация только для train
    train_ds = _AugmentedSubset(train_sub, ThreatAugmentation()) \
               if augment_train else train_sub

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,      # стабильный размер батча для BatchNorm
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_sub,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_sub,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    logger.info(
        "DataLoaders: train=%d | val=%d | test=%d",
        n_train, n_val, n_test
    )
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════
# 5. КЭШИРОВАНИЕ ДАННЫХ
#    Граф генерируется один раз и сохраняется на диск
# ══════════════════════════════════════════════════════════════

def save_graph_data(graph_data: Dict, path: str = None) -> None:
    """Сохраняет graph_data на диск через torch.save."""
    path = path or PATHS["graph_data"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(graph_data, path)
    logger.info("Graph data saved → %s", path)


def load_graph_data(path: str = None) -> Dict:
    """Загружает graph_data с диска."""
    path = path or PATHS["graph_data"]
    data = torch.load(path, map_location="cpu", weights_only=False)
    logger.info("Graph data loaded ← %s", path)
    return data


def get_or_generate_graph_data(force: bool = False) -> Dict:
    """
    Загружает кэш если есть, иначе генерирует и сохраняет.

    Аргументы:
        force: True → всегда генерировать заново

    Использование в main.py:
        graph_data = get_or_generate_graph_data()
        train_loader, val_loader, test_loader = create_dataloaders(graph_data)
    """
    path = PATHS["graph_data"]
    if not force and os.path.exists(path):
        return load_graph_data(path)

    gen        = ThreatDataGenerator()
    graph_data = gen.build_full_graph_data()
    save_graph_data(graph_data, path)
    return graph_data


# ══════════════════════════════════════════════════════════════
# TODO — ЧТО ОСТАВЛЕНО ТЕБЕ
# ══════════════════════════════════════════════════════════════
#
# 1. СТРАТИФИЦИРОВАННЫЙ СПЛИТ:
#    Текущий random_split может нарушать баланс классов в splits.
#    Реализуй через sklearn.model_selection.StratifiedShuffleSplit
#    или вручную через torch.where по каждому классу.
#
# 2. MIXUP АУГМЕНТАЦИЯ:
#    Добавь в ThreatAugmentation или как отдельный collate_fn:
#
#    def mixup_collate(batch, alpha=0.2):
#        X, y = zip(*batch)
#        X = torch.stack(X); y = torch.tensor(y)
#        lam = np.random.beta(alpha, alpha)
#        idx = torch.randperm(len(X))
#        X_mix = lam * X + (1 - lam) * X[idx]
#        return X_mix, y, y[idx], lam
#
# 3. РЕАЛЬНЫЙ ИСТОЧНИК ДАННЫХ:
#    Замени ThreatDataGenerator._generate_incidents() на:
#
#    def load_from_csv(self, path: str) -> Tuple[Tensor, Tensor]:
#        df = pd.read_csv(path)
#        X  = torch.tensor(df[feature_cols].values, dtype=torch.float)
#        y  = torch.tensor(df["label"].values,      dtype=torch.long)
#        return X, y
#
# 4. TORCH GEOMETRIC (опционально):
#    Вместо dict можно использовать torch_geometric.data.Data:
#
#    from torch_geometric.data import Data
#    data = Data(
#        x          = node_features,
#        edge_index = edge_index,
#        edge_attr  = edge_type,
#    )
#    # Тогда model.forward принимает data напрямую
#
# ══════════════════════════════════════════════════════════════