"""
model.py
─────────────────────────────────────────────────────────
РОЛЬ В ПРОЕКТЕ:
    Определяет всю архитектуру нейросети.
    Остальные модули вызывают build_model() и работают
    с интерфейсом model.forward() / model.predict().

ЗАВИСИМОСТИ:
    config.py → MODEL_CFG, DATA_CFG

ЭКСПОРТИРУЕТ:
    CyberThreatGAT  → train.py, inference.py
    build_model()   → main.py

ДАННЫЕ МЕЖДУ МОДУЛЯМИ:
    Входы (приходят из dataset.py / DataLoader):
        incident_x:    Tensor (B, 128)          — батч инцидентов
        node_features: Tensor (10, 128)         — из graph_data
        edge_index:    Tensor (2, E)            — из graph_data

    Выходы (уходят в train.py / inference.py):
        dict {
            "logits":       Tensor (B, 10)      → loss function
            "probs":        Tensor (B, 10)      → inference / метрики
            "graph_emb":    Tensor (10, 256)    → визуализация
            "incident_emb": Tensor (B, 256)     → анализ / кластеризация
        }

АРХИТЕКТУРА:
    ┌──────────────────────────────────────────────────┐
    │  incident_x (B,128)    node_features (10,128)    │
    │       │                       │                  │
    │  IncidentEncoder         GATEncoder (3 слоя)     │
    │  (B,128)→(B,256)        (10,128)→(10,256)        │
    │       │                       │                  │
    │       └──── SemanticAggregator ────┘             │
    │              (B,256) + context → (B,512)         │
    │                       │                         │
    │                  MLPClassifier                   │
    │               (B,512) → (B,10)                   │
    └──────────────────────────────────────────────────┘

МАТЕМАТИКА:
    GAT attention (Veličković et al. 2018):
        e_ij  = LeakyReLU(aᵀ [Wh_i ‖ Wh_j])
        α_ij  = exp(e_ij) / Σ_k exp(e_ik)
        h'_i  = ‖ₖ σ(Σ_j α_ij^k · Wᵏh_j)   (multi-head concat)

    Semantic aggregation (cross-attention):
        scores = Q·Kᵀ / √d
        α      = softmax(scores / τ)
        ctx    = α · graph_emb
        out    = [incident_emb ‖ ctx]

ЧТО ОСТАВЛЕНО ТЕБЕ:
    - Заменить GAT на RGAT (поддержка edge_type) — см. TODO
    - Добавить contrastive loss headку
    - Подключить GNNExplainer для XAI
    - Реализовать Relation-aware message passing
─────────────────────────────────────────────────────────
"""

import math
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MODEL_CFG, DATA_CFG

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 1. GRAPH ATTENTION LAYER
#    Базовый строительный блок. Независим от остальной модели.
#    Можно заменить на torch_geometric.nn.GATConv без изменений
#    в GATEncoder — просто поменяй forward-вызов.
# ══════════════════════════════════════════════════════════════

class GraphAttentionLayer(nn.Module):
    """
    Один GAT-слой с поддержкой multi-head attention.

    Принимает:
        h:          (N, F_in)   — признаки узлов
        edge_index: (2, E)      — рёбра в COO-формате

    Возвращает:
        h':  concat=True  → (N, K * F_out)
             concat=False → (N, F_out)      ← для последнего слоя

    Параметры:
        W: (K, F_in, F_out)      — проекционные матрицы K голов
        a: (K, 2 * F_out)        — векторы внимания K голов

    TODO — Relation-aware GAT (RGAT):
        Добавь отдельные проекции W_r для каждого типа ребра r:
            W_list = nn.ParameterList([
                nn.Parameter(torch.empty(F_in, F_out))
                for _ in range(num_edge_types)
            ])
        В forward используй edge_type для выбора W_r[edge_type[e]]
    """

    def __init__(self,
                 in_features:  int,
                 out_features: int,
                 num_heads:    int   = 8,
                 dropout:      float = 0.3,
                 concat:       bool  = True,
                 edge_dropout: float = 0.1):
        super().__init__()

        self.F_in   = in_features
        self.F_out  = out_features
        self.K      = num_heads
        self.concat = concat
        self.edge_do = edge_dropout

        # W ∈ R^(K, F_in, F_out)
        self.W = nn.Parameter(torch.empty(num_heads, in_features, out_features))
        # a ∈ R^(K, 2*F_out)
        self.a = nn.Parameter(torch.empty(num_heads, 2 * out_features))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.attn_drop  = nn.Dropout(dropout)

        self._init_params()

    def _init_params(self):
        for k in range(self.K):
            nn.init.xavier_uniform_(self.W[k])
            # a инициализируем малыми значениями
            nn.init.normal_(self.a[k], std=0.01)

    def forward(self,
                h:          torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        h:          (N, F_in)
        edge_index: (2, E)
        """
        N      = h.shape[0]
        K      = self.K
        F_out  = self.F_out

        src, dst = edge_index[0], edge_index[1]   # (E,)

        # ── DropEdge: случайно удаляем рёбра во время обучения ──
        # Регуляризация структуры графа (аналог Dropout для рёбер)
        if self.training and self.edge_do > 0:
            keep = torch.rand(len(src), device=h.device) > self.edge_do
            src, dst = src[keep], dst[keep]

        # ── Шаг 1: Линейное преобразование ──────────────────────
        # Wh[n, k, :] = W[k] · h[n]
        # Операция: (N, F_in) × (K, F_in, F_out) → (N, K, F_out)
        Wh = torch.einsum("ni,kio->nko", h, self.W)   # (N, K, F_out)

        # ── Шаг 2: Attention logits e_ij ────────────────────────
        # e = LeakyReLU(aᵀ [Wh_src ‖ Wh_dst])
        Wh_src = Wh[src]   # (E, K, F_out)
        Wh_dst = Wh[dst]   # (E, K, F_out)

        # Конкатенируем по последней оси: (E, K, 2*F_out)
        cat = torch.cat([Wh_src, Wh_dst], dim=-1)

        # Скалярное внимание: (E, K, 2F) × (K, 2F) → (E, K)
        e = (cat * self.a.unsqueeze(0)).sum(-1)   # (E, K)
        e = self.leaky_relu(e)

        # ── Шаг 3: Softmax нормализация α_ij ─────────────────────
        # Для стабильности: вычитаем max по соседям (log-sum-exp trick)
        e_max = torch.full((N, K), float("-inf"), device=h.device)
        e_max.scatter_reduce_(0,
                              dst.unsqueeze(-1).expand(-1, K),
                              e, reduce="amax", include_self=True)
        e_exp = torch.exp(e - e_max[dst])   # (E, K)

        e_sum = torch.zeros(N, K, device=h.device)
        e_sum.scatter_add_(0, dst.unsqueeze(-1).expand(-1, K), e_exp)

        alpha = e_exp / (e_sum[dst] + 1e-9)   # (E, K) — нормированные веса
        alpha = self.attn_drop(alpha)

        # ── Шаг 4: Взвешенная агрегация ──────────────────────────
        # h'_i = Σ_j α_ij · Wh_j
        weighted = alpha.unsqueeze(-1) * Wh[src]   # (E, K, F_out)

        h_agg = torch.zeros(N, K, F_out, device=h.device)
        idx   = dst.unsqueeze(-1).unsqueeze(-1).expand(-1, K, F_out)
        h_agg.scatter_add_(0, idx, weighted)        # (N, K, F_out)

        h_agg = F.elu(h_agg)   # нелинейность после агрегации

        # ── Шаг 5: Объединение голов ─────────────────────────────
        if self.concat:
            return h_agg.reshape(N, K * F_out)   # (N, K*F_out)
        else:
            return h_agg.mean(dim=1)              # (N, F_out)


# ══════════════════════════════════════════════════════════════
# 2. GAT ENCODER
#    Стек GAT-слоёв с LayerNorm + Residual.
#    Обрабатывает Knowledge Graph → обновлённые узловые эмбеддинги.
# ══════════════════════════════════════════════════════════════

class GATEncoder(nn.Module):
    """
    Глубокий GAT: L слоёв с нормализацией и residual connections.

    Residual схема (как в Pre-LN трансформере):
        h^(l) = LayerNorm(GAT(h^(l-1)) + proj(h^(l-1)))

    Рецептивное поле растёт с глубиной:
        Слой 1 → видит 1-hop соседей
        Слой 2 → видит 2-hop соседей
        Слой 3 → видит 3-hop соседей

    Принимает: x (N, in_dim), edge_index (2, E)
    Возвращает: h (N, hidden_dim)

    ИСПОЛЬЗОВАНИЕ В CyberThreatGAT:
        graph_emb = self.gat_encoder(node_features, edge_index)
        # graph_emb: (n_classes, hidden_dim) — обогащённые узлы
    """

    def __init__(self,
                 in_dim:      int,
                 hidden_dim:  int,
                 num_layers:  int,
                 num_heads:   int,
                 dropout:     float,
                 edge_dropout: float):
        super().__init__()

        self.gat_layers = nn.ModuleList()
        self.norms      = nn.ModuleList()
        self.res_projs  = nn.ModuleList()
        self.drop       = nn.Dropout(dropout)

        cur_dim = in_dim

        for l in range(num_layers):
            is_last  = (l == num_layers - 1)
            head_dim = hidden_dim // num_heads

            if is_last:
                # Последний слой: усредняем головы → (N, hidden_dim)
                gat     = GraphAttentionLayer(cur_dim, hidden_dim,
                                             num_heads, dropout,
                                             concat=False,
                                             edge_dropout=edge_dropout)
                out_dim = hidden_dim
            else:
                # Промежуточные: конкатенируем → (N, K * head_dim)
                gat     = GraphAttentionLayer(cur_dim, head_dim,
                                             num_heads, dropout,
                                             concat=True,
                                             edge_dropout=edge_dropout)
                out_dim = num_heads * head_dim   # == hidden_dim

            self.gat_layers.append(gat)
            self.norms.append(nn.LayerNorm(out_dim))

            # Residual projection: выравниваем размерности если нужно
            if cur_dim != out_dim:
                self.res_projs.append(nn.Linear(cur_dim, out_dim, bias=False))
            else:
                self.res_projs.append(nn.Identity())

            cur_dim = out_dim

        self.output_dim = cur_dim   # = hidden_dim

    def forward(self,
                x:          torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        x:          (N, in_dim)
        edge_index: (2, E)
        → h:        (N, hidden_dim)
        """
        h = x
        for gat, norm, res_proj in zip(self.gat_layers,
                                       self.norms,
                                       self.res_projs):
            h_gat = gat(h, edge_index)     # GAT преобразование
            h_res = res_proj(h)             # Residual
            h     = norm(h_gat + h_res)    # Pre-LN стиль
            h     = self.drop(h)
        return h   # (N, hidden_dim)


# ══════════════════════════════════════════════════════════════
# 3. INCIDENT ENCODER
#    Кодирует входной вектор инцидента в hidden_dim пространство.
#    Должен иметь ту же output-размерность что и GATEncoder,
#    потому что SemanticAggregator сравнивает их напрямую.
# ══════════════════════════════════════════════════════════════

class IncidentEncoder(nn.Module):
    """
    MLP-кодировщик: incident (B, in_dim) → (B, hidden_dim)

    Архитектура:
        Linear → LayerNorm → GELU → Dropout →
        Linear → LayerNorm → GELU → Dropout

    GELU предпочтительнее ReLU для трансформероподобных моделей
    (более плавный градиент вблизи нуля).
    """

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,       hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, hidden_dim)


# ══════════════════════════════════════════════════════════════
# 4. SEMANTIC AGGREGATOR  ← ключевой семантический модуль
#
#    Механизм cross-attention между инцидентом и узлами графа.
#    Это и есть "семантическое обогащение":
#    модель спрашивает Knowledge Graph, на какие классы
#    похож данный инцидент — и получает контекстный вектор.
# ══════════════════════════════════════════════════════════════

class SemanticAggregator(nn.Module):
    """
    Cross-attention: инцидент → Knowledge Graph → контекст.

    Математика:
        Q = W_q · incident_emb         (B, D)
        K = W_k · graph_node_emb       (C, D)
        scores = Q · Kᵀ / √D           (B, C)
        α = softmax(scores / τ)         (B, C) — веса по классам
        ctx = α · graph_node_emb        (B, D)
        out = [incident_emb ‖ ctx]      (B, 2D)

    τ (temperature) — обучаемый параметр:
        - τ → 0: жёсткое внимание (winner-take-all)
        - τ → ∞: мягкое равномерное внимание

    Выход (B, 2D) отвечает на вопрос:
        "Этот инцидент похож на X (из его own encoding) +
         Knowledge Graph говорит, что больше всего он похож на Y"

    ИСПОЛЬЗОВАНИЕ В INFERENCE:
        Смотри model.get_attention_weights() —
        возвращает α (B, n_classes) для объяснения предсказания.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Обучаемая температура softmax
        self.log_tau = nn.Parameter(torch.zeros(1))   # τ = exp(log_τ)
        self.scale   = hidden_dim ** 0.5

    def forward(self,
                incident_emb:   torch.Tensor,
                graph_node_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        incident_emb:   (B, D)
        graph_node_emb: (C, D)

        Возвращает:
            enriched: (B, 2D) — конкатенация inc_emb + контекст
            attn:     (B, C)  — веса внимания (для XAI)
        """
        Q = self.W_q(incident_emb)    # (B, D)
        K = self.W_k(graph_node_emb)  # (C, D)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.t()) / self.scale   # (B, C)
        tau    = self.log_tau.exp().clamp(min=0.1)
        attn   = F.softmax(scores / tau, dim=-1)        # (B, C)

        # Взвешенная сумма по классам
        ctx      = torch.matmul(attn, graph_node_emb)  # (B, D)
        enriched = torch.cat([incident_emb, ctx], dim=-1)  # (B, 2D)

        return enriched, attn


# ══════════════════════════════════════════════════════════════
# 5. MLP CLASSIFIER
#    Финальный классификатор поверх enriched_emb.
# ══════════════════════════════════════════════════════════════

class MLPClassifier(nn.Module):
    """
    Стек: Linear → LayerNorm → GELU → Dropout → ... → Linear(n_classes)

    Принимает: (B, 2*hidden_dim)
    Возвращает: (B, n_classes) — raw logits
    """

    def __init__(self,
                 in_dim:      int,
                 hidden_dims: List[int],
                 n_classes:   int,
                 dropout:     float):
        super().__init__()
        layers    = []
        cur       = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(cur, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            cur = h
        layers.append(nn.Linear(cur, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, n_classes)


# ══════════════════════════════════════════════════════════════
# 6. ПОЛНАЯ МОДЕЛЬ
# ══════════════════════════════════════════════════════════════

class CyberThreatGAT(nn.Module):
    """
    Полная модель классификации кибератак.

    Порядок вызова в train.py:
        out   = model(X_batch, node_features, edge_index)
        loss  = criterion(out["logits"], y_batch)
        preds = out["probs"].argmax(dim=-1)

    Для инференса (inference.py):
        preds, confs = model.predict(X, node_features, edge_index)

    Для XAI (объяснимости):
        attn = model.get_attention_weights(X, node_features, edge_index)
        # attn[b, c] — насколько инцидент b похож на класс c по графу
    """

    def __init__(self,
                 in_dim:   int,
                 n_classes: int,
                 cfg       = None):
        super().__init__()
        cfg = cfg or MODEL_CFG

        self.in_dim    = in_dim
        self.n_classes = n_classes
        self.D         = cfg.hidden_dim

        self.incident_enc = IncidentEncoder(in_dim, cfg.hidden_dim, cfg.dropout)

        self.gat_encoder  = GATEncoder(
            in_dim       = in_dim,
            hidden_dim   = cfg.hidden_dim,
            num_layers   = cfg.num_gat_layers,
            num_heads    = cfg.num_heads,
            dropout      = cfg.dropout,
            edge_dropout = cfg.edge_dropout,
        )

        self.semantic_agg = SemanticAggregator(cfg.hidden_dim)

        self.classifier   = MLPClassifier(
            in_dim      = 2 * cfg.hidden_dim,   # incident + ctx
            hidden_dims = cfg.mlp_hidden_dims,
            n_classes   = n_classes,
            dropout     = cfg.dropout,
        )

        self._init_weights(cfg.weight_init)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("CyberThreatGAT: %s trainable parameters", f"{n_params:,}")

    def _init_weights(self, method: str):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif method == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                elif method == "orthogonal":
                    nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── основной forward ──────────────────────────────────

    def forward(self,
                incident_x:    torch.Tensor,
                node_features: torch.Tensor,
                edge_index:    torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        incident_x:    (B, in_dim)
        node_features: (C, in_dim)   — C == n_classes
        edge_index:    (2, E)

        Возвращает dict:
            "logits"       (B, C)   → nn.CrossEntropyLoss
            "probs"        (B, C)   → метрики / inference
            "graph_emb"    (C, D)   → визуализация узлов
            "incident_emb" (B, D)   → кластеризация / анализ
            "attn"         (B, C)   → XAI (SemanticAggregator weights)
        """
        # 1. Кодируем инциденты
        inc_emb   = self.incident_enc(incident_x)             # (B, D)

        # 2. GAT: обновляем представления узлов Knowledge Graph
        graph_emb = self.gat_encoder(node_features, edge_index)  # (C, D)

        # 3. Семантическое обогащение через cross-attention
        enriched, attn = self.semantic_agg(inc_emb, graph_emb)   # (B,2D), (B,C)

        # 4. Классификация
        logits = self.classifier(enriched)                     # (B, C)

        return {
            "logits":       logits,
            "probs":        F.softmax(logits, dim=-1),
            "graph_emb":    graph_emb,
            "incident_emb": inc_emb,
            "attn":         attn,
        }

    # ── удобные методы ────────────────────────────────────

    def predict(self,
                incident_x:    torch.Tensor,
                node_features: torch.Tensor,
                edge_index:    torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает (predicted_class, confidence) без сохранения графа вычислений.
        Используется в inference.py и при валидации.
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(incident_x, node_features, edge_index)
        confs, preds = out["probs"].max(dim=-1)
        return preds, confs

    def get_attention_weights(self,
                              incident_x:    torch.Tensor,
                              node_features: torch.Tensor,
                              edge_index:    torch.Tensor) -> torch.Tensor:
        """
        Возвращает α (B, n_classes) — веса SemanticAggregator.
        Показывает, на какие классы Knowledge Graph
        "обращает внимание" при анализе данного инцидента.

        Используется в inference.py для объяснения предсказания:
            attn = model.get_attention_weights(x, node_feat, edges)
            top_related = attn.topk(3, dim=-1).indices
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(incident_x, node_features, edge_index)
        return out["attn"]   # (B, n_classes)

    def count_params(self) -> Dict[str, int]:
        """Разбивка параметров по компонентам — удобно для отладки."""
        def _count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        return {
            "incident_encoder":   _count(self.incident_enc),
            "gat_encoder":        _count(self.gat_encoder),
            "semantic_aggregator":_count(self.semantic_agg),
            "classifier":         _count(self.classifier),
            "total":              _count(self),
        }


# ══════════════════════════════════════════════════════════════
# 7. ФАБРИКА МОДЕЛИ
# ══════════════════════════════════════════════════════════════

def build_model(in_dim:   int = None,
                n_classes: int = None,
                cfg               = None) -> CyberThreatGAT:
    """
    Создаёт модель с параметрами из config.py (или явными).

    Вызов в main.py:
        model = build_model()
        model = model.to(device)

    Вызов с кастомными параметрами:
        model = build_model(in_dim=256, n_classes=15)
    """
    return CyberThreatGAT(
        in_dim    = in_dim    or DATA_CFG.node_feature_dim,
        n_classes = n_classes or DATA_CFG.num_classes,
        cfg       = cfg       or MODEL_CFG,
    )


# ══════════════════════════════════════════════════════════════
# TODO — ЧТО ОСТАВЛЕНО ТЕБЕ
# ══════════════════════════════════════════════════════════════
#
# 1. RELATION-AWARE GAT (RGAT):
#    Текущий GAT игнорирует edge_type.
#    Чтобы использовать типы рёбер:
#
#    class RGATLayer(nn.Module):
#        def __init__(self, in_f, out_f, n_relations, ...):
#            # Отдельная W для каждого типа ребра
#            self.W_rel = nn.Embedding(n_relations, in_f * out_f)
#
#        def forward(self, h, edge_index, edge_type):
#            # Для каждого ребра берём свою W
#            W_e = self.W_rel(edge_type).view(-1, F_in, F_out)
#            Wh_src = torch.bmm(h[src].unsqueeze(1), W_e).squeeze(1)
#            ...
#
# 2. CONTRASTIVE LOSS HEAD:
#    Добавь проекционную голову для contrastive learning:
#
#    self.proj_head = nn.Sequential(
#        nn.Linear(hidden_dim, hidden_dim),
#        nn.ReLU(),
#        nn.Linear(hidden_dim, 64),
#    )
#    # В forward: proj = F.normalize(self.proj_head(inc_emb), dim=-1)
#    # Loss: NT-Xent loss между парами инцидентов одного класса
#
# 3. GNNExplainer (torch_geometric):
#    from torch_geometric.explain import GNNExplainer
#    explainer = GNNExplainer(model, epochs=100)
#    node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
#
# 4. TORCH GEOMETRIC ЗАМЕНА:
#    GraphAttentionLayer можно заменить на:
#    from torch_geometric.nn import GATConv
#    self.conv = GATConv(in_channels, out_channels,
#                        heads=num_heads, dropout=dropout)
#    Интерфейс: h = self.conv(h, edge_index)  — тот же!
#
# ══════════════════════════════════════════════════════════════