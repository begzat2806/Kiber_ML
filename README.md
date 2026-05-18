# CyberThreat Intelligence AI
## Production-Level GAT + Knowledge Graph

---

## Структура папок

```
cyberthreat_ai/
│
├── config.py           ← гиперпараметры (ЕДИНСТВЕННЫЙ источник правды)
├── dataset.py          ← данные + Knowledge Graph builder
├── model.py            ← архитектура: GAT + SemanticAggregator + MLP
├── train.py            ← training loop + optimizer/scheduler factories
├── inference.py        ← ThreatAnalyzer (inference pipeline)
├── utils.py            ← метрики, логгер, чекпоинты, визуализация
├── main.py             ← точка входа (СОБИРАЕШЬ САМ)
│
├── data/
│   └── threat_graph.pt     ← кэш графа (генерируется автоматически)
│
├── checkpoints/
│   ├── best_model.pt        ← лучшая модель по val_loss
│   └── checkpoint_epoch_*.pt
│
├── logs/
│   ├── train.log            ← полный лог обучения
│   └── metrics.csv          ← метрики по эпохам
│
├── plots/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── semantic_heatmap.png
│
└── requirements.txt
```

---

## Граф зависимостей между модулями

```
config.py  (независимый — импортируется всеми)
    │
    ├──→ dataset.py
    │       │ graph_data (dict)
    │       ├──→ node_features (C, 128) ──→ model.py :: GATEncoder
    │       ├──→ edge_index    (2, E)   ──→ model.py :: GATEncoder
    │       ├──→ incident_X    (N, 128) ──→ DataLoader
    │       └──→ incident_y    (N,)     ──→ DataLoader
    │
    ├──→ model.py
    │       │ CyberThreatGAT
    │       └──→ forward(incident_x, node_features, edge_index)
    │               → {"logits", "probs", "graph_emb", "attn"}
    │
    ├──→ utils.py  (независимый — принимает простые типы)
    │       MetricsTracker, EarlyStopping,
    │       CheckpointManager, plot_*()
    │
    ├──→ train.py
    │       Trainer(model, optimizer, scheduler, criterion,
    │               train_loader, val_loader, graph_data, device)
    │       │
    │       └── .fit() → history dict
    │
    └──→ inference.py
            ThreatAnalyzer(model, graph_data, device)
            └── .analyze(incident) → result dict
                .explain(incident) → explanation dict
```

---

## Данные между модулями

### `graph_data` — центральная структура данных

```python
graph_data = {
    "node_features": Tensor (10, 128),   # признаки узлов Knowledge Graph
    "edge_index":    Tensor (2, E),      # рёбра в COO-формате
    "edge_type":     Tensor (E,),        # тип каждого ребра (0-4)
    "incident_X":    Tensor (N, 128),    # обучающие инциденты
    "incident_y":    Tensor (N,),        # метки классов
    "sim_matrix":    Tensor (10, 10),    # матрица косинусного сходства
    "class_names":   List[str],          # 10 названий классов
    "n_classes":     int,                # = 10
}
```

### `model.forward()` — входы и выходы

```python
# Вход:
incident_x    = Tensor (B, 128)    # батч инцидентов из DataLoader
node_features = Tensor (10, 128)   # из graph_data, на device
edge_index    = Tensor (2, E)      # из graph_data, на device

# Выход (dict):
{
    "logits":       Tensor (B, 10),   # → nn.CrossEntropyLoss
    "probs":        Tensor (B, 10),   # → метрики, inference
    "graph_emb":    Tensor (10, 256), # → визуализация узлов
    "incident_emb": Tensor (B, 256),  # → кластеризация, анализ
    "attn":         Tensor (B, 10),   # → XAI объяснение
}
```

---

## Порядок сборки (чеклист)

### Этап 1 — Проверка данных
```bash
python main.py --mode data
```
→ Смотришь статистику классов, граф, открываешь `plots/semantic_heatmap.png`

### Этап 2 — Реализация обучения в `train.py`

```python
# Реализуй эти три функции / метода:

def build_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CFG.learning_rate,
        weight_decay=TRAIN_CFG.weight_decay,
    )

def build_scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAIN_CFG.cosine_t_max,
        eta_min=TRAIN_CFG.min_lr,
    )

def _validate_one_epoch(self, epoch):
    self.model.eval()
    tracker = MetricsTracker(self.n_classes)
    with torch.no_grad():
        for X, y in self.val_loader:
            X, y = X.to(self.device), y.to(self.device)
            out  = self.model(X, self.node_features, self.edge_index)
            loss = self.criterion(out["logits"], y)
            preds = out["probs"].argmax(dim=-1)
            tracker.update(loss.item(), preds, y, len(y))
    return tracker.compute()
```

### Этап 3 — Сборка в `main.py`

Раскомментируй блоки в `run_full_pipeline()`:
1. `model = build_model().to(device)`
2. `optimizer = build_optimizer(model)`
3. `scheduler = build_scheduler(optimizer)`
4. `trainer = Trainer(...)` и `trainer.fit()`
5. `evaluate_on_test(...)` и `plot_*()` функции

### Этап 4 — Запуск обучения
```bash
python main.py --mode train
# Следи за логами: val_loss должен падать, val_f1 расти
```

### Этап 5 — Полный пайплайн
```bash
python main.py --mode full
```

### Этап 6 — Инференс
```bash
python main.py --mode infer
```

---

## Типичные проблемы и решения

| Симптом | Причина | Решение |
|---------|---------|---------|
| Loss не падает первые 5 эпох | LR слишком мал | Увеличь `learning_rate` в 10x |
| Loss взрывается (NaN/inf) | LR слишком велик | Уменьши в 10x, проверь `grad_clip` |
| CUDA OOM | Батч слишком большой | Уменьши `batch_size`, включи AMP |
| Accuracy < 60% | Дисбаланс классов | Проверь `use_class_weights=True` |
| Early stopping слишком рано | Patience мал | Увеличь `early_stopping_patience` |
| Overfitting (train >> val) | Нет регуляризации | Увеличь `dropout`, `label_smoothing` |

---

## Математика ключевых компонентов

### GAT Attention
```
e_ij  = LeakyReLU(aᵀ [Wh_i ‖ Wh_j])
α_ij  = exp(e_ij) / Σ_k exp(e_ik)
h'_i  = ‖ₖ σ(Σ_j α_ij^k · Wᵏh_j)   ← K голов конкатенируются
```

### Semantic Aggregator (cross-attention)
```
Q = W_q · incident_emb              (B, D)
K = W_k · graph_node_emb            (C, D)
scores = Q·Kᵀ / √D                  (B, C)
α = softmax(scores / τ)             (B, C)   ← τ обучаемый
ctx = α · graph_node_emb            (B, D)
out = [incident_emb ‖ ctx]          (B, 2D)
```

### Loss (CrossEntropy + Label Smoothing)
```
p_smooth[c] = (1-ε)·one_hot[c] + ε/C
loss = -Σ_c p_smooth[c] · log(softmax(logits)[c])
```

---

## Рекомендации по масштабированию

1. **Больше данных**: Подключи реальные SIEM-логи через `ThreatDataGenerator`
2. **Реальные эмбеддинги**: Замени синтетические векторы на BERT-эмбеддинги описаний из MITRE ATT&CK
3. **RGAT**: Реализуй relation-aware GAT с отдельными весами для каждого типа ребра
4. **Multi-GPU**: Оберни модель в `torch.nn.DataParallel` или используй `torch.distributed`
5. **Hyperparameter search**: Подключи Optuna для автоматического поиска гиперпараметров
6. **Мониторинг**: Замени CSV-логгер на Weights & Biases или MLflow
7. **Деплой**: Экспортируй через ONNX, разверни за FastAPI
