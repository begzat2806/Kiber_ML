"""
inference.py
─────────────────────────────────────────────────────────
РОЛЬ b ПРОЕКТЕ:
    Production inference pipeline.
    Загружает обученную модель и анализирует новые инциденты.
    He знает как модель обучалась — только как её использовать.

ЗАВИСИМОСТИ:
    config.py  → INFERENCE_CFG, PATHS, DATA_CFG
    model.py   → CyberThreatGAT, build_model()
    utils.py   → get_device()

ЭКСПОРТИРУЕТ:
    ThreatAnalyzer      → main.py, внешний API
    load_trained_model() → main.py

kak ПОДКЛЮЧИТЬ b main.py:
    analyzer = ThreatAnalyzer.from_checkpoint(
        checkpoint_path = PATHS["best_model"],
        graph_data      = graph_data,
        device          = device,
    )
    result = analyzer.analyze(incident_vector)
    analyzer.explain(incident_vector)

ДАННЫЕ МЕЖДУ МОДУЛЯМИ:
    Вход:
        incident_vector: Tensor (128,) или (B, 128)
        graph_data:      dict из dataset.py (node_features, edge_index)

    Выход (dict):
        "predicted_class":   str    — название угрозы
        "confidence":        float  — уверенность модели [0, 1]
        "severity":          str    — "critical" / "high" / "medium" / "low"
        "top_k_predictions": list   — топ-3 класса c вероятностями
        "semantic_context":  list   — связанные классы по Knowledge Graph
        "alert":             bool   — превышен ли порог уверенности

ЧТО ОСТАВЛЕНО ТЕБЕ:
    - Реализовать REST API (FastAPI/Flask обёртка)
    - Батч-инференс из CSV файла
    - ONNX экспорт для деплоя
    - Threshold calibration (Platt scaling)
─────────────────────────────────────────────────────────
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from config import INFERENCE_CFG, PATHS, DATA_CFG
from utils import get_device

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 1. ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ
# ══════════════════════════════════════════════════════════════

def load_trained_model(checkpoint_path: str = None,
                       device: torch.device = None):
    """
    Загружает модель из чекпоинта.

    Используется в ThreatAnalyzer.from_checkpoint()
    и в main.py для финального тестирования.

    Аргументы:
        checkpoint_path: путь к .pt файлу
        device:          куда загрузить модель

    Возвращает:
        model: CyberThreatGAT в режиме eval()

    Использование:
        model = load_trained_model(PATHS["best_model"], device)

    Примечание:
        Чекпоинт содержит полный конфиг модели на момент сохранения.
        Это позволяет загружать модель без ручного указания параметров.
    """
    from model import build_model

    path   = checkpoint_path or PATHS["best_model"]
    device = device or get_device()

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Восстанавливаем конфиг из чекпоинта (если есть)
    # Иначе используем текущий config.py
    model = build_model()
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    epoch   = checkpoint.get("epoch", "?")
    metrics = checkpoint.get("metrics", {})
    logger.info(
        "Model loaded ← %s | epoch=%s | val_loss=%.4f",
        path, epoch, metrics.get("loss", float("nan"))
    )
    return model


# ══════════════════════════════════════════════════════════════
# 2. PREPROCESSOR
#    Преобразует сырые данные инцидента в тензор нужной формы.
#    В production здесь будет feature extraction pipeline.
# ══════════════════════════════════════════════════════════════

class IncidentPreprocessor:
    """
    Подготавливает входные данные для модели.

    В реальном проекте здесь:
        - Парсинг SIEM-лога / NetFlow / PCAP
        - Feature engineering (статистики трафика и т.д.)
        - Нормализация (StandardScaler / MinMaxScaler)
        - Преобразование в вектор dim=128

    Сейчас: принимает уже готовый вектор или dict c полями,
    нормализует и приводит к правильному типу/форме.

    TODO:
        Добавь реальный feature extractor:

        class RealIncidentPreprocessor(IncidentPreprocessor):
            def __init__(self, scaler_path: str):
                from joblib import load
                self.scaler = load(scaler_path)

            def from_siem_log(self, log_dict: dict) -> torch.Tensor:
                features = self._extract_features(log_dict)
                features = self.scaler.transform([features])[0]
                return torch.tensor(features, dtype=torch.float32)

            def _extract_features(self, log: dict) -> list:
                return [
                    log.get("bytes_sent", 0) / 1e6,
                    log.get("packets_per_sec", 0) / 1000,
                    log.get("unique_ports", 0) / 65535,
                    # ... 128 признаков
                ]
    """

    def __init__(self, expected_dim: int = None):
        self.dim = expected_dim or DATA_CFG.node_feature_dim

    def process(self,
                raw: Union[torch.Tensor, List[float], Dict]
                ) -> torch.Tensor:
        """
        Преобразует сырые данные в Tensor (1, dim).

        Принимает:
            - torch.Tensor (dim,) или (1, dim)
            - list[float] длины dim
            - dict c ключом "features"

        Возвращает:
            Tensor (1, dim) — готово для model.forward()
        """
        if isinstance(raw, dict):
            raw = raw.get("features", raw.get("x", []))

        if isinstance(raw, list):
            x = torch.tensor(raw, dtype=torch.float32)
        elif isinstance(raw, torch.Tensor):
            x = raw.float()
        else:
            raise TypeError(f"Неожиданный тип входных данных: {type(raw)}")

        if x.dim() == 1:
            x = x.unsqueeze(0)   # (dim,) → (1, dim)

        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Ожидался вектор размерности {self.dim}, "
                f"получен {x.shape[-1]}"
            )

        # L2-нормализация (как при генерации данных)
        x = F.normalize(x, dim=-1)
        return x   # (1, dim)

    def process_batch(self,
                      raws: List[Union[torch.Tensor, List[float]]]
                      ) -> torch.Tensor:
        """
        Обрабатывает список инцидентов.
        Возвращает Tensor (B, dim).
        """
        tensors = [self.process(r) for r in raws]
        return torch.cat(tensors, dim=0)   # (B, dim)


# ══════════════════════════════════════════════════════════════
# 3. THREAT ANALYZER — главный класс инференса
# ══════════════════════════════════════════════════════════════

class ThreatAnalyzer:
    """
    Полный inference pipeline для анализа кибер-угроз.

    Использование (три способа создания):

        # 1. Из чекпоинта (production):
        analyzer = ThreatAnalyzer.from_checkpoint(
            checkpoint_path = PATHS["best_model"],
            graph_data      = graph_data,
        )

        # 2. Из уже загруженной модели (тесты / main.py):
        analyzer = ThreatAnalyzer(
            model      = trained_model,
            graph_data = graph_data,
            device     = device,
        )

        # 3. Анализ инцидента:
        result = analyzer.analyze(incident_vector)
        print(result["predicted_class"])  # "Ransomware"
        print(result["severity"])         # "critical"
        print(result["semantic_context"]) # ["APT", "Botnet", "Phishing"]
    """

    def __init__(self,
                 model:        torch.nn.Module,
                 graph_data:   Dict,
                 device:       torch.device = None,
                 cfg                        = None):
        self.model  = model
        self.device = device or get_device()
        self.cfg    = cfg or INFERENCE_CFG

        # Граф (переносим на device один раз при инициализации)
        self.node_features = graph_data["node_features"].to(self.device)
        self.edge_index    = graph_data["edge_index"].to(self.device)
        self.class_names   = graph_data["class_names"]
        self.n_classes     = graph_data["n_classes"]

        self.preprocessor = IncidentPreprocessor()

        self.model.eval()
        logger.info("ThreatAnalyzer ready | %d threat classes", self.n_classes)

    @classmethod
    def from_checkpoint(cls,
                        checkpoint_path: str,
                        graph_data:      Dict,
                        device:          torch.device = None) -> "ThreatAnalyzer":
        """
        Фабричный метод: загружает модель и создаёт анализатор.

        Использование:
            analyzer = ThreatAnalyzer.from_checkpoint(
                PATHS["best_model"], graph_data
            )
        """
        device = device or get_device()
        model  = load_trained_model(checkpoint_path, device)
        return cls(model=model, graph_data=graph_data, device=device)

    # ──────────────────────────────────────────────────────────
    # ОСНОВНОЙ МЕТОД АНАЛИЗА
    # ──────────────────────────────────────────────────────────

    def analyze(self,
                raw_incident: Union[torch.Tensor, List[float], Dict],
                ) -> Dict:
        """
        Анализирует один инцидент и возвращает полный отчёт.

        Аргументы:
            raw_incident: вектор признаков инцидента (dim=128)
                          или dict {"features": [...]}

        Возвращает dict:
            {
              "predicted_class":    "Ransomware",
              "class_idx":          0,
              "confidence":         0.923,
              "severity":           "critical",
              "alert":              True,
              "top_k_predictions":  [
                  {"class": "Ransomware",       "prob": 0.923},
                  {"class": "Zero_Day_Exploit", "prob": 0.041},
                  {"class": "APT",              "prob": 0.018},
              ],
              "semantic_context":   ["APT", "Botnet", "Phishing"],
              "attention_weights":  {class: weight, ...},
            }

        Пример:
            # Создаём синтетический инцидент (в реальности — из SIEM)
            incident = torch.randn(128)
            result   = analyzer.analyze(incident)
            if result["alert"]:
                send_alert(result["predicted_class"], result["severity"])
        """
        # 1. Препроцессинг
        x = self.preprocessor.process(raw_incident).to(self.device)  # (1, dim)

        # 2. Инференс
        with torch.no_grad():
            out = self.model(x, self.node_features, self.edge_index)

        probs = out["probs"][0]       # (n_classes,)
        attn  = out["attn"][0]        # (n_classes,) — semantic attention

        # 3. Основное предсказание
        conf, pred_idx = probs.max(dim=0)
        pred_idx  = pred_idx.item()
        conf      = conf.item()
        pred_name = self.class_names[pred_idx]

        # 4. Тяжесть угрозы по порогу уверенности
        severity = self._get_severity(conf)

        # 5. Топ-K предсказаний
        top_k_vals, top_k_idx = probs.topk(
            min(self.cfg.top_k, self.n_classes)
        )
        top_k = [
            {
                "class": self.class_names[i.item()],
                "prob":  round(v.item(), 4),
            }
            for v, i in zip(top_k_vals, top_k_idx)
        ]

        # 6. Семантический контекст из Knowledge Graph
        #    Топ-3 класса по attention (кроме предсказанного)
        attn_cpu   = attn.cpu()
        attn_items = [
            (self.class_names[i], attn_cpu[i].item())
            for i in range(self.n_classes)
            if i != pred_idx
        ]
        attn_items.sort(key=lambda x: x[1], reverse=True)
        semantic_context = [name for name, _ in attn_items[:3]]

        # 7. Полная карта внимания (для объяснения)
        attention_weights = {
            self.class_names[i]: round(attn_cpu[i].item(), 4)
            for i in range(self.n_classes)
        }

        return {
            "predicted_class":   pred_name,
            "class_idx":         pred_idx,
            "confidence":        round(conf, 4),
            "severity":          severity,
            "alert":             conf >= self.cfg.confidence_threshold,
            "top_k_predictions": top_k,
            "semantic_context":  semantic_context,
            "attention_weights": attention_weights,
        }

    def analyze_batch(self,
                      raw_incidents: List[Union[torch.Tensor, List[float]]]
                      ) -> List[Dict]:
        """
        Пакетный анализ списка инцидентов.

        Эффективнее чем последовательный вызов analyze() —
        один batched forward pass.

        Использование:
            incidents = [vec1, vec2, vec3, ...]
            results   = analyzer.analyze_batch(incidents)
            for r in results:
                print(r["predicted_class"], r["confidence"])
        """
        X = self.preprocessor.process_batch(raw_incidents).to(self.device)

        with torch.no_grad():
            out = self.model(X, self.node_features, self.edge_index)

        probs_batch = out["probs"]   # (B, n_classes)
        attn_batch  = out["attn"]    # (B, n_classes)

        results = []
        for b in range(len(raw_incidents)):
            probs = probs_batch[b]
            attn  = attn_batch[b]

            conf, pred_idx = probs.max(dim=0)
            pred_idx  = pred_idx.item()
            conf      = conf.item()
            pred_name = self.class_names[pred_idx]

            top_k_vals, top_k_idx = probs.topk(min(self.cfg.top_k, self.n_classes))
            top_k = [
                {"class": self.class_names[i.item()], "prob": round(v.item(), 4)}
                for v, i in zip(top_k_vals, top_k_idx)
            ]

            attn_cpu   = attn.cpu()
            attn_items = [
                (self.class_names[i], attn_cpu[i].item())
                for i in range(self.n_classes) if i != pred_idx
            ]
            attn_items.sort(key=lambda x: x[1], reverse=True)

            results.append({
                "predicted_class":  pred_name,
                "class_idx":        pred_idx,
                "confidence":       round(conf, 4),
                "severity":         self._get_severity(conf),
                "alert":            conf >= self.cfg.confidence_threshold,
                "top_k_predictions":top_k,
                "semantic_context": [n for n, _ in attn_items[:3]],
            })

        return results

    # ──────────────────────────────────────────────────────────
    # XAI — ОБЪЯСНЕНИЕ ПРЕДСКАЗАНИЯ
    # ──────────────────────────────────────────────────────────

    def explain(self,
                raw_incident: Union[torch.Tensor, List[float], Dict],
                verbose:      bool = True) -> Dict:
        """
        Объясняет предсказание модели через SemanticAggregator attention.

        Механизм объяснения:
            attention_weights[c] = насколько модель "смотрела" на класс c
            в Knowledge Graph при анализе данного инцидента.

            Высокий вес y класса c означает:
            "По семантике этого инцидента Knowledge Graph
             ассоциирует ego c угрозой типа c"

        Возвращает:
            {
                "predicted_class": "Ransomware",
                "confidence":      0.923,
                "explanation": {
                    "top_attention": [
                        {"class": "APT",      "weight": 0.312,
                         "relation": "Ransomware semantically precedes APT"},
                        {"class": "Botnet",   "weight": 0.241,
                         "relation": "Botnet often deploys Ransomware"},
                        {"class": "Phishing", "weight": 0.187,
                         "relation": "Phishing precedes Ransomware"},
                    ],
                    "interpretation": "Модель уверена на 92.3%...",
                }
            }
        """
        result = self.analyze(raw_incident)

        attn = result["attention_weights"]

        # Сортируем по весу
        sorted_attn = sorted(attn.items(), key=lambda x: x[1], reverse=True)

        # Строим объяснение для топ-3
        top_attention = []
        for cls_name, weight in sorted_attn[:3]:
            relation = self._get_semantic_relation(
                result["predicted_class"], cls_name
            )
            top_attention.append({
                "class":    cls_name,
                "weight":   weight,
                "relation": relation,
            })

        # Текстовая интерпретация
        conf_pct = result["confidence"] * 100
        interpretation = (
            f"Модель классифицирует инцидент как '{result['predicted_class']}' "
            f"с уверенностью {conf_pct:.1f}% (тяжесть: {result['severity']}). "
            f"Knowledge Graph указывает на семантическую близость с: "
            f"{', '.join(result['semantic_context'])}."
        )

        explanation = {
            "predicted_class": result["predicted_class"],
            "confidence":      result["confidence"],
            "severity":        result["severity"],
            "explanation": {
                "top_attention":  top_attention,
                "interpretation": interpretation,
            }
        }

        if verbose:
            self._print_explanation(explanation)

        return explanation

    # ──────────────────────────────────────────────────────────
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ──────────────────────────────────────────────────────────

    def _get_severity(self, confidence: float) -> str:
        """
        Определяет тяжесть угрозы по порогам уверенности модели.

        Пороги задаются в INFERENCE_CFG.severity_thresholds.
        Логика: confidence ≥ threshold → severity level.
        """
        thresholds = self.cfg.severity_thresholds
        # Сортируем по убыванию порога
        for level in ["critical", "high", "medium", "low"]:
            if confidence >= thresholds.get(level, 0.0):
                return level
        return "low"

    def _get_semantic_relation(self,
                               pred_class:    str,
                               related_class: str) -> str:
        """
        Возвращает описание семантической связи между классами.

        В production здесь был бы запрос к Knowledge Graph
        для получения реального типа ребра.

        TODO: подключить к graph_data["edge_index"] + ["edge_type"]
        для получения реального типа ребра между узлами.
        """
        # Упрощённые шаблоны (замени на реальный lookup по графу)
        templates = {
            "precedes":    f"{pred_class} часто предшествует {related_class}",
            "exploits":    f"{pred_class} эксплуатирует уязвимости, типичные для {related_class}",
            "targets":     f"{pred_class} и {related_class} нацелены на одни активы",
            "similar_to":  f"{pred_class} семантически схожа с {related_class}",
            "mitigated_by":f"{pred_class} и {related_class} снижаются одними мерами",
        }
        # По умолчанию — общая связь
        return templates.get(
            "similar_to",
            f"{related_class} семантически связана с {pred_class}"
        )

    def _print_explanation(self, explanation: Dict) -> None:
        """Форматированный вывод объяснения в консоль."""
        sep = "─" * 55
        print(f"\n{sep}")
        print(f"  THREAT ANALYSIS REPORT")
        print(sep)
        print(f"  Predicted:  {explanation['predicted_class']}")
        print(f"  Confidence: {explanation['confidence']:.1%}")
        print(f"  Severity:   {explanation['severity'].upper()}")
        print(f"\n  Semantic Context (Knowledge Graph attention):")
        for item in explanation["explanation"]["top_attention"]:
            bar_len = int(item["weight"] * 30)
            bar     = "█" * bar_len + "░" * (30 - bar_len)
            print(f"    {item['class']:20s} [{bar}] {item['weight']:.3f}")
        print(f"\n  Interpretation:")
        print(f"    {explanation['explanation']['interpretation']}")
        print(sep)

    # ──────────────────────────────────────────────────────────
    # УТИЛИТЫ ДЛЯ ДЕПЛОЯ
    # ──────────────────────────────────────────────────────────

    def get_model_info(self) -> Dict:
        """Возвращает метаданные модели для API / мониторинга."""
        from utils import count_model_params
        params = count_model_params(self.model)
        return {
            "n_classes":        self.n_classes,
            "class_names":      self.class_names,
            "input_dim":        self.model.in_dim,
            "parameters":       params,
            "device":           str(self.device),
            "confidence_threshold": self.cfg.confidence_threshold,
        }

    def export_to_onnx(self, output_path: str) -> None:
        """
        Экспортирует модель в ONNX для деплоя без PyTorch.

        TODO — РЕАЛИЗУЙ:

        Проблема: model.forward() принимает 3 аргумента
        (incident_x, node_features, edge_index), но ONNX
        требует фиксированного числа входов.

        Решение: создай wrapper:

        class ONNXWrapper(nn.Module):
            def __init__(self, model, node_features, edge_index):
                super().__init__()
                self.model         = model
                self.node_features = node_features
                self.edge_index    = edge_index

            def forward(self, incident_x):
                out = self.model(incident_x,
                                 self.node_features,
                                 self.edge_index)
                return out["probs"]

        wrapper = ONNXWrapper(self.model,
                              self.node_features,
                              self.edge_index)
        dummy   = torch.randn(1, self.model.in_dim)
        torch.onnx.export(
            wrapper, dummy, output_path,
            input_names  = ["incident_features"],
            output_names = ["class_probabilities"],
            dynamic_axes = {"incident_features": {0: "batch_size"}},
            opset_version= 17,
        )
        """
        raise NotImplementedError(
            "Реализуй export_to_onnx() по алгоритму в docstring"
        )


# ══════════════════════════════════════════════════════════════
# TODO — ЧТО ОСТАВЛЕНО ТЕБЕ
# ══════════════════════════════════════════════════════════════
#
# 1. REST API (FastAPI):
#
#    from fastapi import FastAPI
#    from pydantic import BaseModel
#
#    app      = FastAPI(title="CyberThreat AI API")
#    analyzer = None   # инициализируется при старте
#
#    class IncidentRequest(BaseModel):
#        features: List[float]   # 128 значений
#
#    @app.on_event("startup")
#    async def startup():
#        global analyzer
#        analyzer = ThreatAnalyzer.from_checkpoint(PATHS["best_model"],
#                                                  graph_data)
#
#    @app.post("/analyze")
#    async def analyze(req: IncidentRequest):
#        return analyzer.analyze(req.features)
#
#    @app.post("/explain")
#    async def explain(req: IncidentRequest):
#        return analyzer.explain(req.features, verbose=False)
#
# 2. БАТЧ-ИНФЕРЕНС ИЗ CSV:
#
#    def analyze_csv(csv_path: str, output_path: str):
#        df = pd.read_csv(csv_path)
#        feature_cols = [c for c in df.columns if c.startswith("feat_")]
#        incidents    = df[feature_cols].values.tolist()
#        results      = analyzer.analyze_batch(incidents)
#        out_df = pd.DataFrame(results)
#        out_df.to_csv(output_path, index=False)
#
# 3. THRESHOLD CALIBRATION (Platt scaling):
#
#    from sklearn.calibration import CalibratedClassifierCV
#    # Или вручную: обучи LogisticRegression на логитах val-выборки
#    # → откалиброванные вероятности лучше соответствуют реальным
#
# 4. ONNX EXPORT — см. export_to_onnx() docstring выше
#
# ══════════════════════════════════════════════════════════════