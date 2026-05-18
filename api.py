# api.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import torch

from config import PATHS
from dataset import get_or_generate_graph_data
from inference import ThreatAnalyzer
from utils import get_device

# ── Глобальное состояние ──────────────────────────────────────
analyzer: ThreatAnalyzer = None


# ── Lifespan (современная замена on_event) ────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружает модель при старте, освобождает при остановке."""
    global analyzer
    print("[startup] Loading graph data...")
    graph_data = get_or_generate_graph_data()
    print("[startup] Loading model...")
    analyzer = ThreatAnalyzer.from_checkpoint(
        checkpoint_path = PATHS["best_model"],
        graph_data      = graph_data,
        device          = get_device(),
    )
    print(f"[startup] Ready. {analyzer.n_classes} threat classes.")
    yield
    # Код после yield — при остановке сервера
    print("[shutdown] Shutting down.")


# ── Приложение ────────────────────────────────────────────────
app = FastAPI(
    title       = "CyberThreat Intelligence API",
    description = "Graph Attention Network + Knowledge Graph для классификации кибератак",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ── Схемы запросов/ответов ────────────────────────────────────
class IncidentRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length = 128,
        max_length = 128,
        description = "Вектор признаков инцидента (128 значений)",
    )

class AnalyzeResponse(BaseModel):
    predicted_class:   str
    class_idx:         int
    confidence:        float
    severity:          str
    alert:             bool
    top_k_predictions: List[dict]
    semantic_context:  List[str]
    attention_weights: dict

class ExplainResponse(BaseModel):
    predicted_class: str
    confidence:      float
    severity:        str
    explanation:     dict


# ── Роуты ─────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "CyberThreat Intelligence API",
        "version": "1.0.0",
        "endpoints": {
            "GET  /":        "Эта страница",
            "GET  /health":  "Статус модели",
            "POST /analyze": "Классификация инцидента",
            "POST /explain": "Классификация + XAI объяснение",
            "GET  /classes": "Список классов угроз",
            "GET  /docs":    "Swagger UI",
        }
    }


@app.get("/health")
async def health():
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    info = analyzer.get_model_info()
    return {
        "status":      "ok",
        "classes":     analyzer.class_names,
        "n_classes":   analyzer.n_classes,
        "parameters":  info["parameters"]["total"],
        "device":      str(analyzer.device),
        "threshold":   analyzer.cfg.confidence_threshold,
    }


@app.get("/classes")
async def get_classes():
    """Возвращает все классы угроз с индексами."""
    return {
        "classes": [
            {"idx": i, "name": name}
            for i, name in enumerate(analyzer.class_names)
        ]
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: IncidentRequest):
    """
    Классифицирует инцидент кибербезопасности.

    Возвращает:
    - predicted_class: название угрозы
    - confidence: уверенность модели [0, 1]
    - severity: critical / high / medium / low
    - alert: True если confidence > threshold (0.70)
    - semantic_context: связанные классы из Knowledge Graph
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = analyzer.analyze(req.features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplainResponse)
async def explain(req: IncidentRequest):
    """
    Классифицирует инцидент и объясняет предсказание через
    attention weights Knowledge Graph (XAI).
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = analyzer.explain(req.features, verbose=False)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch")
async def batch_analyze(requests_list: List[IncidentRequest]):
    """
    Пакетная классификация нескольких инцидентов за один запрос.
    Максимум 100 инцидентов.
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(requests_list) > 100:
        raise HTTPException(status_code=400,
                            detail="Max 100 incidents per batch")
    try:
        features_list = [r.features for r in requests_list]
        results = analyzer.analyze_batch(features_list)
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Запуск напрямую (python api.py) ──────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host    = "127.0.0.1",
        port    = 8000,
        reload  = True,
        workers = 1,
    )