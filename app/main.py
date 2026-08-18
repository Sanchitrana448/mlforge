from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .data import FEATURE_COLUMNS
from .drift import compute_drift
from .train import REGISTRY_DIR, train_and_register

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlforge")

app = FastAPI(
    title="MLForge",
    description="Production ML intelligence & MLOps platform — training, evaluation, serving, drift detection, monitoring.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = Path(__file__).parent.parent / "frontend"

_STATE = {"model": None, "scaler": None, "metadata": None, "live_samples": [], "prediction_log": []}


def _load_current_model():
    current_path = REGISTRY_DIR / "current.json"
    if not current_path.exists():
        return False
    version = json.loads(current_path.read_text())["version"]
    bundle = joblib.load(REGISTRY_DIR / f"{version}.joblib")
    metadata = json.loads((REGISTRY_DIR / f"{version}.json").read_text())
    _STATE["model"] = bundle["model"]
    _STATE["scaler"] = bundle["scaler"]
    _STATE["metadata"] = metadata
    return True


@app.on_event("startup")
def startup():
    if not _load_current_model():
        logger.info("No trained model found — training a fresh one at startup.")
        train_and_register()
        _load_current_model()


class PredictRequest(BaseModel):
    tenure_months: float
    monthly_charge: float
    total_charge: float
    num_support_tickets: float
    contract_type: int = Field(ge=0, le=2)
    has_internet_addon: int = Field(ge=0, le=1)
    avg_monthly_usage_gb: float
    late_payments_last_year: float


class BatchPredictRequest(BaseModel):
    records: list[PredictRequest]


def _predict_one(req: PredictRequest) -> dict:
    if _STATE["model"] is None:
        raise HTTPException(503, "Model not loaded")
    row = [getattr(req, col) for col in FEATURE_COLUMNS]
    X = _STATE["scaler"].transform([row])
    model = _STATE["model"]
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else None

    sample = dict(zip(FEATURE_COLUMNS, row))
    _STATE["live_samples"].append(sample)
    if len(_STATE["live_samples"]) > 2000:
        _STATE["live_samples"] = _STATE["live_samples"][-2000:]

    record = {
        "prediction": pred,
        "probability": round(prob, 4) if prob is not None else None,
        "model_version": _STATE["metadata"]["version"],
        "timestamp": time.time(),
        "request_id": str(uuid.uuid4())[:8],
    }
    _STATE["prediction_log"].append(record)
    if len(_STATE["prediction_log"]) > 500:
        _STATE["prediction_log"] = _STATE["prediction_log"][-500:]
    return record


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _STATE["model"] is not None}


@app.get("/model")
def model_info():
    if not _STATE["metadata"]:
        raise HTTPException(503, "No model loaded")
    return _STATE["metadata"]


@app.post("/predict")
def predict(req: PredictRequest):
    return _predict_one(req)


@app.post("/batch-predict")
def batch_predict(req: BatchPredictRequest):
    return {"predictions": [_predict_one(r) for r in req.records]}


@app.post("/train")
def retrain(n_rows: int = 5000):
    """Trigger a fresh training run and hot-swap the serving model."""
    metadata = train_and_register(n_rows=n_rows)
    _load_current_model()
    return metadata


@app.get("/metrics")
def metrics():
    if not _STATE["metadata"]:
        raise HTTPException(503, "No model loaded")
    log = _STATE["prediction_log"]
    churn_rate = (sum(r["prediction"] for r in log) / len(log)) if log else None
    return {
        "model_version": _STATE["metadata"]["version"],
        "validation_metrics": _STATE["metadata"]["validation_metrics"],
        "test_metrics": _STATE["metadata"]["test_metrics"],
        "predictions_served": len(log),
        "live_predicted_churn_rate": round(churn_rate, 4) if churn_rate is not None else None,
    }


@app.get("/drift")
def drift_report():
    if not _STATE["metadata"]:
        raise HTTPException(503, "No model loaded")
    reports = compute_drift(_STATE["metadata"]["training_feature_stats"], _STATE["live_samples"])
    return {
        "n_live_samples": len(_STATE["live_samples"]),
        "features": [r.__dict__ for r in reports],
        "any_drift_detected": any(r.drifted for r in reports),
    }


@app.get("/predictions/recent")
def recent_predictions(limit: int = 20):
    return list(reversed(_STATE["prediction_log"][-limit:]))


@app.get("/", response_class=HTMLResponse)
def index_page():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>MLForge</h1><p>Frontend not built. See /docs for API.</p>"
