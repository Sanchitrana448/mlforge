# MLForge

**Production ML intelligence & MLOps platform.** A complete customer-churn-prediction system covering the full lifecycle: data validation → multi-model training/benchmarking → model registry → FastAPI serving → live monitoring dashboard → feature-drift detection → one-click retraining.

## Why this project exists

MLOps — not just model training — is what separates an "ML notebook" from an ML *engineer*. This project demonstrates the parts hiring panels actually probe: model comparison methodology, a real model registry, a served API with proper response contracts, and production monitoring including drift detection.

## Business problem

Predict telecom customer churn from account/usage features (tenure, contract type, monthly charge, support tickets, late payments, usage). Dataset is a seeded, reproducible synthetic generator with a documented, known feature→label relationship (see `app/data.py`) — this makes the project fully self-contained (no external data dependency) while still allowing real benchmarking against a known ground truth.

## Architecture

```
Data generation & validation
        │
        ▼
Train/val/test split (stratified)
        │
        ▼
Benchmark: Logistic Regression vs Random Forest vs Gradient Boosting
        │
        ▼
Best model selected by validation ROC-AUC → evaluated on held-out test set
        │
        ▼
Model registry (versioned .joblib + metadata JSON, "current" pointer)
        │
        ▼
FastAPI serving (/predict, /batch-predict, /model, /metrics)
        │
        ▼
Live monitoring: prediction log, feature-drift z-shift detector, dashboard
        │
        ▼
POST /train → hot-swaps a freshly trained model into serving
```

## Run it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# A model trains automatically on first startup if none exists.
# open http://localhost:8000 for the monitoring dashboard
```

Or train explicitly first:

```bash
python -m app.train
```

Docker:

```bash
docker build -t mlforge .
docker run -p 8000:8000 mlforge
```

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/predict` | Single prediction |
| `POST` | `/batch-predict` | Batch predictions |
| `POST` | `/train?n_rows=5000` | Retrain and hot-swap the serving model |
| `GET` | `/model` | Full model metadata: candidates evaluated, metrics, feature stats |
| `GET` | `/metrics` | Live serving metrics (predictions served, live churn rate) |
| `GET` | `/drift` | Per-feature drift report vs training distribution |
| `GET` | `/predictions/recent` | Recent prediction log |

## Verified results (this repo, measured — not illustrative)

On a fresh training run (5,000 synthetic rows, seed=42), three candidate models were benchmarked on a held-out validation set; the best by ROC-AUC was selected and re-evaluated on a separate held-out test set:

- Selected model: **logistic_regression**
- Validation ROC-AUC: **0.806**, F1: **0.459**
- Test ROC-AUC: **0.764**, accuracy: **0.771**

Live-traffic smoke test: a high-risk synthetic customer (2-month tenure, 4 support tickets, 3 late payments, month-to-month contract) scored **94.6% churn probability**; a low-risk customer (60-month tenure, 2-year contract, 0 support tickets) scored **1.3%** — confirming the model has learned the intended risk signal, not noise. Drift endpoint verified against both a matching and an artificially shifted distribution (`tests/test_pipeline.py`).

## Tests

```bash
pytest tests/ -v
```

7/7 passing: dataset generation/reproducibility, data validation (missing columns, invalid target), and drift detection (no false positive on matching distributions, true positive on a large synthetic shift).

## Tech stack

Python · FastAPI · scikit-learn · pandas · joblib (model registry) · Docker · pytest.

## Case study (recruiter summary)

**Problem:** Most "ML portfolio" projects stop at `model.fit()` in a notebook — they don't show what happens after the model works.
**Approach:** Built the full loop — validated data, benchmarked 3 model families with proper train/val/test discipline, a lightweight but real model registry, a served API, and a monitoring dashboard with drift detection based on live-traffic feature statistics vs training statistics.
**Result (measured, this repo):** 7/7 tests passing; trained model achieves 0.806 validation ROC-AUC and correctly separates high/low-risk customers by two orders of magnitude in predicted probability on a live smoke test.
**What I'd do next:** add MLflow/W&B experiment tracking, real PSI-based drift scoring, a Celery-based async retraining trigger, and Prometheus/Grafana for production-grade observability.
