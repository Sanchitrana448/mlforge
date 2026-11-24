import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.main import app
from app.threshold import best_threshold, metrics_at

SAMPLE = dict(
    tenure_months=3, monthly_charge=150, total_charge=450, num_support_tickets=4,
    contract_type=0, has_internet_addon=1, avg_monthly_usage_gb=200, late_payments_last_year=2,
)


def test_metrics_at_counts_cost_correctly():
    y = [1, 1, 0, 0]
    p = [0.9, 0.2, 0.6, 0.1]
    m = metrics_at(y, p, 0.5, offer_cost=20, lost_customer_cost=200)
    # flagged: idx0 (tp) and idx2 (fp) -> 2 offers; missed churner idx1 -> 1 loss
    assert m["flagged"] == 2 and m["total_cost"] == 2 * 20 + 200
    assert m["recall"] == 0.5 and m["precision"] == 0.5


def test_cheap_offers_push_threshold_down_and_dear_offers_push_it_up():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    low = best_threshold(y, p, offer_cost=1, lost_customer_cost=500)
    high = best_threshold(y, p, offer_cost=500, lost_customer_cost=1)
    assert low < high


def test_model_metadata_includes_operating_point_computed_on_validation():
    with TestClient(app) as client:
        meta = client.get("/model").json()
    op = meta["operating_point"]
    assert op["threshold_chosen_on"] == "validation set"
    assert op["test_at_recommended"]["total_cost"] <= op["test_at_default_0.5"]["total_cost"]


def test_predict_threshold_override_and_reset():
    with TestClient(app) as client:
        client.post("/live/reset")
        base = client.post("/predict", json=SAMPLE).json()
        assert base["threshold"] == 0.5
        strict = client.post("/predict?threshold=1.0", json=SAMPLE).json()
        assert strict["prediction"] == 0
        lenient = client.post("/predict?threshold=0.0", json=SAMPLE).json()
        assert lenient["prediction"] == 1
        assert client.post("/predict?threshold=1.5", json=SAMPLE).status_code == 422
        assert client.get("/metrics").json()["predictions_served"] == 3
        client.post("/live/reset")
        assert client.get("/metrics").json()["predictions_served"] == 0


def test_drifted_traffic_is_flagged():
    drifted = dict(SAMPLE, monthly_charge=200, num_support_tickets=8)
    with TestClient(app) as client:
        client.post("/live/reset")
        for _ in range(20):
            client.post("/predict", json=drifted)
        report = client.get("/drift").json()
        client.post("/live/reset")
    assert report["any_drift_detected"] is True
