"""Operating-point analysis: pick the decision threshold from costs, not from habit.

Churn is asymmetric. Missing a customer who leaves (false negative) costs far
more than sending an unnecessary retention offer (false positive), so the 0.5
default is rarely the right threshold. The threshold is chosen on the
validation set and then reported on the held-out test set, so the test number
is not tuned.

Cost assumptions are parameters, not facts about any real business.
"""
from __future__ import annotations

import numpy as np

DEFAULT_COSTS = {"offer_cost": 20.0, "lost_customer_cost": 200.0}
GRID = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]


def metrics_at(y_true, probs, threshold: float, offer_cost: float, lost_customer_cost: float) -> dict:
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(probs) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    # Every flagged customer gets an offer; every missed churner is lost.
    cost = (tp + fp) * offer_cost + fn * lost_customer_cost
    return {
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "flagged": tp + fp,
        "total_cost": round(cost, 2),
    }


def best_threshold(y_val, val_probs, offer_cost: float, lost_customer_cost: float) -> float:
    rows = [metrics_at(y_val, val_probs, t, offer_cost, lost_customer_cost) for t in GRID]
    return min(rows, key=lambda r: (r["total_cost"], r["threshold"]))["threshold"]


def operating_point(y_val, val_probs, y_test, test_probs, costs: dict | None = None) -> dict:
    c = {**DEFAULT_COSTS, **(costs or {})}
    chosen = best_threshold(y_val, val_probs, **c)
    show = sorted({0.5, chosen, 0.4, 0.3, 0.2})
    return {
        "assumptions": c,
        "threshold_chosen_on": "validation set",
        "recommended_threshold": chosen,
        "test_at_default_0.5": metrics_at(y_test, test_probs, 0.5, **c),
        "test_at_recommended": metrics_at(y_test, test_probs, chosen, **c),
        "test_table": [metrics_at(y_test, test_probs, t, **c) for t in show],
    }
