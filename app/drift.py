"""
Feature-drift detection: compares live prediction-request feature statistics
against the training-time distribution using a standardized mean shift
(z-score of the running mean) — a lightweight, dependency-free proxy for
population stability index (PSI) that's easy to reason about and test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

DRIFT_Z_THRESHOLD = 2.5


@dataclass
class FeatureDriftReport:
    feature: str
    training_mean: float
    live_mean: float
    z_shift: float
    drifted: bool


def compute_drift(
    training_stats: Dict[str, dict], live_samples: List[Dict[str, float]]
) -> List[FeatureDriftReport]:
    if not live_samples:
        return []
    reports = []
    for feature, stats in training_stats.items():
        values = [s[feature] for s in live_samples if feature in s]
        if not values:
            continue
        live_mean = sum(values) / len(values)
        std = stats["std"] or 1e-6
        z_shift = abs(live_mean - stats["mean"]) / std
        reports.append(
            FeatureDriftReport(
                feature=feature,
                training_mean=round(stats["mean"], 3),
                live_mean=round(live_mean, 3),
                z_shift=round(z_shift, 3),
                drifted=z_shift > DRIFT_Z_THRESHOLD,
            )
        )
    return reports
