"""
Synthetic customer-churn dataset generator + data validation.

Business problem: predict whether a telecom customer will churn based on
account/usage features. A synthetic generator is used so the project is
100% self-contained and reproducible without needing an external dataset
download — the generation process itself (seeded, documented feature/label
relationships) is part of the engineering artifact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charge",
    "total_charge",
    "num_support_tickets",
    "contract_type",  # 0=month-to-month, 1=one-year, 2=two-year
    "has_internet_addon",
    "avg_monthly_usage_gb",
    "late_payments_last_year",
]
TARGET_COLUMN = "churned"


def generate_dataset(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 72, n_rows)
    contract_type = rng.choice([0, 1, 2], size=n_rows, p=[0.55, 0.25, 0.20])
    monthly_charge = np.round(rng.normal(70, 25, n_rows).clip(15, 200), 2)
    total_charge = np.round(monthly_charge * (tenure + 1) * rng.uniform(0.9, 1.05, n_rows), 2)
    support_tickets = rng.poisson(1.2, n_rows)
    has_internet = rng.choice([0, 1], size=n_rows, p=[0.25, 0.75])
    usage_gb = np.round(rng.normal(180, 80, n_rows).clip(0, None), 1)
    late_payments = rng.poisson(0.6, n_rows)

    # Ground-truth churn probability driven by a known, documented function of
    # the features (so we can sanity-check model performance against it).
    logit = (
        -1.2
        - 0.05 * tenure
        + 0.015 * monthly_charge
        + 0.35 * support_tickets
        - 0.9 * (contract_type == 2)
        - 0.4 * (contract_type == 1)
        + 0.5 * late_payments
        - 0.15 * has_internet
        + rng.normal(0, 0.6, n_rows)
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churned = (rng.uniform(0, 1, n_rows) < prob_churn).astype(int)

    df = pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charge": monthly_charge,
            "total_charge": total_charge,
            "num_support_tickets": support_tickets,
            "contract_type": contract_type,
            "has_internet_addon": has_internet,
            "avg_monthly_usage_gb": usage_gb,
            "late_payments_last_year": late_payments,
            "churned": churned,
        }
    )
    return df


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Data-quality checks run before training. Returns a list of issues (empty = clean)."""
    issues = []
    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        return issues
    if df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().any().any():
        issues.append("Null values found in feature or target columns")
    if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
        issues.append("Target column contains values outside {0,1}")
    if (df["tenure_months"] < 0).any():
        issues.append("Negative tenure_months found")
    if (df["monthly_charge"] <= 0).any():
        issues.append("Non-positive monthly_charge found")
    class_balance = df[TARGET_COLUMN].mean()
    if class_balance < 0.02 or class_balance > 0.98:
        issues.append(f"Severe class imbalance detected: positive rate={class_balance:.3f}")
    return issues
