import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data import generate_dataset, validate_dataset, FEATURE_COLUMNS
from app.drift import compute_drift


def test_generate_dataset_shape_and_validity():
    df = generate_dataset(n_rows=500, seed=1)
    assert len(df) == 500
    assert set(FEATURE_COLUMNS + ["churned"]).issubset(df.columns)
    issues = validate_dataset(df)
    assert issues == []


def test_generate_dataset_is_reproducible_with_seed():
    df1 = generate_dataset(n_rows=200, seed=7)
    df2 = generate_dataset(n_rows=200, seed=7)
    assert df1.equals(df2)


def test_validate_dataset_flags_missing_columns():
    import pandas as pd

    df = pd.DataFrame({"tenure_months": [1, 2]})
    issues = validate_dataset(df)
    assert any("Missing required columns" in i for i in issues)


def test_validate_dataset_flags_bad_target():
    df = generate_dataset(n_rows=100, seed=3)
    df["churned"] = 5  # invalid class
    issues = validate_dataset(df)
    assert any("Target column" in i for i in issues)


def test_compute_drift_no_drift_when_matching_distribution():
    stats = {"tenure_months": {"mean": 30.0, "std": 10.0}}
    live = [{"tenure_months": 30.5}, {"tenure_months": 29.0}, {"tenure_months": 31.0}]
    reports = compute_drift(stats, live)
    assert len(reports) == 1
    assert reports[0].drifted is False


def test_compute_drift_detects_large_shift():
    stats = {"tenure_months": {"mean": 30.0, "std": 5.0}}
    live = [{"tenure_months": 80.0}] * 10
    reports = compute_drift(stats, live)
    assert reports[0].drifted is True


def test_compute_drift_empty_live_samples():
    stats = {"tenure_months": {"mean": 30.0, "std": 5.0}}
    reports = compute_drift(stats, [])
    assert reports == []
