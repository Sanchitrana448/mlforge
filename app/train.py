"""
Training pipeline: data -> validation -> split -> baseline + advanced models
-> evaluation -> model registry.

Run directly: `python -m app.train`
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import FEATURE_COLUMNS, TARGET_COLUMN, generate_dataset, validate_dataset

REGISTRY_DIR = Path(__file__).parent.parent / "model_registry"
REGISTRY_DIR.mkdir(exist_ok=True)


def _evaluate(name: str, model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
    cm = confusion_matrix(y_test, preds).tolist()
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
        "confusion_matrix": cm,
    }


def train_and_register(n_rows: int = 5000, seed: int = 42) -> dict:
    df = generate_dataset(n_rows=n_rows, seed=seed)
    issues = validate_dataset(df)
    if issues:
        raise ValueError(f"Data validation failed: {issues}")

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s, X_test_s = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }

    results = []
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train_s, y_train)
        fitted[name] = model
        results.append(_evaluate(name, model, X_val_s, y_val))

    best = max(results, key=lambda r: r["roc_auc"])
    best_model = fitted[best["model"]]
    test_metrics = _evaluate(best["model"] + "_test", best_model, X_test_s, y_test)

    version = f"v{int(time.time())}"
    model_path = REGISTRY_DIR / f"{version}.joblib"
    joblib.dump({"model": best_model, "scaler": scaler, "features": FEATURE_COLUMNS}, model_path)

    training_stats = {
        col: {"mean": float(np.mean(X_train[:, i])), "std": float(np.std(X_train[:, i]))}
        for i, col in enumerate(FEATURE_COLUMNS)
    }

    metadata = {
        "version": version,
        "trained_at": time.time(),
        "n_rows": n_rows,
        "features": FEATURE_COLUMNS,
        "candidates_evaluated": results,
        "selected_model": best["model"],
        "selection_criterion": "roc_auc (validation set)",
        "validation_metrics": best,
        "test_metrics": test_metrics,
        "training_feature_stats": training_stats,
        "class_balance": float(np.mean(y_train)),
    }
    (REGISTRY_DIR / f"{version}.json").write_text(json.dumps(metadata, indent=2))
    (REGISTRY_DIR / "current.json").write_text(json.dumps({"version": version}, indent=2))

    return metadata


if __name__ == "__main__":
    meta = train_and_register()
    print(json.dumps(meta, indent=2))
