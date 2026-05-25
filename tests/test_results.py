"""Evaluation checks for locally generated prediction artifacts.

The prediction CSV files are generated outputs and are intentionally ignored by
Git. These tests validate them when present and skip in a clean source-only
checkout.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PRED_DIR = Path("outputs/predictions")
MODELS = ["ridge", "random_forest", "xgboost", "lightgbm", "ensemble_ridge_lgb"]


def calc_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))
    pct_error = np.abs((y_pred - y_true) / (y_true + 1e-10)) * 100
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "sMAPE": smape,
        "Within_15": np.mean(pct_error <= 15) * 100,
        "Within_25": np.mean(pct_error <= 25) * 100,
    }


def test_prediction_artifact_metrics_when_available():
    results = {}

    for model in MODELS:
        path = PRED_DIR / f"test_predictions_{model}.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path)
        required_columns = {"actual_power", "ml_scheduled_power"}
        assert required_columns.issubset(df.columns)

        y_true = df["actual_power"].values
        y_pred = df["ml_scheduled_power"].values
        results[model] = calc_metrics(y_true, y_pred)

    if not results:
        pytest.skip("Local prediction artifacts are not available under outputs/predictions.")

    best_model, best_metrics = min(results.items(), key=lambda item: item[1]["MAE"])
    best_path = PRED_DIR / f"test_predictions_{best_model}.csv"
    assert best_path.exists()
    assert np.isfinite(best_metrics["MAE"])
    assert np.isfinite(best_metrics["RMSE"])
