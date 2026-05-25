"""Tests for the 6-block-ahead inference pipeline.

These checks require local trained model artifacts and local prediction outputs.
Those files are intentionally ignored by Git, so the test skips cleanly in a
fresh clone until artifacts are generated locally.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inference.inference_pipeline import FSPInferencePipeline


MODELS_DIR = Path("outputs/models")
PREDICTIONS_FILE = Path("outputs/predictions/test_predictions_ridge.csv")


def test_6block_inference_pipeline_with_local_artifacts():
    if not MODELS_DIR.exists() or not any(MODELS_DIR.glob("*.pkl")):
        pytest.skip("Local model artifacts are not available under outputs/models.")

    if not PREDICTIONS_FILE.exists():
        pytest.skip("Local prediction artifact is not available.")

    try:
        pipeline = FSPInferencePipeline(models_dir=str(MODELS_DIR))
    except ModuleNotFoundError as exc:
        pytest.skip(f"Optional model dependency is not installed: {exc.name}")

    assert pipeline.prediction_horizon_from_metadata == 6
    assert pipeline.model_metadata.get("target_column") == "target_horizon"

    test_data = pd.read_csv(PREDICTIONS_FILE)
    required_columns = {
        "ml_predicted_power",
        "actual_power",
        "selection_confidence",
        "ml_selected_fsp",
    }
    assert required_columns.issubset(test_data.columns)

    mae = np.mean(np.abs(test_data["ml_predicted_power"] - test_data["actual_power"]))
    assert np.isfinite(mae)
    assert len(test_data["ml_selected_fsp"].dropna()) > 0
