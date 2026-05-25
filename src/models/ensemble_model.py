"""
Ensemble Model: Ridge + LightGBM
=================================

Combines Ridge regression and LightGBM using weighted averaging.
Leverages Ridge's simplicity and interpretability with LightGBM's gradient boosting power.

Maintainer: Project Team
Date: January 2026
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer


def _ensure_imputer_compat(imputer):
    """Patch SimpleImputer loaded from pickle with older sklearn (missing _fill_dtype)."""
    if imputer is None:
        return
    if isinstance(imputer, SimpleImputer) and not hasattr(imputer, '_fill_dtype'):
        if hasattr(imputer, 'statistics_') and imputer.statistics_ is not None:
            imputer._fill_dtype = np.asarray(imputer.statistics_).dtype
        else:
            imputer._fill_dtype = np.float64


class RidgeLightGBMEnsemble(BaseEstimator, RegressorMixin):
    """
    Ensemble combining Ridge (scaled features) and LightGBM (raw features).

    Parameters:
    -----------
    ridge_model : sklearn Ridge regressor
        Ridge model trained on scaled features
    lightgbm_model : LightGBM regressor
        LightGBM model trained on raw features
    ridge_weight : float, default=0.4
        Weight for Ridge predictions (0-1). LightGBM gets 1 - ridge_weight.
    """

    def __init__(
        self,
        imputer=None,
        ridge_model=None,
        lightgbm_model=None,
        ridge_weight: float = 0.4,
        scaler=None
    ):
        self.imputer = imputer
        self.ridge_model = ridge_model
        self.lightgbm_model = lightgbm_model
        self.ridge_weight = ridge_weight
        self.scaler = scaler

        # Validate weights
        if not (0 <= ridge_weight <= 1):
            raise ValueError(f"ridge_weight must be between 0 and 1, got {ridge_weight}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using weighted ensemble.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)

        # Apply imputation if needed
        if self.imputer is not None:
            X = self.imputer.transform(X)
        # Final safety: replace any remaining NaNs
        X = np.nan_to_num(X, nan=0.0)

        Returns:
        --------
        np.ndarray
            Ensemble predictions
        """
        if self.ridge_model is None or self.lightgbm_model is None:
            raise ValueError("Models not fitted. Train or load them first.")

        # Ridge uses scaled features
        X_scaled = self.scaler.transform(X) if self.scaler is not None else X
        ridge_pred = self.ridge_model.predict(X_scaled)

        # LightGBM uses raw features
        lgb_pred = self.lightgbm_model.predict(X)

        # Weighted ensemble
        lgb_weight = 1.0 - self.ridge_weight
        ensemble_pred = (self.ridge_weight * ridge_pred +
                        lgb_weight * lgb_pred)

        return ensemble_pred

    def get_component_predictions(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions from individual models for analysis.

        Returns:
        --------
        ridge_pred : np.ndarray
            Ridge predictions
        lgb_pred : np.ndarray
            LightGBM predictions
        """
        if self.imputer is not None:
            X = self.imputer.transform(X)
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X) if self.scaler is not None else X
        ridge_pred = self.ridge_model.predict(X_scaled)
        lgb_pred = self.lightgbm_model.predict(X)
        return ridge_pred, lgb_pred

    def get_feature_importance(self) -> dict:
        """Get feature importance from LightGBM component."""
        if self.lightgbm_model is None:
            raise ValueError("LightGBM model not available")

        if hasattr(self.lightgbm_model, 'feature_importances_'):
            return {
                'feature_importances': self.lightgbm_model.feature_importances_,
                'source': 'lightgbm'
            }
        return {}

    def __repr__(self):
        return (
            f"RidgeLightGBMEnsemble("
            f"ridge_weight={self.ridge_weight}, "
            f"lgb_weight={1 - self.ridge_weight})"
        )


def load_ensemble_from_disk(
    models_dir: Path,
    ridge_weight: float = 0.4
) -> RidgeLightGBMEnsemble:
    """
    Load ensemble from saved model files.

    Parameters:
    -----------
    models_dir : Path
        Directory containing saved models
    ridge_weight : float
        Weight for Ridge component

    Returns:
    --------
    RidgeLightGBMEnsemble
        Loaded ensemble model
    """
    with open(models_dir / 'ridge_model.pkl', 'rb') as f:
        ridge = pickle.load(f)

    with open(models_dir / 'lightgbm_model.pkl', 'rb') as f:
        lgb = pickle.load(f)

    with open(models_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load imputer if exists
    imputer = None
    imputer_path = models_dir / 'imputer.pkl'
    if imputer_path.exists():
        with open(imputer_path, 'rb') as f:
            imputer = pickle.load(f)
        _ensure_imputer_compat(imputer)

    return RidgeLightGBMEnsemble(
        ridge_model=ridge,
        lightgbm_model=lgb,
        ridge_weight=ridge_weight,
        scaler=scaler,
        imputer=imputer
    )


def save_ensemble_config(
    models_dir: Path,
    ridge_weight: float = 0.4,
    filename: str = 'ensemble_config.pkl'
):
    """Save ensemble configuration for reproducibility."""
    config = {
        'ridge_weight': ridge_weight,
        'lgb_weight': 1.0 - ridge_weight,
        'ensemble_type': 'Ridge+LightGBM'
    }

    with open(models_dir / filename, 'wb') as f:
        pickle.dump(config, f)
