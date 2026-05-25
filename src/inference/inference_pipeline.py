"""
Inference Pipeline for FSP Switching
=====================================

Standardized inference pipeline that:
1. Loads models with proper scaling requirements
2. Predicts FSP errors 6 blocks ahead
3. Selects best FSP for scheduling
4. Tracks model version and confidence

Maintainer: Project Team
Date: January 2026
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.feature_engineering import (
    create_time_features,
    encode_categorical_features,
    create_rolling_features,
    get_feature_columns
)
from src.config_loader import load_config

# Load config for FSP providers (single source of truth)
_config = load_config()


class FSPInferencePipeline:
    """
    Standardized inference pipeline for FSP switching predictions.

    This pipeline:
    1. Handles model loading with correct scaling requirements
    2. Predicts FSP performance 6 blocks ahead
    3. Selects the best FSP based on predicted error
    4. Provides selection confidence scores
    5. Tracks model version for each prediction

    Attributes:
    -----------
    models_dir : Path
        Directory containing trained models
    models : dict
        Dictionary of loaded models
    scaler : StandardScaler
        Fitted scaler for feature normalization
    feature_cols : list
        List of feature column names
    model_metadata : dict
        Metadata including scaling requirements
    version : str
        Current model version
    """

    # Model scaling requirements
    REQUIRES_SCALING = {
        'ridge': True,
        'lasso': True,
        'harmonic_regression': True,
        'random_forest': False,
        'xgboost': False,
        'lightgbm': False,
        'ann': True,
        'fcnn': True,
        'lstm': True,
        'gru': True,
        'temporal_cnn': True,
        'custom_architecture': True,
        'bigru_cnn': True,
        'ceemdan_vmd_cnn_bilstm': True,
        'ivmd_fe_ad_informer': True,
        'ensemble': False  # Uses pre-scaled predictions
    }

    # Models requiring 3D input (samples, timesteps, features)
    REQUIRES_3D_INPUT = {
        'lstm': True,
        'gru': True,
        'temporal_cnn': True,
        'custom_architecture': True,
        'bigru_cnn': True,
        'ceemdan_vmd_cnn_bilstm': True,
        'ivmd_fe_ad_informer': True
    }

    # FSP providers loaded from config
    FSP_PROVIDERS = _config.get('fsp_providers', [
        'FA_PROVIDER_A',
        'FA_PROVIDER_B',
        'FA_PROVIDER_C',
        'FA_PROVIDER_D'
    ])

    def __init__(
        self,
        models_dir: str = 'outputs/models',
        prediction_horizon: int = 6
    ):
        """
        Initialize the inference pipeline.

        Parameters:
        -----------
        models_dir : str
            Path to directory containing trained models
        prediction_horizon : int
            Number of blocks ahead to predict (default: 6)
        """
        self.models_dir = Path(models_dir)
        self.prediction_horizon = prediction_horizon
        self.models = {}
        self.scaler = None
        self.imputer = None
        self.feature_cols = None
        self.encoders = None
        self.model_metadata = {}
        self.version = "1.0.0"
        self.prediction_horizon_from_metadata = None  # Will be loaded from metadata

        self._load_artifacts()

    def _load_artifacts(self):
        """Load all model artifacts."""
        # Load scaler
        scaler_path = self.models_dir / 'scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

        # Load imputer (if exists)
        imputer_path = self.models_dir / 'imputer.pkl'
        if imputer_path.exists():
            with open(imputer_path, 'rb') as f:
                self.imputer = pickle.load(f)

        # Load feature columns
        feature_cols_path = self.models_dir / 'feature_columns.json'
        if feature_cols_path.exists():
            with open(feature_cols_path, 'r') as f:
                self.feature_cols = json.load(f)

        # Load model metadata (includes prediction_horizon for 6-block-ahead)
        metadata_path = self.models_dir / 'model_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
                self.version = self.model_metadata.get('version', '1.0.0')
                self.prediction_horizon_from_metadata = self.model_metadata.get('prediction_horizon', 6)

                print(f" Loaded model metadata:")
                print(f"  Version: {self.version}")
                print(f"  Prediction Horizon: {self.prediction_horizon_from_metadata} blocks ahead")
                print(f"  Target Column: {self.model_metadata.get('target_column', 'unknown')}")

        # Load all available models
        self._load_models()

    def _load_models(self):
        """Load all trained models from the models directory."""
        # Traditional ML models (pickle)
        for model_name in ['ridge', 'harmonic_regression', 'random_forest', 'xgboost', 'lightgbm']:
            model_path = self.models_dir / f'{model_name}_model.pkl'
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)

        # Deep learning models (keras)
        for model_name in ['lstm', 'bigru_cnn', 'ceemdan_vmd_cnn_bilstm', 'ivmd_fe_ad_informer']:
            model_path = self.models_dir / f'{model_name}_model.keras'
            if model_path.exists():
                try:
                    from tensorflow import keras
                    self.models[model_name] = keras.models.load_model(model_path)
                except ImportError:
                    pass

    def predict(
        self,
        df: pd.DataFrame,
        model_name: str = 'xgboost'
    ) -> np.ndarray:
        """
        Make predictions using a specific model.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with features
        model_name : str
            Name of model to use

        Returns:
        --------
        np.ndarray : Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded")

        model = self.models[model_name]

        # Prepare features
        X = df[self.feature_cols].values if self.feature_cols else df.values

        # Handle missing values (should already be dropped, but safety check)
        if np.any(np.isnan(X)):
            if self.imputer:
                X = self.imputer.transform(X)
            else:
                # Fill with 0 as last resort (not recommended)
                X = np.nan_to_num(X, nan=0.0)

        # Scale if required
        if self.REQUIRES_SCALING.get(model_name, False) and self.scaler:
            X = self.scaler.transform(X)

        # Special handling for harmonic regression
        if model_name == 'harmonic_regression':
            time_col = None
            if self.model_metadata:
                time_col = self.model_metadata.get('harmonic_time_col')
            time_series = df[time_col] if time_col and time_col in df.columns else df.index
            if np.issubdtype(time_series.dtype, np.datetime64):
                time_index = pd.to_datetime(time_series).view('int64') / 3.6e12
            else:
                time_index = time_series.astype(float).to_numpy()

            periods = self.model_metadata.get('harmonic_periods', [24, 96])
            order = self.model_metadata.get('harmonic_order', 2)

            features = []
            for period in periods:
                for k in range(1, order + 1):
                    omega = 2 * np.pi * k / max(period, 1e-6)
                    features.append(np.sin(omega * time_index))
                    features.append(np.cos(omega * time_index))

            harmonic_features = np.column_stack(features) if features else np.empty((len(time_index), 0))
            use_original = self.model_metadata.get('harmonic_use_original_features', True)
            X = np.column_stack([X, harmonic_features]) if use_original else harmonic_features

        # Reshape for 3D input if required (LSTM, GRU, CNN models)
        if self.REQUIRES_3D_INPUT.get(model_name, False):
            if hasattr(model, 'input_shape') and len(model.input_shape) >= 3:
                timesteps = model.input_shape[1] or 1
                features_per_step = model.input_shape[2] or X.shape[1]
                total_features = timesteps * features_per_step
                if X.shape[1] < total_features:
                    pad_width = total_features - X.shape[1]
                    X = np.pad(X, ((0, 0), (0, pad_width)), mode='constant')
                X_trimmed = X[:, :total_features]
                X = X_trimmed.reshape((X.shape[0], timesteps, features_per_step))
            else:
                X = X.reshape((X.shape[0], 1, X.shape[1]))

        # Make predictions
        predictions = model.predict(X)

        # Flatten if needed
        if predictions.ndim > 1:
            predictions = predictions.flatten()

        return predictions

    def predict_fsp_errors(
        self,
        df: pd.DataFrame,
        model_name: str = 'xgboost'
    ) -> pd.DataFrame:
        """
        Predict error for each FSP 6 blocks ahead.

        For each FSP, calculates expected absolute error between
        FSP prediction and actual power.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with FSP predictions
        model_name : str
            Model to use for predictions

        Returns:
        --------
        pd.DataFrame : DataFrame with predicted errors for each FSP
        """
        results = df.copy()

        # Get FSP prediction columns
        fsp_cols = [c for c in df.columns if any(
            fsp.lower() in c.lower() for fsp in self.FSP_PROVIDERS
        ) and 'power' in c.lower()]

        # Predict actual power
        predicted_power = self.predict(df, model_name)
        results['ml_predicted_power'] = predicted_power

        # Calculate expected error for each FSP
        for fsp_col in fsp_cols:
            if fsp_col in df.columns:
                fsp_name = self._extract_fsp_name(fsp_col)
                expected_error = np.abs(df[fsp_col].values - predicted_power)
                results[f'expected_error_{fsp_name}'] = expected_error

        return results

    def select_best_fsp(
        self,
        df: pd.DataFrame,
        model_name: str = 'xgboost'
    ) -> pd.DataFrame:
        """
        Select the best FSP for each timestep based on predicted error.

        This is the main FSP switching function that:
        1. Predicts expected error for each FSP
        2. Selects FSP with minimum predicted error
        3. Calculates selection confidence

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with FSP predictions
        model_name : str
            Model to use

        Returns:
        --------
        pd.DataFrame : DataFrame with FSP selection results
        """
        # Get error predictions
        results = self.predict_fsp_errors(df, model_name)

        # Get error columns
        error_cols = [c for c in results.columns if c.startswith('expected_error_')]

        if not error_cols:
            results['ml_selected_fsp'] = 'UNKNOWN'
            results['selection_confidence'] = 0.0
            return results

        # Create error matrix
        error_matrix = results[error_cols].values

        # Select FSP with minimum error
        min_indices = np.argmin(error_matrix, axis=1)
        fsp_names = [col.replace('expected_error_', '') for col in error_cols]

        results['ml_selected_fsp'] = [fsp_names[i] for i in min_indices]

        # Calculate confidence (inverse of error normalized)
        min_errors = np.min(error_matrix, axis=1)
        max_errors = np.max(error_matrix, axis=1)

        # Confidence = how much better is the best FSP vs worst
        error_range = max_errors - min_errors
        results['selection_confidence'] = np.where(
            error_range > 0,
            error_range / (max_errors + 1e-8),
            0.5  # Default confidence if all FSPs have same error
        )

        # Add model version
        results['model_version'] = self.version

        return results

    def _extract_fsp_name(self, col_name: str) -> str:
        """Extract FSP name from column name."""
        col_lower = col_name.lower()
        for fsp in self.FSP_PROVIDERS:
            if fsp.lower() in col_lower:
                return fsp
        return col_name

    def format_output_csv(
        self,
        df: pd.DataFrame,
        actual_col: str = 'actual_power',
        scheduled_col: str = 'schedule_power'
    ) -> pd.DataFrame:
        """
        Format output DataFrame with required columns.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with predictions
        actual_col : str
            Column name for actual power
        scheduled_col : str
            Column name for manually scheduled power

        Returns:
        --------
        pd.DataFrame : Formatted output DataFrame
        """
        output = pd.DataFrame()

        # Required columns
        output['timestamp'] = df.get('timestamp', pd.NaT)
        output['date'] = df.get('date', '')
        output['block'] = df.get('block', 0)

        # Actual and scheduled power
        output['actual_power'] = df.get(actual_col, np.nan)
        output['manual_scheduled_power'] = df.get(scheduled_col, np.nan)

        # FSP predictions
        for fsp in self.FSP_PROVIDERS:
            fsp_col = f'forecast_power_{fsp.lower()}'
            alt_col = f'{fsp.lower()}_power'
            if fsp_col in df.columns:
                output[f'fsp_{fsp.lower()}_power'] = df[fsp_col]
            elif alt_col in df.columns:
                output[f'fsp_{fsp.lower()}_power'] = df[alt_col]

        # ML predictions
        output['ml_selected_fsp'] = df.get('ml_selected_fsp', '')
        output['ml_predicted_power'] = df.get('ml_predicted_power', np.nan)
        output['selection_confidence'] = df.get('selection_confidence', np.nan)
        output['model_version'] = df.get('model_version', self.version)

        return output

    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.models.keys())


def create_model_metadata(
    models_dir: str,
    version: str,
    training_date: str = None,
    metrics: dict = None
) -> dict:
    """
    Create and save model metadata.

    Parameters:
    -----------
    models_dir : str
        Path to models directory
    version : str
        Model version string
    training_date : str, optional
        Training date (defaults to now)
    metrics : dict, optional
        Model performance metrics

    Returns:
    --------
    dict : Metadata dictionary
    """
    metadata = {
        'version': version,
        'training_date': training_date or datetime.now().isoformat(),
        'metrics': metrics or {},
        'scaling_requirements': FSPInferencePipeline.REQUIRES_SCALING,
        'fsp_providers': FSPInferencePipeline.FSP_PROVIDERS
    }

    # Save to file
    metadata_path = Path(models_dir) / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata
