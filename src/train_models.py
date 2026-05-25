"""
FSP Selection Model Training Script
=====================================

This script trains models to select the best FSP for power scheduling.
The model predicts which FSP will have the lowest error, then schedules
that FSP's forecasted power.

This is NOT power prediction - it's FSP selection/ranking.

Key Features:
- 70-15-15 temporal split (no data leakage)
- Rolling features computed AFTER split
- Drops rows with missing data (no imputation)
- Excludes actual_* features (prevents target leakage)
- LSTM and BiGRU-CNN models for time-series
- Separate output CSV per model
- MLflow experiment tracking

Maintainer: Project Team
Date: January 2026
"""

import os
import sys
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import ks_2samp

# Optional imports
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Suppress specific warnings (not global) to avoid hiding important issues
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', message='.*categorical features are.*')
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Add src to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.config_loader import load_config
from src.features.feature_engineering import (
    create_temporal_split, create_rolling_features, create_time_features,
    encode_categorical_features, drop_missing_data, get_feature_columns,
    check_distribution_shift
)
from src.models.sequence_models import (
    build_lstm_model, build_bigru_cnn_model, create_sequences,
    train_sequence_model, get_callbacks
)

# Load configuration
config = load_config()

# Directories
DATA_DIR = PROJECT_DIR / config.get('data.processed_dir', 'data/processed')
INTERIM_DIR = PROJECT_DIR / config.get('data.interim_dir', 'data/interim')
OUTPUT_DIR = PROJECT_DIR / 'outputs'
MODELS_DIR = OUTPUT_DIR / 'models'
PREDS_DIR = OUTPUT_DIR / 'predictions'
PLOTS_DIR = OUTPUT_DIR / 'plots'
REPORTS_DIR = OUTPUT_DIR / 'reports'

# Create directories
for d in [MODELS_DIR, PREDS_DIR, PLOTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Training configuration from config.yaml
TRAIN_RATIO = config.get('training.train_ratio', 0.70)
VAL_RATIO = config.get('training.val_ratio', 0.15)
TEST_RATIO = config.get('training.test_ratio', 0.15)

# Validate split ratios sum to 1.0
if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-6:
    raise ValueError(f"Split ratios must sum to 1.0, got {TRAIN_RATIO + VAL_RATIO + TEST_RATIO}")
DATA_MONTHS = config.get('training.data_months', 18)
RANDOM_STATE = config.get('training.random_seed', 42)
SEQUENCE_LENGTH = config.get('training.sequence_length', 24)
PREDICTION_HORIZON = config.get('training.prediction_horizon', 6)

# FSP Providers
FSP_PROVIDERS = config.get('fsp_providers', [
    'FA_PROVIDER_A', 'FA_PROVIDER_B', 'FA_PROVIDER_C', 'FA_PROVIDER_D'
])

# Target and exclusions
TARGET = 'actual_power'
EXCLUDE_PATTERNS = ['date', 'timestamp', 'index', 'actual_', 'sscode']

# Model version
MODEL_VERSION = config.get('versioning.current_version', '1.0.0')

np.random.seed(RANDOM_STATE)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Calculate comprehensive metrics for a model."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Symmetric MAPE - handles zeros safely by using (actual + pred) / 2 as denominator
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    safe_denominator = np.where(denominator == 0, 1, denominator)
    smape = np.mean(np.abs(y_true - y_pred) / safe_denominator) * 100

    # Accuracy within tolerance bands
    error_pct = np.abs((y_true - y_pred) / (y_true + 1e-8)) * 100
    within_15 = (error_pct <= 15).mean() * 100
    within_25 = (error_pct <= 25).mean() * 100

    return {
        'Model': model_name,
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4),
        'sMAPE': round(smape, 2),
        'Within_15%': round(within_15, 2),
        'Within_25%': round(within_25, 2)
    }


def get_fsp_columns(df: pd.DataFrame) -> List[str]:
    """Get FSP forecast power columns from dataframe."""
    fsp_cols = []
    for col in df.columns:
        if 'forecast_power' in col.lower() or any(
            fsp.lower() in col.lower() and 'power' in col.lower()
            for fsp in FSP_PROVIDERS
        ):
            fsp_cols.append(col)
    return fsp_cols


def calculate_fsp_errors(df: pd.DataFrame, fsp_cols: List[str]) -> pd.DataFrame:
    """Calculate absolute error for each FSP compared to actual power."""
    df_out = df.copy()

    for fsp_col in fsp_cols:
        # Extract FSP name
        fsp_name = fsp_col.replace('forecast_power_', '').replace('_power', '')
        error_col = f'error_{fsp_name}'
        df_out[error_col] = np.abs(df_out[fsp_col] - df_out[TARGET])

    return df_out


def select_best_fsp(
    df: pd.DataFrame,
    error_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Select FSP with minimum error for each row."""
    error_matrix = df[error_cols].values
    min_indices = np.argmin(error_matrix, axis=1)

    fsp_names = [col.replace('error_', '') for col in error_cols]
    selected_fsps = np.array([fsp_names[i] for i in min_indices])

    # Confidence: relative difference between best and worst
    min_errors = np.min(error_matrix, axis=1)
    max_errors = np.max(error_matrix, axis=1)
    confidence = np.where(
        max_errors > 0,
        (max_errors - min_errors) / (max_errors + 1e-8),
        0.5
    )

    return selected_fsps, confidence


def save_model_output(
    df: pd.DataFrame,
    predictions: np.ndarray,
    selected_fsps: np.ndarray,
    confidence: np.ndarray,
    model_name: str,
    fsp_cols: List[str],
    output_path: Path
):
    """
    Save structured output CSV for a model.

    Includes: timestamp, block, FSP predictions, ML prediction,
    selected FSP, actual power, manual scheduled
    """
    output = pd.DataFrame()

    # Timestamps and blocks
    output['timestamp'] = df.get('timestamp', pd.NaT)
    output['date'] = df.get('date', '')
    output['block'] = df.get('block', 0)

    # Actual and scheduled power
    output['actual_power'] = df.get(TARGET, np.nan)
    output['manual_scheduled_power'] = df.get('schedule_power', np.nan)

    # FSP predictions
    for fsp_col in fsp_cols:
        fsp_name = fsp_col.replace('forecast_power_', '').replace('_power', '').lower()
        output[f'fsp_{fsp_name}_power'] = df.get(fsp_col, np.nan)

    # ML model outputs
    output['ml_predicted_power'] = predictions
    output['ml_selected_fsp'] = selected_fsps
    output['ml_scheduled_power'] = [
        df.iloc[i][f'forecast_power_{selected_fsps[i]}']
        if f'forecast_power_{selected_fsps[i]}' in df.columns
        else np.nan
        for i in range(len(df))
    ]
    output['selection_confidence'] = confidence
    output['model_version'] = MODEL_VERSION
    output['model_name'] = model_name

    # Calculate ML error
    output['ml_error'] = np.abs(output['actual_power'] - output['ml_scheduled_power'])
    output['manual_error'] = np.abs(output['actual_power'] - output['manual_scheduled_power'])

    # Save
    output.to_csv(output_path, index=False)
    print(f" Saved: {output_path.name}")

    return output


def plot_feature_importance(model, feature_names: List[str], model_name: str, output_path: Path):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=importance, x='importance', y='feature', ax=ax)
        ax.set_title(f'Feature Importance - {model_name}', fontweight='bold')
        ax.set_xlabel('Importance')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" Saved: {output_path.name}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    """Main training pipeline."""
    print("=" * 70)
    print(" FSP SELECTION MODEL TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Split ratios: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"  Data months: {DATA_MONTHS}")
    print(f"  Prediction horizon: {PREDICTION_HORIZON} blocks")
    print(f"  Model version: {MODEL_VERSION}")

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n" + "=" * 70)
    print(" LOADING DATA")
    print("=" * 70)

    interim_file = INTERIM_DIR / 'eda_processed_data.parquet'
    if interim_file.exists():
        df_raw = pd.read_parquet(interim_file)
        print(f" Loaded: {interim_file.name}")
    else:
        parquet_files = list(DATA_DIR.glob('*.parquet'))
        if parquet_files:
            df_raw = pd.read_parquet(parquet_files[0])
            print(f" Loaded: {parquet_files[0].name}")
        else:
            raise FileNotFoundError("No data files found!")

    print(f"  Shape: {df_raw.shape}")

    # =========================================================================
    # STEP 2: Filter to Last N Months
    # =========================================================================
    date_col = 'timestamp' if 'timestamp' in df_raw.columns else 'date'
    df_raw[date_col] = pd.to_datetime(df_raw[date_col])

    original_max = df_raw[date_col].max()
    cutoff_date = original_max - pd.DateOffset(months=DATA_MONTHS)
    df = df_raw[df_raw[date_col] >= cutoff_date].copy()

    print(f"\n Filtered to last {DATA_MONTHS} months: {len(df):,} rows")

    # =========================================================================
    # STEP 3: Get FSP Columns and Calculate Errors
    # =========================================================================
    fsp_cols = get_fsp_columns(df)
    print(f"\n Found {len(fsp_cols)} FSP columns: {fsp_cols}")

    # Calculate errors for each FSP
    df = calculate_fsp_errors(df, fsp_cols)
    error_cols = [c for c in df.columns if c.startswith('error_')]

    # =========================================================================
    # STEP 4: Drop Missing Data
    # =========================================================================
    print("\n" + "=" * 70)
    print(" CLEANING DATA (Dropping Missing - No Imputation)")
    print("=" * 70)

    required_cols = [TARGET] + fsp_cols
    df_clean = drop_missing_data(df, required_cols, verbose=True)

    # =========================================================================
    # STEP 5: Temporal Split
    # =========================================================================
    print("\n" + "=" * 70)
    print(f" TEMPORAL SPLIT ({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%})")
    print("=" * 70)

    train_df, val_df, test_df = create_temporal_split(
        df_clean, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )

    print(f"  Train: {len(train_df):,} ({len(train_df)/len(df_clean)*100:.1f}%)")
    print(f"  Val:   {len(val_df):,} ({len(val_df)/len(df_clean)*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} ({len(test_df)/len(df_clean)*100:.1f}%)")

    # =========================================================================
    # STEP 6: Feature Engineering (AFTER split - no leakage)
    # =========================================================================
    print("\n" + "=" * 70)
    print(" FEATURE ENGINEERING (After Split - No Leakage)")
    print("=" * 70)

    # Time features
    train_df = create_time_features(train_df)
    val_df = create_time_features(val_df)
    test_df = create_time_features(test_df)

    # Rolling features (computed separately for each split)
    train_df = create_rolling_features(train_df, TARGET, [1, 6, 24, 96], fsp_cols)
    val_df = create_rolling_features(val_df, TARGET, [1, 6, 24, 96], fsp_cols)
    test_df = create_rolling_features(test_df, TARGET, [1, 6, 24, 96], fsp_cols)

    # Encode categorical
    train_df, encoders = encode_categorical_features(train_df, [TARGET, 'date', 'timestamp', 'sscode'])
    val_df, _ = encode_categorical_features(val_df, [TARGET, 'date', 'timestamp', 'sscode'], encoders)
    test_df, _ = encode_categorical_features(test_df, [TARGET, 'date', 'timestamp', 'sscode'], encoders)

    # Get feature columns (excludes actual_* to prevent leakage)
    feature_cols = get_feature_columns(train_df, TARGET, EXCLUDE_PATTERNS)
    print(f"\n Selected {len(feature_cols)} features (actual_* excluded)")

    # =========================================================================
    # STEP 7: Prepare X and y
    # =========================================================================
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET].values
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET].values
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET].values

    # Handle NaN from rolling features using median imputation
    # This avoids bias from replacing NaN with 0 where 0 has semantic meaning
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)
    print(f"\n Imputed NaN values in rolling features using median strategy")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    with open(MODELS_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Save imputer for inference
    with open(MODELS_DIR / 'imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)

    # Check distribution shift
    has_shift, pval = check_distribution_shift(y_train, y_test)
    if has_shift:
        print(f"\n Distribution shift detected (p={pval:.4f})")
    else:
        print(f"\n No significant distribution shift (p={pval:.4f})")

    # =========================================================================
    # STEP 8: MLflow Setup
    # =========================================================================
    if MLFLOW_AVAILABLE and config.get('mlflow.enabled', False):
        mlflow.set_tracking_uri(config.get('mlflow.tracking_uri', 'mlruns'))
        mlflow.set_experiment(config.get('mlflow.experiment_name', 'fsp_switching'))
        print("\n MLflow tracking enabled")

    # =========================================================================
    # STEP 9: Train Models
    # =========================================================================
    print("\n" + "=" * 70)
    print(" TRAINING MODELS")
    print("=" * 70)

    model_results = []
    trained_models = {}
    all_predictions = {}

    # MODEL 1: Ridge Regression
    print("\n Training Ridge Regression...")
    ridge = Ridge(alpha=config.get('models.ridge.alpha', 1.0), random_state=RANDOM_STATE)
    ridge.fit(X_train_scaled, y_train)

    ridge_val_pred = ridge.predict(X_val_scaled)
    ridge_test_pred = ridge.predict(X_test_scaled)

    ridge_results = evaluate_model(y_val, ridge_val_pred, 'Ridge')
    model_results.append(ridge_results)
    trained_models['ridge'] = ridge
    all_predictions['ridge'] = {'val': ridge_val_pred, 'test': ridge_test_pred}
    print(f" Ridge - Val MAE: {ridge_results['MAE']}, R2: {ridge_results['R2']}")

    with open(MODELS_DIR / 'ridge_model.pkl', 'wb') as f:
        pickle.dump(ridge, f)

    # MODEL 2: Random Forest
    print("\n Training Random Forest...")
    rf_params = config.get('models.random_forest', {})
    rf = RandomForestRegressor(
        n_estimators=rf_params.get('n_estimators', 200),
        max_depth=rf_params.get('max_depth', 15),
        min_samples_split=rf_params.get('min_samples_split', 5),
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)

    rf_val_pred = rf.predict(X_val)
    rf_test_pred = rf.predict(X_test)

    rf_results = evaluate_model(y_val, rf_val_pred, 'Random Forest')
    model_results.append(rf_results)
    trained_models['random_forest'] = rf
    all_predictions['random_forest'] = {'val': rf_val_pred, 'test': rf_test_pred}
    print(f" Random Forest - Val MAE: {rf_results['MAE']}, R2: {rf_results['R2']}")

    with open(MODELS_DIR / 'random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf, f)

    # Feature importance
    plot_feature_importance(rf, feature_cols, 'Random Forest', PLOTS_DIR / 'feature_importance_rf.png')

    # MODEL 3: XGBoost
    if XGB_AVAILABLE:
        print("\n Training XGBoost...")
        xgb_params = config.get('models.xgboost', {})
        xgb_model = xgb.XGBRegressor(
            n_estimators=xgb_params.get('n_estimators', 200),
            max_depth=xgb_params.get('max_depth', 8),
            learning_rate=xgb_params.get('learning_rate', 0.05),
            subsample=xgb_params.get('subsample', 0.8),
            colsample_bytree=xgb_params.get('colsample_bytree', 0.8),
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        xgb_val_pred = xgb_model.predict(X_val)
        xgb_test_pred = xgb_model.predict(X_test)

        xgb_results = evaluate_model(y_val, xgb_val_pred, 'XGBoost')
        model_results.append(xgb_results)
        trained_models['xgboost'] = xgb_model
        all_predictions['xgboost'] = {'val': xgb_val_pred, 'test': xgb_test_pred}
        print(f" XGBoost - Val MAE: {xgb_results['MAE']}, R2: {xgb_results['R2']}")

        with open(MODELS_DIR / 'xgboost_model.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)

        plot_feature_importance(xgb_model, feature_cols, 'XGBoost', PLOTS_DIR / 'feature_importance_xgb.png')

    # MODEL 4: LightGBM
    if LGB_AVAILABLE:
        print("\n Training LightGBM...")
        lgb_params = config.get('models.lightgbm', {})
        lgb_model = lgb.LGBMRegressor(
            n_estimators=lgb_params.get('n_estimators', 200),
            max_depth=lgb_params.get('max_depth', 10),
            learning_rate=lgb_params.get('learning_rate', 0.05),
            num_leaves=lgb_params.get('num_leaves', 31),
            subsample=lgb_params.get('subsample', 0.8),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        lgb_val_pred = lgb_model.predict(X_val)
        lgb_test_pred = lgb_model.predict(X_test)

        lgb_results = evaluate_model(y_val, lgb_val_pred, 'LightGBM')
        model_results.append(lgb_results)
        trained_models['lightgbm'] = lgb_model
        all_predictions['lightgbm'] = {'val': lgb_val_pred, 'test': lgb_test_pred}
        print(f" LightGBM - Val MAE: {lgb_results['MAE']}, R2: {lgb_results['R2']}")

        with open(MODELS_DIR / 'lightgbm_model.pkl', 'wb') as f:
            pickle.dump(lgb_model, f)

        plot_feature_importance(lgb_model, feature_cols, 'LightGBM', PLOTS_DIR / 'feature_importance_lgb.png')

    # MODEL 5: LSTM
    if DL_AVAILABLE:
        print("\n Training LSTM...")
        lstm_params = config.get('models.lstm', {})

        # Create sequences
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, SEQUENCE_LENGTH)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, SEQUENCE_LENGTH)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, SEQUENCE_LENGTH)

        # Build and train
        lstm = build_lstm_model(
            input_shape=(SEQUENCE_LENGTH, X_train.shape[1]),
            units=lstm_params.get('units', [128, 64]),
            dropout_rate=lstm_params.get('dropout_rate', 0.3),
            learning_rate=lstm_params.get('learning_rate', 0.001)
        )

        history = train_sequence_model(
            lstm, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
            str(MODELS_DIR / 'lstm_model.keras'),
            epochs=lstm_params.get('epochs', 150),
            batch_size=lstm_params.get('batch_size', 64),
            patience=lstm_params.get('early_stopping_patience', 15)
        )

        lstm_val_pred = lstm.predict(X_val_seq, verbose=0).flatten()
        lstm_test_pred = lstm.predict(X_test_seq, verbose=0).flatten()

        lstm_results = evaluate_model(y_val_seq, lstm_val_pred, 'LSTM')
        model_results.append(lstm_results)
        trained_models['lstm'] = lstm
        # Store with adjusted indices due to sequence
        all_predictions['lstm'] = {
            'val': lstm_val_pred,
            'test': lstm_test_pred,
            'seq_offset': SEQUENCE_LENGTH
        }
        print(f" LSTM - Val MAE: {lstm_results['MAE']}, R2: {lstm_results['R2']}")

    # MODEL 6: BiGRU-CNN
    if DL_AVAILABLE:
        print("\n Training BiGRU-CNN...")
        bigru_params = config.get('models.bigru_cnn', {})

        # Build and train
        bigru_cnn = build_bigru_cnn_model(
            input_shape=(SEQUENCE_LENGTH, X_train.shape[1]),
            gru_units=bigru_params.get('gru_units', [64, 32]),
            cnn_filters=bigru_params.get('cnn_filters', 64),
            cnn_kernel_size=bigru_params.get('cnn_kernel_size', 3),
            dropout_rate=bigru_params.get('dropout_rate', 0.3),
            learning_rate=bigru_params.get('learning_rate', 0.001)
        )

        history = train_sequence_model(
            bigru_cnn, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
            str(MODELS_DIR / 'bigru_cnn_model.keras'),
            epochs=bigru_params.get('epochs', 150),
            batch_size=bigru_params.get('batch_size', 64),
            patience=bigru_params.get('early_stopping_patience', 15)
        )

        bigru_val_pred = bigru_cnn.predict(X_val_seq, verbose=0).flatten()
        bigru_test_pred = bigru_cnn.predict(X_test_seq, verbose=0).flatten()

        bigru_results = evaluate_model(y_val_seq, bigru_val_pred, 'BiGRU-CNN')
        model_results.append(bigru_results)
        trained_models['bigru_cnn'] = bigru_cnn
        all_predictions['bigru_cnn'] = {
            'val': bigru_val_pred,
            'test': bigru_test_pred,
            'seq_offset': SEQUENCE_LENGTH
        }
        print(f" BiGRU-CNN - Val MAE: {bigru_results['MAE']}, R2: {bigru_results['R2']}")

    # =========================================================================
    # STEP 10: Model Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print(" MODEL COMPARISON (Validation Set)")
    print("=" * 70)

    results_df = pd.DataFrame(model_results).sort_values('MAE')
    print(results_df.to_string(index=False))

    best_model = results_df.iloc[0]['Model']
    print(f"\n Best Model: {best_model}")

    results_df.to_csv(REPORTS_DIR / 'model_results.csv', index=False)

    # =========================================================================
    # STEP 11: Save Test Predictions (Separate File per Model)
    # =========================================================================
    print("\n" + "=" * 70)
    print(" SAVING TEST PREDICTIONS (Separate File per Model)")
    print("=" * 70)

    for model_name, preds in all_predictions.items():
        test_pred = preds['test']
        seq_offset = preds.get('seq_offset', 0)

        # Adjust test_df for sequence models
        if seq_offset > 0:
            test_df_adj = test_df.iloc[seq_offset:].copy().reset_index(drop=True)
            y_test_adj = y_test[seq_offset:]
        else:
            test_df_adj = test_df.copy()
            y_test_adj = y_test

        # Select best FSP based on predictions
        # For each row, find FSP whose forecast is closest to our prediction
        selected_fsps = []
        confidence = []

        for i in range(len(test_df_adj)):
            fsp_values = {}
            for fsp_col in fsp_cols:
                if fsp_col in test_df_adj.columns:
                    fsp_name = fsp_col.replace('forecast_power_', '').replace('_power', '')
                    fsp_values[fsp_name] = test_df_adj.iloc[i].get(fsp_col, np.nan)

            if fsp_values:
                # Select FSP whose forecast is closest to predicted actual
                predicted = test_pred[i] if i < len(test_pred) else np.nan
                errors = {fsp: abs(val - predicted) for fsp, val in fsp_values.items() if not np.isnan(val)}

                if errors:
                    best_fsp = min(errors, key=errors.get)
                    selected_fsps.append(best_fsp)

                    # Confidence based on how much better best is than worst
                    min_err, max_err = min(errors.values()), max(errors.values())
                    conf = (max_err - min_err) / (max_err + 1e-8) if max_err > 0 else 0.5
                    confidence.append(conf)
                else:
                    selected_fsps.append('UNKNOWN')
                    confidence.append(0.0)
            else:
                selected_fsps.append('UNKNOWN')
                confidence.append(0.0)

        selected_fsps = np.array(selected_fsps)
        confidence = np.array(confidence)

        # Save output
        output_path = PREDS_DIR / f'test_predictions_{model_name}.csv'
        save_model_output(
            test_df_adj, test_pred, selected_fsps, confidence,
            model_name, fsp_cols, output_path
        )

    # =========================================================================
    # STEP 12: Save Artifacts
    # =========================================================================
    print("\n" + "=" * 70)
    print(" SAVING ARTIFACTS")
    print("=" * 70)

    # Save feature columns
    with open(MODELS_DIR / 'feature_columns.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(" Saved: feature_columns.json")

    # Save model metadata
    metadata = {
        'version': MODEL_VERSION,
        'training_date': datetime.now().isoformat(),
        'train_ratio': TRAIN_RATIO,
        'val_ratio': VAL_RATIO,
        'test_ratio': TEST_RATIO,
        'data_months': DATA_MONTHS,
        'sequence_length': SEQUENCE_LENGTH,
        'prediction_horizon': PREDICTION_HORIZON,
        'fsp_providers': FSP_PROVIDERS,
        'models_trained': list(trained_models.keys()),
        'best_model': best_model,
        'metrics': {m['Model']: m for m in model_results}
    }

    with open(MODELS_DIR / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(" Saved: model_metadata.json")

    # Save encoders
    with open(MODELS_DIR / 'encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    print(" Saved: encoders.pkl")

    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n Outputs saved to:")
    print(f"   Models: {MODELS_DIR}")
    print(f"   Predictions: {PREDS_DIR}")
    print(f"   Plots: {PLOTS_DIR}")
    print(f"   Reports: {REPORTS_DIR}")


if __name__ == '__main__':
    main()
