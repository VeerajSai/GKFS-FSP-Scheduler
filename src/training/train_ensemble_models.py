"""
FSP Selection Model Training Script - Ensemble Edition
======================================================

Trains shallow ML models (Ridge, RF, XGB, LGB) and combines Ridge + LightGBM
into an optimized ensemble. Deep learning models removed.

Updated Features:
- Ridge + LightGBM ensemble with weighted averaging
- No deep learning overhead
- Faster training and inference
- Better interpretability
- Follows ML project review best practices

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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

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
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Suppress only specific warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', message='.*categorical features are.*')
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.config_loader import load_config
from src.data.preprocessing import (
    pivot_fsp_data, calculate_fsp_errors, format_output_csv,
    get_fsp_forecast_columns, FSP_PROVIDERS
)
from src.features.feature_engineering import (
    create_temporal_split, create_rolling_features, create_time_features,
    encode_categorical_features, drop_missing_data, get_feature_columns,
    check_distribution_shift
)
from src.models.ensemble_model import RidgeLightGBMEnsemble, save_ensemble_config

config = load_config()

# Directories
DATA_DIR = PROJECT_DIR / config.get('data.processed_dir', 'data/processed')
INTERIM_DIR = PROJECT_DIR / config.get('data.interim_dir', 'data/interim')
OUTPUT_DIR = PROJECT_DIR / 'outputs'
MODELS_DIR = OUTPUT_DIR / 'models'
PREDS_DIR = OUTPUT_DIR / 'predictions'
PLOTS_DIR = OUTPUT_DIR / 'plots'
REPORTS_DIR = OUTPUT_DIR / 'reports'

for d in [MODELS_DIR, PREDS_DIR, PLOTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Training config
TRAIN_RATIO = config.get('training.train_ratio', 0.70)
VAL_RATIO = config.get('training.val_ratio', 0.15)
TEST_RATIO = config.get('training.test_ratio', 0.15)

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-6:
    raise ValueError(f"Split ratios must sum to 1.0, got {TRAIN_RATIO + VAL_RATIO + TEST_RATIO}")

DATA_MONTHS = config.get('training.data_months', 18)
RANDOM_STATE = config.get('training.random_seed', 42)
MODEL_VERSION = config.get('versioning.current_version', '1.0.0')
PREDICTION_HORIZON = config.get('training.prediction_horizon', 6) # 6-block-ahead forecasting

# Ensemble config
ENSEMBLE_RIDGE_WEIGHT = 0.4 # Ridge: 40%, LightGBM: 60%

TARGET = 'actual_power'
TARGET_HORIZON = 'target_horizon' # Forward-shifted target for 6-block-ahead forecasting
# Do NOT feed manual schedule inputs into model features (only used for comparison)
EXCLUDE_PATTERNS = ['date', 'timestamp', 'index', 'actual_', 'sscode', 'error_', 'schedule_']

np.random.seed(RANDOM_STATE)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Calculate comprehensive metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Symmetric MAPE - handles zeros safely
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    safe_denominator = np.where(denominator == 0, 1, denominator)
    smape = np.mean(np.abs(y_true - y_pred) / safe_denominator) * 100

    return {
        'Model': model_name,
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4),
        'sMAPE': round(smape, 2)
    }


def select_best_fsp_by_prediction(
    df: pd.DataFrame,
    predictions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select FSP whose forecast is closest to predicted actual power.

    Returns:
    - selected_fsps: FSP names (e.g., 'FA_PROVIDER_A')
    - scheduled_power: The selected FSP's forecast power
    - confidence: How much better the best FSP is vs others
    """
    fsp_cols = get_fsp_forecast_columns(df)

    selected_fsps = []
    scheduled_power = []
    confidence = []

    for i in range(len(df)):
        pred = predictions[i] if i < len(predictions) else np.nan

        # Get FSP values for this row
        fsp_values = {}
        for fsp_col in fsp_cols:
            val = df.iloc[i].get(fsp_col, np.nan)
            if not np.isnan(val):
                fsp_name = fsp_col.replace('forecast_power_', '').upper()
                fsp_values[fsp_name] = val

        if fsp_values and not np.isnan(pred):
            # Find FSP closest to prediction
            errors = {fsp: abs(val - pred) for fsp, val in fsp_values.items()}
            best_fsp = min(errors, key=errors.get)

            selected_fsps.append(best_fsp)
            scheduled_power.append(fsp_values[best_fsp])

            # Confidence
            min_err, max_err = min(errors.values()), max(errors.values())
            conf = (max_err - min_err) / (max_err + 1e-8) if max_err > 0 else 0.5
            confidence.append(conf)
        else:
            selected_fsps.append('UNKNOWN')
            scheduled_power.append(np.nan)
            confidence.append(0.0)

    return np.array(selected_fsps), np.array(scheduled_power), np.array(confidence)


def save_model_output(
    df: pd.DataFrame,
    predictions: np.ndarray,
    selected_fsps: np.ndarray,
    scheduled_power: np.ndarray,
    confidence: np.ndarray,
    model_name: str,
    output_path: Path
) -> pd.DataFrame:
    """Save structured output CSV with all FSP forecasts."""
    output = pd.DataFrame()

    # Time columns
    output['timestamp'] = df.get('timestamp', pd.NaT)
    output['date'] = df.get('date', '')
    output['block'] = df.get('block', 0)

    # All FSP forecasts
    for fsp in FSP_PROVIDERS:
        col = f'forecast_power_{fsp.lower()}'
        output[f'{fsp}_power'] = df.get(col, np.nan)

    # Actual and manual
    output['actual_power'] = df.get(TARGET_HORIZON, np.nan)  # Use 6-block-ahead target for alignment
    output['manual_scheduled_power'] = df.get('schedule_power', np.nan)

    # ML outputs
    output['ml_predicted_power'] = predictions
    output['ml_selected_fsp'] = selected_fsps
    output['ml_scheduled_power'] = scheduled_power
    output['selection_confidence'] = confidence
    output['model_version'] = MODEL_VERSION
    output['model_name'] = model_name

    # Errors
    output['ml_error'] = np.abs(output['actual_power'] - output['ml_scheduled_power'])
    output['manual_error'] = np.abs(output['actual_power'] - output['manual_scheduled_power'])

    output.to_csv(output_path, index=False)
    print(f" Saved: {output_path.name}")

    return output


def plot_feature_importance(model, feature_names: List[str], model_name: str, output_path: Path):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        # Ensure feature_names and importances have same length
        importances = model.feature_importances_

        if len(feature_names) != len(importances):
            # Tree models may drop constant features internally
            # Use only the features that were actually used
            print(f"i  Model uses {len(importances)} features (dropped {len(feature_names) - len(importances)} constant/low-variance features)")
            feature_names_used = feature_names[:len(importances)]
        else:
            feature_names_used = feature_names

        importance = pd.DataFrame({
            'feature': feature_names_used,
            'importance': importances
        }).sort_values('importance', ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=importance, x='importance', y='feature', ax=ax)
        ax.set_title(f'Feature Importance - {model_name}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" Saved: {output_path.name}")


def plot_ensemble_comparison(ridge_preds, lgb_preds, y_val, output_path: Path):
    """Compare Ridge and LightGBM component predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Ridge vs actual
    axes[0].scatter(y_val, ridge_preds, alpha=0.5, s=20)
    axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Ridge Predictions')
    axes[0].set_title('Ridge Component')

    # LightGBM vs actual
    axes[1].scatter(y_val, lgb_preds, alpha=0.5, s=20)
    axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('LightGBM Predictions')
    axes[1].set_title('LightGBM Component')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Saved: {output_path.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(" FSP SELECTION - ENSEMBLE MODEL TRAINING (Ridge + LightGBM)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Split ratios: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"  Data months: {DATA_MONTHS}")
    print(f"  Model version: {MODEL_VERSION}")
    print(f"  Ensemble weights: Ridge {ENSEMBLE_RIDGE_WEIGHT*100:.0f}% | LightGBM {(1-ENSEMBLE_RIDGE_WEIGHT)*100:.0f}%")

    # =========================================================================
    # STEP 1: Load and Pivot Data
    # =========================================================================
    print("\n" + "=" * 70)
    print(" LOADING AND PIVOTING DATA")
    print("=" * 70)

    interim_file = INTERIM_DIR / 'eda_processed_data.parquet'
    if interim_file.exists():
        df_raw = pd.read_parquet(interim_file)
        print(f" Loaded: {interim_file.name} ({df_raw.shape})")
    else:
        raise FileNotFoundError("No data files found!")

    df_pivoted = pivot_fsp_data(df_raw)

    # =========================================================================
    # STEP 2: Filter to Last N Months
    # =========================================================================
    date_col = 'timestamp' if 'timestamp' in df_pivoted.columns else 'date'
    df_pivoted[date_col] = pd.to_datetime(df_pivoted[date_col])

    max_date = df_pivoted[date_col].max()
    cutoff = max_date - pd.DateOffset(months=DATA_MONTHS)
    df = df_pivoted[df_pivoted[date_col] >= cutoff].copy()

    print(f"\n Filtered to last {DATA_MONTHS} months: {len(df):,} rows")

    fsp_cols = get_fsp_forecast_columns(df)
    print(f" FSP columns: {fsp_cols}")

    df = calculate_fsp_errors(df, TARGET)

    # =========================================================================
    # STEP 3: Drop Missing Data
    # =========================================================================
    print("\n" + "=" * 70)
    print(" CLEANING DATA")
    print("=" * 70)

    required = [TARGET]
    df_clean = df.dropna(subset=required).copy()

    fsp_mask = df_clean[fsp_cols].notna().any(axis=1)
    df_clean = df_clean[fsp_mask].copy()

    print(f" Dropped {len(df) - len(df_clean):,} rows with missing required data")

    # =========================================================================
    # STEP 4: Create Forward-Shifted Target for 6-Block-Ahead Forecasting
    # =========================================================================
    print("\n" + "=" * 70)
    print(f" CREATE 6-BLOCK-AHEAD TARGET (shift=-{PREDICTION_HORIZON})")
    print("=" * 70)

    # Create forward-shifted target: predict actual_power from {PREDICTION_HORIZON} blocks in the future
    df_clean[TARGET_HORIZON] = df_clean[TARGET].shift(-PREDICTION_HORIZON)

    # Drop the trailing PREDICTION_HORIZON rows where target_horizon is NaN
    rows_before = len(df_clean)
    df_clean = df_clean.dropna(subset=[TARGET_HORIZON])
    rows_dropped = rows_before - len(df_clean)

    print(f" Created target_horizon (shift of -6 blocks)")
    print(f" Dropped {rows_dropped:,} trailing rows with NaN target_horizon")
    print(f" Remaining rows: {len(df_clean):,}")

    # =========================================================================
    # STEP 5: Temporal Split
    # =========================================================================
    print("\n" + "=" * 70)
    print(f" TEMPORAL SPLIT ({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%})")
    print("=" * 70)

    train_df, val_df, test_df = create_temporal_split(df_clean, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # =========================================================================
    # STEP 6: Feature Engineering
    # =========================================================================
    print("\n" + "=" * 70)
    print(" FEATURE ENGINEERING")
    print("=" * 70)

    train_df = create_time_features(train_df)
    val_df = create_time_features(val_df)
    test_df = create_time_features(test_df)

    train_df = create_rolling_features(train_df, TARGET, [1, 6, 24, 96])
    val_df = create_rolling_features(val_df, TARGET, [1, 6, 24, 96])
    test_df = create_rolling_features(test_df, TARGET, [1, 6, 24, 96])

    train_df, encoders = encode_categorical_features(train_df, [TARGET, 'date', 'timestamp', 'sscode'])
    val_df, _ = encode_categorical_features(val_df, [TARGET, 'date', 'timestamp', 'sscode'], encoders)
    test_df, _ = encode_categorical_features(test_df, [TARGET, 'date', 'timestamp', 'sscode'], encoders)

    feature_cols = get_feature_columns(train_df, TARGET, EXCLUDE_PATTERNS)

    # Remove target_horizon from features (it's the target we're predicting)
    if TARGET_HORIZON in feature_cols:
        feature_cols.remove(TARGET_HORIZON)

    print(f"\n Selected {len(feature_cols)} features")

    # Drop any feature columns that are entirely NaN to avoid shape mismatch downstream
    nan_only_cols = [c for c in feature_cols if train_df[c].isna().all()]
    if nan_only_cols:
        print(f" Dropping {len(nan_only_cols)} all-NaN features: {nan_only_cols}")
        feature_cols = [c for c in feature_cols if c not in nan_only_cols]
        train_df = train_df.drop(columns=nan_only_cols)
        val_df = val_df.drop(columns=nan_only_cols)
        test_df = test_df.drop(columns=nan_only_cols)
        print(f" Using {len(feature_cols)} features after dropping NaN-only columns")

    # =========================================================================
    # STEP 7: Prepare X and y (using 6-block-ahead target)
    # =========================================================================
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_HORIZON].values  # 6-block-ahead target
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_HORIZON].values  # 6-block-ahead target
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_HORIZON].values  # 6-block-ahead target

    print(f"\n Prepared training data")
    print(f"  Target: {TARGET_HORIZON} (6-block-ahead shifted actual_power)")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")

    # Imputation for rolling features
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)
    print(f"\n Imputed NaN values using median strategy")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    with open(MODELS_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open(MODELS_DIR / 'imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)

    has_shift, pval = check_distribution_shift(y_train, y_test)
    print(f"\n{' Distribution shift detected' if has_shift else ' No distribution shift'} (p={pval:.4f})")

    # =========================================================================
    # STEP 8: Training Models
    # =========================================================================
    print("\n" + "=" * 70)
    print(" TRAINING MODELS")
    print("=" * 70)

    model_results = []
    all_predictions = {}

    # Ridge
    print("\n Training Ridge...")
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge.fit(X_train_scaled, y_train)
    ridge_val = ridge.predict(X_val_scaled)
    ridge_test = ridge.predict(X_test_scaled)
    model_results.append(evaluate_model(y_val, ridge_val, 'Ridge'))
    all_predictions['ridge'] = {'val': ridge_val, 'test': ridge_test}
    print(f" Ridge - Val MAE: {model_results[-1]['MAE']}, R2: {model_results[-1]['R2']}")
    with open(MODELS_DIR / 'ridge_model.pkl', 'wb') as f:
        pickle.dump(ridge, f)

    # Random Forest
    print("\n Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    rf_val = rf.predict(X_val)
    rf_test = rf.predict(X_test)
    model_results.append(evaluate_model(y_val, rf_val, 'Random Forest'))
    all_predictions['random_forest'] = {'val': rf_val, 'test': rf_test}
    print(f" RF - Val MAE: {model_results[-1]['MAE']}, R2: {model_results[-1]['R2']}")
    with open(MODELS_DIR / 'random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    plot_feature_importance(rf, feature_cols, 'Random Forest', PLOTS_DIR / 'feature_importance_rf.png')

    # XGBoost
    if XGB_AVAILABLE:
        print("\n Training XGBoost...")
        xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, n_jobs=-1, random_state=RANDOM_STATE)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_val = xgb_model.predict(X_val)
        xgb_test = xgb_model.predict(X_test)
        model_results.append(evaluate_model(y_val, xgb_val, 'XGBoost'))
        all_predictions['xgboost'] = {'val': xgb_val, 'test': xgb_test}
        print(f" XGBoost - Val MAE: {model_results[-1]['MAE']}, R2: {model_results[-1]['R2']}")
        with open(MODELS_DIR / 'xgboost_model.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)
        plot_feature_importance(xgb_model, feature_cols, 'XGBoost', PLOTS_DIR / 'feature_importance_xgb.png')

    # LightGBM
    if LGB_AVAILABLE:
        print("\n Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(n_estimators=200, max_depth=10, learning_rate=0.05, n_jobs=-1, random_state=RANDOM_STATE, verbose=-1)
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        lgb_val = lgb_model.predict(X_val)
        lgb_test = lgb_model.predict(X_test)
        model_results.append(evaluate_model(y_val, lgb_val, 'LightGBM'))
        all_predictions['lightgbm'] = {'val': lgb_val, 'test': lgb_test}
        print(f" LightGBM - Val MAE: {model_results[-1]['MAE']}, R2: {model_results[-1]['R2']}")
        with open(MODELS_DIR / 'lightgbm_model.pkl', 'wb') as f:
            pickle.dump(lgb_model, f)
        plot_feature_importance(lgb_model, feature_cols, 'LightGBM', PLOTS_DIR / 'feature_importance_lgb.png')

    # =========================================================================
    # STEP 9: Create Ridge + LightGBM Ensemble
    # =========================================================================
    print("\n" + "=" * 70)
    print(" CREATING RIDGE + LIGHTGBM ENSEMBLE")
    print("=" * 70)

    if LGB_AVAILABLE:
        print("\n Ensemble: Ridge + LightGBM")
        ensemble = RidgeLightGBMEnsemble(
            ridge_model=ridge,
            lightgbm_model=lgb_model,
            ridge_weight=ENSEMBLE_RIDGE_WEIGHT,
            scaler=scaler,
            imputer=imputer
        )

        ensemble_val = ensemble.predict(X_val)
        ensemble_test = ensemble.predict(X_test)

        model_results.append(evaluate_model(y_val, ensemble_val, 'Ensemble (Ridge+LGB)'))
        all_predictions['ensemble_ridge_lgb'] = {'val': ensemble_val, 'test': ensemble_test}

        print(f" Ensemble - Val MAE: {model_results[-1]['MAE']}, R2: {model_results[-1]['R2']}")

        # Save ensemble
        with open(MODELS_DIR / 'ensemble_ridge_lgb.pkl', 'wb') as f:
            pickle.dump(ensemble, f)

        save_ensemble_config(MODELS_DIR, ENSEMBLE_RIDGE_WEIGHT)

        # Plot ensemble component comparison
        ridge_preds, lgb_preds = ensemble.get_component_predictions(X_val)
        plot_ensemble_comparison(ridge_preds, lgb_preds, y_val, PLOTS_DIR / 'ensemble_components.png')

    # =========================================================================
    # STEP 10: Results Summary - Validation & Test Set Comparison
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
    # Evaluate all models on Test Set
    # =========================================================================
    print("\n" + "=" * 70)
    print(" MODEL COMPARISON (Test Set)")
    print("=" * 70)

    test_results = []
    for model_name, preds in all_predictions.items():
        test_pred = preds['test']
        test_metrics = evaluate_model(y_test, test_pred, model_name)
        test_results.append(test_metrics)

    test_results_df = pd.DataFrame(test_results).sort_values('MAE')
    print(test_results_df.to_string(index=False))

    # Save test results
    test_results_df.to_csv(REPORTS_DIR / 'model_results_test.csv', index=False)

    # =========================================================================
    # STEP 11: Save Test Predictions
    # =========================================================================
    print("\n" + "=" * 70)
    print(" SAVING TEST PREDICTIONS")
    print("=" * 70)

    for model_name, preds in all_predictions.items():
        test_pred = preds['test']
        test_df_adj = test_df.reset_index(drop=True)

        selected_fsps, scheduled_power, confidence = select_best_fsp_by_prediction(
            test_df_adj, test_pred
        )

        output_path = PREDS_DIR / f'test_predictions_{model_name}.csv'
        save_model_output(
            test_df_adj,
            test_pred, selected_fsps, scheduled_power, confidence,
            model_name, output_path
        )

    # =========================================================================
    # STEP 11B: Save Validation & Test Sets
    # =========================================================================
    print("\n" + "=" * 70)
    print(" SAVING VALIDATION & TEST SETS")
    print("=" * 70)

    # Save validation set (features + target)
    val_df_copy = val_df.copy()
    val_df_copy['target'] = y_val
    val_output_path = PREDS_DIR / 'val_set.csv'
    val_df_copy.to_csv(val_output_path, index=False)
    print(f" Saved: {val_output_path.name} ({len(val_df_copy)} samples)")

    # Save test set (features + target)
    test_df_copy = test_df.copy()
    test_df_copy['target'] = y_test
    test_output_path = PREDS_DIR / 'test_set.csv'
    test_df_copy.to_csv(test_output_path, index=False)
    print(f" Saved: {test_output_path.name} ({len(test_df_copy)} samples)")

    # =========================================================================
    # STEP 12: Save Artifacts
    # =========================================================================
    print("\n" + "=" * 70)
    print(" SAVING ARTIFACTS")
    print("=" * 70)

    with open(MODELS_DIR / 'feature_columns.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)

    # Extract time periods for monitoring
    time_periods = {}
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        if 'date' in split_df.columns:
            time_periods[split_name] = {
                'start': str(split_df['date'].min()),
                'end': str(split_df['date'].max()),
                'n_samples': len(split_df)
            }
        elif 'timestamp' in split_df.columns:
            time_periods[split_name] = {
                'start': str(pd.to_datetime(split_df['timestamp']).min()),
                'end': str(pd.to_datetime(split_df['timestamp']).max()),
                'n_samples': len(split_df)
            }

    metadata = {
        'version': MODEL_VERSION,
        'training_date': datetime.now().isoformat(),
        'prediction_horizon': PREDICTION_HORIZON,  # 6-block-ahead forecasting
        'target_column': TARGET_HORIZON,  # Using forward-shifted target
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'time_periods': time_periods,
        'fsp_providers': FSP_PROVIDERS,
        'models_trained': list(all_predictions.keys()),
        'best_model': best_model,
        'ensemble_config': {
            'type': 'Ridge + LightGBM',
            'ridge_weight': ENSEMBLE_RIDGE_WEIGHT,
            'lightgbm_weight': 1.0 - ENSEMBLE_RIDGE_WEIGHT
        },
        'metrics': {m['Model']: m for m in model_results},
        'data_info': {
            'months_used': DATA_MONTHS,
            'total_rows': len(df_clean),
            'features_count': len(feature_cols)
        }
    }
    with open(MODELS_DIR / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(" Saved all artifacts")

    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nBest model: {best_model}")
    print(f"Models saved in: {MODELS_DIR}")
    print(f"Predictions saved in: {PREDS_DIR}")


if __name__ == '__main__':
    main()
