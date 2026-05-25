"""
FSP Selection Model Training Script v2
=======================================

Updated to handle pivoted FSP data (1 row per time block).

Key Changes from v1:
- Pivots data from 5 FSP rows per block to 1 row with separate columns
- Shows all FSP forecasts in output
- Correctly selects and names best FSP
- No duplicate rows in output

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
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor

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

# Deep Learning models removed (v2 uses shallow models only)
DL_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Suppress only specific warnings, not all
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

# Validate split ratios sum to 1.0
if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-6:
    raise ValueError(f"Split ratios must sum to 1.0, got {TRAIN_RATIO + VAL_RATIO + TEST_RATIO}")

DATA_MONTHS = config.get('training.data_months', 18)
RANDOM_STATE = config.get('training.random_seed', 42)
MODEL_VERSION = config.get('versioning.current_version', '1.0.0')

TARGET = 'actual_power'
# Keep manual schedule data out of model features (only for reporting/comparison)
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

    # Symmetric MAPE - handles zeros safely by using (actual + pred) / 2 as denominator
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    safe_denominator = np.where(denominator == 0, 1, denominator)
    smape = np.mean(np.abs(y_true - y_pred) / safe_denominator) * 100

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
    output['actual_power'] = df.get(TARGET, np.nan)
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
    """Plot feature importance."""
    if hasattr(model, 'feature_importances_'):
        importances = np.array(model.feature_importances_)

        # Align lengths defensively in case the estimator drops features internally
        if len(importances) != len(feature_names):
            warnings.warn(
                f"Feature importance length mismatch: {len(importances)} importances vs {len(feature_names)} names. "
                "Truncating to the smaller size to keep training flowing."
            )
        n = min(len(importances), len(feature_names))
        importance = pd.DataFrame({
            'feature': feature_names[:n],
            'importance': importances[:n]
        }).sort_values('importance', ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=importance, x='importance', y='feature', ax=ax)
        ax.set_title(f'Feature Importance - {model_name}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" Saved: {output_path.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(" FSP SELECTION MODEL TRAINING v2")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Split ratios: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"  Data months: {DATA_MONTHS}")
    print(f"  Model version: {MODEL_VERSION}")

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

    # Pivot FSP data: 5 rows per block  1 row per block
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

    # Get FSP columns
    fsp_cols = get_fsp_forecast_columns(df)
    print(f" FSP columns: {fsp_cols}")

    # Calculate FSP errors for oracle comparison
    df = calculate_fsp_errors(df, TARGET)

    # =========================================================================
    # STEP 3: Drop Missing Data
    # =========================================================================
    print("\n" + "=" * 70)
    print(" CLEANING DATA")
    print("=" * 70)

    # Require at least one FSP forecast and actual power
    required = [TARGET]
    df_clean = df.dropna(subset=required).copy()

    # Also need at least one FSP forecast
    fsp_mask = df_clean[fsp_cols].notna().any(axis=1)
    df_clean = df_clean[fsp_mask].copy()

    print(f" Dropped {len(df) - len(df_clean):,} rows with missing required data")

    # =========================================================================
    # STEP 4: Temporal Split
    # =========================================================================
    print("\n" + "=" * 70)
    print(f" TEMPORAL SPLIT ({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%})")
    print("=" * 70)

    train_df, val_df, test_df = create_temporal_split(df_clean, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # =========================================================================
    # STEP 5: Feature Engineering
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
    print(f"\n Selected {len(feature_cols)} features")

    # =========================================================================
    # STEP 6: Prepare X and y
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    with open(MODELS_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Save imputer for inference
    with open(MODELS_DIR / 'imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)

    has_shift, pval = check_distribution_shift(y_train, y_test)
    print(f"\n{' Distribution shift detected' if has_shift else ' No distribution shift'} (p={pval:.4f})")

    # =========================================================================
    # STEP 7: Train Models
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

    # SVM (Support Vector Machine) - Using LinearSVR for scalability
    print("\n Training SVM (LinearSVR)...")
    svm_model = LinearSVR(C=100, random_state=RANDOM_STATE, max_iter=2000)
    svm_model.fit(X_train_scaled, y_train)
    svm_val = svm_model.predict(X_val_scaled)
    svm_test = svm_model.predict(X_test_scaled)
    model_results.append(evaluate_model(y_val, svm_val, 'SVM'))
    all_predictions['svm'] = {'val': svm_val, 'test': svm_test}
    print(f" SVM - Val MAE: {model_results[-1]['MAE']}, R2: {model_results[-1]['R2']}")
    with open(MODELS_DIR / 'svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)

    # KNN (K-Nearest Neighbors)
    print("\n Training KNN...")
    knn_model = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    knn_model.fit(X_train_scaled, y_train)
    knn_val = knn_model.predict(X_val_scaled)
    knn_test = knn_model.predict(X_test_scaled)
    model_results.append(evaluate_model(y_val, knn_val, 'KNN'))
    all_predictions['knn'] = {'val': knn_val, 'test': knn_test}
    print(f" KNN - Val MAE: {model_results[-1]['MAE']}, R2: {model_results[-1]['R2']}")
    with open(MODELS_DIR / 'knn_model.pkl', 'wb') as f:
        pickle.dump(knn_model, f)

    # Bayesian Ridge (Proper Bayesian Regression - not a classifier)
    print("\n Training Bayesian Ridge...")
    bayesian_model = BayesianRidge()
    bayesian_model.fit(X_train_scaled, y_train)
    bayesian_val = bayesian_model.predict(X_val_scaled)
    bayesian_test = bayesian_model.predict(X_test_scaled)
    model_results.append(evaluate_model(y_val, bayesian_val, 'Bayesian Ridge'))
    all_predictions['bayesian_ridge'] = {'val': bayesian_val, 'test': bayesian_test}
    print(f" Bayesian Ridge - Val MAE: {model_results[-1]['MAE']}, R2: {model_results[-1]['R2']}")
    with open(MODELS_DIR / 'bayesian_ridge_model.pkl', 'wb') as f:
        pickle.dump(bayesian_model, f)

    # =========================================================================
    # STEP 8: Results
    # =========================================================================
    print("\n" + "=" * 70)
    print(" MODEL COMPARISON (Validation)")
    print("=" * 70)

    results_df = pd.DataFrame(model_results).sort_values('MAE')
    print(results_df.to_string(index=False))
    best_model = results_df.iloc[0]['Model']
    print(f"\n Best: {best_model}")
    results_df.to_csv(REPORTS_DIR / 'model_results.csv', index=False)

    # =========================================================================
    # STEP 9: Save Predictions with FSP Selection
    # =========================================================================
    print("\n" + "=" * 70)
    print(" SAVING TEST PREDICTIONS")
    print("=" * 70)

    for model_name, preds in all_predictions.items():
        test_pred = preds['test']
        seq_offset = preds.get('seq_offset', 0)

        # Adjust test_df for sequence models (they skip first seq_offset rows)
        if seq_offset > 0:
            test_df_adj = test_df.iloc[seq_offset:].reset_index(drop=True)
        else:
            test_df_adj = test_df.reset_index(drop=True)

        # Select best FSP based on predictions
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
    # STEP 10: Save Artifacts
    # =========================================================================
    print("\n" + "=" * 70)
    print(" SAVING ARTIFACTS")
    print("=" * 70)

    with open(MODELS_DIR / 'feature_columns.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)

    metadata = {
        'version': MODEL_VERSION,
        'training_date': datetime.now().isoformat(),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'fsp_providers': FSP_PROVIDERS,
        'models_trained': list(all_predictions.keys()),
        'best_model': best_model,
        'metrics': {m['Model']: m for m in model_results}
    }
    with open(MODELS_DIR / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(" Saved all artifacts")

    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
