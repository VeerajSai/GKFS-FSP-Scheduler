"""
FSP Selection Model Training Script - Variance-Based Temporal Split
====================================================================

Trains ensemble models on 24 months of data, split by wind speed variance:
- High variance months: Train on 2024, Test on 2025
- Low variance months: Train on 2024, Test on 2025

Uses Ridge + LightGBM ensemble with weighted averaging.

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
MODELS_DIR = OUTPUT_DIR / 'models_variance_temporal'
PREDS_DIR = OUTPUT_DIR / 'predictions_variance_temporal'
PLOTS_DIR = OUTPUT_DIR / 'plots_variance_temporal'
REPORTS_DIR = OUTPUT_DIR / 'reports_variance_temporal'

for d in [MODELS_DIR, PREDS_DIR, PLOTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Training config
DATA_MONTHS = 24 # 24 months of data
RANDOM_STATE = config.get('training.random_seed', 42)
MODEL_VERSION = config.get('versioning.current_version', '1.0.0')
PREDICTION_HORIZON = config.get('training.prediction_horizon', 6) # 6-block-ahead forecasting

# Ensemble config
ENSEMBLE_RIDGE_WEIGHT = 0.4 # Ridge: 40%, LightGBM: 60%

TARGET = 'actual_power'
TARGET_HORIZON = 'target_horizon' # Forward-shifted target for 6-block-ahead forecasting
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
    output['actual_power'] = df.get(TARGET_HORIZON, np.nan)
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
        importances = model.feature_importances_

        if len(feature_names) != len(importances):
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


def calculate_monthly_wind_variance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate wind speed variance for each month.

    Returns:
        DataFrame with columns: year, month, wind_variance
    """
    # Group by year and month
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month

    # Calculate variance of actual_power (proxy for wind speed variance)
    monthly_variance = df.groupby(['year', 'month'])[TARGET].var().reset_index()
    monthly_variance.columns = ['year', 'month', 'wind_variance']

    return monthly_variance


def split_months_by_variance(monthly_variance: pd.DataFrame, threshold_percentile: float = 50.0) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Split months into high and low variance groups based on threshold percentile.

    Args:
        monthly_variance: DataFrame with year, month, wind_variance
        threshold_percentile: Percentile threshold (default 50 for median split)

    Returns:
        high_variance_months: List of (year, month) tuples
        low_variance_months: List of (year, month) tuples
    """
    threshold = np.percentile(monthly_variance['wind_variance'], threshold_percentile)

    high_variance_months = monthly_variance[monthly_variance['wind_variance'] >= threshold][['year', 'month']].values.tolist()
    low_variance_months = monthly_variance[monthly_variance['wind_variance'] < threshold][['year', 'month']].values.tolist()

    # Convert to list of tuples
    high_variance_months = [tuple(m) for m in high_variance_months]
    low_variance_months = [tuple(m) for m in low_variance_months]

    return high_variance_months, low_variance_months


def filter_data_by_months(df: pd.DataFrame, months_list: List[Tuple[int, int]]) -> pd.DataFrame:
    """
    Filter dataframe to include only specified months.

    Args:
        df: DataFrame with 'timestamp' column
        months_list: List of (year, month) tuples

    Returns:
        Filtered DataFrame
    """
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month

    # Create a set of (year, month) tuples for fast lookup
    months_set = set(months_list)

    # Filter
    mask = df.apply(lambda row: (row['year'], row['month']) in months_set, axis=1)
    filtered_df = df[mask].copy()

    # Drop temporary columns
    filtered_df = filtered_df.drop(columns=['year', 'month'])

    return filtered_df


def split_by_year(df: pd.DataFrame, train_year: int = 2024, test_year: int = 2025) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by year for training and testing.

    Args:
        df: DataFrame with 'timestamp' column
        train_year: Year to use for training
        test_year: Year to use for testing

    Returns:
        train_df: Data from train_year
        test_df: Data from test_year
    """
    df['year'] = df['timestamp'].dt.year

    train_df = df[df['year'] == train_year].copy()
    test_df = df[df['year'] == test_year].copy()

    # Drop temporary column
    train_df = train_df.drop(columns=['year'])
    test_df = test_df.drop(columns=['year'])

    return train_df, test_df


def train_and_evaluate_variance_group(
    df: pd.DataFrame,
    group_name: str,
    months_list: List[Tuple[int, int]],
    train_year: int = 2024,
    test_year: int = 2025
) -> Dict:
    """
    Train and evaluate models for a specific variance group.

    Args:
        df: Full dataframe
        group_name: Name of the group ('high_variance' or 'low_variance')
        months_list: List of (year, month) tuples for this group
        train_year: Year to use for training
        test_year: Year to use for testing

    Returns:
        Dictionary with results
    """
    print("\n" + "=" * 70)
    print(f" PROCESSING {group_name.upper()} GROUP")
    print("=" * 70)

    # Get FSP columns
    fsp_cols = get_fsp_forecast_columns(df)

    # Filter to this variance group
    df_group = filter_data_by_months(df, months_list)
    print(f" Filtered to {len(months_list)} months: {len(df_group):,} rows")

    # Create output directories for this group
    group_models_dir = MODELS_DIR / group_name
    group_preds_dir = PREDS_DIR / group_name
    group_plots_dir = PLOTS_DIR / group_name

    for d in [group_models_dir, group_preds_dir, group_plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Split by year (train on 2024, test on 2025)
    train_df, test_df = split_by_year(df_group, train_year, test_year)
    print(f" Train ({train_year}): {len(train_df):,} rows")
    print(f" Test ({test_year}): {len(test_df):,} rows")

    # Diagnostic: Check data distribution
    print(f"\n DATA DISTRIBUTION DIAGNOSTICS:")
    print(f"  Train {TARGET} - Mean: {train_df[TARGET].mean():.2f}, Std: {train_df[TARGET].std():.2f}, Min: {train_df[TARGET].min():.2f}, Max: {train_df[TARGET].max():.2f}")
    print(f"  Test {TARGET} - Mean: {test_df[TARGET].mean():.2f}, Std: {test_df[TARGET].std():.2f}, Min: {test_df[TARGET].min():.2f}, Max: {test_df[TARGET].max():.2f}")

    # Diagnostic: Check FSP forecast quality
    print(f"\n FSP FORECAST DIAGNOSTICS:")
    for fsp_col in fsp_cols:
        train_fsp_mae = (train_df[TARGET] - train_df[fsp_col]).abs().mean()
        test_fsp_mae = (test_df[TARGET] - test_df[fsp_col]).abs().mean()
        print(f"  {fsp_col}: Train MAE={train_fsp_mae:.2f}, Test MAE={test_fsp_mae:.2f}")

    # Check if we have data for both years
    if len(train_df) == 0:
        print(f"  No training data for {train_year} in {group_name} group")
        return None

    if len(test_df) == 0:
        print(f"  No test data for {test_year} in {group_name} group")
        return None

    # Create forward-shifted target
    train_df[TARGET_HORIZON] = train_df[TARGET].shift(-PREDICTION_HORIZON)
    test_df[TARGET_HORIZON] = test_df[TARGET].shift(-PREDICTION_HORIZON)

    # Drop trailing rows with NaN target_horizon
    train_df = train_df.dropna(subset=[TARGET_HORIZON])
    test_df = test_df.dropna(subset=[TARGET_HORIZON])

    print(f" After creating target_horizon: Train {len(train_df):,} | Test {len(test_df):,}")

    # Feature engineering
    print("\n  FEATURE ENGINEERING")
    train_df = create_time_features(train_df)
    test_df = create_time_features(test_df)

    # Add year feature to help with distribution shift
    train_df['year'] = train_df['timestamp'].dt.year
    test_df['year'] = test_df['timestamp'].dt.year

    # Enhanced rolling features with more windows
    train_df = create_rolling_features(train_df, TARGET, [1, 2, 3, 6, 12, 24, 48, 96])
    test_df = create_rolling_features(test_df, TARGET, [1, 2, 3, 6, 12, 24, 48, 96])

    # Add interaction features for FSP forecasts
    fsp_cols = get_fsp_forecast_columns(train_df)
    if len(fsp_cols) >= 2:
        # Average of all FSP forecasts
        train_df['fsp_avg'] = train_df[fsp_cols].mean(axis=1)
        test_df['fsp_avg'] = test_df[fsp_cols].mean(axis=1)

        # Std of FSP forecasts (measure of disagreement)
        train_df['fsp_std'] = train_df[fsp_cols].std(axis=1)
        test_df['fsp_std'] = test_df[fsp_cols].std(axis=1)

        # Min and Max of FSP forecasts
        train_df['fsp_min'] = train_df[fsp_cols].min(axis=1)
        test_df['fsp_min'] = test_df[fsp_cols].min(axis=1)
        train_df['fsp_max'] = train_df[fsp_cols].max(axis=1)
        test_df['fsp_max'] = test_df[fsp_cols].max(axis=1)

    train_df, encoders = encode_categorical_features(train_df, [TARGET, 'date', 'timestamp', 'sscode'])
    test_df, _ = encode_categorical_features(test_df, [TARGET, 'date', 'timestamp', 'sscode'], encoders)

    feature_cols = get_feature_columns(train_df, TARGET, EXCLUDE_PATTERNS)

    # Remove target_horizon from features
    if TARGET_HORIZON in feature_cols:
        feature_cols.remove(TARGET_HORIZON)

    print(f" Selected {len(feature_cols)} features")

    # Drop NaN-only features
    nan_only_cols = [c for c in feature_cols if train_df[c].isna().all()]
    if nan_only_cols:
        print(f" Dropping {len(nan_only_cols)} all-NaN features")
        feature_cols = [c for c in feature_cols if c not in nan_only_cols]
        train_df = train_df.drop(columns=nan_only_cols)
        test_df = test_df.drop(columns=nan_only_cols)

    # Prepare X and y
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_HORIZON].values
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_HORIZON].values

    # Imputation
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Robust scaling (better for outliers than StandardScaler)
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save preprocessing objects
    with open(group_models_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(group_models_dir / 'imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)
    with open(group_models_dir / 'feature_columns.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)

    # Check distribution shift
    has_shift, pval = check_distribution_shift(y_train, y_test)
    print(f"{' Distribution shift detected' if has_shift else ' No distribution shift'} (p={pval:.4f})")

    # Train models
    print("\n TRAINING MODELS")
    model_results = []
    all_predictions = {}

    # Ridge - Improved hyperparameters
    print("\n Training Ridge...")
    ridge = Ridge(alpha=10.0, random_state=RANDOM_STATE)  # Increased alpha for regularization
    ridge.fit(X_train_scaled, y_train)
    ridge_test = ridge.predict(X_test_scaled)
    model_results.append(evaluate_model(y_test, ridge_test, 'Ridge'))
    all_predictions['ridge'] = ridge_test
    print(f" Ridge - Test MAE: {model_results[-1]['MAE']}, R2: {model_results[-1]['R2']}")
    with open(group_models_dir / 'ridge_model.pkl', 'wb') as f:
        pickle.dump(ridge, f)

    # Random Forest - Improved hyperparameters
    print("\n Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=300,  # Increased from 200
        max_depth=20,  # Increased from 15
        min_samples_split=5,  # Added to prevent overfitting
        min_samples_leaf=2,  # Added to prevent overfitting
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)
    rf_test = rf.predict(X_test)
    model_results.append(evaluate_model(y_test, rf_test, 'Random Forest'))
    all_predictions['random_forest'] = rf_test
    print(f" RF - Test MAE: {model_results[-1]['MAE']}, R2: {model_results[-1]['R2']}")
    with open(group_models_dir / 'random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    plot_feature_importance(rf, feature_cols, 'Random Forest', group_plots_dir / 'feature_importance_rf.png')

    # XGBoost - Improved hyperparameters
    if XGB_AVAILABLE:
        print("\n Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,  # Increased from 200
            max_depth=10,  # Increased from 8
            learning_rate=0.03,  # Decreased from 0.05 for better generalization
            subsample=0.8,  # Added to prevent overfitting
            colsample_bytree=0.8,  # Added to prevent overfitting
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=False
        )
        xgb_model.fit(X_train, y_train, verbose=False)
        xgb_test = xgb_model.predict(X_test)
        model_results.append(evaluate_model(y_test, xgb_test, 'XGBoost'))
        all_predictions['xgboost'] = xgb_test
        print(f" XGBoost - Test MAE: {model_results[-1]['MAE']}, R2: {model_results[-1]['R2']}")
        with open(group_models_dir / 'xgboost_model.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)
        plot_feature_importance(xgb_model, feature_cols, 'XGBoost', group_plots_dir / 'feature_importance_xgb.png')

    # LightGBM - Improved hyperparameters
    if LGB_AVAILABLE:
        print("\n Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=300,  # Increased from 200
            max_depth=10,  # Increased from 8
            learning_rate=0.03,  # Decreased from 0.05
            num_leaves=31,  # Added for better control
            subsample=0.8,  # Added to prevent overfitting
            colsample_bytree=0.8,  # Added to prevent overfitting
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_test = lgb_model.predict(X_test)
        model_results.append(evaluate_model(y_test, lgb_test, 'LightGBM'))
        all_predictions['lightgbm'] = lgb_test
        print(f" LightGBM - Test MAE: {model_results[-1]['MAE']}, R2: {model_results[-1]['R2']}")
        with open(group_models_dir / 'lightgbm_model.pkl', 'wb') as f:
            pickle.dump(lgb_model, f)
        plot_feature_importance(lgb_model, feature_cols, 'LightGBM', group_plots_dir / 'feature_importance_lgb.png')

    # Ensemble (Ridge + LightGBM)
    if LGB_AVAILABLE:
        print("\n Creating Ridge + LightGBM Ensemble...")
        ensemble = RidgeLightGBMEnsemble(
            ridge_model=ridge,
            lightgbm_model=lgb_model,
            ridge_weight=ENSEMBLE_RIDGE_WEIGHT,
            scaler=scaler,
            imputer=imputer
        )
        ensemble_test = ensemble.predict(X_test)
        model_results.append(evaluate_model(y_test, ensemble_test, 'Ridge+LightGBM Ensemble'))
        all_predictions['ensemble'] = ensemble_test
        print(f" Ensemble - Test MAE: {model_results[-1]['MAE']}, R2: {model_results[-1]['R2']}")
        with open(group_models_dir / 'ensemble_model.pkl', 'wb') as f:
            pickle.dump(ensemble, f)
        save_ensemble_config(group_models_dir, ENSEMBLE_RIDGE_WEIGHT)

    # Select best model
    results_df = pd.DataFrame(model_results)
    best_model = results_df.loc[results_df['MAE'].idxmin(), 'Model']

    # Map model name to prediction key
    model_key_map = {
        'Ridge': 'ridge',
        'Random Forest': 'random_forest',
        'XGBoost': 'xgboost',
        'LightGBM': 'lightgbm',
        'Ridge+LightGBM Ensemble': 'ensemble'
    }
    best_predictions = all_predictions[model_key_map[best_model]]

    print(f"\n Best model: {best_model}")

    # FSP selection
    print("\n FSP SELECTION")
    selected_fsps, scheduled_power, confidence = select_best_fsp_by_prediction(test_df, best_predictions)

    # Save outputs
    output_df = save_model_output(
        test_df, best_predictions, selected_fsps, scheduled_power, confidence,
        best_model, group_preds_dir / f'{group_name}_predictions.csv'
    )

    # Calculate improvement over manual schedule
    manual_mae = output_df['manual_error'].mean()
    ml_mae = output_df['ml_error'].mean()
    improvement = ((manual_mae - ml_mae) / manual_mae) * 100

    print(f"\n RESULTS SUMMARY")
    print(f"  Manual Schedule MAE: {manual_mae:.4f}")
    print(f"  ML Model MAE: {ml_mae:.4f}")
    print(f"  Improvement: {improvement:.2f}%")

    # Save results
    results_summary = {
        'group_name': group_name,
        'months': months_list,
        'train_year': train_year,
        'test_year': test_year,
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'best_model': best_model,
        'manual_mae': float(manual_mae),
        'ml_mae': float(ml_mae),
        'improvement_percent': float(improvement),
        'model_results': results_df.to_dict('records')
    }

    with open(group_models_dir / 'results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    return results_summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(" FSP SELECTION - VARIANCE-BASED TEMPORAL SPLIT TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data months: {DATA_MONTHS}")
    print(f"  Model version: {MODEL_VERSION}")
    print(f"  Train year: 2024 | Test year: 2025")
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
    # STEP 2: Filter to Last 24 Months
    # =========================================================================
    date_col = 'timestamp' if 'timestamp' in df_pivoted.columns else 'date'
    df_pivoted[date_col] = pd.to_datetime(df_pivoted[date_col])

    max_date = df_pivoted[date_col].max()
    cutoff = max_date - pd.DateOffset(months=DATA_MONTHS)
    df = df_pivoted[df_pivoted[date_col] >= cutoff].copy()

    print(f"\n Filtered to last {DATA_MONTHS} months: {len(df):,} rows")
    print(f"  Date range: {df[date_col].min()} to {df[date_col].max()}")

    fsp_cols = get_fsp_forecast_columns(df)
    print(f" FSP columns: {fsp_cols}")

    df = calculate_fsp_errors(df, TARGET)

    # =========================================================================
    # STEP 3: Calculate Monthly Wind Variance
    # =========================================================================
    print("\n" + "=" * 70)
    print(" CALCULATING MONTHLY WIND VARIANCE")
    print("=" * 70)

    monthly_variance = calculate_monthly_wind_variance(df)
    print(f"\n Calculated variance for {len(monthly_variance)} months")
    print("\nMonthly Wind Variance:")
    print(monthly_variance.to_string(index=False))

    # Save monthly variance
    monthly_variance.to_csv(REPORTS_DIR / 'monthly_wind_variance.csv', index=False)

    # =========================================================================
    # STEP 4: Split Months by Variance
    # =========================================================================
    print("\n" + "=" * 70)
    print(" SPLITTING MONTHS BY VARIANCE")
    print("=" * 70)

    high_var_months, low_var_months = split_months_by_variance(monthly_variance, threshold_percentile=50.0)

    print(f"\n High variance months ({len(high_var_months)}):")
    for m in sorted(high_var_months):
        var_val = monthly_variance[(monthly_variance['year'] == m[0]) & (monthly_variance['month'] == m[1])]['wind_variance'].values[0]
        print(f"    {m[0]}-{m[1]:02d}: variance = {var_val:.4f}")

    print(f"\n Low variance months ({len(low_var_months)}):")
    for m in sorted(low_var_months):
        var_val = monthly_variance[(monthly_variance['year'] == m[0]) & (monthly_variance['month'] == m[1])]['wind_variance'].values[0]
        print(f"    {m[0]}-{m[1]:02d}: variance = {var_val:.4f}")

    # =========================================================================
    # STEP 5: Train and Evaluate for Each Variance Group
    # =========================================================================
    print("\n" + "=" * 70)
    print(" TRAINING AND EVALUATION")
    print("=" * 70)

    all_results = {}

    # High variance group
    high_var_results = train_and_evaluate_variance_group(
        df, 'high_variance', high_var_months, train_year=2024, test_year=2025
    )
    if high_var_results:
        all_results['high_variance'] = high_var_results

    # Low variance group
    low_var_results = train_and_evaluate_variance_group(
        df, 'low_variance', low_var_months, train_year=2024, test_year=2025
    )
    if low_var_results:
        all_results['low_variance'] = low_var_results

    # =========================================================================
    # STEP 6: Summary Report
    # =========================================================================
    print("\n" + "=" * 70)
    print(" FINAL SUMMARY")
    print("=" * 70)

    if all_results:
        print("\nResults by Variance Group:")
        print("-" * 70)

        for group_name, results in all_results.items():
            print(f"\n{group_name.upper()}:")
            print(f"  Train samples: {results['train_samples']:,}")
            print(f"  Test samples: {results['test_samples']:,}")
            print(f"  Best model: {results['best_model']}")
            print(f"  Manual MAE: {results['manual_mae']:.4f}")
            print(f"  ML MAE: {results['ml_mae']:.4f}")
            print(f"  Improvement: {results['improvement_percent']:.2f}%")

        # Save combined results
        with open(REPORTS_DIR / 'combined_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n All results saved to: {REPORTS_DIR}")
        print(f" Models saved to: {MODELS_DIR}")
        print(f" Predictions saved to: {PREDS_DIR}")
        print(f" Plots saved to: {PLOTS_DIR}")

    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
