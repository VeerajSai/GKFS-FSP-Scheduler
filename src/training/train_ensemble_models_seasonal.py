"""
FSP Selection Model Training Script - Seasonal Edition
======================================================

Trains models on 24 months of data with seasonal splits:
- Train: Season X in 2024
- Val/Test: Same Season X in 2025

Seasons:
- Winter: December, January, February
- Spring: March, April, May
- Summer: June, July, August
- Fall: September, October, November

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

# Suppress warnings
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
    create_rolling_features, create_time_features,
    encode_categorical_features, drop_missing_data, get_feature_columns,
    check_distribution_shift
)
from src.models.ensemble_model import RidgeLightGBMEnsemble, save_ensemble_config

config = load_config()

# Directories
DATA_DIR = PROJECT_DIR / config.get('data.processed_dir', 'data/processed')
INTERIM_DIR = PROJECT_DIR / config.get('data.interim_dir', 'data/interim')
OUTPUT_DIR = PROJECT_DIR / 'outputs'
MODELS_DIR = OUTPUT_DIR / 'models_seasonal'
PREDS_DIR = OUTPUT_DIR / 'predictions_seasonal'
PLOTS_DIR = OUTPUT_DIR / 'plots_seasonal'
REPORTS_DIR = OUTPUT_DIR / 'reports_seasonal'

for d in [MODELS_DIR, PREDS_DIR, PLOTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Training config - Use 24 months for seasonal analysis
DATA_MONTHS = 24 # 2 full years
VAL_RATIO = 0.5 # Split 2025 season data: 50% val, 50% test
RANDOM_STATE = config.get('training.random_seed', 42)
MODEL_VERSION = config.get('versioning.current_version', '1.0.0')
PREDICTION_HORIZON = config.get('training.prediction_horizon', 6)

# Ensemble config
ENSEMBLE_RIDGE_WEIGHT = 0.4

TARGET = 'actual_power'
TARGET_HORIZON = 'target_horizon'
EXCLUDE_PATTERNS = ['date', 'timestamp', 'index', 'actual_', 'sscode', 'error_', 'schedule_']

np.random.seed(RANDOM_STATE)

# Season definitions
SEASONS = {
    'winter': [12, 1, 2],
    'spring': [3, 4, 5],
    'summer': [6, 7, 8],
    'fall': [9, 10, 11]
}


# =============================================================================
# SEASONAL SPLIT FUNCTIONS
# =============================================================================

def assign_season(date):
    """Assign season based on month."""
    month = date.month
    for season_name, months in SEASONS.items():
        if month in months:
            return season_name
    return None


def create_seasonal_split(
    df: pd.DataFrame,
    season: str,
    val_ratio: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data seasonally:
    - Train: Given season from year 1 (2024)
    - Val/Test: Same season from year 2 (2025), split 50/50

    Args:
        df: Full dataframe with 'date' column
        season: Season name ('winter', 'spring', 'summer', 'fall')
        val_ratio: Ratio to split year 2 season data between val and test

    Returns:
        train_df, val_df, test_df
    """
    if 'date' not in df.columns:
        raise ValueError("DataFrame must have 'date' column")

    # Add season column
    df = df.copy()
    df['season'] = df['date'].apply(assign_season)
    df['year'] = df['date'].dt.year

    # Filter for the specified season
    season_df = df[df['season'] == season].copy()

    if len(season_df) == 0:
        raise ValueError(f"No data found for season: {season}")

    # Get unique years
    years = sorted(season_df['year'].unique())

    if len(years) < 2:
        raise ValueError(f"Need at least 2 years of {season} data. Found only: {years}")

    # Use first year for training, last year for val/test
    train_year = years[0]
    test_year = years[-1]

    print(f"\n Seasonal Split for {season.upper()}:")
    print(f"  Train Year: {train_year}")
    print(f"  Val/Test Year: {test_year}")

    # Split by year
    train_df = season_df[season_df['year'] == train_year].copy()
    year2_df = season_df[season_df['year'] == test_year].copy()

    # Split year 2 into val and test (chronologically)
    year2_sorted = year2_df.sort_values('date').reset_index(drop=True)
    val_size = int(len(year2_sorted) * val_ratio)

    val_df = year2_sorted.iloc[:val_size].copy()
    test_df = year2_sorted.iloc[val_size:].copy()

    # Drop helper columns
    for split_df in [train_df, val_df, test_df]:
        split_df.drop(columns=['season', 'year'], inplace=True, errors='ignore')

    print(f"  Train: {len(train_df)} samples ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"  Val:   {len(val_df)} samples ({val_df['date'].min()} to {val_df['date'].max()})")
    print(f"  Test:  {len(test_df)} samples ({test_df['date'].min()} to {test_df['date'].max()})")

    return train_df, val_df, test_df


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Calculate comprehensive metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Symmetric MAPE
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
    """Select FSP whose forecast is closest to predicted actual power."""
    fsp_cols = get_fsp_forecast_columns(df)

    selected_fsps = []
    scheduled_power = []
    confidence = []

    for i in range(len(df)):
        pred = predictions[i] if i < len(predictions) else np.nan

        fsp_values = {}
        for fsp_col in fsp_cols:
            val = df.iloc[i].get(fsp_col, np.nan)
            if not np.isnan(val):
                fsp_name = fsp_col.replace('forecast_power_', '').upper()
                fsp_values[fsp_name] = val

        if fsp_values and not np.isnan(pred):
            errors = {fsp: abs(val - pred) for fsp, val in fsp_values.items()}
            best_fsp = min(errors, key=errors.get)

            selected_fsps.append(best_fsp)
            scheduled_power.append(fsp_values[best_fsp])

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
    output_path: Path,
    season: str
):
    """Save model predictions with FSP selection."""
    output_df = df.copy()
    output_df['ml_predicted_power'] = predictions
    output_df['ml_selected_fsp'] = selected_fsps
    output_df['ml_schedule_power'] = scheduled_power
    output_df['ml_confidence'] = confidence

    # Calculate error for ML prediction
    if 'actual_power' in output_df.columns:
        output_df['ml_error'] = np.abs(output_df['actual_power'] - output_df['ml_predicted_power'])

    # Add season info
    output_df['season'] = season

    output_df.to_csv(output_path, index=False)
    print(f"   Saved: {output_path.name}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_season(season: str):
    """Train models for a specific season."""

    print("\n" + "=" * 80)
    print(f" TRAINING MODELS FOR {season.upper()} SEASON")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n" + "=" * 70)
    print(" LOADING DATA")
    print("=" * 70)

    processed_path = DATA_DIR / 'sample_pss_dataset.parquet'
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found: {processed_path}")

    df = pd.read_parquet(processed_path)
    print(f" Loaded: {len(df):,} rows")

    # Ensure date column
    if 'date' not in df.columns and 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        raise ValueError("DataFrame must have 'date' or 'timestamp' column")

    # =========================================================================
    # STEP 2: Filter to Last 24 Months
    # =========================================================================
    print("\n" + "=" * 70)
    print(f" FILTERING TO LAST {DATA_MONTHS} MONTHS")
    print("=" * 70)

    df = df.sort_values('date').reset_index(drop=True)
    max_date = df['date'].max()
    cutoff = max_date - pd.DateOffset(months=DATA_MONTHS)

    df = df[df['date'] >= cutoff].copy()
    print(f" Filtered to last {DATA_MONTHS} months: {len(df):,} rows")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    # =========================================================================
    # STEP 3: Create Target Horizon (6-block-ahead)
    # =========================================================================
    print("\n" + "=" * 70)
    print(f" CREATING {PREDICTION_HORIZON}-BLOCK-AHEAD TARGET")
    print("=" * 70)

    df = df.sort_values('date').reset_index(drop=True)
    df[TARGET_HORIZON] = df[TARGET].shift(-PREDICTION_HORIZON)

    initial_rows = len(df)
    df = df.dropna(subset=[TARGET_HORIZON]).reset_index(drop=True)
    print(f" Target created, dropped {initial_rows - len(df)} rows with NaN targets")
    print(f"  Remaining: {len(df):,} rows")

    # =========================================================================
    # STEP 4: Seasonal Split
    # =========================================================================
    print("\n" + "=" * 70)
    print(" SEASONAL DATA SPLIT")
    print("=" * 70)

    try:
        train_df, val_df, test_df = create_seasonal_split(df, season, val_ratio=VAL_RATIO)
    except ValueError as e:
        print(f" Error: {e}")
        print(f" Skipping {season} season - insufficient data")
        return None

    # =========================================================================
    # STEP 5: Feature Engineering
    # =========================================================================
    print("\n" + "=" * 70)
    print(" FEATURE ENGINEERING")
    print("=" * 70)

    # Create time and rolling features
    train_df = create_time_features(train_df)
    train_df = create_rolling_features(train_df, target_col=TARGET)

    val_df = create_time_features(val_df)
    val_df = create_rolling_features(val_df, target_col=TARGET)

    test_df = create_time_features(test_df)
    test_df = create_rolling_features(test_df, target_col=TARGET)

    # Encode categorical features (train first to fit encoders)
    exclude_cols = ['date', 'timestamp', TARGET, TARGET_HORIZON]
    train_df, label_encoders = encode_categorical_features(train_df, exclude_cols)
    val_df, _ = encode_categorical_features(val_df, exclude_cols, label_encoders)
    test_df, _ = encode_categorical_features(test_df, exclude_cols, label_encoders)

    print(" Features created")

    # Drop missing data - specify required columns
    required_cols = [TARGET_HORIZON] + get_fsp_forecast_columns(train_df)
    train_df = drop_missing_data(train_df, required_cols)
    val_df = drop_missing_data(val_df, required_cols)
    test_df = drop_missing_data(test_df, required_cols)

    print(f"\nAfter dropping missing data:")
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        print(f" Insufficient data after cleaning for {season} season")
        return None

    # =========================================================================
    # STEP 6: Feature Selection
    # =========================================================================
    print("\n" + "=" * 70)
    print(" FEATURE SELECTION")
    print("=" * 70)

    feature_cols = get_feature_columns(train_df, target_col=TARGET_HORIZON, exclude_patterns=EXCLUDE_PATTERNS)
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    print(f" Selected {len(feature_cols)} features")

    # Prepare datasets
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_HORIZON].values

    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_HORIZON].values

    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_HORIZON].values

    # Check for NaN values and impute if necessary
    if np.isnan(X_train).any():
        print("\n NaN values detected in features, applying imputation...")
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)
        X_test = imputer.transform(X_test)
        print(" NaN values imputed")

    # =========================================================================
    # STEP 7: Scaling
    # =========================================================================
    print("\n" + "=" * 70)
    print(" FEATURE SCALING")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(" Features scaled using StandardScaler")

    # Save scaler
    scaler_path = MODELS_DIR / f'scaler_{season}.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f" Scaler saved: {scaler_path.name}")

    # =========================================================================
    # STEP 8: Train Models
    # =========================================================================
    print("\n" + "=" * 70)
    print(" TRAINING MODELS")
    print("=" * 70)

    models = {}
    all_predictions = {}

    # Ridge Regression
    print("\n1 Training Ridge Regression...")
    ridge = Ridge(alpha=10.0, random_state=RANDOM_STATE)
    ridge.fit(X_train_scaled, y_train)
    models['ridge'] = ridge

    ridge_val_pred = ridge.predict(X_val_scaled)
    ridge_test_pred = ridge.predict(X_test_scaled)
    all_predictions['ridge'] = {'val': ridge_val_pred, 'test': ridge_test_pred}
    print(" Ridge trained")

    # Random Forest
    print("\n2 Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    models['random_forest'] = rf

    rf_val_pred = rf.predict(X_val_scaled)
    rf_test_pred = rf.predict(X_test_scaled)
    all_predictions['random_forest'] = {'val': rf_val_pred, 'test': rf_test_pred}
    print(" Random Forest trained")

    # XGBoost
    if XGB_AVAILABLE:
        print("\n3 Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        xgb_model.fit(X_train_scaled, y_train)
        models['xgboost'] = xgb_model

        xgb_val_pred = xgb_model.predict(X_val_scaled)
        xgb_test_pred = xgb_model.predict(X_test_scaled)
        all_predictions['xgboost'] = {'val': xgb_val_pred, 'test': xgb_test_pred}
        print(" XGBoost trained")

    # LightGBM
    if LGB_AVAILABLE:
        print("\n4 Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train_scaled, y_train)
        models['lightgbm'] = lgb_model

        lgb_val_pred = lgb_model.predict(X_val_scaled)
        lgb_test_pred = lgb_model.predict(X_test_scaled)
        all_predictions['lightgbm'] = {'val': lgb_val_pred, 'test': lgb_test_pred}
        print(" LightGBM trained")

    # Ensemble: Ridge + LightGBM
    if LGB_AVAILABLE:
        print("\n5 Creating Ensemble (Ridge + LightGBM)...")
        ensemble = RidgeLightGBMEnsemble(
            ridge_model=ridge,
            lightgbm_model=lgb_model,
            ridge_weight=ENSEMBLE_RIDGE_WEIGHT,
            scaler=scaler
        )
        models['ensemble_ridge_lgb'] = ensemble

        ens_val_pred = ensemble.predict(X_val)
        ens_test_pred = ensemble.predict(X_test)
        all_predictions['ensemble_ridge_lgb'] = {'val': ens_val_pred, 'test': ens_test_pred}
        print(" Ensemble created")

    # =========================================================================
    # STEP 9: Evaluate on Validation Set
    # =========================================================================
    print("\n" + "=" * 70)
    print(" VALIDATION RESULTS")
    print("=" * 70)

    val_results = []
    for model_name, preds in all_predictions.items():
        metrics = evaluate_model(y_val, preds['val'], model_name)
        val_results.append(metrics)

    val_results_df = pd.DataFrame(val_results).sort_values('MAE')
    print(val_results_df.to_string(index=False))

    best_model = val_results_df.iloc[0]['Model']
    print(f"\n Best model on validation: {best_model}")

    # =========================================================================
    # STEP 10: Evaluate on Test Set
    # =========================================================================
    print("\n" + "=" * 70)
    print(" TEST RESULTS")
    print("=" * 70)

    test_results = []
    for model_name, preds in all_predictions.items():
        metrics = evaluate_model(y_test, preds['test'], model_name)
        test_results.append(metrics)

    test_results_df = pd.DataFrame(test_results).sort_values('MAE')
    print(test_results_df.to_string(index=False))

    # =========================================================================
    # STEP 11: Save Models
    # =========================================================================
    print("\n" + "=" * 70)
    print(" SAVING MODELS")
    print("=" * 70)

    for model_name, model in models.items():
        model_path = MODELS_DIR / f'{model_name}_{season}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f" Saved: {model_path.name}")

    # =========================================================================
    # STEP 12: Save Predictions with FSP Selection
    # =========================================================================
    print("\n" + "=" * 70)
    print(" SAVING PREDICTIONS")
    print("=" * 70)

    for model_name, preds in all_predictions.items():
        # Test predictions
        test_pred = preds['test']
        selected_fsps, scheduled_power, confidence = select_best_fsp_by_prediction(
            test_df, test_pred
        )

        output_path = PREDS_DIR / f'test_predictions_{model_name}_{season}.csv'
        save_model_output(
            test_df, test_pred, selected_fsps, scheduled_power, confidence,
            model_name, output_path, season
        )

    # Save test set
    test_df_copy = test_df.copy()
    test_df_copy['target'] = y_test
    test_df_copy['season'] = season
    test_output_path = PREDS_DIR / f'test_set_{season}.csv'
    test_df_copy.to_csv(test_output_path, index=False)
    print(f"   Saved: test_set_{season}.csv")

    # =========================================================================
    # STEP 13: Save Metadata
    # =========================================================================
    print("\n" + "=" * 70)
    print(" SAVING METADATA")
    print("=" * 70)

    with open(MODELS_DIR / f'feature_columns_{season}.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)

    metadata = {
        'version': MODEL_VERSION,
        'season': season,
        'training_date': datetime.now().isoformat(),
        'data_months': DATA_MONTHS,
        'prediction_horizon': PREDICTION_HORIZON,
        'target_column': TARGET_HORIZON,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'train_period': {
            'start': str(train_df['date'].min()),
            'end': str(train_df['date'].max())
        },
        'val_period': {
            'start': str(val_df['date'].min()),
            'end': str(val_df['date'].max())
        },
        'test_period': {
            'start': str(test_df['date'].min()),
            'end': str(test_df['date'].max())
        },
        'models_trained': list(models.keys()),
        'best_model': best_model,
        'val_metrics': val_results_df.to_dict('records'),
        'test_metrics': test_results_df.to_dict('records'),
        'features_count': len(feature_cols)
    }

    with open(MODELS_DIR / f'model_metadata_{season}.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f" Saved metadata for {season}")

    print("\n" + "=" * 70)
    print(f" {season.upper()} SEASON TRAINING COMPLETE!")
    print("=" * 70)

    return {
        'season': season,
        'best_model': best_model,
        'val_results': val_results_df,
        'test_results': test_results_df
    }


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Train models for all seasons."""

    print("=" * 80)
    print(" SEASONAL MODEL TRAINING - 24 MONTHS DATA")
    print("=" * 80)
    print(f"\nTraining Configuration:")
    print(f"  Data Duration: {DATA_MONTHS} months (2 years)")
    print(f"  Prediction Horizon: {PREDICTION_HORIZON} blocks ahead")
    print(f"  Seasons: {', '.join(SEASONS.keys())}")
    print(f"  Split Strategy: Train on Year 1, Val/Test on Year 2")

    all_results = []

    # Train for each season
    for season in SEASONS.keys():
        print("\n\n")
        result = train_season(season)
        if result:
            all_results.append(result)

    # =========================================================================
    # SUMMARY ACROSS ALL SEASONS
    # =========================================================================
    if all_results:
        print("\n\n" + "=" * 80)
        print(" SUMMARY ACROSS ALL SEASONS")
        print("=" * 80)

        summary_data = []
        for result in all_results:
            season = result['season']
            best_model = result['best_model']

            # Get best model's test metrics
            test_metrics = result['test_results']
            best_test = test_metrics[test_metrics['Model'] == best_model].iloc[0]

            summary_data.append({
                'Season': season.upper(),
                'Best Model': best_model,
                'Test MAE': best_test['MAE'],
                'Test RMSE': best_test['RMSE'],
                'Test R2': best_test['R2'],
                'Test sMAPE': best_test['sMAPE']
            })

        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))

        # Save summary
        summary_path = REPORTS_DIR / 'seasonal_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\n Summary saved: {summary_path}")

        print("\n" + "=" * 80)
        print(" ALL SEASONS TRAINING COMPLETE!")
        print("=" * 80)
        print(f"\nModels saved in: {MODELS_DIR}")
        print(f"Predictions saved in: {PREDS_DIR}")
        print(f"Reports saved in: {REPORTS_DIR}")
    else:
        print("\n No seasons were successfully trained")


if __name__ == '__main__':
    main()
