"""
Standalone script to train and save ensemble models for all 5 target plants.
This script works without Streamlit context.

Target Plants:
- Plant Alpha
- Plant Beta
- Plant Gamma
- Sample Plant
- Plant Delta
"""

import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
import pickle
import json
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import lightgbm as lgb

# Add project to path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from src.config_loader import load_config
from src.data.preprocessing import (
    pivot_fsp_data, calculate_fsp_errors, get_fsp_forecast_columns
)
from src.features.feature_engineering import (
    create_temporal_split, create_rolling_features, create_time_features,
    encode_categorical_features, get_feature_columns
)
from src.models.ensemble_model import RidgeLightGBMEnsemble

# Configuration
config = load_config()
DATA_MONTHS = 18
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
ENSEMBLE_RIDGE_WEIGHT = 0.4
PREDICTION_HORIZON = 6
TARGET = 'actual_power'
TARGET_HORIZON = 'target_horizon'
EXCLUDE_PATTERNS = ['date', 'timestamp', 'index', 'actual_', 'sscode', 'error_', 'schedule_']
RANDOM_STATE = 42

DATA_INTERIM = PROJECT_DIR / config.get('data.interim_dir', 'data/interim')
DATA_PROCESSED = PROJECT_DIR / config.get('data.processed_dir', 'data/processed')
MODEL_SAVESSS_DIR = PROJECT_DIR / "model_savesss"

# Plant names to train models for
TARGET_PLANTS = ['Plant Alpha', 'Plant Beta', 'Plant Gamma', 'Sample Plant', 'Plant Delta']


def calculate_accuracy(y_true, y_pred):
    """Calculate R2 accuracy score."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    # Handle NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return 0.0
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    if len(y_true_clean) == 0:
        return 0.0
    return r2_score(y_true_clean, y_pred_clean)


def get_plant_filename(plant_name: str) -> str:
    """Convert plant name to filename format."""
    # Handle special cases
    plant_mapping = {
        'Plant Alpha': 'plant_alpha_pss',
        'Plant Beta': 'plant_beta_pss',
        'Plant Gamma': 'plant_gamma_pss',
        'Sample Plant': 'sample_pss',
        'Plant Delta': 'plant_delta_pss'
    }
    return plant_mapping.get(plant_name, plant_name.lower().replace(' ', '_'))


def train_and_save_plant_model(plant_name: str, save_dir: Path = None):
    """Train and save ridge-lightgbm ensemble model for a specific plant.

    Parameters:
    -----------
    plant_name : str
        Name of the plant
    save_dir : Path, optional
        Directory to save the model. If None, uses MODEL_SAVESSS_DIR
    """
    print(f"\n{'='*70}")
    print(f"Training model for: {plant_name}")
    print(f"{'='*70}")

    plant_filename = get_plant_filename(plant_name)

    # Load plant data
    parquet_dir = DATA_PROCESSED / 'parquet'
    plant_file = parquet_dir / f"{plant_filename}_dataset.parquet"

    if not plant_file.exists():
        # Try root processed directory
        plant_file = DATA_PROCESSED / f"{plant_filename}_dataset.parquet"

    if not plant_file.exists():
        print(f"[ERROR] Data file not found for {plant_name}: {plant_file}")
        return False, None

        print(f"[OK] Found data file: {plant_file}")

    try:
        # Load data
        print("  Loading data...")
        df_raw = pd.read_parquet(plant_file)
        print(f"  [OK] Loaded {len(df_raw)} rows")

        # Pivot FSP data
        print("  Pivoting FSP data...")
        df_pivoted = pivot_fsp_data(df_raw)
        print(f"  [OK] Pivoted to {len(df_pivoted)} rows")

        # Filter to last 18 months
        print("  Filtering to last 18 months...")
        date_col = 'timestamp' if 'timestamp' in df_pivoted.columns else 'date'
        df_pivoted[date_col] = pd.to_datetime(df_pivoted[date_col])
        max_date = df_pivoted[date_col].max()
        cutoff = max_date - pd.DateOffset(months=DATA_MONTHS)
        df = df_pivoted[df_pivoted[date_col] >= cutoff].copy()
        print(f"  [OK] Filtered to {len(df)} rows")

        # Calculate FSP errors
        print("  Calculating FSP errors...")
        fsp_cols = get_fsp_forecast_columns(df)
        df = calculate_fsp_errors(df, TARGET)

        # Drop missing data
        df_clean = df.dropna(subset=[TARGET]).copy()
        fsp_mask = df_clean[fsp_cols].notna().any(axis=1)
        df_clean = df_clean[fsp_mask].copy()
        print(f"  [OK] Cleaned to {len(df_clean)} rows")

        # Create forward-shifted target
        print("  Creating target horizon...")
        df_clean[TARGET_HORIZON] = df_clean[TARGET].shift(-PREDICTION_HORIZON)
        df_clean = df_clean.dropna(subset=[TARGET_HORIZON])
        print(f"  [OK] Final dataset: {len(df_clean)} rows")

        # Temporal split
        print("  Creating temporal split...")
        train_df, val_df, test_df = create_temporal_split(
            df_clean, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
        )
        print(f"  [OK] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Feature engineering
        print("  Engineering features...")
        train_df = create_time_features(train_df)
        val_df = create_time_features(val_df)
        test_df = create_time_features(test_df)

        train_df = create_rolling_features(train_df, TARGET, [1, 6, 24, 96])
        val_df = create_rolling_features(val_df, TARGET, [1, 6, 24, 96])
        test_df = create_rolling_features(test_df, TARGET, [1, 6, 24, 96])

        train_df, encoders = encode_categorical_features(
            train_df, [TARGET, 'date', 'timestamp', 'sscode']
        )
        val_df, _ = encode_categorical_features(
            val_df, [TARGET, 'date', 'timestamp', 'sscode'], encoders
        )
        test_df, _ = encode_categorical_features(
            test_df, [TARGET, 'date', 'timestamp', 'sscode'], encoders
        )

        feature_cols = get_feature_columns(train_df, TARGET, EXCLUDE_PATTERNS)
        if TARGET_HORIZON in feature_cols:
            feature_cols.remove(TARGET_HORIZON)

        # Drop all-NaN features
        nan_only_cols = [c for c in feature_cols if train_df[c].isna().all()]
        if nan_only_cols:
            feature_cols = [c for c in feature_cols if c not in nan_only_cols]
            train_df = train_df.drop(columns=nan_only_cols)
            val_df = val_df.drop(columns=nan_only_cols)
            test_df = test_df.drop(columns=nan_only_cols)

        print(f"  [OK] Using {len(feature_cols)} features")

        # Prepare X and y
        print("  Preparing training data...")
        X_train = train_df[feature_cols].values
        y_train = train_df[TARGET_HORIZON].values
        X_val = val_df[feature_cols].values
        y_val = val_df[TARGET_HORIZON].values
        X_test = test_df[feature_cols].values
        y_test = test_df[TARGET_HORIZON].values

        # Imputation and scaling
        print("  Applying imputation and scaling...")
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)
        X_test = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Train Ridge
        print("  Training Ridge model...")
        ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        ridge.fit(X_train_scaled, y_train)
        ridge_val = ridge.predict(X_val_scaled)
        ridge_test = ridge.predict(X_test_scaled)
        ridge_val_acc = calculate_accuracy(y_val, ridge_val)
        ridge_test_acc = calculate_accuracy(y_test, ridge_test)
        print(f"    [OK] Ridge - Val R2: {ridge_val_acc:.4f}, Test R2: {ridge_test_acc:.4f}")

        # Train LightGBM
        print("  Training LightGBM model...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.05,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        lgb_val = lgb_model.predict(X_val)
        lgb_test = lgb_model.predict(X_test)
        lgb_val_acc = calculate_accuracy(y_val, lgb_val)
        lgb_test_acc = calculate_accuracy(y_test, lgb_test)
        print(f"    [OK] LightGBM - Val R2: {lgb_val_acc:.4f}, Test R2: {lgb_test_acc:.4f}")

        # Create Ensemble
        print("  Creating ensemble...")
        ensemble = RidgeLightGBMEnsemble(
            ridge_model=ridge,
            lightgbm_model=lgb_model,
            ridge_weight=ENSEMBLE_RIDGE_WEIGHT,
            scaler=scaler,
            imputer=imputer
        )

        ensemble_val = ensemble.predict(X_val)
        ensemble_test = ensemble.predict(X_test)

        # Calculate metrics
        val_accuracy = calculate_accuracy(y_val, ensemble_val)
        test_accuracy = calculate_accuracy(y_test, ensemble_test)
        val_rmse = np.sqrt(np.mean((y_val - ensemble_val) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - ensemble_test) ** 2))

        print(f"    [OK] Ensemble - Val R2: {val_accuracy:.4f}, Test R2: {test_accuracy:.4f}")
        print(f"    [OK] Ensemble - Val RMSE: {val_rmse:.2f} MW, Test RMSE: {test_rmse:.2f} MW")

        # Save model
        print("  Saving model...")
        plant_upper = plant_name.upper().replace(' ', '_')
        if save_dir is None:
            model_dir = MODEL_SAVESSS_DIR / plant_upper / "ridge_lightgbm_ensemble" / "v1"
        else:
            model_dir = save_dir / plant_upper / "ridge_lightgbm_ensemble" / "v1"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save components
        with open(model_dir / "ridge.pkl", 'wb') as f:
            pickle.dump(ridge, f)
        with open(model_dir / "lgbm.pkl", 'wb') as f:
            pickle.dump(lgb_model, f)
        with open(model_dir / "scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        with open(model_dir / "imputer.pkl", 'wb') as f:
            pickle.dump(imputer, f)

        # Save feature columns
        with open(model_dir / "feature_columns.json", 'w') as f:
            json.dump(feature_cols, f)

        # Save config with stats
        config_data = {
            'plant_name': plant_name,
            'model_type': 'RidgeLightGBMEnsemble',
            'ridge_weight': ENSEMBLE_RIDGE_WEIGHT,
            'feature_count': len(feature_cols),
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'val_accuracy': float(val_accuracy),
            'test_accuracy': float(test_accuracy),
            'val_rmse': float(val_rmse),
            'test_rmse': float(test_rmse),
            'data_months': DATA_MONTHS
        }
        with open(model_dir / "config.json", 'w') as f:
            json.dump(config_data, f, indent=2)

        print(f"  [OK] Model saved to: {model_dir}")
        print(f"[SUCCESS] Successfully trained and saved model for {plant_name}")

        return True, config_data

    except Exception as e:
        print(f"[ERROR] Training model for {plant_name}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False, None


def main():
    """Main function to train and save models for all target plants."""
    print("=" * 70)
    print("Training Ridge-LightGBM Ensemble Models for All Target Plants")
    print("=" * 70)
    print(f"Target Plants: {', '.join(TARGET_PLANTS)}")
    print(f"Save Directory: {MODEL_SAVESSS_DIR}")
    print()

    # Create save directory
    MODEL_SAVESSS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, plant in enumerate(TARGET_PLANTS, 1):
        print(f"\n[{idx}/{len(TARGET_PLANTS)}] Processing {plant}...")
        success, stats = train_and_save_plant_model(plant, save_dir=MODEL_SAVESSS_DIR)
        results.append((plant, success, stats))

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    if successful:
        print(f"\n[SUCCESS] Successfully trained {len(successful)} models:")
        for plant, success, stats in successful:
            if stats:
                print(f"   - {plant}: Test R2 = {stats.get('test_accuracy', 0):.4f}, "
                      f"Test RMSE = {stats.get('test_rmse', 0):.2f} MW")

    if failed:
        print(f"\n[FAILED] Failed to train {len(failed)} models:")
        for plant, success, stats in failed:
            print(f"   - {plant}")

    print(f"\nModels saved to: {MODEL_SAVESSS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
