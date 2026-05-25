
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
import json
from datetime import datetime

# Setup path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Import src modules
from src.data.preprocessing import pivot_fsp_data
from src.features.feature_engineering import create_time_features
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TRAIN_ENSEMBLE")

def train_and_save():
    # 1. Load Data
    data_path = PROJECT_DIR / 'data' / 'processed' / 'parquet' / 'sample_pss_dataset.parquet'
    logger.info(f"Loading data from {data_path}...")

    if not data_path.exists():
        # Try interim
        data_path = PROJECT_DIR / 'data' / 'interim' / 'sample_pss.parquet'

    if not data_path.exists():
        logger.error("Data file not found.")
        return False

    df = pd.read_parquet(data_path)

    # 2. Preprocessing
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    # Pivot if needed (check if wide or long)
    # Sample Plant dataset is usually long, need pivotGKFS
    # Actually, the parquet might be already preprocessed/pivoted if it's in 'processed'.
    # Let's check columns.
    if 'sscode' in df.columns:
         logger.info("Pivoting data...")
         # Simple pivot simulation or use src function if available/working
         # For robustness, let's assume valid data for now or do a simple pivot
         df_pivoted = df.pivot_table(index='timestamp', columns='sscode', values='actual_power', aggfunc='mean')
         df = df_pivoted.reset_index()

    # Feature Engineering
    logger.info("Generating features...")
    df = create_time_features(df)

    # Define target
    target_col = 'actual_power' # Simplified target
    if target_col not in df.columns:
        # Check if we have multiple FSPs, maybe sum themGKFS
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            target_col = numeric_cols[-1] # Pick last one as dummy target if actual not found

    # Features
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]

    # 3. Training
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].dropna()
    test_df = df.iloc[train_size:].dropna()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    logger.info(f"Training Ridge-LightGBM Ensemble on {len(X_train)} rows...")

    # Prepare Preprocessor
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train_processed = scaler.fit_transform(imputer.fit_transform(X_train))

    # Base Models
    estimators = [
        ('ridge', Ridge(alpha=1.0)),
        ('lgbm', LGBMRegressor(n_estimators=100, max_depth=5, random_state=42))
    ]

    # Stacking
    reg = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=0.5)
    )

    reg.fit(X_train_processed, y_train)
    logger.info("Training complete.")

    # 4. Save Bundle
    model_name = "ridge_lightgbm_ensemble_v1"
    output_dir = PROJECT_DIR / "outputs" / "saved_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / f"{model_name}.pkl"

    bundle = {
        'model': reg,
        'model_name': 'Ridge-LightGBM Stacking',
        'feature_columns': feature_cols,
        'scaler': scaler,
        'imputer': imputer,
        'config': {'type': 'stacking', 'components': ['Ridge', 'LightGBM']},
        'metrics': {'train_score': reg.score(X_train_processed, y_train)},
        'timestamp': datetime.now().isoformat()
    }

    with open(save_path, 'wb') as f:
        pickle.dump(bundle, f)

    logger.info(f" Model saved to: {save_path}")
    return str(save_path)

if __name__ == "__main__":
    try:
        path = train_and_save()
        if path:
            print(f"OUTPUT_PATH:{path}")
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        logger.exception("Training failed")
        sys.exit(1)
