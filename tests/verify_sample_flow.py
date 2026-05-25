
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Mock streamlit session state if needed, though this is a standalone script
import streamlit as st
class MockSessionState(dict):
    def __getattr__(self, key):
        return self.get(key)
    def __setattr__(self, key, value):
        self[key] = value

if not hasattr(st, 'session_state'):
    st.session_state = MockSessionState()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SAMPLE_TEST")

def verify_sample_flow():
    dataset_path = PROJECT_DIR / 'data' / 'processed' / 'parquet' / 'sample_pss_dataset.parquet'
    logger.info(f"1. Locating Sample Plant Data at {dataset_path}...")

    if not dataset_path.exists():
        logger.error(f"Data file not found: {dataset_path}")
        # Try interim if not in processed
        interim_path = PROJECT_DIR / 'data' / 'interim' / 'sample_pss.parquet'
        if interim_path.exists():
            dataset_path = interim_path
            logger.info(f"Found in interim: {dataset_path}")
        else:
            return False

    try:
        df = pd.read_parquet(dataset_path)
    except Exception as e:
        logger.error(f"Failed to read parquet: {e}")
        return False

    logger.info(f"Loaded {len(df)} rows. Columns: {list(df.columns[:5])}...")

    logger.info("2. Checking for critical columns...")
    # Typically 'timestamp' or 'date', and target columns
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    target_col = 'target_horizon'
    # Check if target exists or if we need to derive it (e.g. from actual_power)
    if target_col not in df.columns:
        # Check for 'actual_power'
        if 'actual_power' in df.columns:
            logger.info(f"Using 'actual_power' as target for verification.")
            target_col = 'actual_power'
        else:
            # Check for any float column to use as dummy target
            float_cols = df.select_dtypes(include=[np.number]).columns
            if len(float_cols) > 0:
                target_col = float_cols[-1]
                logger.warning(f"Target column not found. Using {target_col} as dummy target for flow verification.")
            else:
                logger.error("No numeric columns found for target.")
                return False

    # Feature selection (simple heuristic)
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col][:10]
    if not feature_cols:
        logger.error("No feature columns found.")
        return False

    # Drop NaNs
    df_clean = df.dropna(subset=feature_cols + [target_col])
    if len(df_clean) == 0:
        logger.error("Data is empty after dropping NaNs.")
        return False

    logger.info(f"Data after cleaning: {len(df_clean)} rows. Features: {len(feature_cols)}")

    logger.info("3. Split Data...")
    train_size = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:train_size]
    test_df = df_clean.iloc[train_size:]

    if len(train_df) < 10 or len(test_df) < 10:
        logger.error("Insufficient data for split.")
        return False

    logger.info("4. Training (Ridge Scikit-Learn)...")
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error, r2_score

    # Simple pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ])

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    pipeline.fit(X_train, y_train)
    logger.info("Model trained successfully.")

    logger.info("5. Prediction & Evaluation...")
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    logger.info(f" Success! Feature flow verified. MAE: {mae:.4f}, R2: {r2:.4f}")

    # 6. Verify Visualization Data Structure compatibility
    logger.info("6. Verifying Visualization DataFrame Structure...")
    # Simulate create_prediction_dataframe logic
    output = pd.DataFrame()
    if 'timestamp' in test_df.columns:
        output['timestamp'] = test_df['timestamp']
    output['actual_power'] = y_test
    output['ml_predicted_power'] = preds
    # Add dummy 'ml_scheduled_power'
    output['ml_scheduled_power'] = preds * 0.95

    # Check simple error calc
    output['error'] = np.abs(output['actual_power'] - output['ml_predicted_power'])
    logger.info(f"Visualization DataFrame created with {len(output)} rows.")

    return True

if __name__ == "__main__":
    try:
        sys.exit(0 if verify_sample_flow() else 1)
    except Exception as e:
        logger.exception("Verification failed with exception")
        sys.exit(1)
