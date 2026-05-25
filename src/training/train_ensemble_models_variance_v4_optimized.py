"""
Variance-Based Ensemble Model Training - V4 (OPTIMIZED)
========================================================

Optimized version with faster training times:
- Uses SGDRegressor as fast alternative to linear SVM
- Samples data for SVM training to reduce O(n^3) complexity
- Uses LinearSVR for faster linear SVM
- Adds progress indicators
- Optimized hyperparameters for speed
"""

from __future__ import annotations

import json
import os
import pickle
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR

# Optional dependencies
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    xgb = None
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    lgb = None
    LGB_AVAILABLE = False

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(__file__).parent.parent

try:
    from src.config_loader import load_config
    _config = load_config()
except Exception:
    _config = {}

DATA_DIR = PROJECT_DIR / _config.get("data.processed_dir", "data/processed")
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models_variance_v4"
PREDS_DIR = OUTPUT_DIR / "predictions_variance_v4"
REPORTS_DIR = OUTPUT_DIR / "reports_variance_v4"

for d in [MODELS_DIR, PREDS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingConfig:
    """Configuration for variance-based training."""
    max_actual_power: float = 600.0
    min_actual_power: float = 0.0
    exclude_fsps: Tuple[str, ...] = ("FA_PROVIDER_E",)

    train_year: int = 2024
    val_test_year: int = 2025
    val_ratio: float = 0.5

    prediction_horizon: int = 6
    random_state: int = 42
    n_cv_folds: int = 5

    feature_importance_threshold: float = 0.01

    # Optimization settings
    svm_sample_size: int = 20000  # Sample size for SVM training
    use_fast_svm: bool = True  # Use LinearSVR instead of SVR

    model_version: str = "4.0.0"


CONFIG = TrainingConfig()

TARGET = "actual_power"

# Variance classification (from analysis)
LOW_VARIANCE_MONTHS = [1, 2, 3, 4, 10, 11]
HIGH_VARIANCE_MONTHS = [5, 6, 7, 8, 9, 12]


# =============================================================================
# ENSEMBLE CLASSES
# =============================================================================

class WeightedEnsemble(BaseEstimator, RegressorMixin):
    """Weighted ensemble of two models."""

    def __init__(self, model1, model2, weight1=0.4, weight2=0.6):
        self.model1 = model1
        self.model2 = model2
        self.weight1 = weight1
        self.weight2 = weight2

    def fit(self, X, y):
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        return self

    def predict(self, X):
        pred1 = self.model1.predict(X)
        pred2 = self.model2.predict(X)
        return self.weight1 * pred1 + self.weight2 * pred2


# =============================================================================
# MODEL FACTORIES
# =============================================================================

def get_low_variance_models(random_state: int = 42, config: TrainingConfig = CONFIG) -> Dict[str, BaseEstimator]:
    """Get models for LOW_VARIANCE periods (simpler, faster)."""
    models = {}

    # Ridge Regression (proven excellent in V3)
    models['ridge'] = Ridge(alpha=10.0, random_state=random_state)

    # LightGBM
    if LGB_AVAILABLE:
        models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )

    # Fast Linear SVM (LinearSVR is much faster than SVR with linear kernel)
    if config.use_fast_svm:
        models['svm_linear'] = LinearSVR(
            C=1.0,
            epsilon=0.1,
            max_iter=1000,
            random_state=random_state
        )
    else:
        models['svm_linear'] = SVR(
            kernel='linear',
            C=1.0,
            epsilon=0.1
        )

    # SGDRegressor (fast alternative to linear SVM)
    models['sgd'] = SGDRegressor(
        loss='epsilon_insensitive',
        epsilon=0.1,
        alpha=0.0001,
        max_iter=1000,
        random_state=random_state
    )

    return models


def get_high_variance_models(random_state: int = 42, config: TrainingConfig = CONFIG) -> Dict[str, BaseEstimator]:
    """Get models for HIGH_VARIANCE periods (more complex, robust)."""
    models = {}

    # Random Forest (handles non-linearity well)
    models['random_forest'] = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        random_state=random_state,
        n_jobs=-1
    )

    # XGBoost
    if XGB_AVAILABLE:
        models['xgboost'] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1
        )

    # Fast Linear SVM
    if config.use_fast_svm:
        models['svm_linear'] = LinearSVR(
            C=10.0,
            epsilon=0.5,
            max_iter=1000,
            random_state=random_state
        )
    else:
        models['svm_linear'] = SVR(
            kernel='linear',
            C=10.0,
            epsilon=0.5
        )

    # SGDRegressor (fast alternative)
    models['sgd'] = SGDRegressor(
        loss='epsilon_insensitive',
        epsilon=0.5,
        alpha=0.0001,
        max_iter=1000,
        random_state=random_state
    )

    return models


def get_stacking_ensemble_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """
    Get stacking ensemble models for HIGH_VARIANCE periods.
    Uses non-tree base models to avoid overfitting.
    """
    models = {}

    # Base models (non-tree to avoid overfitting)
    base_models = [
        ('ridge', Ridge(alpha=10.0, random_state=random_state)),
        ('sgd', SGDRegressor(loss='epsilon_insensitive', epsilon=0.1, alpha=0.0001,
                          max_iter=1000, random_state=random_state)),
        ('linear_svr', LinearSVR(C=10.0, epsilon=0.1, max_iter=1000, random_state=random_state)),
    ]

    # Stacking with Linear meta-learner
    models['stacking_linear'] = StackingRegressor(
        estimators=base_models,
        final_estimator=LinearRegression(),
        cv=3,
        n_jobs=-1
    )

    # Stacking with Ridge meta-learner
    models['stacking_ridge'] = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=1.0),
        cv=3,
        n_jobs=-1
    )

    return models


# =============================================================================
# UTILITIES
# =============================================================================

def _print_section(title: str, char: str = "=", width: int = 80) -> None:
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def _print_subsection(title: str, char: str = "-", width: int = 70) -> None:
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # sMAPE (symmetric Mean Absolute Percentage Error)
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'smape': smape,
        'mape': mape,
    }


# =============================================================================
# MAIN TRAINING CLASS
# =============================================================================

class VarianceModelTrainer:
    """Trains models for variance-based splitting."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.scalers: Dict[str, StandardScaler] = {}
        self.imputers: Dict[str, SimpleImputer] = {}
        self.feature_columns: Dict[str, List[str]] = {}
        self.models: Dict[str, Dict[str, BaseEstimator]] = {}
        self.metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess data."""
        _print_section("LOADING DATA")

        df = pd.read_parquet(DATA_DIR / "sample_pss_dataset.parquet")

        # Filter valid data
        df = df.dropna(subset=['actual_windspeed', 'actual_power'])

        # Ensure datetime column exists
        if 'date' not in df.columns and 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'])
        elif 'date' not in df.columns:
            df['date'] = pd.to_datetime(df.index)
        else:
            df['date'] = pd.to_datetime(df['date'])

        # Extract temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['hour'] = df['date'].dt.hour

        print(f"Loaded {len(df)} valid records")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer variance-specific features."""
        _print_section("FEATURE ENGINEERING")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from features.variance_features import VarianceFeatureEngineer, VarianceFeatureConfig

        config = VarianceFeatureConfig()
        engineer = VarianceFeatureEngineer(config)
        df = engineer.engineer_features(df)

        return df

    def create_variance_splits(self, df: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Create train/val/test splits for each variance type."""
        _print_section("CREATING VARIANCE SPLITS")

        splits = {}

        for var_type, months in [('low_variance', LOW_VARIANCE_MONTHS),
                                  ('high_variance', HIGH_VARIANCE_MONTHS)]:
            print(f"\n{var_type.upper()} Months: {months}")

            # Train: All data for variance type in train_year
            train_mask = (df['year'] == self.config.train_year) & (df['month'].isin(months))
            train_df = df[train_mask].copy()

            # Val/Test: Split data for variance type in val_test_year
            val_test_mask = (df['year'] == self.config.val_test_year) & (df['month'].isin(months))
            val_test_df = df[val_test_mask].copy().sort_values('date')

            # Split val_test into val and test
            val_size = int(len(val_test_df) * self.config.val_ratio)
            val_df = val_test_df.iloc[:val_size].copy()
            test_df = val_test_df.iloc[val_size:].copy()

            splits[var_type] = {
                'train': train_df,
                'val': val_df,
                'test': test_df,
            }

            print(f"  Train: {len(train_df)} records")
            print(f"  Val:   {len(val_df)} records")
            print(f"  Test:  {len(test_df)} records")

        return splits

    def prepare_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        var_type: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Prepare data for training."""
        _print_subsection(f"Preparing {var_type.upper()} Data")

        # Get feature columns (exclude target, date, and non-numeric columns)
        exclude_cols = [TARGET, 'date', 'timestamp']

        # Select only numeric columns
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        # Store feature columns
        self.feature_columns[var_type] = feature_cols

        print(f"  Features: {len(feature_cols)}")

        # Prepare X and y
        X_train = train_df[feature_cols].values
        y_train = train_df[TARGET].values

        X_val = val_df[feature_cols].values
        y_val = val_df[TARGET].values

        X_test = test_df[feature_cols].values
        y_test = test_df[TARGET].values

        # Replace infinity values with NaN
        X_train = np.where(np.isinf(X_train), np.nan, X_train)
        X_val = np.where(np.isinf(X_val), np.nan, X_val)
        X_test = np.where(np.isinf(X_test), np.nan, X_test)

        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)
        X_test = imputer.transform(X_test)

        self.imputers[var_type] = imputer

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        self.scalers[var_type] = scaler

        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

    def sample_data_for_svm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample data for SVM training to reduce training time."""
        if len(X) <= sample_size:
            return X, y

        # Random sampling
        indices = np.random.choice(len(X), sample_size, replace=False)
        return X[indices], y[indices]

    def train_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        var_type: str
    ) -> Dict[str, BaseEstimator]:
        """Train models for a variance type."""
        _print_section(f"TRAINING {var_type.upper()} MODELS")

        models = {}

        if var_type == 'low_variance':
            # Get low variance models
            base_models = get_low_variance_models(self.config.random_state, self.config)

            # Train base models
            for name, model in base_models.items():
                print(f"\nTraining {name}...")
                start_time = time.time()

                # Sample data for SVM models
                if 'svm' in name or 'sgd' in name:
                    X_train_sampled, y_train_sampled = self.sample_data_for_svm(
                        X_train, y_train, self.config.svm_sample_size
                    )
                    print(f"  Using {len(X_train_sampled)} samples for training")
                    model.fit(X_train_sampled, y_train_sampled)
                else:
                    model.fit(X_train, y_train)

                elapsed_time = time.time() - start_time
                models[name] = model

                # Evaluate on validation set
                y_pred = model.predict(X_val)
                metrics = calculate_metrics(y_val, y_pred)
                print(f"  Val MAE: {metrics['mae']:.4f}")
                print(f"  Time: {elapsed_time:.2f}s")

        elif var_type == 'high_variance':
            # Get high variance models
            base_models = get_high_variance_models(self.config.random_state, self.config)

            # Train base models
            for name, model in base_models.items():
                print(f"\nTraining {name}...")
                start_time = time.time()

                # Sample data for SVM models
                if 'svm' in name or 'sgd' in name:
                    X_train_sampled, y_train_sampled = self.sample_data_for_svm(
                        X_train, y_train, self.config.svm_sample_size
                    )
                    print(f"  Using {len(X_train_sampled)} samples for training")
                    model.fit(X_train_sampled, y_train_sampled)
                else:
                    model.fit(X_train, y_train)

                elapsed_time = time.time() - start_time
                models[name] = model

                # Evaluate on validation set
                y_pred = model.predict(X_val)
                metrics = calculate_metrics(y_val, y_pred)
                print(f"  Val MAE: {metrics['mae']:.4f}")
                print(f"  Time: {elapsed_time:.2f}s")

            # Train stacking ensembles (research-grade approach for high variance)
            print("\nTraining Stacking Ensembles...")
            stacking_models = get_stacking_ensemble_models(self.config.random_state)

            for name, model in stacking_models.items():
                print(f"\nTraining {name}...")
                start_time = time.time()

                # Sample data for stacking
                X_train_sampled, y_train_sampled = self.sample_data_for_svm(
                    X_train, y_train, self.config.svm_sample_size
                )
                print(f"  Using {len(X_train_sampled)} samples for training")
                model.fit(X_train_sampled, y_train_sampled)

                elapsed_time = time.time() - start_time
                models[name] = model

                # Evaluate on validation set
                y_pred = model.predict(X_val)
                metrics = calculate_metrics(y_val, y_pred)
                print(f"  Val MAE: {metrics['mae']:.4f}")
                print(f"  Time: {elapsed_time:.2f}s")

        self.models[var_type] = models
        return models

    def create_weighted_ensemble(
        self,
        models: Dict[str, BaseEstimator],
        X_val: np.ndarray,
        y_val: np.ndarray,
        var_type: str
    ) -> BaseEstimator:
        """Create weighted ensemble based on validation performance."""
        _print_subsection(f"Creating Weighted Ensemble for {var_type.upper()}")

        # Calculate validation MAE for each model
        val_maes = {}
        for name, model in models.items():
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            val_maes[name] = mae

        # Sort by MAE (lower is better)
        sorted_models = sorted(val_maes.items(), key=lambda x: x[1])

        # Select top 2 models
        if len(sorted_models) >= 2:
            model1_name, mae1 = sorted_models[0]
            model2_name, mae2 = sorted_models[1]

            # Calculate weights (inverse of MAE)
            total_inv_mae = (1/mae1) + (1/mae2)
            weight1 = (1/mae1) / total_inv_mae
            weight2 = (1/mae2) / total_inv_mae

            print(f"  Top models: {model1_name} (MAE: {mae1:.4f}), {model2_name} (MAE: {mae2:.4f})")
            print(f"  Weights: {weight1:.2f}, {weight2:.2f}")

            # Create weighted ensemble
            ensemble = WeightedEnsemble(
                models[model1_name],
                models[model2_name],
                weight1=weight1,
                weight2=weight2
            )

            # Fit ensemble
            ensemble.fit(X_val, y_val)

            return ensemble
        else:
            # Return best single model
            best_model_name = sorted_models[0][0]
            print(f"  Using best single model: {best_model_name}")
            return models[best_model_name]

    def evaluate_models(
        self,
        models: Dict[str, BaseEstimator],
        X_test: np.ndarray,
        y_test: np.ndarray,
        var_type: str
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate models on test set."""
        _print_section(f"EVALUATING {var_type.upper()} MODELS")

        metrics = {}

        for name, model in models.items():
            y_pred = model.predict(X_test)
            model_metrics = calculate_metrics(y_test, y_pred)
            metrics[name] = model_metrics

            print(f"\n{name}:")
            print(f"  MAE:  {model_metrics['mae']:.4f}")
            print(f"  RMSE: {model_metrics['rmse']:.4f}")
            print(f"  R2:   {model_metrics['r2']:.4f}")
            print(f"  sMAPE: {model_metrics['smape']:.2f}%")

        self.metrics[var_type] = metrics
        return metrics

    def save_models(self, var_type: str):
        """Save trained models and artifacts."""
        _print_subsection(f"Saving {var_type.upper()} Models")

        var_dir = MODELS_DIR / var_type
        var_dir.mkdir(parents=True, exist_ok=True)

        # Save models
        for name, model in self.models[var_type].items():
            model_path = var_dir / f"{name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"  Saved: {model_path}")

        # Save scaler
        scaler_path = var_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers[var_type], f)
        print(f"  Saved: {scaler_path}")

        # Save imputer
        imputer_path = var_dir / "imputer.pkl"
        with open(imputer_path, 'wb') as f:
            pickle.dump(self.imputers[var_type], f)
        print(f"  Saved: {imputer_path}")

        # Save feature columns
        feature_cols_path = var_dir / "feature_columns.json"
        with open(feature_cols_path, 'w') as f:
            json.dump(self.feature_columns[var_type], f, indent=2)
        print(f"  Saved: {feature_cols_path}")

        # Save metrics
        metrics_path = var_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics[var_type], f, indent=2)
        print(f"  Saved: {metrics_path}")

    def generate_predictions(
        self,
        models: Dict[str, BaseEstimator],
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        var_type: str
    ):
        """Generate and save predictions."""
        _print_subsection(f"Generating {var_type.upper()} Predictions")

        pred_dir = PREDS_DIR / var_type
        pred_dir.mkdir(parents=True, exist_ok=True)

        # Validation predictions
        val_preds = val_df[['date', TARGET]].copy()
        for name, model in models.items():
            val_preds[f'pred_{name}'] = model.predict(X_val)

        val_preds_path = pred_dir / "val_predictions.csv"
        val_preds.to_csv(val_preds_path, index=False)
        print(f"  Saved: {val_preds_path}")

        # Test predictions
        test_preds = test_df[['date', TARGET]].copy()
        for name, model in models.items():
            test_preds[f'pred_{name}'] = model.predict(X_test)

        test_preds_path = pred_dir / "test_predictions.csv"
        test_preds.to_csv(test_preds_path, index=False)
        print(f"  Saved: {test_preds_path}")

    def run(self):
        """Run complete training pipeline."""
        _print_section("VARIANCE-BASED MODEL TRAINING - V4 (OPTIMIZED)")
        print(f"Version: {self.config.model_version}")
        print(f"Train Year: {self.config.train_year}")
        print(f"Val/Test Year: {self.config.val_test_year}")
        print(f"LOW_VARIANCE Months: {LOW_VARIANCE_MONTHS}")
        print(f"HIGH_VARIANCE Months: {HIGH_VARIANCE_MONTHS}")
        print(f"SVM Sample Size: {self.config.svm_sample_size}")
        print(f"Use Fast SVM: {self.config.use_fast_svm}")

        # Load data
        df = self.load_data()

        # Engineer features
        df = self.engineer_features(df)

        # Create variance splits
        splits = self.create_variance_splits(df)

        # Train and evaluate for each variance type
        for var_type in ['low_variance', 'high_variance']:
            _print_section(f"PROCESSING {var_type.upper()}")

            # Prepare data
            X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = self.prepare_data(
                splits[var_type]['train'],
                splits[var_type]['val'],
                splits[var_type]['test'],
                var_type
            )

            # Train models
            models = self.train_models(X_train, y_train, X_val, y_val, var_type)

            # Create weighted ensemble
            weighted_ensemble = self.create_weighted_ensemble(models, X_val, y_val, var_type)
            models['weighted_ensemble'] = weighted_ensemble

            # Evaluate models
            metrics = self.evaluate_models(models, X_test, y_test, var_type)

            # Save models
            self.save_models(var_type)

            # Generate predictions
            self.generate_predictions(
                models,
                X_val, y_val,
                X_test, y_test,
                splits[var_type]['val'],
                splits[var_type]['test'],
                var_type
            )

        # Generate summary report
        self.generate_summary_report()

        _print_section("TRAINING COMPLETE")

    def generate_summary_report(self):
        """Generate summary report."""
        _print_section("GENERATING SUMMARY REPORT")

        summary = {
            'version': self.config.model_version,
            'train_year': self.config.train_year,
            'val_test_year': self.config.val_test_year,
            'low_variance_months': LOW_VARIANCE_MONTHS,
            'high_variance_months': HIGH_VARIANCE_MONTHS,
            'low_variance_metrics': self.metrics.get('low_variance', {}),
            'high_variance_metrics': self.metrics.get('high_variance', {}),
        }

        # Save summary
        summary_path = REPORTS_DIR / "variance_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {summary_path}")

        # Create CSV summary
        summary_data = []
        for var_type in ['low_variance', 'high_variance']:
            for model_name, metrics in self.metrics.get(var_type, {}).items():
                summary_data.append({
                    'variance_type': var_type,
                    'model': model_name,
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'r2': metrics['r2'],
                    'smape': metrics['smape'],
                    'mape': metrics['mape'],
                })

        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = REPORTS_DIR / "variance_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"  Saved: {summary_csv_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("=" * 80)


def main():
    """Main function."""
    config = TrainingConfig()
    trainer = VarianceModelTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
