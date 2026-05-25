"""
FSP Selection Model Training Script - Seasonal Edition V2
==========================================================

Production-grade ML pipeline for Sample Plant Wind Plant FSP selection.
Implements comprehensive data preprocessing, seasonal training, and advanced
ensemble methods for automated power scheduling.

Key Features:
- Data pivoting from multi-row FSP format to single-row format
- Outlier cleaning and FA_PROVIDER_E exclusion
- Proper year-over-year seasonal splits (2024 train 2025 val/test)
- Feature selection with permutation importance
- Hyperparameter tuning with TimeSeriesSplit CV
- Model stacking with Ridge meta-learner
- Prediction intervals via quantile regression
- Comprehensive evaluation metrics

Seasons:
- Winter: December, January, February
- Spring: March, April, May
- Summer: June, July, August
- Fall: September, October, November

Maintainer: Project Team
Date: January 2026
Version: 2.0.0
"""

import os
import sys
import json
import pickle
import warnings
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from sklearn.base import BaseEstimator, RegressorMixin, clone

# Optional imports with availability flags
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print(" XGBoost not available - will skip XGB models")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print(" LightGBM not available - will skip LGB models")

try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print(" Optuna not available - will use default hyperparameters")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(x, **kwargs): return x

# Suppress warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', message='.*categorical features are.*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.config_loader import load_config
from src.features.feature_engineering import (
    create_rolling_features, create_time_features,
    encode_categorical_features, get_feature_columns
)

config = load_config()

# Directories
DATA_DIR = PROJECT_DIR / config.get('data.processed_dir', 'data/processed')
RAW_DIR = PROJECT_DIR / config.get('data.raw_dir', 'data/raw')
INTERIM_DIR = PROJECT_DIR / config.get('data.interim_dir', 'data/interim')
OUTPUT_DIR = PROJECT_DIR / 'outputs'
MODELS_DIR = OUTPUT_DIR / 'models_seasonal_v2'
PREDS_DIR = OUTPUT_DIR / 'predictions_seasonal_v2'
PLOTS_DIR = OUTPUT_DIR / 'plots_seasonal_v2'
REPORTS_DIR = OUTPUT_DIR / 'reports_seasonal_v2'

for d in [MODELS_DIR, PREDS_DIR, PLOTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration with all hyperparameters."""
    # Data configuration
    max_actual_power: float = 600.0  # MW - cap outliers
    min_actual_power: float = 0.0    # MW - floor
    exclude_fsps: List[str] = None   # FSPs to exclude

    # Seasonal split configuration
    train_year: int = 2024           # Year for training
    val_test_year: int = 2025        # Year for validation/test
    val_ratio: float = 0.5           # Split ratio for val/test in 2025

    # Model configuration
    prediction_horizon: int = 6      # Blocks ahead (6 = 1.5 hours)
    random_state: int = 42
    n_cv_folds: int = 3              # TimeSeriesSplit folds
    n_optuna_trials: int = 5         # Hyperparameter tuning trials (reduced for testing)

    # Feature selection
    feature_importance_threshold: float = 0.01  # Min importance to keep
    use_rfe: bool = False            # Use Recursive Feature Elimination
    rfe_n_features: int = 30         # Features to keep if using RFE

    # Ensemble configuration
    ridge_weight: float = 0.3        # Weight for Ridge in simple ensemble
    use_stacking: bool = True        # Use stacking ensemble

    # Prediction intervals
    quantiles: List[float] = None    # Quantiles for prediction intervals

    # Output configuration
    model_version: str = "2.0.0"

    def __post_init__(self):
        if self.exclude_fsps is None:
            self.exclude_fsps = ['FA_PROVIDER_E']
        if self.quantiles is None:
            self.quantiles = [0.05, 0.5, 0.95]  # 90% prediction interval


# Initialize config
CONFIG = TrainingConfig()

# FSP Providers (excluding those in exclude list)
FSP_PROVIDERS = [
    fsp for fsp in config.get('fsp_providers', [
        'FA_PROVIDER_A', 'FA_PROVIDER_B', 'FA_PROVIDER_C', 'FA_PROVIDER_D'
    ]) if fsp not in CONFIG.exclude_fsps
]

# Season definitions
SEASONS = {
    'winter': [12, 1, 2],
    'spring': [3, 4, 5],
    'summer': [6, 7, 8],
    'fall': [9, 10, 11]
}

TARGET = 'actual_power'
TARGET_HORIZON = 'target_horizon'
EXCLUDE_PATTERNS = ['date', 'timestamp', 'index', 'actual_', 'sscode', 'error_', 'schedule_', 'target']

np.random.seed(CONFIG.random_state)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_git_commit_hash() -> str:
    """Get current git commit hash for versioning."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, cwd=PROJECT_DIR
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def print_section(title: str, char: str = "=", width: int = 80):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_subsection(title: str, char: str = "-", width: int = 70):
    """Print a formatted subsection header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def pivot_fsp_data(
    df: pd.DataFrame,
    fsp_col: str = 'forecast_facode',
    exclude_fsps: List[str] = None
) -> pd.DataFrame:
    """
    Pivot FSP data from multiple rows per time block to single row format.

    Input: DataFrame with 1 row per (date, block, FSP)
    Output: DataFrame with 1 row per (date, block), separate columns for each FSP

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with multiple FSP rows per time block
    fsp_col : str
        Column containing FSP provider names
    exclude_fsps : List[str]
        FSP providers to exclude from pivot

    Returns:
    --------
    pd.DataFrame : Pivoted dataframe with 1 row per time block
    """
    exclude_fsps = exclude_fsps or []

    # Filter out excluded FSPs
    df_filtered = df[~df[fsp_col].isin(exclude_fsps)].copy()

    # Get unique FSPs
    unique_fsps = sorted(df_filtered[fsp_col].unique())
    print(f"  FSP providers after filtering: {unique_fsps}")
    print(f"  Excluded FSPs: {exclude_fsps}")

    # Identify key columns for grouping
    group_cols = ['date', 'block']
    if 'timestamp' in df_filtered.columns:
        group_cols.append('timestamp')
    if 'sscode' in df_filtered.columns:
        group_cols.append('sscode')

    # Common columns (same for all FSPs in a block)
    common_cols = [
        'actual_power', 'actual_avc', 'actual_windspeed', 'actual_ghirr',
        'actual_flowrate', 'actual_time', 'actual_source',
        'schedule_power', 'schedule_avc', 'schedule_windspeed', 'schedule_ghirr',
        'schedule_flowrate', 'schedule_time', 'schedule_source', 'schedule_revno'
    ]
    existing_common = [c for c in common_cols if c in df_filtered.columns]

    # Create base with common columns
    base_df = df_filtered.groupby(group_cols, as_index=False)[existing_common].first()

    # FSP-specific columns to pivot
    fsp_cols_to_pivot = [
        'forecast_power', 'forecast_avc', 'forecast_windspeed',
        'forecast_ghirr', 'forecast_flowrate', 'forecast_revno'
    ]

    for col in fsp_cols_to_pivot:
        if col in df_filtered.columns:
            # Pivot this column by FSP
            pivot_data = df_filtered.pivot_table(
                index=group_cols,
                columns=fsp_col,
                values=col,
                aggfunc='first'
            )

            # Rename columns
            rename_dict = {fsp: f'{col}_{fsp.lower()}' for fsp in pivot_data.columns}
            pivot_data = pivot_data.rename(columns=rename_dict)
            pivot_data = pivot_data.reset_index()

            # Merge with base
            cols_to_merge = [c for c in pivot_data.columns if c not in group_cols]
            if cols_to_merge:
                base_df = base_df.merge(
                    pivot_data[group_cols + cols_to_merge],
                    on=group_cols,
                    how='left'
                )

    # Sort by time
    base_df = base_df.sort_values(group_cols).reset_index(drop=True)

    rows_before = len(df)
    rows_after = len(base_df)
    print(f" Pivoted data: {rows_before:,} rows  {rows_after:,} rows")

    return base_df


def clean_data(df: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    """
    Clean data by handling outliers and missing values.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    config : TrainingConfig
        Training configuration

    Returns:
    --------
    pd.DataFrame : Cleaned dataframe
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)

    # Cap outliers in actual_power
    outliers_high = (df_clean['actual_power'] > config.max_actual_power).sum()
    outliers_low = (df_clean['actual_power'] < config.min_actual_power).sum()

    df_clean['actual_power'] = df_clean['actual_power'].clip(
        lower=config.min_actual_power,
        upper=config.max_actual_power
    )

    print(f"  Capped {outliers_high} high outliers (>{config.max_actual_power} MW)")
    print(f"  Capped {outliers_low} low outliers (<{config.min_actual_power} MW)")

    # Drop rows with missing actual_power
    df_clean = df_clean.dropna(subset=['actual_power'])

    # Drop rows where ALL FSP forecasts are missing
    fsp_cols = [c for c in df_clean.columns if 'forecast_power_' in c]
    if fsp_cols:
        all_fsp_missing = df_clean[fsp_cols].isna().all(axis=1)
        df_clean = df_clean[~all_fsp_missing]

    final_rows = len(df_clean)
    print(f"  Rows after cleaning: {initial_rows:,}  {final_rows:,} ({initial_rows - final_rows:,} dropped)")

    return df_clean


def get_fsp_forecast_columns(df: pd.DataFrame) -> List[str]:
    """Get list of FSP forecast power columns."""
    return [f'forecast_power_{fsp.lower()}' for fsp in FSP_PROVIDERS
            if f'forecast_power_{fsp.lower()}' in df.columns]


# =============================================================================
# SEASONAL SPLIT FUNCTIONS
# =============================================================================

def assign_season(date) -> Optional[str]:
    """Assign season based on month."""
    month = date.month
    for season_name, months in SEASONS.items():
        if month in months:
            return season_name
    return None


def create_seasonal_split(
    df: pd.DataFrame,
    season: str,
    train_year: int,
    val_test_year: int,
    val_ratio: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create proper year-over-year seasonal split.

    Strategy:
    - Train: Season X in train_year (e.g., 2024)
    - Val: First half of Season X in val_test_year (e.g., Jan 2025)
    - Test: Second half of Season X in val_test_year (e.g., Feb 2025)

    Parameters:
    -----------
    df : pd.DataFrame
        Full dataframe with 'date' column
    season : str
        Season name ('winter', 'spring', 'summer', 'fall')
    train_year : int
        Year for training data
    val_test_year : int
        Year for validation/test data
    val_ratio : float
        Ratio to split val_test_year data between val and test

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        train_df, val_df, test_df
    """
    if 'date' not in df.columns:
        raise ValueError("DataFrame must have 'date' column")

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['season'] = df['date'].apply(assign_season)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # Filter for the specified season
    season_df = df[df['season'] == season].copy()

    if len(season_df) == 0:
        raise ValueError(f"No data found for season: {season}")

    # Get season months
    season_months = SEASONS[season]

    # Handle winter specially (crosses year boundary)
    if season == 'winter':
        # Winter train: Dec of (train_year-1), Jan-Feb of train_year
        train_mask = (
            ((df['year'] == train_year - 1) & (df['month'] == 12)) |
            ((df['year'] == train_year) & (df['month'].isin([1, 2])))
        )
        # Winter val/test: Dec of (val_test_year-1), Jan-Feb of val_test_year
        val_test_mask = (
            ((df['year'] == val_test_year - 1) & (df['month'] == 12)) |
            ((df['year'] == val_test_year) & (df['month'].isin([1, 2])))
        )
    else:
        # Other seasons: simple year filter
        train_mask = (df['year'] == train_year) & (df['month'].isin(season_months))
        val_test_mask = (df['year'] == val_test_year) & (df['month'].isin(season_months))

    train_df = df[train_mask].copy()
    val_test_df = df[val_test_mask].copy()

    if len(train_df) == 0:
        raise ValueError(f"No training data for {season} in {train_year}")

    if len(val_test_df) == 0:
        raise ValueError(f"No val/test data for {season} in {val_test_year}")

    # Split val_test chronologically
    val_test_sorted = val_test_df.sort_values('date').reset_index(drop=True)
    val_size = int(len(val_test_sorted) * val_ratio)

    val_df = val_test_sorted.iloc[:val_size].copy()
    test_df = val_test_sorted.iloc[val_size:].copy()

    # Drop helper columns
    for split_df in [train_df, val_df, test_df]:
        split_df.drop(columns=['season', 'year', 'month'], inplace=True, errors='ignore')

    print(f"\n Seasonal Split for {season.upper()}:")
    print(f"  Train ({train_year}): {len(train_df):,} samples")
    print(f"    Period: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
    print(f"  Val ({val_test_year}): {len(val_df):,} samples")
    print(f"    Period: {val_df['date'].min().date()} to {val_df['date'].max().date()}")
    print(f"  Test ({val_test_year}): {len(test_df):,} samples")
    print(f"    Period: {test_df['date'].min().date()} to {test_df['date'].max().date()}")

    return train_df, val_df, test_df


# =============================================================================
# EVALUATION METRICS
# =============================================================================

@dataclass
class ModelMetrics:
    """Comprehensive model evaluation metrics."""
    model_name: str
    mae: float
    rmse: float
    r2: float
    mape: float
    smape: float
    within_5pct: float    # % predictions within 5%
    within_10pct: float   # % predictions within 10%
    within_15pct: float   # % predictions within 15%
    directional_accuracy: float  # % correct direction of change
    peak_mae: float       # MAE during peak hours (10am-4pm)

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    timestamps: pd.Series = None
) -> ModelMetrics:
    """
    Calculate comprehensive evaluation metrics.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    model_name : str
        Model name for identification
    timestamps : pd.Series, optional
        Timestamps for peak hour calculation

    Returns:
    --------
    ModelMetrics : Dataclass with all metrics
    """
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE (with zero handling)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    # Symmetric MAPE
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    safe_denominator = np.where(denominator == 0, 1, denominator)
    smape = np.mean(np.abs(y_true - y_pred) / safe_denominator) * 100

    # Within percentage thresholds
    abs_errors = np.abs(y_true - y_pred)
    safe_true = np.where(y_true == 0, 1, y_true)
    pct_errors = abs_errors / np.abs(safe_true) * 100

    within_5pct = (pct_errors <= 5).mean() * 100
    within_10pct = (pct_errors <= 10).mean() * 100
    within_15pct = (pct_errors <= 15).mean() * 100

    # Directional accuracy (did we predict increase/decrease correctlyGKFS)
    if len(y_true) > 1:
        actual_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = (actual_direction == pred_direction).mean() * 100
    else:
        directional_accuracy = np.nan

    # Peak hour MAE (10am - 4pm)
    if timestamps is not None:
        timestamps = pd.to_datetime(timestamps)
        peak_mask = (timestamps.dt.hour >= 10) & (timestamps.dt.hour <= 16)
        if peak_mask.sum() > 0:
            peak_mae = mean_absolute_error(y_true[peak_mask], y_pred[peak_mask])
        else:
            peak_mae = mae
    else:
        peak_mae = mae

    return ModelMetrics(
        model_name=model_name,
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        r2=round(r2, 4),
        mape=round(mape, 2) if not np.isnan(mape) else -1,
        smape=round(smape, 2),
        within_5pct=round(within_5pct, 2),
        within_10pct=round(within_10pct, 2),
        within_15pct=round(within_15pct, 2),
        directional_accuracy=round(directional_accuracy, 2) if not np.isnan(directional_accuracy) else -1,
        peak_mae=round(peak_mae, 4)
    )


def calculate_fsp_selection_accuracy(
    df: pd.DataFrame,
    ml_selected_fsps: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate FSP selection accuracy metrics.

    Returns:
    --------
    Tuple[float, float]
        (oracle_match_pct, ml_improvement_pct)
        - oracle_match_pct: % times ML picked the oracle-best FSP
        - ml_improvement_pct: % improvement in error vs. always picking most common FSP
    """
    fsp_cols = get_fsp_forecast_columns(df)

    if len(fsp_cols) == 0:
        return 0.0, 0.0

    # Find oracle best FSP (lowest error)
    oracle_errors = []
    ml_errors = []
    baseline_errors = []
    oracle_matches = 0

    # Get baseline FSP (most common in schedule)
    if 'schedule_power' in df.columns:
        baseline_error = np.abs(df['actual_power'].values - df['schedule_power'].values)
    else:
        baseline_error = np.full(len(df), np.inf)

    for i in range(len(df)):
        # Get all FSP values for this row
        fsp_values = {}
        for fsp_col in fsp_cols:
            val = df.iloc[i].get(fsp_col, np.nan)
            if not np.isnan(val):
                fsp_name = fsp_col.replace('forecast_power_', '').upper()
                fsp_values[fsp_name] = val

        if not fsp_values:
            continue

        actual = df.iloc[i]['actual_power']

        # Oracle best
        errors = {fsp: abs(val - actual) for fsp, val in fsp_values.items()}
        oracle_fsp = min(errors, key=errors.get)
        oracle_errors.append(errors[oracle_fsp])

        # ML selection
        ml_fsp = ml_selected_fsps[i]
        if ml_fsp in fsp_values:
            ml_errors.append(abs(fsp_values[ml_fsp] - actual))
            if ml_fsp == oracle_fsp:
                oracle_matches += 1
        else:
            ml_errors.append(np.nan)

        baseline_errors.append(baseline_error[i])

    oracle_match_pct = (oracle_matches / len(df)) * 100 if len(df) > 0 else 0

    # Calculate improvement over baseline
    ml_errors_clean = np.array([e for e in ml_errors if not np.isnan(e)])
    baseline_errors_clean = np.array([e for e in baseline_errors if not np.isnan(e) and not np.isinf(e)])

    if len(ml_errors_clean) > 0 and len(baseline_errors_clean) > 0:
        ml_mean_error = np.mean(ml_errors_clean)
        baseline_mean_error = np.mean(baseline_errors_clean)
        if baseline_mean_error > 0:
            improvement_pct = ((baseline_mean_error - ml_mean_error) / baseline_mean_error) * 100
        else:
            improvement_pct = 0.0
    else:
        improvement_pct = 0.0

    return round(oracle_match_pct, 2), round(improvement_pct, 2)


# =============================================================================
# FEATURE SELECTION
# =============================================================================

def select_features_by_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_cols: List[str],
    threshold: float = 0.01,
    n_repeats: int = 5
) -> Tuple[List[str], np.ndarray]:
    """
    Select features using permutation importance.

    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training target
    feature_cols : List[str]
        Feature column names
    threshold : float
        Minimum importance to keep feature
    n_repeats : int
        Number of permutation repeats

    Returns:
    --------
    Tuple[List[str], np.ndarray]
        Selected feature names and their importance scores
    """
    print("\n Feature Selection via Permutation Importance...")

    # Train a quick RF model for importance
    rf = RandomForestRegressor(
        n_estimators=50, max_depth=10,
        random_state=CONFIG.random_state, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Calculate permutation importance
    result = permutation_importance(
        rf, X_train, y_train,
        n_repeats=n_repeats,
        random_state=CONFIG.random_state,
        n_jobs=-1
    )

    importances = result.importances_mean

    # Normalize importances
    importances = importances / importances.sum() if importances.sum() > 0 else importances

    # Select features above threshold
    mask = importances >= threshold
    selected_features = [f for f, m in zip(feature_cols, mask) if m]
    selected_importances = importances[mask]

    print(f"  Features before selection: {len(feature_cols)}")
    print(f"  Features after selection: {len(selected_features)}")

    # Show top 10 features
    sorted_idx = np.argsort(importances)[::-1][:10]
    print("\n  Top 10 features:")
    for idx in sorted_idx:
        print(f"    {feature_cols[idx]}: {importances[idx]:.4f}")

    return selected_features, importances


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    n_trials: int = 30,
    n_cv_folds: int = 3
) -> dict:
    """
    Tune hyperparameters using Optuna with TimeSeriesSplit.

    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training target
    model_type : str
        Model type ('ridge', 'rf', 'xgb', 'lgb')
    n_trials : int
        Number of Optuna trials
    n_cv_folds : int
        Number of CV folds

    Returns:
    --------
    dict : Best hyperparameters
    """
    if not OPTUNA_AVAILABLE:
        return get_default_hyperparameters(model_type)

    tscv = TimeSeriesSplit(n_splits=n_cv_folds)

    def objective(trial):
        if model_type == 'ridge':
            params = {
                'alpha': trial.suggest_float('alpha', 0.01, 100.0, log=True)
            }
            model = Ridge(**params, random_state=CONFIG.random_state)

        elif model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
            model = RandomForestRegressor(
                **params, random_state=CONFIG.random_state, n_jobs=-1
            )

        elif model_type == 'xgb' and XGB_AVAILABLE:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
            }
            model = xgb.XGBRegressor(
                **params, random_state=CONFIG.random_state, n_jobs=-1
            )

        elif model_type == 'lgb' and LGB_AVAILABLE:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            model = lgb.LGBMRegressor(
                **params, random_state=CONFIG.random_state, n_jobs=-1, verbose=-1
            )
        else:
            return float('inf')

        # Cross-validation
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr, y_va = y_train[train_idx], y_train[val_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            scores.append(mean_absolute_error(y_va, y_pred))

        return np.mean(scores)

    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=CONFIG.random_state)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params


def get_default_hyperparameters(model_type: str) -> dict:
    """Get default hyperparameters when Optuna is not available."""
    defaults = {
        'ridge': {'alpha': 10.0},
        'rf': {
            'n_estimators': 100, 'max_depth': 15,
            'min_samples_split': 10, 'min_samples_leaf': 5
        },
        'xgb': {
            'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.1, 'reg_lambda': 1.0
        },
        'lgb': {
            'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
            'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8
        }
    }
    return defaults.get(model_type, {})


# =============================================================================
# STACKING ENSEMBLE
# =============================================================================

class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Stacking ensemble with Ridge meta-learner.

    Uses out-of-fold predictions from base models as features
    for a Ridge regression meta-learner.
    """

    def __init__(
        self,
        base_models: Dict[str, Any],
        meta_model: Any = None,
        n_folds: int = 3,
        scaler: StandardScaler = None,
        imputer: SimpleImputer = None
    ):
        self.base_models = base_models
        self.meta_model = meta_model or Ridge(alpha=1.0)
        self.n_folds = n_folds
        self.scaler = scaler
        self.imputer = imputer
        self.fitted_base_models_ = {}
        self.meta_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """Fit the stacking ensemble."""
        n_samples = X.shape[0]
        n_models = len(self.base_models)

        # Initialize out-of-fold predictions
        oof_predictions = np.zeros((n_samples, n_models))

        tscv = TimeSeriesSplit(n_splits=self.n_folds)

        # Generate out-of-fold predictions for each base model
        for model_idx, (model_name, model) in enumerate(self.base_models.items()):
            print(f"    Fitting {model_name} for stacking...")

            # Store clones for each fold
            fold_models = []

            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]

                # Clone and fit model
                model_clone = clone(model)
                model_clone.fit(X_train_fold, y_train_fold)
                fold_models.append(model_clone)

                # Generate OOF predictions
                oof_predictions[val_idx, model_idx] = model_clone.predict(X_val_fold)

            # Fit final model on all data
            final_model = clone(model)
            final_model.fit(X, y)
            self.fitted_base_models_[model_name] = final_model

        # Fit meta-learner on OOF predictions
        # Only use samples that have OOF predictions (exclude first folds)
        valid_mask = ~np.isnan(oof_predictions).any(axis=1)
        if valid_mask.sum() < len(y) * 0.5:
            # If too few valid samples, use all predictions
            valid_mask = np.ones(n_samples, dtype=bool)
            # Fill NaN with predictions from full models
            for model_idx, (model_name, model) in enumerate(self.fitted_base_models_.items()):
                nan_mask = np.isnan(oof_predictions[:, model_idx])
                if nan_mask.any():
                    oof_predictions[nan_mask, model_idx] = model.predict(X[nan_mask])

        self.meta_model.fit(oof_predictions[valid_mask], y[valid_mask])
        self.meta_features_ = oof_predictions

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the stacking ensemble."""
        # Handle preprocessing
        if self.imputer is not None:
            X = self.imputer.transform(X)
        X = np.nan_to_num(X, nan=0.0)

        # Get predictions from all base models
        meta_features = np.column_stack([
            model.predict(X) for model in self.fitted_base_models_.values()
        ])

        # Meta-learner prediction
        return self.meta_model.predict(meta_features)

    def get_base_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from each base model."""
        if self.imputer is not None:
            X = self.imputer.transform(X)
        X = np.nan_to_num(X, nan=0.0)

        return {
            name: model.predict(X)
            for name, model in self.fitted_base_models_.items()
        }


# =============================================================================
# PREDICTION INTERVALS
# =============================================================================

class QuantileRegressorWrapper:
    """Wrapper for quantile regression prediction intervals."""

    def __init__(
        self,
        quantiles: List[float] = [0.05, 0.5, 0.95],
        alpha: float = 1.0
    ):
        self.quantiles = quantiles
        self.alpha = alpha
        self.models = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit quantile regressors."""
        for q in self.quantiles:
            print(f"    Fitting quantile regressor (q={q})...")
            model = QuantileRegressor(
                quantile=q,
                alpha=self.alpha,
                solver='highs'
            )
            model.fit(X, y)
            self.models[q] = model
        return self

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict quantiles."""
        return {
            f'q{int(q*100)}': model.predict(X)
            for q, model in self.models.items()
        }

    def get_prediction_intervals(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get lower, median, upper predictions."""
        preds = self.predict(X)
        lower = preds.get('q5', preds.get('q10', None))
        median = preds.get('q50', None)
        upper = preds.get('q95', preds.get('q90', None))
        return lower, median, upper


# =============================================================================
# FSP SELECTION
# =============================================================================

def select_best_fsp_by_prediction(
    df: pd.DataFrame,
    predictions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select FSP whose forecast is closest to predicted actual power.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with FSP forecast columns
    predictions : np.ndarray
        Model predictions of actual power

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        selected_fsps, scheduled_power, confidence
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
            if pd.notna(val):
                fsp_name = fsp_col.replace('forecast_power_', '').upper()
                fsp_values[fsp_name] = val

        if fsp_values and not np.isnan(pred):
            # Select FSP with forecast closest to prediction
            errors = {fsp: abs(val - pred) for fsp, val in fsp_values.items()}
            best_fsp = min(errors, key=errors.get)

            selected_fsps.append(best_fsp)
            scheduled_power.append(fsp_values[best_fsp])

            # Confidence based on spread of FSP forecasts
            min_err, max_err = min(errors.values()), max(errors.values())
            conf = (max_err - min_err) / (max_err + 1e-8) if max_err > 0 else 0.5
            confidence.append(min(conf, 1.0))
        else:
            selected_fsps.append('UNKNOWN')
            scheduled_power.append(np.nan)
            confidence.append(0.0)

    return np.array(selected_fsps), np.array(scheduled_power), np.array(confidence)


# =============================================================================
# OUTPUT SAVING
# =============================================================================

def save_predictions(
    df: pd.DataFrame,
    predictions: np.ndarray,
    selected_fsps: np.ndarray,
    scheduled_power: np.ndarray,
    confidence: np.ndarray,
    model_name: str,
    output_path: Path,
    season: str,
    prediction_intervals: Tuple[np.ndarray, np.ndarray, np.ndarray] = None
):
    """Save model predictions with FSP selection and prediction intervals."""
    output_df = df.copy()

    # Core predictions
    output_df['ml_predicted_power'] = predictions
    output_df['ml_selected_fsp'] = selected_fsps
    output_df['ml_schedule_power'] = scheduled_power
    output_df['ml_confidence'] = confidence

    # Prediction intervals
    if prediction_intervals is not None:
        lower, median, upper = prediction_intervals
        if lower is not None:
            output_df['ml_pred_lower_5'] = lower
        if upper is not None:
            output_df['ml_pred_upper_95'] = upper

    # Calculate errors
    if 'actual_power' in output_df.columns:
        output_df['ml_error'] = np.abs(output_df['actual_power'] - output_df['ml_predicted_power'])
        output_df['ml_schedule_error'] = np.abs(output_df['actual_power'] - output_df['ml_schedule_power'])

    # Metadata
    output_df['season'] = season
    output_df['model_name'] = model_name
    output_df['model_version'] = CONFIG.model_version

    output_df.to_csv(output_path, index=False)
    print(f"   Saved: {output_path.name}")


def save_metadata(
    season: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    best_model: str,
    val_results: List[ModelMetrics],
    test_results: List[ModelMetrics],
    hyperparameters: Dict[str, dict],
    fsp_accuracy: Tuple[float, float]
):
    """Save comprehensive training metadata."""
    metadata = {
        'version': CONFIG.model_version,
        'git_commit': get_git_commit_hash(),
        'season': season,
        'training_date': datetime.now().isoformat(),
        'config': asdict(CONFIG),
        'data_info': {
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
            }
        },
        'features': {
            'count': len(feature_cols),
            'names': feature_cols
        },
        'best_model': best_model,
        'hyperparameters': hyperparameters,
        'val_metrics': [m.to_dict() for m in val_results],
        'test_metrics': [m.to_dict() for m in test_results],
        'fsp_selection': {
            'oracle_match_pct': fsp_accuracy[0],
            'improvement_vs_baseline_pct': fsp_accuracy[1]
        }
    }

    # Save with proper JSON handling
    output_path = MODELS_DIR / f'model_metadata_{season}.json'
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f" Saved metadata: {output_path.name}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_season(season: str, df_pivoted: pd.DataFrame) -> Optional[dict]:
    """
    Train models for a specific season.

    Parameters:
    -----------
    season : str
        Season name ('winter', 'spring', 'summer', 'fall')
    df_pivoted : pd.DataFrame
        Pivoted and cleaned dataframe

    Returns:
    --------
    dict : Training results or None if failed
    """
    print_section(f" TRAINING MODELS FOR {season.upper()} SEASON")

    # =========================================================================
    # STEP 1: Create Seasonal Split
    # =========================================================================
    print_subsection(" Creating Seasonal Split")

    try:
        train_df, val_df, test_df = create_seasonal_split(
            df_pivoted,
            season,
            train_year=CONFIG.train_year,
            val_test_year=CONFIG.val_test_year,
            val_ratio=CONFIG.val_ratio
        )
    except ValueError as e:
        print(f" Error: {e}")
        print(f" Skipping {season} season - insufficient data")
        return None

    # =========================================================================
    # STEP 2: Create Target (6-block-ahead prediction)
    # =========================================================================
    print_subsection(f" Creating {CONFIG.prediction_horizon}-Block-Ahead Target")

    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        split_df[TARGET_HORIZON] = split_df[TARGET].shift(-CONFIG.prediction_horizon)

    # Drop rows with NaN targets
    train_df = train_df.dropna(subset=[TARGET_HORIZON]).reset_index(drop=True)
    val_df = val_df.dropna(subset=[TARGET_HORIZON]).reset_index(drop=True)
    test_df = test_df.dropna(subset=[TARGET_HORIZON]).reset_index(drop=True)

    print(f"  After target creation:")
    print(f"    Train: {len(train_df):,} samples")
    print(f"    Val: {len(val_df):,} samples")
    print(f"    Test: {len(test_df):,} samples")

    if len(train_df) < 100 or len(val_df) < 50 or len(test_df) < 50:
        print(f" Insufficient data for {season} season")
        return None

    # =========================================================================
    # STEP 3: Feature Engineering
    # =========================================================================
    print_subsection(" Feature Engineering")

    # Create time features
    train_df = create_time_features(train_df)
    val_df = create_time_features(val_df)
    test_df = create_time_features(test_df)

    # Create rolling features (after split to prevent leakage)
    fsp_cols = get_fsp_forecast_columns(train_df)
    train_df = create_rolling_features(train_df, target_col=TARGET, fsp_cols=fsp_cols)
    val_df = create_rolling_features(val_df, target_col=TARGET, fsp_cols=fsp_cols)
    test_df = create_rolling_features(test_df, target_col=TARGET, fsp_cols=fsp_cols)

    # Encode categorical features
    exclude_cols = ['date', 'timestamp', TARGET, TARGET_HORIZON]
    train_df, label_encoders = encode_categorical_features(train_df, exclude_cols)
    val_df, _ = encode_categorical_features(val_df, exclude_cols, label_encoders)
    test_df, _ = encode_categorical_features(test_df, exclude_cols, label_encoders)

    print(" Features created")

    # =========================================================================
    # STEP 4: Feature Selection
    # =========================================================================
    print_subsection(" Feature Selection")

    feature_cols = get_feature_columns(
        train_df,
        target_col=TARGET_HORIZON,
        exclude_patterns=EXCLUDE_PATTERNS
    )
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    print(f"  Initial features: {len(feature_cols)}")

    # Prepare data
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_HORIZON].values
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_HORIZON].values
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_HORIZON].values

    # Impute NaN values
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    # Feature selection via permutation importance
    selected_features, importances = select_features_by_importance(
        X_train, y_train, feature_cols,
        threshold=CONFIG.feature_importance_threshold
    )

    # Update feature set
    selected_indices = [feature_cols.index(f) for f in selected_features]
    X_train = X_train[:, selected_indices]
    X_val = X_val[:, selected_indices]
    X_test = X_test[:, selected_indices]
    feature_cols = selected_features

    # =========================================================================
    # STEP 5: Scaling
    # =========================================================================
    print_subsection(" Feature Scaling")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(" Features scaled using StandardScaler")

    # Save scaler and imputer
    with open(MODELS_DIR / f'scaler_{season}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(MODELS_DIR / f'imputer_{season}.pkl', 'wb') as f:
        pickle.dump(imputer, f)

    # =========================================================================
    # STEP 6: Hyperparameter Tuning
    # =========================================================================
    print_subsection(" Hyperparameter Tuning")

    hyperparameters = {}

    print("\n  Tuning Ridge...")
    hyperparameters['ridge'] = tune_hyperparameters(
        X_train_scaled, y_train, 'ridge',
        n_trials=CONFIG.n_optuna_trials, n_cv_folds=CONFIG.n_cv_folds
    )
    print(f"    Best params: {hyperparameters['ridge']}")

    print("\n  Tuning Random Forest...")
    hyperparameters['rf'] = tune_hyperparameters(
        X_train_scaled, y_train, 'rf',
        n_trials=CONFIG.n_optuna_trials, n_cv_folds=CONFIG.n_cv_folds
    )
    print(f"    Best params: {hyperparameters['rf']}")

    if XGB_AVAILABLE:
        print("\n  Tuning XGBoost...")
        hyperparameters['xgb'] = tune_hyperparameters(
            X_train_scaled, y_train, 'xgb',
            n_trials=CONFIG.n_optuna_trials, n_cv_folds=CONFIG.n_cv_folds
        )
        print(f"    Best params: {hyperparameters['xgb']}")

    if LGB_AVAILABLE:
        print("\n  Tuning LightGBM...")
        hyperparameters['lgb'] = tune_hyperparameters(
            X_train_scaled, y_train, 'lgb',
            n_trials=CONFIG.n_optuna_trials, n_cv_folds=CONFIG.n_cv_folds
        )
        print(f"    Best params: {hyperparameters['lgb']}")

    # =========================================================================
    # STEP 7: Train Models
    # =========================================================================
    print_subsection(" Training Models")

    models = {}
    all_predictions = {}

    # Ridge Regression
    print("\n  1 Training Ridge Regression...")
    ridge = Ridge(**hyperparameters['ridge'], random_state=CONFIG.random_state)
    ridge.fit(X_train_scaled, y_train)
    models['ridge'] = ridge
    all_predictions['ridge'] = {
        'val': ridge.predict(X_val_scaled),
        'test': ridge.predict(X_test_scaled)
    }
    print("     Ridge trained")

    # Random Forest
    print("\n  2 Training Random Forest...")
    rf = RandomForestRegressor(
        **hyperparameters['rf'],
        random_state=CONFIG.random_state,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    models['random_forest'] = rf
    all_predictions['random_forest'] = {
        'val': rf.predict(X_val_scaled),
        'test': rf.predict(X_test_scaled)
    }
    print("     Random Forest trained")

    # XGBoost
    if XGB_AVAILABLE:
        print("\n  3 Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            **hyperparameters['xgb'],
            random_state=CONFIG.random_state,
            n_jobs=-1
        )
        xgb_model.fit(X_train_scaled, y_train)
        models['xgboost'] = xgb_model
        all_predictions['xgboost'] = {
            'val': xgb_model.predict(X_val_scaled),
            'test': xgb_model.predict(X_test_scaled)
        }
        print("     XGBoost trained")

    # LightGBM
    if LGB_AVAILABLE:
        print("\n  4 Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            **hyperparameters['lgb'],
            random_state=CONFIG.random_state,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train_scaled, y_train)
        models['lightgbm'] = lgb_model
        all_predictions['lightgbm'] = {
            'val': lgb_model.predict(X_val_scaled),
            'test': lgb_model.predict(X_test_scaled)
        }
        print("     LightGBM trained")

    # =========================================================================
    # STEP 8: Stacking Ensemble (Option C)
    # =========================================================================
    if CONFIG.use_stacking and len(models) >= 2:
        print_subsection(" Building Stacking Ensemble")

        # Prepare base models for stacking
        base_models = {}
        if 'ridge' in models:
            base_models['ridge'] = Ridge(**hyperparameters['ridge'], random_state=CONFIG.random_state)
        if 'random_forest' in models:
            base_models['rf'] = RandomForestRegressor(**hyperparameters['rf'], random_state=CONFIG.random_state, n_jobs=-1)
        if 'xgboost' in models and XGB_AVAILABLE:
            base_models['xgb'] = xgb.XGBRegressor(**hyperparameters['xgb'], random_state=CONFIG.random_state, n_jobs=-1)
        if 'lightgbm' in models and LGB_AVAILABLE:
            base_models['lgb'] = lgb.LGBMRegressor(**hyperparameters['lgb'], random_state=CONFIG.random_state, n_jobs=-1, verbose=-1)

        stacking = StackingEnsemble(
            base_models=base_models,
            meta_model=Ridge(alpha=1.0),
            n_folds=CONFIG.n_cv_folds,
            scaler=None,  # Already scaled
            imputer=None  # Already imputed
        )
        stacking.fit(X_train_scaled, y_train)
        models['stacking_ensemble'] = stacking

        all_predictions['stacking_ensemble'] = {
            'val': stacking.predict(X_val_scaled),
            'test': stacking.predict(X_test_scaled)
        }
        print("   Stacking ensemble created")

    # =========================================================================
    # STEP 9: Prediction Intervals (Option D)
    # =========================================================================
    print_subsection(" Building Prediction Intervals")

    try:
        quantile_reg = QuantileRegressorWrapper(
            quantiles=CONFIG.quantiles,
            alpha=1.0
        )
        quantile_reg.fit(X_train_scaled, y_train)

        # Get intervals for test set
        lower, median, upper = quantile_reg.get_prediction_intervals(X_test_scaled)
        prediction_intervals = (lower, median, upper)

        # Coverage analysis
        if lower is not None and upper is not None:
            coverage = ((y_test >= lower) & (y_test <= upper)).mean() * 100
            print(f"   90% prediction interval coverage: {coverage:.1f}%")

        models['quantile_regressor'] = quantile_reg
    except Exception as e:
        print(f"   Quantile regression failed: {e}")
        prediction_intervals = None

    # =========================================================================
    # STEP 10: Evaluate on Validation Set
    # =========================================================================
    print_subsection(" Validation Results")

    val_results = []
    for model_name, preds in all_predictions.items():
        metrics = calculate_metrics(
            y_val, preds['val'], model_name,
            timestamps=val_df['date'] if 'date' in val_df.columns else None
        )
        val_results.append(metrics)

    val_results_df = pd.DataFrame([m.to_dict() for m in val_results])
    val_results_df = val_results_df.sort_values('mae')
    print("\n" + val_results_df.to_string(index=False))

    best_model = val_results_df.iloc[0]['model_name']
    print(f"\n Best model on validation: {best_model}")

    # =========================================================================
    # STEP 11: Evaluate on Test Set
    # =========================================================================
    print_subsection(" Test Results")

    test_results = []
    for model_name, preds in all_predictions.items():
        metrics = calculate_metrics(
            y_test, preds['test'], model_name,
            timestamps=test_df['date'] if 'date' in test_df.columns else None
        )
        test_results.append(metrics)

    test_results_df = pd.DataFrame([m.to_dict() for m in test_results])
    test_results_df = test_results_df.sort_values('mae')
    print("\n" + test_results_df.to_string(index=False))

    # =========================================================================
    # STEP 12: FSP Selection Accuracy
    # =========================================================================
    print_subsection(" FSP Selection Accuracy")

    best_test_pred = all_predictions[best_model]['test']
    selected_fsps, scheduled_power, confidence = select_best_fsp_by_prediction(
        test_df, best_test_pred
    )

    oracle_match, improvement = calculate_fsp_selection_accuracy(
        test_df, selected_fsps, best_test_pred
    )

    print(f"  Oracle FSP match rate: {oracle_match:.1f}%")
    print(f"  Improvement vs baseline: {improvement:.1f}%")

    # =========================================================================
    # STEP 13: Save Models
    # =========================================================================
    print_subsection(" Saving Models")

    for model_name, model in models.items():
        if model_name != 'quantile_regressor':
            model_path = MODELS_DIR / f'{model_name}_{season}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"   Saved: {model_path.name}")

    # Save quantile regressor separately
    if 'quantile_regressor' in models:
        qr_path = MODELS_DIR / f'quantile_regressor_{season}.pkl'
        with open(qr_path, 'wb') as f:
            pickle.dump(models['quantile_regressor'], f)
        print(f"   Saved: {qr_path.name}")

    # =========================================================================
    # STEP 14: Save Predictions
    # =========================================================================
    print_subsection(" Saving Predictions")

    # Create season subdirectory
    season_pred_dir = PREDS_DIR / season
    season_pred_dir.mkdir(parents=True, exist_ok=True)

    for model_name, preds in all_predictions.items():
        # Validation predictions
        val_selected_fsps, val_scheduled_power, val_confidence = select_best_fsp_by_prediction(
            val_df, preds['val']
        )
        save_predictions(
            val_df, preds['val'], val_selected_fsps, val_scheduled_power, val_confidence,
            model_name, season_pred_dir / f'val_predictions_{model_name}.csv', season,
            prediction_intervals=None
        )

        # Test predictions
        test_selected_fsps, test_scheduled_power, test_confidence = select_best_fsp_by_prediction(
            test_df, preds['test']
        )
        save_predictions(
            test_df, preds['test'], test_selected_fsps, test_scheduled_power, test_confidence,
            model_name, season_pred_dir / f'test_predictions_{model_name}.csv', season,
            prediction_intervals=prediction_intervals if model_name == best_model else None
        )

    # Save raw datasets
    val_df_save = val_df.copy()
    val_df_save['target'] = y_val
    val_df_save.to_csv(season_pred_dir / 'val_set.csv', index=False)
    print(f"   Saved: val_set.csv")

    test_df_save = test_df.copy()
    test_df_save['target'] = y_test
    test_df_save.to_csv(season_pred_dir / 'test_set.csv', index=False)
    print(f"   Saved: test_set.csv")

    # =========================================================================
    # STEP 15: Save Metadata
    # =========================================================================
    print_subsection(" Saving Metadata")

    # Save feature columns
    with open(MODELS_DIR / f'feature_columns_{season}.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)

    # Save comprehensive metadata
    save_metadata(
        season=season,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        best_model=best_model,
        val_results=val_results,
        test_results=test_results,
        hyperparameters=hyperparameters,
        fsp_accuracy=(oracle_match, improvement)
    )

    print_section(f" {season.upper()} SEASON TRAINING COMPLETE!")

    return {
        'season': season,
        'best_model': best_model,
        'val_results': val_results,
        'test_results': test_results,
        'fsp_accuracy': (oracle_match, improvement)
    }


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main training pipeline."""

    print_section(" SEASONAL MODEL TRAINING PIPELINE V2.0", "=", 80)
    print("\n Sample Plant Wind Plant - FSP Selection Optimization")
    print(f"   Version: {CONFIG.model_version}")
    print(f"   Git Commit: {get_git_commit_hash()}")
    print(f"   Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n Configuration:")
    print(f"   Train Year: {CONFIG.train_year}")
    print(f"   Val/Test Year: {CONFIG.val_test_year}")
    print(f"   Prediction Horizon: {CONFIG.prediction_horizon} blocks (1.5 hours)")
    print(f"   Excluded FSPs: {CONFIG.exclude_fsps}")
    print(f"   Max Power Cap: {CONFIG.max_actual_power} MW")
    print(f"   Use Stacking: {CONFIG.use_stacking}")
    print(f"   Optuna Trials: {CONFIG.n_optuna_trials}")

    # =========================================================================
    # STEP 1: Load and Pivot Data
    # =========================================================================
    print_section(" Loading and Preprocessing Data")

    processed_path = DATA_DIR / 'sample_pss_dataset.parquet'
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found: {processed_path}")

    df = pd.read_parquet(processed_path)
    print(f" Loaded: {len(df):,} rows")

    # Check if data needs pivoting
    if 'forecast_facode' in df.columns:
        print("\n Pivoting FSP data...")
        df_pivoted = pivot_fsp_data(df, exclude_fsps=CONFIG.exclude_fsps)
    else:
        print(" Data already pivoted")
        df_pivoted = df

    # Clean data
    print("\n Cleaning data...")
    df_pivoted = clean_data(df_pivoted, CONFIG)

    # Ensure date column
    if 'date' not in df_pivoted.columns and 'timestamp' in df_pivoted.columns:
        df_pivoted['date'] = pd.to_datetime(df_pivoted['timestamp'])
    else:
        df_pivoted['date'] = pd.to_datetime(df_pivoted['date'])

    print(f"\n Final dataset: {len(df_pivoted):,} rows")
    print(f"  Date range: {df_pivoted['date'].min().date()} to {df_pivoted['date'].max().date()}")

    # =========================================================================
    # STEP 2: Train for Each Season
    # =========================================================================
    all_results = []

    for season in SEASONS.keys():
        result = train_season(season, df_pivoted)
        if result:
            all_results.append(result)

    # =========================================================================
    # STEP 3: Summary
    # =========================================================================
    if all_results:
        print_section(" SUMMARY ACROSS ALL SEASONS", "=", 80)

        summary_data = []
        for result in all_results:
            season = result['season']
            best_model = result['best_model']

            # Get best model's test metrics
            test_metrics = [m for m in result['test_results'] if m.model_name == best_model][0]

            summary_data.append({
                'Season': season.upper(),
                'Best Model': best_model,
                'Test MAE': test_metrics.mae,
                'Test RMSE': test_metrics.rmse,
                'Test R2': test_metrics.r2,
                'Within 15%': f"{test_metrics.within_15pct:.1f}%",
                'FSP Oracle Match': f"{result['fsp_accuracy'][0]:.1f}%",
                'Improvement': f"{result['fsp_accuracy'][1]:.1f}%"
            })

        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))

        # Save summary
        summary_path = REPORTS_DIR / 'seasonal_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\n Summary saved: {summary_path}")

        # Real-world impact statement
        print_section(" REAL-WORLD IMPACT - SAMPLE WIND PLANT")

        avg_improvement = np.mean([r['fsp_accuracy'][1] for r in all_results])
        avg_within_15 = np.mean([
            [m.within_15pct for m in r['test_results'] if m.model_name == r['best_model']][0]
            for r in all_results
        ])

        print(f"""
    Sample Plant Wind Plant Scheduling Optimization Results:

    Average FSP selection improvement: {avg_improvement:.1f}%
    Predictions within 15% of actual: {avg_within_15:.1f}%

    Estimated Benefits:
    Reduced deviation settlement charges
    Better grid stability compliance
    Improved revenue from accurate scheduling

    Production Deployment:
    Models saved in: {MODELS_DIR}
    Predictions saved in: {PREDS_DIR}
    Reports saved in: {REPORTS_DIR}

    Next Steps:
   1. Validate models on live data
   2. A/B test against current scheduling
   3. Integrate with SCADA system
""")

        print_section(" ALL SEASONS TRAINING COMPLETE!", "=", 80)
    else:
        print("\n No seasons were successfully trained")


if __name__ == '__main__':
    main()
