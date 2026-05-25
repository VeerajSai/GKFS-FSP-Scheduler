"""
Seasonal FSP Selection Model Training - V3
=========================================

End-to-end ML lifecycle for Sample Plant Wind Plant FSP selection:
- Data pivoting and cleaning
- Season-aware splits (train year -> val/test year)
- Feature engineering and selection (Option B)
- Hyperparameter tuning with TimeSeriesSplit
- Stacking ensemble (Option C)
- Prediction intervals (Option D)
- Robust metrics (MAE, RMSE, R^2, MAPE, within thresholds)

This script is self-contained and does not modify other code.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

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

try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except Exception:
    optuna = None
    TPESampler = None
    OPTUNA_AVAILABLE = False

warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*valid feature names.*",
    category=UserWarning,
)


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
MODELS_DIR = OUTPUT_DIR / "models_seasonal_v3"
PREDS_DIR = OUTPUT_DIR / "predictions_seasonal_v3"
REPORTS_DIR = OUTPUT_DIR / "reports_seasonal_v3"

for d in [MODELS_DIR, PREDS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingConfig:
    max_actual_power: float = 600.0
    min_actual_power: float = 0.0
    exclude_fsps: Tuple[str, ...] = ("FA_PROVIDER_E",)

    train_year: int = 2024
    val_test_year: int = 2025
    val_ratio: float = 0.5

    prediction_horizon: int = 6
    random_state: int = 42
    n_cv_folds: int = 5
    n_optuna_trials: int = 60
    tuning_sample_size: int = 50000

    feature_importance_threshold: float = 0.01
    use_stacking: bool = False
    use_weighted_ensemble: bool = True
    ridge_weight: float = 0.4
    quantiles: Tuple[float, float, float] = (0.05, 0.5, 0.95)

    model_version: str = "3.0.0"


CONFIG = TrainingConfig()

SEASONS = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "fall": [9, 10, 11],
}

TARGET = "actual_power"
TARGET_HORIZON = "target_horizon"


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


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError("Expected 'date' or 'timestamp' column in data.")
    return df


def _pivot_fsp_data(df: pd.DataFrame, fsp_col: str = "forecast_facode") -> pd.DataFrame:
    exclude = set(CONFIG.exclude_fsps)
    filtered = df[~df[fsp_col].isin(exclude)].copy()

    group_cols = ["date", "block"]
    if "timestamp" in filtered.columns:
        group_cols.append("timestamp")
    if "sscode" in filtered.columns:
        group_cols.append("sscode")

    common_cols = [
        "actual_power",
        "actual_avc",
        "actual_windspeed",
        "actual_ghirr",
        "actual_flowrate",
        "actual_time",
        "actual_source",
        "schedule_power",
        "schedule_avc",
        "schedule_windspeed",
        "schedule_ghirr",
        "schedule_flowrate",
        "schedule_time",
        "schedule_source",
        "schedule_revno",
    ]
    existing_common = [c for c in common_cols if c in filtered.columns]
    base_df = filtered.groupby(group_cols, as_index=False)[existing_common].first()

    pivot_cols = [
        "forecast_power",
        "forecast_avc",
        "forecast_windspeed",
        "forecast_ghirr",
        "forecast_flowrate",
        "forecast_revno",
    ]

    for col in pivot_cols:
        if col not in filtered.columns:
            continue
        pivoted = filtered.pivot_table(
            index=group_cols,
            columns=fsp_col,
            values=col,
            aggfunc="first",
        )
        rename_map = {fsp: f"{col}_{fsp.lower()}" for fsp in pivoted.columns}
        pivoted = pivoted.rename(columns=rename_map).reset_index()
        merge_cols = [c for c in pivoted.columns if c not in group_cols]
        if merge_cols:
            base_df = base_df.merge(pivoted[group_cols + merge_cols], on=group_cols, how="left")

    base_df = base_df.sort_values(group_cols).reset_index(drop=True)
    print(f" Pivoted rows: {len(df):,} -> {len(base_df):,}")
    return base_df


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)

    if TARGET in df.columns:
        df[TARGET] = df[TARGET].clip(CONFIG.min_actual_power, CONFIG.max_actual_power)
        df = df.dropna(subset=[TARGET])

    fsp_cols = [c for c in df.columns if c.startswith("forecast_power_")]
    if fsp_cols:
        all_missing = df[fsp_cols].isna().all(axis=1)
        df = df[~all_missing]

    after = len(df)
    print(f" Cleaned rows: {before:,} -> {after:,} (dropped {before - after:,})")
    return df


def _get_fsp_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("forecast_power_")]


# =============================================================================
# SEASONAL SPLITS
# =============================================================================

def _season_for_date(ts: pd.Timestamp) -> Optional[str]:
    month = ts.month
    for name, months in SEASONS.items():
        if month in months:
            return name
    return None


def _seasonal_split(
    df: pd.DataFrame,
    season: str,
    train_year: int,
    val_test_year: int,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = df["date"].apply(_season_for_date)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    season_df = df[df["season"] == season].copy()
    if season_df.empty:
        raise ValueError(f"No data for season {season}.")

    if season == "winter":
        train_mask = (
            ((df["year"] == train_year - 1) & (df["month"] == 12))
            | ((df["year"] == train_year) & (df["month"].isin([1, 2])))
        )
        val_test_mask = (
            ((df["year"] == val_test_year - 1) & (df["month"] == 12))
            | ((df["year"] == val_test_year) & (df["month"].isin([1, 2])))
        )
    else:
        months = SEASONS[season]
        train_mask = (df["year"] == train_year) & (df["month"].isin(months))
        val_test_mask = (df["year"] == val_test_year) & (df["month"].isin(months))

    train_df = df[train_mask].copy()
    val_test_df = df[val_test_mask].copy()

    if train_df.empty or val_test_df.empty:
        raise ValueError(f"Insufficient data for season {season}.")

    val_test_sorted = val_test_df.sort_values("date").reset_index(drop=True)
    val_size = int(len(val_test_sorted) * val_ratio)
    val_df = val_test_sorted.iloc[:val_size].copy()
    test_df = val_test_sorted.iloc[val_size:].copy()

    for split_df in [train_df, val_df, test_df]:
        split_df.drop(columns=["season", "year", "month"], inplace=True, errors="ignore")

    print(f"\n {season.upper()} split")
    print(f"  Train: {len(train_df):,} ({train_df['date'].min().date()} -> {train_df['date'].max().date()})")
    print(f"  Val:   {len(val_df):,} ({val_df['date'].min().date()} -> {val_df['date'].max().date()})")
    print(f"  Test:  {len(test_df):,} ({test_df['date'].min().date()} -> {test_df['date'].max().date()})")

    return train_df, val_df, test_df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["date"])
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_year"] = dt.dt.dayofyear
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    if "block" in df.columns:
        df["block"] = pd.to_numeric(df["block"], errors="coerce")
    return df


def _add_lag_features(df: pd.DataFrame, lags: Iterable[int]) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date")
    for lag in lags:
        df[f"lag_{TARGET}_{lag}"] = df[TARGET].shift(lag)
    return df


def _encode_categoricals(
    train_df: pd.DataFrame,
    other_dfs: List[pd.DataFrame],
    exclude_cols: Iterable[str],
) -> Tuple[pd.DataFrame, List[pd.DataFrame], Dict[str, Dict[str, int]]]:
    exclude = set(exclude_cols)
    mappings: Dict[str, Dict[str, int]] = {}

    def _encode(col: str, series: pd.Series, mapping: Optional[Dict[str, int]] = None) -> Tuple[pd.Series, Dict[str, int]]:
        if mapping is None:
            codes, uniques = pd.factorize(series.astype(str), sort=True)
            return pd.Series(codes, index=series.index), {u: int(i) for i, u in enumerate(uniques)}
        mapped = series.astype(str).map(mapping)
        mapped = mapped.fillna(-1).astype(int)
        return mapped, mapping

    for col in train_df.columns:
        if col in exclude:
            continue
        if train_df[col].dtype == "object":
            encoded, mapping = _encode(col, train_df[col])
            train_df[col] = encoded
            mappings[col] = mapping
            for i, df in enumerate(other_dfs):
                other_dfs[i][col], _ = _encode(col, df[col], mapping)

    return train_df, other_dfs, mappings


def _get_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude_prefixes = ("actual_", "error_", "target", "date", "timestamp")
    exclude_exact = {TARGET, TARGET_HORIZON}
    feature_cols = []
    for col in df.columns:
        if col in exclude_exact:
            continue
        if any(col.startswith(p) for p in exclude_prefixes) and not col.startswith("lag_"):
            continue
        if df[col].dtype in ("object", "string"):
            continue
        feature_cols.append(col)
    return feature_cols


# =============================================================================
# FEATURE SELECTION (Option B)
# =============================================================================

def _select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_cols: List[str],
    threshold: float,
) -> List[str]:
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=CONFIG.random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    result = permutation_importance(
        rf,
        X_train,
        y_train,
        n_repeats=5,
        random_state=CONFIG.random_state,
        n_jobs=-1,
    )
    importances = result.importances_mean
    if importances.sum() > 0:
        importances = importances / importances.sum()
    keep_mask = importances >= threshold
    selected = [f for f, keep in zip(feature_cols, keep_mask) if keep]
    if not selected:
        selected = feature_cols
    return selected


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

def _default_params(model_type: str) -> Dict[str, Any]:
    defaults = {
        "ridge": {"alpha": 10.0},
        "rf": {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
        },
        "xgb": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
            "eval_metric": "mae",
            "tree_method": "hist",
        },
        "lgb": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    }
    return defaults.get(model_type, {})


def _tuning_sample(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) <= max_samples:
        return X, y
    idx = np.linspace(0, len(X) - 1, max_samples, dtype=int)
    return X[idx], y[idx]


def _tune_params(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    n_trials: int,
    n_folds: int,
) -> Dict[str, Any]:
    if not OPTUNA_AVAILABLE or optuna is None or TPESampler is None:
        return _default_params(model_type)

    X_tune, y_tune = _tuning_sample(X, y, CONFIG.tuning_sample_size)
    tscv = TimeSeriesSplit(n_splits=n_folds)

    def objective(trial) -> float:
        if model_type == "ridge":
            params = {"alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True)}
            model = Ridge(**params, random_state=CONFIG.random_state)
        elif model_type == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }
            model = RandomForestRegressor(
                **params, random_state=CONFIG.random_state, n_jobs=-1
            )
        elif model_type == "xgb" and XGB_AVAILABLE and xgb is not None:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "eval_metric": "mae",
                "tree_method": "hist",
            }
            model = xgb.XGBRegressor(
                **params, random_state=CONFIG.random_state, n_jobs=-1
            )
        elif model_type == "lgb" and LGB_AVAILABLE and lgb is not None:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
            model = lgb.LGBMRegressor(
                **params, random_state=CONFIG.random_state, n_jobs=-1, verbose=-1
            )
        else:
            return float("inf")

        scores = []
        for train_idx, val_idx in tscv.split(X_tune):
            X_tr, X_va = X_tune[train_idx], X_tune[val_idx]
            y_tr, y_va = y_tune[train_idx], y_tune[val_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            scores.append(mean_absolute_error(y_va, preds))

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=CONFIG.random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# =============================================================================
# WEIGHTED ENSEMBLE (Proven Method from Seasonal V1)
# =============================================================================

class WeightedEnsemble(BaseEstimator, RegressorMixin):
    """Simple weighted ensemble - proven to work better than stacking."""

    def __init__(
        self,
        ridge_model: Any,
        lgb_model: Any,
        ridge_weight: float = 0.4,
    ):
        self.ridge_model = ridge_model
        self.lgb_model = lgb_model
        self.ridge_weight = ridge_weight
        self.lgb_weight = 1.0 - ridge_weight

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WeightedEnsemble":
        # Models are already fitted
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        ridge_pred = self.ridge_model.predict(X)
        lgb_pred = self.lgb_model.predict(X)
        return self.ridge_weight * ridge_pred + self.lgb_weight * lgb_pred


# =============================================================================
# STACKING ENSEMBLE (Option C - Currently Disabled)
# =============================================================================

class StackingEnsemble(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        base_models: Dict[str, Any],
        meta_model: Optional[Any] = None,
        n_folds: int = 3,
    ):
        self.base_models = base_models
        self.meta_model = meta_model or Ridge(alpha=1.0)
        self.n_folds = n_folds
        self.fitted_base_models_: Dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingEnsemble":
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        oof_preds = np.zeros((n_samples, n_models))

        tscv = TimeSeriesSplit(n_splits=self.n_folds)
        for model_idx, (name, model) in enumerate(self.base_models.items()):
            for train_idx, val_idx in tscv.split(X):
                model_clone = clone(model)
                model_clone.fit(X[train_idx], y[train_idx])
                oof_preds[val_idx, model_idx] = model_clone.predict(X[val_idx])

            final_model = clone(model)
            final_model.fit(X, y)
            self.fitted_base_models_[name] = final_model

        self.meta_model.fit(oof_preds, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        meta_features = np.column_stack([
            model.predict(X) for model in self.fitted_base_models_.values()
        ])
        return self.meta_model.predict(meta_features)


# =============================================================================
# PREDICTION INTERVALS (Option D)
# =============================================================================

class QuantileIntervalModel:
    def __init__(self, quantiles: Tuple[float, float, float]):
        self.quantiles = quantiles
        self.models: Dict[float, QuantileRegressor] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileIntervalModel":
        for q in self.quantiles:
            model = QuantileRegressor(quantile=q, alpha=1.0, solver="highs")
            model.fit(X, y)
            self.models[q] = model
        return self

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        outputs: Dict[str, np.ndarray] = {}
        for q, model in self.models.items():
            key = f"q{int(q * 100)}"
            outputs[key] = model.predict(X)
        return outputs


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class ModelMetrics:
    model_name: str
    mae: float
    rmse: float
    r2: float
    mape: float
    smape: float
    directional_accuracy: float
    peak_mae: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _calculate_baseline_metrics(
    df: pd.DataFrame,
    season: str,
) -> Optional[ModelMetrics]:
    """Calculate baseline metrics using manual scheduling (schedule_power)."""
    if TARGET not in df.columns or "schedule_power" not in df.columns:
        return None

    y_true = df[TARGET].values
    y_pred = df["schedule_power"].values

    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return None

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    return _calculate_metrics(y_true, y_pred, "baseline_manual_schedule", timestamps=df["date"][mask] if "date" in df.columns else None)


def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    timestamps: Optional[pd.Series] = None,
) -> ModelMetrics:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.where(denom == 0, 1, denom)
    smape = np.mean(np.abs(y_true - y_pred) / denom) * 100

    if len(y_true) > 1:
        actual_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = (actual_direction == pred_direction).mean() * 100
    else:
        directional_accuracy = np.nan

    if timestamps is not None:
        ts = pd.to_datetime(timestamps)
        peak_mask = (ts.dt.hour >= 10) & (ts.dt.hour <= 16)
        peak_mae = mean_absolute_error(y_true[peak_mask], y_pred[peak_mask]) if peak_mask.any() else mae
    else:
        peak_mae = mae

    return ModelMetrics(
        model_name=model_name,
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        r2=round(r2, 4),
        mape=round(mape, 2) if not np.isnan(mape) else -1,
        smape=round(smape, 2),
        directional_accuracy=round(directional_accuracy, 2) if not np.isnan(directional_accuracy) else -1,
        peak_mae=round(peak_mae, 4),
    )


def _fsp_selection_accuracy(
    df: pd.DataFrame,
    selected_fsps: np.ndarray,
) -> Tuple[float, float]:
    fsp_cols = _get_fsp_cols(df)
    if not fsp_cols:
        return 0.0, 0.0

    oracle_matches = 0
    ml_errors = []
    baseline_errors = []

    if "schedule_power" in df.columns:
        baseline_error = np.abs(df[TARGET].values - df["schedule_power"].values)
    else:
        baseline_error = np.full(len(df), np.inf)

    for i in range(len(df)):
        actual = df.iloc[i][TARGET]
        fsp_vals = {}
        for col in fsp_cols:
            val = df.iloc[i].get(col, np.nan)
            if not np.isnan(val):
                fsp_vals[col.replace("forecast_power_", "").upper()] = val
        if not fsp_vals:
            continue
        errors = {fsp: abs(val - actual) for fsp, val in fsp_vals.items()}
        oracle_fsp = min(errors.items(), key=lambda item: item[1])[0]
        chosen = selected_fsps[i]
        if chosen in fsp_vals:
            ml_errors.append(abs(fsp_vals[chosen] - actual))
            if chosen == oracle_fsp:
                oracle_matches += 1
        baseline_errors.append(baseline_error[i])

    oracle_match_pct = (oracle_matches / len(df)) * 100 if len(df) else 0.0

    ml_errors = np.array([e for e in ml_errors if not np.isnan(e)])
    baseline_errors = np.array([e for e in baseline_errors if not np.isnan(e) and not np.isinf(e)])
    if ml_errors.size and baseline_errors.size:
        improvement = ((baseline_errors.mean() - ml_errors.mean()) / baseline_errors.mean()) * 100
    else:
        improvement = 0.0

    return round(oracle_match_pct, 2), round(improvement, 2)


# =============================================================================
# TRAINING
# =============================================================================

def _select_best_fsp(df: pd.DataFrame, preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fsp_cols = _get_fsp_cols(df)
    selected = []
    schedule = []
    confidence = []

    for i in range(len(df)):
        pred = preds[i]
        fsp_vals = {}
        for col in fsp_cols:
            val = df.iloc[i].get(col, np.nan)
            if pd.notna(val):
                fsp_vals[col.replace("forecast_power_", "").upper()] = val
        if not fsp_vals or np.isnan(pred):
            selected.append("UNKNOWN")
            schedule.append(np.nan)
            confidence.append(0.0)
            continue
        errors = {fsp: abs(val - pred) for fsp, val in fsp_vals.items()}
        best_fsp = min(errors.items(), key=lambda item: item[1])[0]
        selected.append(best_fsp)
        schedule.append(fsp_vals[best_fsp])
        spread = max(errors.values()) - min(errors.values())
        conf = spread / (max(errors.values()) + 1e-8) if max(errors.values()) > 0 else 0.5
        confidence.append(min(conf, 1.0))

    return np.array(selected), np.array(schedule), np.array(confidence)


def _save_predictions(
    df: pd.DataFrame,
    preds: np.ndarray,
    selected: np.ndarray,
    schedule: np.ndarray,
    confidence: np.ndarray,
    model_name: str,
    output_path: Path,
    season: str,
    intervals: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> None:
    out = df.copy()
    out["ml_predicted_power"] = preds
    out["ml_selected_fsp"] = selected
    out["ml_schedule_power"] = schedule
    out["ml_confidence"] = confidence
    if intervals is not None:
        lower, median, upper = intervals
        if lower is not None:
            out["ml_pred_lower_5"] = lower
        if upper is not None:
            out["ml_pred_upper_95"] = upper
    if TARGET in out.columns:
        out["ml_error"] = np.abs(out[TARGET] - out["ml_predicted_power"])
        out["ml_schedule_error"] = np.abs(out[TARGET] - out["ml_schedule_power"])
    out["season"] = season
    out["model_name"] = model_name
    out["model_version"] = CONFIG.model_version
    out.to_csv(output_path, index=False)


def _train_season(season: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    _print_section(f"Training {season.upper()} season")

    train_df, val_df, test_df = _seasonal_split(
        df,
        season,
        CONFIG.train_year,
        CONFIG.val_test_year,
        CONFIG.val_ratio,
    )

    for split_df in [train_df, val_df, test_df]:
        split_df[TARGET_HORIZON] = split_df[TARGET].shift(-CONFIG.prediction_horizon)

    train_df = train_df.dropna(subset=[TARGET_HORIZON]).reset_index(drop=True)
    val_df = val_df.dropna(subset=[TARGET_HORIZON]).reset_index(drop=True)
    test_df = test_df.dropna(subset=[TARGET_HORIZON]).reset_index(drop=True)

    if len(train_df) < 100 or len(val_df) < 50 or len(test_df) < 50:
        print(" Insufficient samples after target creation; skipping.")
        return None

    train_df = _add_time_features(train_df)
    val_df = _add_time_features(val_df)
    test_df = _add_time_features(test_df)

    train_df = _add_lag_features(train_df, [1, 2, 6])
    val_df = _add_lag_features(val_df, [1, 2, 6])
    test_df = _add_lag_features(test_df, [1, 2, 6])

    exclude = ["date", "timestamp", TARGET, TARGET_HORIZON]
    train_df, [val_df, test_df], _ = _encode_categoricals(train_df, [val_df, test_df], exclude)

    feature_cols = _get_feature_columns(train_df)
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_HORIZON].values
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_HORIZON].values
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_HORIZON].values

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    selected_features = _select_features(
        X_train,
        y_train,
        feature_cols,
        CONFIG.feature_importance_threshold,
    )

    selected_idx = [feature_cols.index(c) for c in selected_features]
    X_train = X_train[:, selected_idx]
    X_val = X_val[:, selected_idx]
    X_test = X_test[:, selected_idx]
    feature_cols = selected_features

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    with open(MODELS_DIR / f"scaler_{season}.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODELS_DIR / f"imputer_{season}.pkl", "wb") as f:
        pickle.dump(imputer, f)

    _print_subsection("Hyperparameter tuning")
    params: Dict[str, Any] = {}
    params["ridge"] = _tune_params(X_train_scaled, y_train, "ridge", CONFIG.n_optuna_trials, CONFIG.n_cv_folds)
    params["rf"] = _tune_params(X_train_scaled, y_train, "rf", CONFIG.n_optuna_trials, CONFIG.n_cv_folds)
    if XGB_AVAILABLE and xgb is not None:
        params["xgb"] = _tune_params(X_train_scaled, y_train, "xgb", CONFIG.n_optuna_trials, CONFIG.n_cv_folds)
    if LGB_AVAILABLE and lgb is not None:
        params["lgb"] = _tune_params(X_train_scaled, y_train, "lgb", CONFIG.n_optuna_trials, CONFIG.n_cv_folds)

    _print_subsection("Model training")
    models: Dict[str, Any] = {}
    predictions: Dict[str, Dict[str, np.ndarray]] = {}

    ridge = Ridge(**params["ridge"], random_state=CONFIG.random_state)
    ridge.fit(X_train_scaled, y_train)
    models["ridge"] = ridge
    predictions["ridge"] = {
        "val": ridge.predict(X_val_scaled),
        "test": ridge.predict(X_test_scaled),
    }

    rf = RandomForestRegressor(
        **params["rf"],
        random_state=CONFIG.random_state,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)
    models["random_forest"] = rf
    predictions["random_forest"] = {
        "val": rf.predict(X_val_scaled),
        "test": rf.predict(X_test_scaled),
    }

    if XGB_AVAILABLE and xgb is not None:
        xgb_params = params["xgb"].copy()
        xgb_params.setdefault("eval_metric", "mae")
        xgb_params.setdefault("tree_method", "hist")
        xgb_model = xgb.XGBRegressor(
            **xgb_params,
            random_state=CONFIG.random_state,
            n_jobs=-1,
        )
        xgb_model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False,
        )
        models["xgboost"] = xgb_model
        predictions["xgboost"] = {
            "val": xgb_model.predict(X_val_scaled),
            "test": xgb_model.predict(X_test_scaled),
        }

    if LGB_AVAILABLE and lgb is not None:
        lgb_model = lgb.LGBMRegressor(
            **params["lgb"],
            random_state=CONFIG.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_val_scaled, y_val)],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        models["lightgbm"] = lgb_model
        predictions["lightgbm"] = {
            "val": lgb_model.predict(X_val_scaled),
            "test": lgb_model.predict(X_test_scaled),
        }

    # Weighted Ensemble: Ridge + LightGBM (proven to work better than stacking)
    if CONFIG.use_weighted_ensemble and LGB_AVAILABLE and lgb is not None:
        weighted_ensemble = WeightedEnsemble(
            ridge_model=ridge,
            lgb_model=lgb_model,
            ridge_weight=CONFIG.ridge_weight,
        )
        weighted_ensemble.fit(X_train_scaled, y_train)
        models["weighted_ensemble"] = weighted_ensemble
        predictions["weighted_ensemble"] = {
            "val": weighted_ensemble.predict(X_val_scaled),
            "test": weighted_ensemble.predict(X_test_scaled),
        }

    # Stacking Ensemble (disabled by default - weighted ensemble performs better)
    if CONFIG.use_stacking and len(models) >= 2:
        base_models = {}
        base_models["ridge"] = Ridge(**params["ridge"], random_state=CONFIG.random_state)
        base_models["rf"] = RandomForestRegressor(**params["rf"], random_state=CONFIG.random_state, n_jobs=-1)
        if XGB_AVAILABLE and xgb is not None:
            base_models["xgb"] = xgb.XGBRegressor(**params["xgb"], random_state=CONFIG.random_state, n_jobs=-1)
        if LGB_AVAILABLE and lgb is not None:
            base_models["lgb"] = lgb.LGBMRegressor(**params["lgb"], random_state=CONFIG.random_state, n_jobs=-1, verbose=-1)

        stacking = StackingEnsemble(base_models=base_models, meta_model=Ridge(alpha=1.0), n_folds=CONFIG.n_cv_folds)
        stacking.fit(X_train_scaled, y_train)
        models["stacking_ensemble"] = stacking
        predictions["stacking_ensemble"] = {
            "val": stacking.predict(X_val_scaled),
            "test": stacking.predict(X_test_scaled),
        }

    _print_subsection("Prediction intervals")
    intervals = None
    try:
        interval_model = QuantileIntervalModel(CONFIG.quantiles)
        interval_model.fit(X_train_scaled, y_train)
        q_preds = interval_model.predict(X_test_scaled)
        intervals = (
            q_preds.get("q5"),
            q_preds.get("q50"),
            q_preds.get("q95"),
        )
        models["quantile_interval"] = interval_model
    except Exception as exc:
        print(f" Prediction intervals skipped: {exc}")

    _print_subsection("Validation metrics")
    val_metrics = [
        _calculate_metrics(y_val, preds["val"], name, timestamps=val_df["date"])
        for name, preds in predictions.items()
    ]

    # Add baseline metrics for comparison
    baseline_val = _calculate_baseline_metrics(val_df, season)
    if baseline_val:
        val_metrics.append(baseline_val)

    val_df_metrics = pd.DataFrame([m.to_dict() for m in val_metrics]).sort_values("mae")
    print(val_df_metrics.to_string(index=False))

    best_model = val_df_metrics.iloc[0]["model_name"]

    _print_subsection("Test metrics")
    test_metrics = [
        _calculate_metrics(y_test, preds["test"], name, timestamps=test_df["date"])
        for name, preds in predictions.items()
    ]

    # Add baseline metrics for comparison
    baseline_test = _calculate_baseline_metrics(test_df, season)
    if baseline_test:
        test_metrics.append(baseline_test)

    test_df_metrics = pd.DataFrame([m.to_dict() for m in test_metrics]).sort_values("mae")
    print(test_df_metrics.to_string(index=False))

    best_test_pred = predictions[best_model]["test"]
    selected_fsps, schedule_power, confidence = _select_best_fsp(test_df, best_test_pred)
    oracle_match, improvement = _fsp_selection_accuracy(test_df, selected_fsps)

    _print_subsection("Saving outputs")
    season_dir = PREDS_DIR / season
    season_dir.mkdir(parents=True, exist_ok=True)

    for model_name, preds in predictions.items():
        val_selected, val_schedule, val_conf = _select_best_fsp(val_df, preds["val"])
        _save_predictions(
            val_df,
            preds["val"],
            val_selected,
            val_schedule,
            val_conf,
            model_name,
            season_dir / f"val_predictions_{model_name}.csv",
            season,
            intervals=None,
        )

        test_selected, test_schedule, test_conf = _select_best_fsp(test_df, preds["test"])
        _save_predictions(
            test_df,
            preds["test"],
            test_selected,
            test_schedule,
            test_conf,
            model_name,
            season_dir / f"test_predictions_{model_name}.csv",
            season,
            intervals=intervals if model_name == best_model else None,
        )

    val_set = val_df.copy()
    val_set["target"] = y_val
    val_set.to_csv(season_dir / "val_set.csv", index=False)

    test_set = test_df.copy()
    test_set["target"] = y_test
    test_set.to_csv(season_dir / "test_set.csv", index=False)

    for model_name, model in models.items():
        if model_name == "quantile_interval":
            model_path = MODELS_DIR / f"quantile_interval_{season}.pkl"
        else:
            model_path = MODELS_DIR / f"{model_name}_{season}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    with open(MODELS_DIR / f"feature_columns_{season}.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    metadata = {
        "version": CONFIG.model_version,
        "git_commit": _git_commit(),
        "season": season,
        "training_date": datetime.now().isoformat(),
        "config": asdict(CONFIG),
        "data_info": {
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "train_period": {
                "start": str(train_df["date"].min()),
                "end": str(train_df["date"].max()),
            },
            "val_period": {
                "start": str(val_df["date"].min()),
                "end": str(val_df["date"].max()),
            },
            "test_period": {
                "start": str(test_df["date"].min()),
                "end": str(test_df["date"].max()),
            },
        },
        "features": {
            "count": len(feature_cols),
            "names": feature_cols,
        },
        "best_model": best_model,
        "hyperparameters": params,
        "val_metrics": [m.to_dict() for m in val_metrics],
        "test_metrics": [m.to_dict() for m in test_metrics],
        "fsp_selection": {
            "oracle_match_pct": oracle_match,
            "improvement_vs_baseline_pct": improvement,
        },
    }

    with open(MODELS_DIR / f"model_metadata_{season}.json", "w") as f:
        json.dump(metadata, f, indent=2, default=_json_safe)

    return {
        "season": season,
        "best_model": best_model,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "fsp_accuracy": (oracle_match, improvement),
    }


def main() -> None:
    _print_section("Seasonal FSP Selection Training V3")
    print("Sample Plant Wind Plant - ML scheduling optimization")
    print(f"Version: {CONFIG.model_version}")
    print(f"Git: {_git_commit()}")

    data_path = DATA_DIR / "sample_pss_dataset.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")

    _print_section("Loading data")
    df = pd.read_parquet(data_path)
    df = _ensure_datetime(df)
    print(f"Loaded rows: {len(df):,}")

    if "forecast_facode" in df.columns:
        _print_subsection("Pivoting FSP data")
        df = _pivot_fsp_data(df)
    else:
        print("Data already pivoted.")

    _print_subsection("Cleaning data")
    df = _clean_data(df)
    df = _ensure_datetime(df)

    results = []
    for season in SEASONS.keys():
        try:
            res = _train_season(season, df)
            if res:
                results.append(res)
        except Exception as exc:
            print(f" {season} skipped: {exc}")

    if not results:
        print("No seasons completed.")
        return

    _print_section("Cross-season summary")
    summary = []
    for res in results:
        season = res["season"]
        best = res["best_model"]
        best_metrics = [m for m in res["test_metrics"] if m.model_name == best][0]

        # Find baseline metrics for comparison
        baseline_metrics = [m for m in res["test_metrics"] if m.model_name == "baseline_manual_schedule"]
        baseline_mae = baseline_metrics[0].mae if baseline_metrics else None

        summary_row = {
            "Season": season.upper(),
            "Best Model": best,
            "Test MAE": best_metrics.mae,
            "Test RMSE": best_metrics.rmse,
            "Test R^2": best_metrics.r2,
            "Test sMAPE": f"{best_metrics.smape:.1f}%",
            "FSP Oracle Match": f"{res['fsp_accuracy'][0]:.1f}%",
            "Improvement vs Baseline": f"{res['fsp_accuracy'][1]:.1f}%",
        }

        if baseline_mae:
            improvement_pct = ((baseline_mae - best_metrics.mae) / baseline_mae) * 100
            summary_row["MAE vs Manual Schedule"] = f"{improvement_pct:.1f}%"

        summary.append(summary_row)

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    summary_path = REPORTS_DIR / "seasonal_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")

    avg_improvement = np.mean([r["fsp_accuracy"][1] for r in results])

    _print_section("Real-world impact")
    print("Sample Plant Wind Plant scheduling impact")
    print(f"Average FSP selection improvement: {avg_improvement:.1f}%")

    # Calculate average improvement vs manual schedule
    manual_improvements = []
    for res in results:
        best_metrics = [m for m in res["test_metrics"] if m.model_name == res["best_model"]][0]
        baseline_metrics = [m for m in res["test_metrics"] if m.model_name == "baseline_manual_schedule"]
        if baseline_metrics:
            improvement = ((baseline_metrics[0].mae - best_metrics.mae) / baseline_metrics[0].mae) * 100
            manual_improvements.append(improvement)

    if manual_improvements:
        avg_manual_improvement = np.mean(manual_improvements)
        print(f"Average improvement vs manual scheduling: {avg_manual_improvement:.1f}%")

    print(f"Models: {MODELS_DIR}")
    print(f"Predictions: {PREDS_DIR}")
    print(f"Reports: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
