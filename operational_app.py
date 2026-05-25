"""
GKFS Operational Dashboard - Automated Training & Predictions
============================================================

Automated Streamlit app that:
- Loads 18 months of Sample Plant data
- Performs 70-15-15 temporal split
- Trains Ridge-LightGBM ensemble model
- Displays predictions with quantile graphs and all visualizations

Maintainer: Project Team
Date: February 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import date as date_type
import pickle
import sys
import io
import zipfile
import json
import re
from typing import Dict, Tuple, Optional, List, Any
from scipy.stats import norm
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
from app.pages.predictions_viz import (
    select_best_fsp_by_prediction, create_prediction_dataframe,
    build_fsp_palette, render_fsp_color_legend, get_fsp_color_map
)


def _ensure_imputer_compat(imputer):
    """Patch SimpleImputer loaded from pickle with older sklearn (missing _fill_dtype)."""
    if imputer is None:
        return
    if isinstance(imputer, SimpleImputer) and not hasattr(imputer, '_fill_dtype'):
        if hasattr(imputer, 'statistics_') and imputer.statistics_ is not None:
            imputer._fill_dtype = np.asarray(imputer.statistics_).dtype
        else:
            imputer._fill_dtype = np.float64


# Page configuration
st.set_page_config(
    page_title="GKFS Operational Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Light mode only
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

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
SAVED_MODELS_DIR = PROJECT_DIR / "outputs" / "saved_models"
MODEL_SAVESSS_DIR = PROJECT_DIR / "model_savesss"

# Plant names to train models for
TARGET_PLANTS = ['Plant Alpha', 'Plant Beta', 'Plant Gamma', 'Sample Plant', 'Plant Delta']

# DSM deviation band config (15-min block = 0.25 h for MWh)
DEVIATION_BAND_CONFIG_PATH = PROJECT_DIR / "deviation_band_configs.json"
BLOCK_HOURS = 0.25 # 15-min block in hours (for penalty per MWh deviation)

# --- AI insights: dynamic commentary on the current view (like streamlit_app / predictions_viz) ---

def _insight_daily_quantile(**kwargs):
    plot_df = kwargs.get("plot_df")
    date = kwargs.get("date")
    if plot_df is None or getattr(plot_df, "empty", True):
        return "- Select a date to see day-specific commentary.\n- Quantile bands (F10F90) show forecast uncertainty.\n- Ribbons show 50% and 80% confidence intervals; markers show FSP selection per block."
    actual = plot_df.get("actual_power")
    ml_sched = plot_df.get("ml_scheduled_power")
    date_str = str(date) if date else "this day"
    if actual is None or ml_sched is None:
        return f"- **{date_str}:** Quantile bands show uncertainty around ML Scheduled.\n- F10F90 = 80% confidence interval; F25F75 = 50%.\n- Dots show which FSP was selected per 15-min block."
    actual_arr = actual.values if hasattr(actual, "values") else actual
    ml_arr = ml_sched.values if hasattr(ml_sched, "values") else ml_sched
    peak = float(np.nanmax(actual_arr))
    low = float(np.nanmin(actual_arr))
    err = np.abs(actual_arr - ml_arr)
    mae = float(np.nanmean(err))
    return (
        f"- **{date_str}:** Peak actual **{peak:.1f} MW**, range **{low:.1f}{peak:.1f} MW**.\n"
        f"- Mean absolute error (ML scheduled vs actual): **{mae:.2f} MW**.\n"
        "- Ribbons show 50% and 80% confidence intervals; markers show FSP selection per block.\n"
        "- Use bands to assess forecast uncertainty and schedule risk."
    )


def _insight_fsp_waterfall(**kwargs):
    plot_df = kwargs.get("plot_df")
    if plot_df is None or getattr(plot_df, "empty", True):
        return "- Chart shows which FSP was selected per 15-min block.\n- Waterfall encodes scheduled power and FSP choice across the day.\n- Use it to see time-of-day mix of providers."
    col = plot_df.get("ml_selected_fsp")
    if col is None:
        return "- FSP selection per block for the selected day.\n- Waterfall shows scheduled power by block.\n- Compare with actual to assess selection quality."
    counts = col.value_counts()
    top = counts.index[0] if len(counts) else ""
    n_blocks = len(plot_df)
    n_top = counts.get(top, 0)
    return (
        f"- **This day ({n_blocks} blocks):** Most selected FSP is **{top}** ({n_top} blocks).\n"
        "- Waterfall shows scheduled power and FSP choice per 15-min block.\n"
        "- Use it to see time-of-day mix and which provider dominates.\n"
        "- Compare with actual power to assess selection accuracy."
    )


def _insight_block_accuracy_table(**kwargs):
    plot_df = kwargs.get("plot_df")
    date = kwargs.get("date", "")
    if plot_df is not None and not getattr(plot_df, "empty", True):
        return f"**For {date}:** Block-wise accuracy and improvement %; positive improvement means ML is better than manual for that block."
    return "Table shows block-wise R2 and improvement % (ML vs manual) for the selected day."


def _insight_block_wise_power(**kwargs):
    plot_df = kwargs.get("plot_df")
    if plot_df is not None and not getattr(plot_df, "empty", True):
        actual = plot_df.get("actual_power")
        if actual is not None:
            a = actual.values if hasattr(actual, "values") else actual
            lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
            return (
                f"- **This day:** Actual power range **{lo:.1f}{hi:.1f} MW**.\n"
                "- Lines show ML scheduled, manual scheduled, actual, and FSP forecasts across 96 blocks.\n"
                "- Use to spot blocks where ML or manual deviates from actual.\n"
                "- Toggle FSP series in the legend to compare individual forecasts."
            )
    return "- Compares ML scheduled, manual scheduled, actual power, and FSP forecasts by block.\n- Use to spot blocks where schedules deviate from actual.\n- Toggle series in the legend to focus on specific curves."


def _insight_forecast_heatmap(**kwargs):
    plot_df = kwargs.get("plot_df")
    if plot_df is not None and not getattr(plot_df, "empty", True):
        return "Heatmap encodes forecasted MW and FSP selection by block for the selected day; patterns show time-of-day and provider mix."
    return "Heatmap shows forecast (MW) and scheduled FSP by 15-min block."


def _insight_fsp_selection(**kwargs):
    pred_df = kwargs.get("pred_df")
    if pred_df is None or getattr(pred_df, "empty", True):
        return "Selection confidence and per-FSP performance when that FSP was chosen."
    conf = pred_df.get("selection_confidence")
    if conf is not None:
        c = conf.values if hasattr(conf, "values") else conf
        return f"**Test set:** Mean selection confidence **{float(np.nanmean(c)):.3f}** (n={len(pred_df):,} blocks). Charts show distribution and average error by FSP."
    return "Distribution of selection confidence and average error per selected FSP."


def _insight_selection_confidence_dist(**kwargs):
    pred_df = kwargs.get("pred_df")
    if pred_df is not None and not getattr(pred_df, "empty", True):
        conf = pred_df.get("selection_confidence")
        if conf is not None:
            c = conf.values if hasattr(conf, "values") else conf
            return f"**This run:** Mean **{float(np.nanmean(c)):.3f}**, std **{float(np.nanstd(c)):.3f}**. Higher values = more confident FSP choice per block."
    return "Selection confidence = model confidence in the chosen FSP per block."


def _insight_fsp_performance_after(**kwargs):
    pred_df = kwargs.get("pred_df")
    if pred_df is None or getattr(pred_df, "empty", True):
        return "Average error (MW) when each FSP was selected; fewer selections with low error is better."
    return "**Test set:** Per-FSP average error and selection count; use to see which FSP performed best when selected."


def _insight_test_set_aggregates(**kwargs):
    pred_df = kwargs.get("pred_df")
    if pred_df is None or getattr(pred_df, "empty", True):
        return "Totals and differences for actual, ML predicted, ML scheduled, and manual scheduled."
    total_actual = pred_df["actual_power"].sum()
    total_ml_sched = pred_df["ml_scheduled_power"].sum()
    total_manual = pred_df.get("manual_scheduled_power")
    total_manual = total_manual.sum() if total_manual is not None and hasattr(total_manual, "sum") else None
    diff_ml = total_ml_sched - total_actual
    pct_ml = (total_ml_sched / total_actual * 100) if total_actual > 0 else 0
    line = f"**Test set:** Total actual **{total_actual:,.0f} MW**; ML scheduled **{total_ml_sched:,.0f} MW** ({pct_ml:.1f}%), difference **{diff_ml:+,.0f} MW**. "
    if total_manual is not None and not (hasattr(total_manual, "__iter__") and np.any(np.isnan(total_manual))):
        try:
            tm = float(total_manual)
            line += f"Manual scheduled **{tm:,.0f} MW**."
        except Exception:
            pass
    return line


def _insight_aggregate_power_summary(**kwargs):
    pred_df = kwargs.get("pred_df")
    if pred_df is None or getattr(pred_df, "empty", True):
        return "Table: total actual, ML/manual scheduled, % of actual, and difference from actual."
    total_actual = pred_df["actual_power"].sum()
    total_ml_sched = pred_df["ml_scheduled_power"].sum()
    diff = total_ml_sched - total_actual
    return f"**This test set:** ML scheduled sum **{total_ml_sched:,.0f} MW** (diff from actual **{diff:+,.0f} MW**). Table shows all four totals and differences."


def _insight_performance_metrics_r2(**kwargs):
    pred_df = kwargs.get("pred_df")
    accuracy_ml = kwargs.get("accuracy_ml")
    accuracy_manual = kwargs.get("accuracy_manual")
    improvement = kwargs.get("improvement")
    better_pct = kwargs.get("better_pct")
    if pred_df is not None and accuracy_ml is not None and accuracy_manual is not None:
        imp = improvement if improvement is not None and not (isinstance(improvement, float) and np.isinf(improvement)) else ((accuracy_ml - accuracy_manual) / abs(accuracy_manual) * 100) if accuracy_manual != 0 else 0
        bp = better_pct
        if bp is None and "ml_scheduled_error" in pred_df.columns and "manual_error" in pred_df.columns and len(pred_df):
            bp = (pred_df["ml_scheduled_error"] < pred_df["manual_error"]).sum() / len(pred_df) * 100
        elif bp is None:
            bp = 0
        return (
            f"**Test set:** ML scheduled R2 **{accuracy_ml:.4f}**, manual R2 **{accuracy_manual:.4f}**. "
            f"ML improvement **{imp:.2f}%**; ML better than manual in **{float(bp):.1f}%** of blocks."
        )
    return "R2 = fit between actual and scheduled power; improvement % = (ML R2  manual R2) / |manual R2|  100."


def _insight_error_analysis(**kwargs):
    pred_df = kwargs.get("pred_df")
    if pred_df is None or getattr(pred_df, "empty", True):
        return "ML error = actual  ML scheduled; manual error = actual  manual scheduled (MW)."
    ml_err = pred_df.get("ml_error") or pred_df.get("ml_scheduled_error")
    man_err = pred_df.get("manual_error")
    if ml_err is not None:
        ml_arr = ml_err.values if hasattr(ml_err, "values") else ml_err
        mean_ml = float(np.nanmean(np.abs(ml_arr)))
        line = f"**This run:** Mean |ML error| **{mean_ml:.2f} MW**. "
        if man_err is not None:
            man_arr = man_err.values if hasattr(man_err, "values") else man_err
            mean_man = float(np.nanmean(np.abs(man_arr)))
            better = (np.abs(ml_arr) < np.abs(man_arr)).sum()
            pct = better / len(ml_arr) * 100 if len(ml_arr) else 0
            line += f"Mean |manual error| **{mean_man:.2f} MW**. ML better in **{pct:.1f}%** of blocks. Points below the diagonal = ML better."
        return line
    return "Distributions and scatter show ML vs manual error per block."


def _insight_ml_error_dist(**kwargs):
    pred_df = kwargs.get("pred_df")
    if pred_df is not None and not getattr(pred_df, "empty", True):
        col = pred_df.get("ml_error") or pred_df.get("ml_scheduled_error")
        if col is not None:
            c = col.values if hasattr(col, "values") else col
            return f"**This run:** Mean ML error **{float(np.nanmean(c)):.2f} MW** (positive = under-schedule). Distribution over {len(pred_df):,} blocks."
    return "ML error = actual  ML scheduled (MW) per block."


def _insight_manual_error_dist(**kwargs):
    pred_df = kwargs.get("pred_df")
    if pred_df is not None and not getattr(pred_df, "empty", True) and "manual_error" in pred_df.columns:
        c = pred_df["manual_error"].values
        return f"**This run:** Mean manual error **{float(np.nanmean(c)):.2f} MW**. Compare with ML distribution to see which schedule is closer to actual."
    return "Manual error = actual  manual scheduled (MW) per block."


def _insight_ml_vs_manual_error(**kwargs):
    pred_df = kwargs.get("pred_df")
    if pred_df is None or getattr(pred_df, "empty", True):
        return "Each point = one block; below diagonal = ML better than manual."
    ml_err = pred_df.get("ml_error") or pred_df.get("ml_scheduled_error")
    man_err = pred_df.get("manual_error")
    if ml_err is not None and man_err is not None:
        ml_arr = ml_err.values if hasattr(ml_err, "values") else ml_err
        man_arr = man_err.values if hasattr(man_err, "values") else man_err
        below = (ml_arr < man_arr).sum()
        pct = below / len(ml_arr) * 100 if len(ml_arr) else 0
        return f"**This run:** **{pct:.1f}%** of blocks below the diagonal (ML error < manual error). Color = selection confidence."
    return "Scatter: X = manual error, Y = ML error; below diagonal = ML better."


def _insight_block_wise_analysis(**kwargs):
    pred_df = kwargs.get("pred_df")
    block_metrics = kwargs.get("block_metrics")
    if pred_df is not None and block_metrics is not None and not getattr(block_metrics, "empty", True):
        acc_ml = block_metrics.get("accuracy_ml_scheduled") or block_metrics.get("accuracy_ml")
        if acc_ml is not None:
            best_block = int(acc_ml.idxmax()) + 1 if hasattr(acc_ml, "idxmax") else None
            return f"**Test set:** Block-wise R2 and improvement %. Best ML block (by R2) **{best_block}**. Improvement % shows where ML beats manual by time of day."
    return "Block-wise R2 and improvement % (ML vs manual) for each of 96 blocks."


def _insight_overall_day_accuracy(**kwargs):
    pred_df = kwargs.get("pred_df")
    if pred_df is not None and not getattr(pred_df, "empty", True):
        acc_ml = calculate_accuracy(pred_df["actual_power"].values, pred_df["ml_scheduled_power"].values)
        acc_man = None
        if "manual_scheduled_power" in pred_df.columns:
            acc_man = calculate_accuracy(pred_df["actual_power"].values, pred_df["manual_scheduled_power"].values)
        imp = ((acc_ml - acc_man) / abs(acc_man) * 100) if acc_man is not None and acc_man != 0 else 0
        line = f"**Overall test set:** ML scheduled R2 **{acc_ml:.4f}**"
        if acc_man is not None:
            line += f", manual R2 **{acc_man:.4f}**, improvement **{imp:.2f}%**."
        else:
            line += "."
        return line
    return "Single R2 and improvement over the entire test set."


def _insight_block_wise_summary(**kwargs):
    block_metrics = kwargs.get("block_metrics")
    if block_metrics is not None and not getattr(block_metrics, "empty", True):
        acc = block_metrics.get("accuracy_ml_scheduled") or block_metrics.get("accuracy_ml")
        imp = block_metrics.get("improvement_pct")
        if acc is not None:
            return f"**This run:** Avg ML block R2 **{acc.mean():.4f}**, max **{acc.max():.4f}**. " + (f"Avg improvement **{imp.mean():.2f}%**." if imp is not None else "")
    return "Averages and min/max of block-wise accuracy and improvement %."


def _insight_dsm_penalty(**kwargs):
    total_penalty_ml = kwargs.get("total_penalty_ml")
    total_penalty_manual = kwargs.get("total_penalty_manual")
    penalty_reduction_rs = kwargs.get("penalty_reduction_rs")
    penalty_reduction_pct = kwargs.get("penalty_reduction_pct")
    date = kwargs.get("date")
    if total_penalty_ml is not None and date is not None:
        date_str = str(date)
        lines = [f"- **{date_str}:** ML scheduled penalty ** {total_penalty_ml:,.2f}**."]
        if total_penalty_manual is not None and penalty_reduction_rs is not None and penalty_reduction_pct is not None:
            lines.append(f"- Manual penalty ** {total_penalty_manual:,.2f}**; ML reduces by ** {penalty_reduction_rs:,.2f}** ({penalty_reduction_pct:.1f}%).")
        else:
            lines.append("- Add manual schedule to compare penalty reduction.")
        lines.append("- Penalty = band rate  |actual  scheduled| (MW)  0.25 h per block.")
        lines.append("- Lower ML penalty means the schedule stays closer to actual within bands.")
        return "\n".join(lines)
    return "- DSM penalty = band rate  |actual  scheduled| (MW)  0.25 h per 15-min block.\n- Compares ML vs manual for the selected day.\n- Lower penalty indicates better alignment with actual generation."


def _insight_within_band(**kwargs):
    n_within_ml = kwargs.get("n_within_ml")
    pct_within_ml = kwargs.get("pct_within_ml")
    n_within_manual = kwargs.get("n_within_manual")
    pct_within_manual = kwargs.get("pct_within_manual")
    if n_within_ml is not None and pct_within_ml is not None:
        line = f"**This day:** ML scheduled **{n_within_ml}/96** blocks within band (**{pct_within_ml:.1f}%**). "
        if n_within_manual is not None and pct_within_manual is not None:
            line += f"Manual **{n_within_manual}/96** (**{pct_within_manual:.1f}%**). Higher % = fewer penalty blocks."
        else:
            line += "Compare with manual when available."
        return line
    return "Within band = blocks with zero penalty; higher % is better."


def _insight_nov_dec_penalty(**kwargs):
    monthly = kwargs.get("monthly_penalty_df")
    if monthly is not None and not getattr(monthly, "empty", True):
        rows = monthly.to_dict("records") if hasattr(monthly, "to_dict") else []
        parts = []
        for r in rows:
            m = r.get("month_name", "")
            pm = r.get("penalty_ml")
            pman = r.get("penalty_manual")
            if pm is not None:
                parts.append(f"{m}: ML  {pm:,.0f}" + (f", manual  {pman:,.0f}" if pman is not None else ""))
        return "**Monthly (NovDec):** " + "; ".join(parts) + "." if parts else "Monthly DSM penalty: ML vs manual for November and December."
    return "Monthly total DSM penalty (Rs) for November and December: ML vs manual."


# AI insights only for: Daily Quantile Forecast, FSP Waterfall, Block-wise Power, DSM Penalty Comparison
SECTION_INSIGHTS = {
    "daily_quantile_forecast": _insight_daily_quantile,
    "fsp_waterfall": _insight_fsp_waterfall,
    "block_wise_power": _insight_block_wise_power,
    "dsm_penalty_comparison": _insight_dsm_penalty,
}


def render_section_insight(section_key: str, **context) -> None:
    """Render an expandable AI insights box; content can be dynamic from context (e.g. that day's graph/penalty)."""
    value = SECTION_INSIGHTS.get(section_key)
    if value is None:
        return
    try:
        text = value(**context) if callable(value) else value
    except Exception:
        text = str(value) if not callable(value) else ""
    if not (text and text.strip()):
        return
    with st.expander(" **AI insights**", expanded=False):
        st.markdown(text)


def get_plant_sscode(plant_name: str) -> str:
    """Map dashboard plant name to sscode used in deviation band config."""
    plant_to_sscode = {
        'Plant Alpha': 'PLANT_ALPHA_PSS',
        'Plant Beta': 'PLANT_BETA_PSS',
        'Plant Gamma': 'PLANT_GAMMA_PSS',
        'Sample Plant': 'SAMPLE_PSS',
        'Plant Delta': 'PLANT_DELTA_PSS',
    }
    return plant_to_sscode.get(plant_name, plant_name.upper().replace(' ', '_') + '_PSS')


def _normalize_mongo_json(text: str) -> str:
    """Replace MongoDB extended JSON so standard json.loads can parse."""
    text = re.sub(r'\bNaN\b', 'null', text)
    text = re.sub(r'ObjectId\s*\(\s*"[^"]*"\s*\)', 'null', text)
    text = re.sub(r'ISODate\s*\(\s*"[^"]*"\s*\)', 'null', text)
    return text


def load_deviation_band_config(sscode: str) -> Optional[Dict[str, Any]]:
    """Load deviation band config for a plant from GKFS config JSON. Returns first matching doc with enabled=True."""
    if not DEVIATION_BAND_CONFIG_PATH.exists():
        return None
    try:
        raw = DEVIATION_BAND_CONFIG_PATH.read_text(encoding='utf-8')
        raw = _normalize_mongo_json(raw)
        # File is concatenated JSON objects: split by }\n{
        parts = raw.split('}\n{')
        for i, part in enumerate(parts):
            if not part.strip():
                continue
            if i > 0:
                part = '{' + part
            if i < len(parts) - 1:
                part = part + '}'
            try:
                doc = json.loads(part)
            except json.JSONDecodeError:
                continue
            if doc.get('sscode') == sscode and doc.get('enabled', True):
                bands = doc.get('bands')
                if bands:
                    return {'bands': bands, 'sscode': sscode}
    except Exception:
        pass
    return None


def get_band_for_deviation(deviation_pct: float, bands: List[Dict]) -> Dict[str, Any]:
    """Return band info (penalty, category, colorcode) for a deviation %; bands have 'from', 'to', 'penalty', 'category', 'colorcode'."""
    for b in bands:
        lo = b.get('from')
        hi = b.get('to')
        if lo is None:
            lo = float('-inf')
        if hi is None:
            hi = float('inf')
        if lo <= deviation_pct < hi:
            return {
                'penalty': float(b.get('penalty', 0)),
                'category': b.get('category', ''),
                'colorcode': b.get('colorcode', '#888888'),
            }
    return {'penalty': 0.0, 'category': 'UNKNOWN', 'colorcode': '#888888'}


def compute_penalty_rs(actual: np.ndarray, scheduled: np.ndarray, bands: List[Dict],
                       block_hours: float = BLOCK_HOURS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute deviation % and penalty (Rs) per block. Returns (deviation_pct, penalty_rs).
    Penalty = band penalty rate * |actual - scheduled| (MW) * block_hours (h) = Rs per block (treating unit as MWh).
    """
    actual = np.asarray(actual, dtype=float)
    scheduled = np.asarray(scheduled, dtype=float)
    eps = 1e-6
    deviation_pct = np.where(scheduled > eps, (actual - scheduled) / scheduled * 100.0, 0.0)
    penalty_rs = np.zeros_like(actual)
    for i in range(len(actual)):
        band_info = get_band_for_deviation(float(deviation_pct[i]), bands)
        rate = band_info['penalty']
        penalty_rs[i] = rate * abs(actual[i] - scheduled[i]) * block_hours
    return deviation_pct, penalty_rs


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


def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        'data_loaded': False,
        'model_trained': False,
        'model_loaded': False,
        'predictions_generated': False,
        'df_raw': None,
        'df_pivoted': None,
        'train_df': None,
        'val_df': None,
        'test_df': None,
        'model': None,
        'feature_columns': [],
        'model_feature_columns': [],  # Feature columns from loaded model
        'scaler': None,
        'imputer': None,
        'encoders': None,
        'prediction_dfs': {},
        'selected_fsps': [],
        'show_heatmap': False,
        'plant_selected': None,
        'model_stats': None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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


def discover_available_plants():
    """Discover available plants from data files."""
    plants = []

    # Check processed parquet directory
    parquet_dir = DATA_PROCESSED / 'parquet'
    if parquet_dir.exists():
        parquet_files = list(parquet_dir.glob("*_dataset.parquet"))
        if parquet_files:
            # Extract plant names from filenames
            for f in parquet_files:
                name = f.stem.replace('_dataset', '')
                # Convert to display name
                display_name = name.replace('_pss', '').replace('_ss', '').replace('_', ' ').title()
                plants.append(display_name)

    # Also check root processed directory
    if DATA_PROCESSED.exists():
        parquet_files = list(DATA_PROCESSED.glob("*_dataset.parquet"))
        if parquet_files:
            for f in parquet_files:
                name = f.stem.replace('_dataset', '')
                display_name = name.replace('_pss', '').replace('_ss', '').replace('_', ' ').title()
                if display_name not in plants:
                    plants.append(display_name)

    # Remove duplicates and sort
    plants = sorted(list(set(plants)))

    # Filter to target plants if available
    available_targets = [p for p in TARGET_PLANTS if p in plants]
    if available_targets:
        return available_targets

    # Fallback to default if nothing found
    if not plants:
        plants = ["Sample Plant"]

    return plants


def train_and_save_plant_model(plant_name: str, save_dir: Path = None):
    """Train and save ridge-lightgbm ensemble model for a specific plant.

    Parameters:
    -----------
    plant_name : str
        Name of the plant
    save_dir : Path, optional
        Directory to save the model. If None, uses SAVED_MODELS_DIR
    """
    plant_filename = get_plant_filename(plant_name)

    # Load plant data
    parquet_dir = DATA_PROCESSED / 'parquet'
    plant_file = parquet_dir / f"{plant_filename}_dataset.parquet"

    if not plant_file.exists():
        # Try root processed directory
        plant_file = DATA_PROCESSED / f"{plant_filename}_dataset.parquet"

    if not plant_file.exists():
        st.error(f"Data file not found for {plant_name}: {plant_file}")
        return False, None

    with st.spinner(f"Training model for {plant_name}..."):
        try:
            # Load data
            df_raw = pd.read_parquet(plant_file)

            # Pivot FSP data
            df_pivoted = pivot_fsp_data(df_raw)

            # Filter to last 18 months
            date_col = 'timestamp' if 'timestamp' in df_pivoted.columns else 'date'
            df_pivoted[date_col] = pd.to_datetime(df_pivoted[date_col])
            max_date = df_pivoted[date_col].max()
            cutoff = max_date - pd.DateOffset(months=DATA_MONTHS)
            df = df_pivoted[df_pivoted[date_col] >= cutoff].copy()

            # Calculate FSP errors
            fsp_cols = get_fsp_forecast_columns(df)
            df = calculate_fsp_errors(df, TARGET)

            # Drop missing data
            df_clean = df.dropna(subset=[TARGET]).copy()
            fsp_mask = df_clean[fsp_cols].notna().any(axis=1)
            df_clean = df_clean[fsp_mask].copy()

            # Create forward-shifted target
            df_clean[TARGET_HORIZON] = df_clean[TARGET].shift(-PREDICTION_HORIZON)
            df_clean = df_clean.dropna(subset=[TARGET_HORIZON])

            # Temporal split
            train_df, val_df, test_df = create_temporal_split(
                df_clean, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
            )

            # Feature engineering
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

            # Prepare X and y
            X_train = train_df[feature_cols].values
            y_train = train_df[TARGET_HORIZON].values
            X_val = val_df[feature_cols].values
            y_val = val_df[TARGET_HORIZON].values
            X_test = test_df[feature_cols].values
            y_test = test_df[TARGET_HORIZON].values

            # Imputation and scaling
            imputer = SimpleImputer(strategy='median')
            X_train = imputer.fit_transform(X_train)
            X_val = imputer.transform(X_val)
            X_test = imputer.transform(X_test)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            # Train Ridge
            ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
            ridge.fit(X_train_scaled, y_train)
            ridge_val = ridge.predict(X_val_scaled)
            ridge_test = ridge.predict(X_test_scaled)

            # Train LightGBM
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

            # Create Ensemble
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

            # Save model
            plant_upper = plant_name.upper().replace(' ', '_')
            if save_dir is None:
                model_dir = SAVED_MODELS_DIR / plant_upper / "ridge_lightgbm_ensemble" / "v1"
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

            return True, config_data

        except Exception as e:
            st.error(f"Error training model for {plant_name}: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return False, None


def save_all_target_plant_models_to_savesss():
    """Save ridge-lightgbm ensemble models for all target plants to model_savesss folder."""
    MODEL_SAVESSS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for plant in TARGET_PLANTS:
        success, stats = train_and_save_plant_model(plant, save_dir=MODEL_SAVESSS_DIR)
        results.append((plant, success, stats))

    return results


def load_and_prepare_data(plant_name: str = None, silent: bool = False):
    """Load 18 months of plant data and prepare it.
    If silent=True, no st.info/st.success messages (errors still shown).
    """
    if plant_name is None:
        plant_name = st.session_state.get('plant_selected', 'Sample Plant')

    with st.spinner(f"Loading data for {plant_name}..."):
        # Get plant filename
        plant_filename = get_plant_filename(plant_name)

        # Try to load from processed directory
        parquet_dir = DATA_PROCESSED / 'parquet'
        plant_file = parquet_dir / f"{plant_filename}_dataset.parquet"

        if not plant_file.exists():
            # Try root processed directory
            plant_file = DATA_PROCESSED / f"{plant_filename}_dataset.parquet"

        if not plant_file.exists():
            # Fallback to interim directory
            interim_file = DATA_INTERIM / 'eda_processed_data.parquet'
            if interim_file.exists():
                plant_file = interim_file
            else:
                if not silent:
                    st.error(f"Data file not found for {plant_name}. Tried: {plant_file}")
                return False

        df_raw = pd.read_parquet(plant_file)

        # Pivot FSP data
        df_pivoted = pivot_fsp_data(df_raw)

        # Filter to last 18 months
        date_col = 'timestamp' if 'timestamp' in df_pivoted.columns else 'date'
        df_pivoted[date_col] = pd.to_datetime(df_pivoted[date_col])
        max_date = df_pivoted[date_col].max()
        cutoff = max_date - pd.DateOffset(months=DATA_MONTHS)
        df = df_pivoted[df_pivoted[date_col] >= cutoff].copy()

        # Calculate FSP errors
        fsp_cols = get_fsp_forecast_columns(df)
        df = calculate_fsp_errors(df, TARGET)

        # Set selected FSPs (all available FSPs)
        fsp_names = [col.replace('forecast_power_', '').upper() for col in fsp_cols]
        st.session_state.selected_fsps = fsp_names

        # Drop missing data
        df_clean = df.dropna(subset=[TARGET]).copy()
        fsp_mask = df_clean[fsp_cols].notna().any(axis=1)
        df_clean = df_clean[fsp_mask].copy()

        # Create forward-shifted target
        df_clean[TARGET_HORIZON] = df_clean[TARGET].shift(-PREDICTION_HORIZON)
        df_clean = df_clean.dropna(subset=[TARGET_HORIZON])

        # Temporal split
        train_df, val_df, test_df = create_temporal_split(
            df_clean, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
        )

        # Apply feature engineering to identify feature columns (for display purposes)
        if not silent:
            st.info("Identifying feature columns...")
        train_df_fe = create_time_features(train_df.copy())
        train_df_fe = create_rolling_features(train_df_fe, TARGET, [1, 6, 24, 96])
        train_df_fe, encoders_fe = encode_categorical_features(
            train_df_fe, [TARGET, 'date', 'timestamp', 'sscode']
        )

        # Get feature columns
        feature_cols = get_feature_columns(train_df_fe, TARGET, EXCLUDE_PATTERNS)
        if TARGET_HORIZON in feature_cols:
            feature_cols.remove(TARGET_HORIZON)

        # Drop all-NaN features from the list
        nan_only_cols = [c for c in feature_cols if train_df_fe[c].isna().all()]
        if nan_only_cols:
            feature_cols = [c for c in feature_cols if c not in nan_only_cols]

        # Store feature columns and encoders in session state
        st.session_state.feature_columns = feature_cols
        st.session_state.encoders = encoders_fe

        # Store in session state
        st.session_state.df_raw = df_raw
        st.session_state.df_pivoted = df_pivoted
        st.session_state.train_df = train_df
        st.session_state.val_df = val_df
        st.session_state.test_df = test_df
        st.session_state.data_loaded = True

        return True


def load_trained_model(plant_name: str = None, use_model_savesss: bool = False, silent: bool = False):
    """Load a trained ensemble model from saved files.

    Parameters:
    -----------
    plant_name : str, optional
        Name of the plant. If None, uses session state.
    use_model_savesss : bool, default=False
        If True, loads from model_savesss folder instead of default saved_models folder.
    silent : bool, default=False
        If True, no st.info/st.success/headers (errors still shown).
    """
    if plant_name is None:
        plant_name = st.session_state.get('plant_selected', 'Sample Plant')

    if not silent:
        st.markdown("### Load Trained Model")

    # Get plant directory name
    plant_upper = plant_name.upper().replace(' ', '_')

    # Choose model directory based on toggle
    if use_model_savesss:
        base_dir = MODEL_SAVESSS_DIR
        if not silent:
            st.info(f" Loading from model_savesss folder")
    else:
        base_dir = SAVED_MODELS_DIR
        if not silent:
            st.info(f" Loading from saved_models folder")

    plant_model_dir = base_dir / plant_upper / "ridge_lightgbm_ensemble"

    # Try to find the latest version
    ensemble_paths = []
    if plant_model_dir.exists():
        for version_dir in sorted(plant_model_dir.iterdir(), reverse=True):
            if version_dir.is_dir():
                ensemble_paths.append(version_dir)

    # Fallback: check for any ensemble models in the selected base directory
    if not ensemble_paths and base_dir.exists():
        for path in base_dir.rglob("*ridge_lightgbm_ensemble"):
            if path.is_dir():
                for version_dir in sorted(path.iterdir(), reverse=True):
                    if version_dir.is_dir():
                        ensemble_paths.append(version_dir)

    if not ensemble_paths:
        folder_name = "model_savesss" if use_model_savesss else "saved_models"
        st.error(f"No saved ensemble models found for {plant_name} in {folder_name}. Please train a model first.")
        return False

    # Use the latest version
    model_path = ensemble_paths[0]

    with st.spinner("Loading model..." if not silent else ""):
        try:
            # Try to load bundle first
            bundle_files = list(model_path.glob("*bundle*.pkl"))
            if bundle_files:
                with open(bundle_files[0], 'rb') as f:
                    bundle = pickle.load(f)
                    if isinstance(bundle, dict) and 'model' in bundle:
                        st.session_state.model = bundle['model']
                        st.session_state.scaler = bundle.get('scaler')
                        imputer = bundle.get('imputer')
                        _ensure_imputer_compat(imputer)
                        st.session_state.imputer = imputer
                        bundle_feature_cols = bundle.get('feature_columns', [])
                        if bundle_feature_cols:
                            st.session_state.model_feature_columns = bundle_feature_cols
                            st.session_state.feature_columns = bundle_feature_cols
                        st.session_state.encoders = bundle.get('encoders')
                    else:
                        st.session_state.model = bundle
            else:
                # Load components separately - use more specific file patterns
                # Look for files ending with specific patterns to avoid mismatches
                ridge_files = [f for f in model_path.glob("*.pkl") if "ridge" in f.name.lower() and "lgbm" not in f.name.lower() and "scaler" not in f.name.lower() and "imputer" not in f.name.lower()]
                lgbm_files = [f for f in model_path.glob("*.pkl") if "lgbm" in f.name.lower() and "ridge" not in f.name.lower() and "scaler" not in f.name.lower() and "imputer" not in f.name.lower()]
                scaler_files = [f for f in model_path.glob("*.pkl") if "scaler" in f.name.lower()]
                imputer_files = [f for f in model_path.glob("*.pkl") if "imputer" in f.name.lower()]

                # If no specific matches, try the original pattern but validate
                if not ridge_files:
                    ridge_files = list(model_path.glob("*ridge*.pkl"))
                if not lgbm_files:
                    lgbm_files = list(model_path.glob("*lgbm*.pkl"))

                if not ridge_files or not lgbm_files:
                    st.error("Model files not found in selected directory")
                    if not silent:
                        st.info("Please train a new model instead.")
                    return False

                # Load and validate each component
                with open(ridge_files[0], 'rb') as f:
                    ridge = pickle.load(f)
                    # Validate it's actually a Ridge model
                    if not hasattr(ridge, 'predict'):
                        st.error(f"Error: File {ridge_files[0].name} is not a Ridge model. It appears to be a {type(ridge).__name__}")
                        if not silent:
                            st.info("Please train a new model instead.")
                        return False

                with open(lgbm_files[0], 'rb') as f:
                    lgbm = pickle.load(f)
                    # Validate it's actually a LightGBM model
                    if not hasattr(lgbm, 'predict'):
                        st.error(f"Error: File {lgbm_files[0].name} is not a LightGBM model. It appears to be a {type(lgbm).__name__}")
                        if not silent:
                            st.info("Please train a new model instead.")
                        return False

                scaler = None
                if scaler_files:
                    with open(scaler_files[0], 'rb') as f:
                        scaler = pickle.load(f)

                imputer = None
                if imputer_files:
                    with open(imputer_files[0], 'rb') as f:
                        imputer = pickle.load(f)
                    _ensure_imputer_compat(imputer)

                # Create ensemble
                ensemble = RidgeLightGBMEnsemble(
                    ridge_model=ridge,
                    lightgbm_model=lgbm,
                    ridge_weight=ENSEMBLE_RIDGE_WEIGHT,
                    scaler=scaler,
                    imputer=imputer
                )
                st.session_state.model = ensemble
                st.session_state.scaler = scaler
                st.session_state.imputer = imputer

            # Load feature columns from model directory - these are the ones the model was trained with
            feature_cols_file = model_path / "feature_columns.json"
            loaded_feature_cols = None
            if feature_cols_file.exists():
                with open(feature_cols_file, 'r') as f:
                    loaded_feature_cols = json.load(f)
                    # Store model-specific feature columns separately
                    if loaded_feature_cols:
                        st.session_state.model_feature_columns = loaded_feature_cols
                        # Also update main feature_columns to use model's features for predictions
                        st.session_state.feature_columns = loaded_feature_cols
                        if not silent:
                            st.info(f" Loaded {len(loaded_feature_cols)} feature columns from model")

            # If model doesn't have feature columns saved, warn user
            if not loaded_feature_cols:
                if not silent:
                    st.warning(" Model doesn't have feature_columns.json. Using features from Step 1, which may not match the model's training features.")
                    if st.session_state.feature_columns:
                        st.info(f"Using {len(st.session_state.feature_columns)} features from Step 1")

            st.session_state.model_loaded = True
            if not silent:
                st.success(" Model loaded successfully!")

            # Load model stats from config (always for session state)
            config_file = model_path / "config.json"
            model_stats = None
            if config_file.exists():
                with open(config_file, 'r') as f:
                    model_stats = json.load(f)
                    st.session_state.model_stats = model_stats

            # Display model summary only when not silent
            if not silent:
                st.markdown("#### Model Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Model Type", "Ridge-LightGBM Ensemble")
                with col2:
                    st.metric("Ridge Weight", f"{ENSEMBLE_RIDGE_WEIGHT:.1%}")
                with col3:
                    if st.session_state.model_feature_columns:
                        feature_count = len(st.session_state.model_feature_columns)
                    else:
                        feature_count = len(st.session_state.feature_columns) if st.session_state.feature_columns else 0
                    st.metric("Features", f"{feature_count:,}" if feature_count > 0 else "N/A")
                with col4:
                    if model_stats:
                        st.metric("Test Accuracy (R2)", f"{model_stats.get('test_accuracy', 0):.4f}")

                if model_stats:
                    st.markdown("#### Model Training Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Train Size", f"{model_stats.get('train_size', 0):,}")
                    with col2:
                        st.metric("Val Size", f"{model_stats.get('val_size', 0):,}")
                    with col3:
                        st.metric("Test Size", f"{model_stats.get('test_size', 0):,}")
                    with col4:
                        st.metric("Val Accuracy (R2)", f"{model_stats.get('val_accuracy', 0):.4f}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Val RMSE", f"{model_stats.get('val_rmse', 0):.2f} MW")
                    with col2:
                        st.metric("Test RMSE", f"{model_stats.get('test_rmse', 0):.2f} MW")

            return True

        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False


def train_model():
    """Train Ridge-LightGBM ensemble model."""
    st.markdown("### Step 2: Training Model")

    if not st.session_state.data_loaded:
        st.error("Please load data first!")
        return False

    train_df = st.session_state.train_df
    val_df = st.session_state.val_df
    test_df = st.session_state.test_df

    with st.spinner("Training model..."):
        # Feature engineering
        st.info("Creating features...")
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

        st.info(f" Created {len(feature_cols)} features")

        # Prepare X and y
        X_train = train_df[feature_cols].values
        y_train = train_df[TARGET_HORIZON].values
        X_val = val_df[feature_cols].values
        y_val = val_df[TARGET_HORIZON].values
        X_test = test_df[feature_cols].values
        y_test = test_df[TARGET_HORIZON].values

        # Imputation and scaling
        st.info("Preprocessing data...")
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)
        X_test = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Train Ridge
        st.info("Training Ridge Regression...")
        ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        ridge.fit(X_train_scaled, y_train)
        ridge_val = ridge.predict(X_val_scaled)
        ridge_test = ridge.predict(X_test_scaled)

        # Train LightGBM
        st.info("Training LightGBM...")
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

        # Create Ensemble
        st.info("Creating Ridge-LightGBM Ensemble...")
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
        def calculate_metrics(y_true, y_pred):
            accuracy = calculate_accuracy(y_true, y_pred)
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            return {'Accuracy': accuracy, 'RMSE': rmse}

        val_metrics = calculate_metrics(y_val, ensemble_val)
        test_metrics = calculate_metrics(y_test, ensemble_test)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Val Accuracy (R2)", f"{val_metrics['Accuracy']:.4f}")
        with col2:
            st.metric("Val RMSE", f"{val_metrics['RMSE']:.2f} MW")
        with col3:
            st.metric("Test Accuracy (R2)", f"{test_metrics['Accuracy']:.4f}")
        with col4:
            st.metric("Test RMSE", f"{test_metrics['RMSE']:.2f} MW")

        # Store in session state
        st.session_state.model = ensemble
        st.session_state.feature_columns = feature_cols
        st.session_state.scaler = scaler
        st.session_state.imputer = imputer
        st.session_state.encoders = encoders
        st.session_state.train_df = train_df
        st.session_state.val_df = val_df
        st.session_state.test_df = test_df
        st.session_state.model_trained = True

        # Save model automatically
        plant_name = st.session_state.get('plant_selected', 'Sample Plant')
        plant_upper = plant_name.upper().replace(' ', '_')
        model_dir = SAVED_MODELS_DIR / plant_upper / "ridge_lightgbm_ensemble" / "v1"
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
            'val_accuracy': float(val_metrics['Accuracy']),
            'test_accuracy': float(test_metrics['Accuracy']),
            'val_rmse': float(val_metrics['RMSE']),
            'test_rmse': float(test_metrics['RMSE']),
            'data_months': DATA_MONTHS
        }
        with open(model_dir / "config.json", 'w') as f:
            json.dump(config_data, f, indent=2)

        st.session_state.model_stats = config_data
        st.success(" Model trained and saved successfully!")
        return True


def generate_predictions(silent: bool = False):
    """Generate predictions and create prediction dataframes.
    If silent=True, no st.info/st.success (errors still shown).
    """
    if not (st.session_state.model_trained or st.session_state.model_loaded):
        st.error("Please train or load model first!")
        return False

    if not st.session_state.data_loaded:
        st.error("Please load data first!")
        return False

    model = st.session_state.model
    test_df = st.session_state.test_df.copy()

    # Prioritize model feature columns (from loaded model) over Step 1 features
    if st.session_state.model_loaded and st.session_state.model_feature_columns:
        feature_cols = st.session_state.model_feature_columns
        if not silent:
            st.info(f"Using {len(feature_cols)} feature columns from loaded model")
    else:
        feature_cols = st.session_state.feature_columns

    if not feature_cols:
        st.error("Feature columns not available. Please train or load a model with feature columns.")
        return False

    # Validate feature count matches model expectations
    if model.imputer is not None:
        # Try to get expected feature count from imputer
        expected_features = None
        if hasattr(model.imputer, 'n_features_in_'):
            expected_features = model.imputer.n_features_in_
        elif hasattr(model.imputer, 'statistics_'):
            # Statistics array length indicates number of features
            expected_features = len(model.imputer.statistics_) if model.imputer.statistics_ is not None else None

        if expected_features and len(feature_cols) != expected_features:
            if not silent:
                st.warning(f" Feature count mismatch: Model expects {expected_features} features, but we have {len(feature_cols)}.")
                st.warning("This usually means the model was trained with different features than identified in Step 1.")

            # If we have model feature columns, use those
            if st.session_state.model_feature_columns and len(st.session_state.model_feature_columns) == expected_features:
                if not silent:
                    st.info(f"Using {len(st.session_state.model_feature_columns)} feature columns from loaded model")
                feature_cols = st.session_state.model_feature_columns
            else:
                # Try to use only the features that match
                if len(feature_cols) > expected_features:
                    st.warning(f" Truncating to first {expected_features} features. This may cause incorrect predictions!")
                    feature_cols = feature_cols[:expected_features]
                elif len(feature_cols) < expected_features:
                    st.error(f" Not enough features: Model needs {expected_features}, but only {len(feature_cols)} available!")
                    st.error("Please ensure the model's feature_columns.json file exists and matches the training data.")
                    return False

    with st.spinner("Generating predictions..."):
        # Apply feature engineering if not already done (for loaded models)
        # Check if feature columns exist in test_df
        missing_features = [col for col in feature_cols if col not in test_df.columns]
        if missing_features:
            if not silent:
                st.info("Applying feature engineering to test data...")
            # Apply feature engineering
            test_df = create_time_features(test_df)
            test_df = create_rolling_features(test_df, TARGET, [1, 6, 24, 96])

            # Use encoders if available, otherwise create new ones
            encoders = st.session_state.get('encoders')
            if encoders is None:
                # Create encoders (this shouldn't happen, but handle it)
                test_df, encoders = encode_categorical_features(
                    test_df, [TARGET, 'date', 'timestamp', 'sscode']
                )
            else:
                test_df, _ = encode_categorical_features(
                    test_df, [TARGET, 'date', 'timestamp', 'sscode'], encoders
                )

            # Check again for missing features
            missing_features = [col for col in feature_cols if col not in test_df.columns]
            if missing_features:
                if not silent:
                    st.warning(f"Some features are still missing: {missing_features[:5]}...")
                # Filter feature_cols to only include columns that exist
                feature_cols = [col for col in feature_cols if col in test_df.columns]
                if not feature_cols:
                    st.error("No valid feature columns found in test data!")
                    return False

        # Prepare test data
        X_test = test_df[feature_cols].values

        # Check if X_test is empty
        if X_test.shape[1] == 0:
            st.error(f"No features available. Expected {len(feature_cols)} features but got 0.")
            return False

        if model.imputer is not None:
            _ensure_imputer_compat(model.imputer)
            X_test_imputed = model.imputer.transform(X_test)
        else:
            X_test_imputed = X_test

        # Get predictions
        predictions = model.predict(X_test_imputed)
        predictions = np.maximum(predictions, 0)  # Clip to non-negative

        # Get component predictions for comparison
        ridge_pred, lgb_pred = model.get_component_predictions(X_test_imputed)
        ridge_pred = np.maximum(ridge_pred, 0)
        lgb_pred = np.maximum(lgb_pred, 0)

        # Select best FSP - use original test_df (before feature engineering) for FSP columns
        # The feature-engineered test_df should still have all original columns, but use original to be safe
        original_test_df = st.session_state.test_df.copy()

        # Ensure row alignment
        if len(original_test_df) != len(predictions):
            # If lengths don't match, try using feature-engineered test_df
            if len(test_df) == len(predictions):
                original_test_df = test_df.copy()
            else:
                st.error(f"Row count mismatch: original has {len(original_test_df)}, predictions has {len(predictions)}")
                return False

        selected_fsps, scheduled_power, confidence = select_best_fsp_by_prediction(
            original_test_df, predictions
        )

        # Create prediction dataframe
        pred_df = create_prediction_dataframe(
            original_test_df, predictions, selected_fsps, scheduled_power, confidence,
            'ridge_lightgbm_ensemble'
        )

        # Store predictions
        st.session_state.prediction_dfs = {
            'ridge_lightgbm_ensemble': pred_df
        }
        st.session_state.predictions_generated = True

        # Save to outputs/predictions so manual_val_check.py can load ML scheduled without retraining
        try:
            plant_name = st.session_state.get('plant_selected', 'Sample Plant')
            plant_slug = re.sub(r'[^a-z0-9]+', '_', str(plant_name).lower()).strip('_') or 'plant'
            pred_dir = PROJECT_DIR / config.get('outputs.predictions_dir', 'outputs/predictions')
            pred_dir.mkdir(parents=True, exist_ok=True)
            out_path = pred_dir / f"{plant_slug}_ridge_lightgbm_ensemble.csv"
            pred_df.to_csv(out_path, index=False)
        except Exception:
            pass  # non-fatal

        if not silent:
            st.success(" Predictions generated successfully!")
        return True


def visualize_quantile_forecasts():
    """Visualize quantile forecast bands with daily drill-down, block axis, and FSP selection markers."""
    st.markdown("#### Quantile Forecasts")
    st.info("Quantile bands (F10F90) centered on ML Scheduled (F50). Actual, Manual Scheduled, and FSP selection by block.")

    prediction_dfs = st.session_state.prediction_dfs
    test_df = st.session_state.test_df

    if not prediction_dfs:
        st.warning("No predictions available")
        return

    model_name = list(prediction_dfs.keys())[0]
    pred_df = prediction_dfs[model_name].copy()

    # Quantiles centered on ML scheduled (F50 = ML scheduled); no ML predicted
    residuals = pred_df['actual_power'] - pred_df['ml_scheduled_power']
    residual_std = float(np.nanstd(residuals))
    if np.isnan(residual_std) or residual_std == 0:
        residual_std = 1e-3

    quantile_levels = {"F10": 0.10, "F25": 0.25, "F50": 0.50, "F75": 0.75, "F90": 0.90}
    z_scores = {label: norm.ppf(q) for label, q in quantile_levels.items()}
    for label, z_val in z_scores.items():
        pred_df[label] = pred_df['ml_scheduled_power'] + z_val * residual_std
    # F50 is exactly ML scheduled
    pred_df['F50'] = pred_df['ml_scheduled_power'].values

    pred_df['actual_percentile'] = np.clip(
        norm.cdf((pred_df['actual_power'] - pred_df['ml_scheduled_power']) / residual_std) * 100,
        0, 100
    )
    pred_df['ml_sched_percentile'] = 50.0  # ML scheduled is the center (F50)

    # Time column
    time_col = None
    for cand in ["timestamp", "date"]:
        if cand in pred_df.columns:
            time_col = cand
            break

    if time_col:
        pred_df[time_col] = pd.to_datetime(pred_df[time_col])
        pred_df = pred_df.sort_values(time_col)
    else:
        pred_df = pred_df.reset_index(drop=True)

    # Merge FSP data
    fsp_cols = []
    if time_col and time_col in test_df.columns:
        fsp_cols = get_fsp_forecast_columns(test_df)
        fsp_merge = test_df[[time_col] + fsp_cols].copy()
        fsp_merge[time_col] = pd.to_datetime(fsp_merge[time_col])
        pred_df = pred_df.merge(fsp_merge, on=time_col, how='left')
    elif not time_col:
        fsp_cols = get_fsp_forecast_columns(test_df)
        pred_df = pred_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        pred_df['row_idx'] = np.arange(len(pred_df))
        test_df['row_idx'] = np.arange(len(test_df))
        pred_df = pred_df.merge(test_df[['row_idx'] + fsp_cols], on='row_idx', how='left')
    else:
        fsp_cols = []

    if time_col:
        pred_df['date_key'] = pred_df[time_col].dt.date
    else:
        pred_df['date_key'] = pd.Series(["All"] * len(pred_df))

    # Only day-wise selection (removed date range and month options)
    available_dates = sorted(pred_df['date_key'].dropna().unique())
    selected_date = st.date_input(
        " Select Date for Daily Ribbon",
        value=available_dates[0] if available_dates else None,
        min_value=min(available_dates) if available_dates else None,
        max_value=max(available_dates) if available_dates else None,
        key="quantile_single_date_calendar"
    )
    plot_df = pred_df[pred_df['date_key'] == selected_date].copy()
    block_axis = np.arange(1, len(plot_df) + 1)

    if plot_df.empty:
        st.warning("No data available for the selected date.")
        return

    show_all_fsps = st.checkbox("Show all FSP forecast lines", value=False, key="show_all_fsps_quantile")

    # Build FSP palette
    available_fsp_names = [col.replace('forecast_power_', '').upper() for col in fsp_cols]
    fsp_colors = build_fsp_palette(available_fsp_names)
    render_fsp_color_legend(fsp_colors)

    # Compute FSP percentiles vs band centered on ML scheduled
    def compute_fsp_percentiles(df):
        if not fsp_cols:
            return {}
        pct_cols = {}
        for col in fsp_cols:
            fsp_name = col.replace('forecast_power_', '').upper()
            pct_series = np.clip(
                norm.cdf((df[col] - df['ml_scheduled_power']) / residual_std) * 100,
                0, 100
            )
            pct_cols[fsp_name] = pct_series
        return pct_cols

    fsp_percentiles = compute_fsp_percentiles(pred_df)

    # Build quantile figure function
    def build_quantile_figure(df, title, x_override=None, include_fsps=False):
        x_axis = x_override if x_override is not None else (df[time_col] if time_col else df.index)
        x_label = 'Block (15-min)' if x_override is not None else ('Timestamp' if time_col else 'Index')
        customdata_actual = np.column_stack([df['actual_percentile']])
        customdata_sched = np.column_stack([df['ml_sched_percentile'], df['ml_selected_fsp']])

        fig = go.Figure()

        # Quantile ribbons
        fig.add_trace(go.Scatter(x=x_axis, y=df['F90'], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=x_axis, y=df['F10'], mode='lines', name='80% CI (F10-F90)',
                               line=dict(width=0), fill='tonexty', fillcolor='rgba(33, 158, 188, 0.2)'))
        fig.add_trace(go.Scatter(x=x_axis, y=df['F75'], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=x_axis, y=df['F25'], mode='lines', name='50% CI (F25-F75)',
                               line=dict(width=0), fill='tonexty', fillcolor='rgba(33, 158, 188, 0.35)'))
        fig.add_trace(go.Scatter(x=x_axis, y=df['F50'], mode='lines', name='F50 (ML Scheduled)',
                               line=dict(color='#1f77b4', width=2)))

        # FSP selection markers (dots colored by selected FSP per block)
        if 'ml_selected_fsp' in df.columns:
            fig.add_trace(go.Scatter(
                x=x_axis, y=df['ml_scheduled_power'], mode='markers', name='FSP selection',
                marker=dict(size=7, color=[fsp_colors.get(fsp, '#7f7f7f') for fsp in df['ml_selected_fsp']],
                          line=dict(width=1, color='DarkSlateGray'), symbol='circle'),
                customdata=customdata_sched,
                hovertemplate="Block: %{x}<br>ML Scheduled: %{y:.2f} MW<br>FSP: %{customdata[1]}<extra></extra>"
            ))

        # FSP overlays (full forecast lines when checkbox on)
        if include_fsps and fsp_cols:
            for col in fsp_cols:
                fsp_name = col.replace('forecast_power_', '').upper()
                pct_series = fsp_percentiles.get(fsp_name)
                fig.add_trace(go.Scatter(
                    x=x_axis, y=df[col], mode='lines', name=f"FSP {fsp_name}",
                    line=dict(color=fsp_colors.get(fsp_name, '#7f7f7f'), width=1),
                    customdata=np.column_stack([pct_series]) if pct_series is not None else None,
                    hovertemplate="FSP percentile vs band: %{customdata[0]:.1f}th<extra></extra>" if pct_series is not None else "FSP percentile unavailable<extra></extra>",
                    opacity=0.6
                ))

        # Actual power
        fig.add_trace(go.Scatter(x=x_axis, y=df['actual_power'], mode='lines', name='Actual',
                               line=dict(color='#111111', width=2), customdata=customdata_actual,
                               hovertemplate="Point percentile vs band: %{customdata[0]:.1f}th<extra></extra>"))

        # Manual scheduled
        if 'manual_scheduled_power' in df.columns and df['manual_scheduled_power'].notna().any():
            fig.add_trace(go.Scatter(x=x_axis, y=df['manual_scheduled_power'], mode='lines', name='Manual Scheduled',
                                   line=dict(color='#006400', width=2),
                                   hovertemplate="Manual Scheduled: %{y:.2f} MW<extra></extra>"))

        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title='Power (MW)', height=550,
                         hovermode='x unified', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                         plot_bgcolor='white', paper_bgcolor='white')
        return fig

    # Display day-wise only
    st.markdown("##### Daily Quantile Forecast (Blocks 196)")
    st.caption("Quantile bands (F10F90) show forecast uncertainty around ML Scheduled (F50); the center line is the schedule. Actual power and which FSP was selected per 15-min block are overlaid.")
    title = f"Quantile Forecast  {plot_df['date_key'].iloc[0]}"
    st.plotly_chart(build_quantile_figure(plot_df, title, x_override=block_axis, include_fsps=show_all_fsps),
                   use_container_width=True)
    render_section_insight("daily_quantile_forecast", plot_df=plot_df, date=selected_date)

    # Heatmap button - hidden by default
    if st.button(" Show Forecast Heatmap", key="show_heatmap_btn"):
        st.session_state.show_heatmap = not st.session_state.show_heatmap

    if st.session_state.show_heatmap:
        render_daily_forecast_heatmap(plot_df, block_axis, fsp_cols, fsp_colors)

    st.markdown("---")
    st.markdown("##### FSP Selection Waterfall (All 96 Blocks)")
    st.caption("Shows which FSP was selected by the ML model for each 15-minute block and the scheduled power per block.")
    render_fsp_waterfall_chart(plot_df, block_axis)


def render_fsp_waterfall_chart(plot_df, block_axis):
    """Create waterfall chart showing FSP selection for all blocks."""
    selected_fsps_list = st.session_state.get('selected_fsps', [])

    # Get full test set for per-block accuracy calculation
    prediction_dfs = st.session_state.prediction_dfs
    full_pred_df = None
    if prediction_dfs:
        model_name = list(prediction_dfs.keys())[0]
        full_pred_df = prediction_dfs[model_name].copy()

    # Extract FSP columns
    fsp_columns = [col for col in plot_df.columns if col.endswith('_power') and col not in
                   ['actual_power', 'manual_scheduled_power', 'ml_predicted_power', 'ml_scheduled_power']]

    fsp_info = []
    for col in fsp_columns:
        fsp_name = col.replace('_power', '').replace('_POWER', '').upper()
        fsp_info.append((fsp_name, col))

    if selected_fsps_list:
        filtered = [info for info in fsp_info if info[0] in selected_fsps_list]
        if filtered:
            fsp_info = filtered

    available_fsps = [name for name, _ in fsp_info]

    if not available_fsps:
        st.warning("No FSP forecast columns available")
        return

    # Unique color per FSP (build_fsp_palette uses base map + qualitative fallback by index)
    fsp_palette = build_fsp_palette(available_fsps)

    # Map FSP forecasts
    fsp_forecasts = {}
    for fsp_name, col in fsp_info:
        if col in plot_df.columns:
            fsp_forecasts[fsp_name] = plot_df[col].to_numpy()
        else:
            fsp_forecasts[fsp_name] = np.zeros(len(plot_df))

    ml_selected_fsp = plot_df['ml_selected_fsp'].values

    fig = go.Figure()

    # Create FSP data
    fsp_data = {fsp: [] for fsp in available_fsps}

    for idx, block in enumerate(block_axis):
        selected = ml_selected_fsp[idx] if idx < len(ml_selected_fsp) else 'UNKNOWN'
        scheduled_power = plot_df['ml_scheduled_power'].values[idx] if idx < len(plot_df) else 0

        for fsp in available_fsps:
            if fsp == selected:
                fsp_data[fsp].append(scheduled_power)
            else:
                fsp_data[fsp].append(0)

    # Add traces for each FSP
    for fsp in available_fsps:
        forecast_values = fsp_forecasts.get(fsp, np.zeros(len(block_axis)))
        if len(forecast_values) != len(block_axis):
            aligned = np.zeros(len(block_axis))
            limit = min(len(block_axis), len(forecast_values))
            if limit > 0:
                aligned[:limit] = forecast_values[:limit]
            forecast_values = aligned

        fig.add_trace(go.Bar(
            x=block_axis,
            y=fsp_data[fsp],
            name=fsp,
            marker_color=fsp_palette[fsp],
            customdata=forecast_values,
            hovertemplate=f"{fsp}: %{{customdata:.2f}} MW<extra></extra>"
        ))

    # Add actual power line
    fig.add_trace(go.Scatter(
        x=block_axis,
        y=plot_df['actual_power'].values,
        name='Actual Power',
        mode='lines',
        line=dict(color='black', width=2)
    ))

    # Add ML scheduled line (highlighted)
    fig.add_trace(go.Scatter(
        x=block_axis,
        y=plot_df['ml_scheduled_power'].values,
        name='ML Scheduled',
        mode='lines',
        line=dict(color='#2ca02c', width=3)
    ))

    fig.update_layout(
        title=f"FSP Selection Waterfall - {plot_df['date_key'].iloc[0] if 'date_key' in plot_df.columns else 'Daily'}",
        xaxis_title="Block Number (15-min intervals)",
        yaxis_title="Power (MW)",
        barmode='stack',
        height=500,
        hovermode='x unified',
        plot_bgcolor='white', paper_bgcolor='white'
    )

    st.plotly_chart(fig, use_container_width=True)
    render_section_insight("fsp_waterfall", plot_df=plot_df, date=plot_df["date_key"].iloc[0] if "date_key" in plot_df.columns else None)

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        unique_fsps_used = len([fsp for fsp in available_fsps if sum(fsp_data[fsp]) > 0])
        st.metric("FSPs Used", unique_fsps_used)
    with col2:
        total_scheduled = plot_df['ml_scheduled_power'].sum()
        st.metric("Total Scheduled", f"{total_scheduled:,.2f} MW")
    with col3:
        daily_accuracy_ml = calculate_accuracy(plot_df['actual_power'].values, plot_df['ml_scheduled_power'].values)
        st.metric("Daily Accuracy (R2)", f"{daily_accuracy_ml:.4f}")
    with col4:
        avg_confidence = plot_df['selection_confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")

    # Block-wise Accuracy & Delta Table
    st.markdown("---")
    st.markdown("##### Block-wise Accuracy & Improvement Table")
    st.info("Accuracy (R2) = calculated per block across full test set. Improvement % = based on THIS DAY's absolute error reduction: (|Manual Delta| - |ML Delta|) / |Manual Delta|  100. Positive % means ML is better for this day. Delta = Actual - Scheduled.")

    # Calculate accuracy for each block (need to group by block number)
    # For single day view, we can calculate accuracy using a rolling window or just show per-block values
    # Since it's a single day, we'll calculate accuracy for the entire day and per-block deltas
    overall_accuracy_ml = calculate_accuracy(plot_df['actual_power'].values, plot_df['ml_scheduled_power'].values)
    overall_accuracy_manual = calculate_accuracy(plot_df['actual_power'].values, plot_df['manual_scheduled_power'].values)

    # Calculate overall day improvement based on absolute errors (MAE)
    ml_mae = np.mean(np.abs(plot_df['actual_power'].values - plot_df['ml_scheduled_power'].values))
    manual_mae = np.mean(np.abs(plot_df['actual_power'].values - plot_df['manual_scheduled_power'].values))
    if manual_mae > 0:
        overall_improvement = ((manual_mae - ml_mae) / manual_mae) * 100
    else:
        overall_improvement = 0.0 if ml_mae == 0 else -100.0

    # Show overall day accuracy first
    st.markdown("**Overall Day Accuracy:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ML Scheduled Accuracy (R2)", f"{overall_accuracy_ml:.4f}")
    with col2:
        st.metric("Manual Scheduled Accuracy (R2)", f"{overall_accuracy_manual:.4f}")
    with col3:
        st.metric("Improvement % (Error Reduction)", f"{overall_improvement:.2f}%", delta=f"{overall_improvement:.2f}%")

    st.markdown("---")
    st.markdown("**Per-Block Details (Accuracy calculated across full test set):**")

    # Calculate per-block accuracy from full test set (once, before the loop)
    per_block_accuracy_ml = {}
    per_block_accuracy_manual = {}

    if full_pred_df is not None and len(full_pred_df) > 0:
        # Create block column if it doesn't exist
        if 'block' not in full_pred_df.columns:
            if 'timestamp' in full_pred_df.columns:
                full_pred_df['timestamp'] = pd.to_datetime(full_pred_df['timestamp'])
                full_pred_df['block'] = (full_pred_df['timestamp'].dt.hour * 4 + (full_pred_df['timestamp'].dt.minute // 15) + 1)
            elif 'date' in full_pred_df.columns:
                full_pred_df['date'] = pd.to_datetime(full_pred_df['date'])
                full_pred_df['block'] = (full_pred_df['date'].dt.hour * 4 + (full_pred_df['date'].dt.minute // 15) + 1)
            else:
                full_pred_df['block'] = (full_pred_df.index % 96) + 1

        full_pred_df['block'] = full_pred_df['block'].clip(1, 96)

        # Calculate accuracy per block across all days
        for block_num in range(1, 97):
            block_data = full_pred_df[full_pred_df['block'] == block_num]
            if len(block_data) > 1:  # Need at least 2 points for R2
                per_block_accuracy_ml[block_num] = calculate_accuracy(
                    block_data['actual_power'].values,
                    block_data['ml_scheduled_power'].values
                )
                per_block_accuracy_manual[block_num] = calculate_accuracy(
                    block_data['actual_power'].values,
                    block_data['manual_scheduled_power'].values
                )
            else:
                per_block_accuracy_ml[block_num] = 0.0
                per_block_accuracy_manual[block_num] = 0.0

    # Create block-wise table
    block_table_data = []
    for idx, block_num in enumerate(block_axis):
        if idx >= len(plot_df):
            break

        row = plot_df.iloc[idx]
        actual = row.get('actual_power', 0) if 'actual_power' in plot_df.columns else 0
        ml_scheduled = row.get('ml_scheduled_power', 0)
        manual_scheduled = row.get('manual_scheduled_power', 0) if 'manual_scheduled_power' in plot_df.columns else 0

        # Calculate deltas (Actual - Predicted/Scheduled)
        ml_scheduled_delta = actual - ml_scheduled if actual > 0 else 0
        manual_delta = actual - manual_scheduled if actual > 0 and manual_scheduled > 0 else 0

        # Get per-block accuracy from pre-calculated dictionary
        block_num_int = int(block_num)
        if block_num_int in per_block_accuracy_ml:
            block_accuracy_ml = per_block_accuracy_ml[block_num_int]
            block_accuracy_manual = per_block_accuracy_manual.get(block_num_int, 0.0)
        else:
            # Fallback to overall day accuracy if per-block not available
            block_accuracy_ml = overall_accuracy_ml
            block_accuracy_manual = overall_accuracy_manual

        # If ML Scheduled = Manual Scheduled for this block, they must have same accuracy
        if abs(ml_scheduled - manual_scheduled) < 0.01 and manual_scheduled > 0:
            block_accuracy_manual = block_accuracy_ml

        # Calculate improvement percentage based on THIS DAY's absolute errors
        # Improvement = (Manual Error - ML Error) / Manual Error * 100
        # Positive means ML is better (smaller error), negative means ML is worse
        ml_abs_error = abs(ml_scheduled_delta)
        manual_abs_error = abs(manual_delta)

        # If ML Scheduled = Manual Scheduled, improvement should be 0%
        if abs(ml_scheduled - manual_scheduled) < 0.01 and manual_scheduled > 0:
            block_improvement_pct = 0.0
        elif manual_abs_error > 0 and actual > 0 and manual_scheduled > 0:
            # Improvement based on absolute error reduction
            block_improvement_pct = ((manual_abs_error - ml_abs_error) / manual_abs_error) * 100
        elif ml_abs_error == 0 and manual_abs_error > 0:
            # ML is perfect, 100% improvement
            block_improvement_pct = 100.0
        elif ml_abs_error > 0 and manual_abs_error == 0:
            # Manual is perfect, ML is worse (negative improvement)
            block_improvement_pct = -100.0
        else:
            block_improvement_pct = 0.0

        block_table_data.append({
            'Block': int(block_num),
            'Actual (MW)': f"{actual:.2f}" if actual > 0 else "N/A",
            'ML Scheduled (MW)': f"{ml_scheduled:.2f}",
            'Manual Scheduled (MW)': f"{manual_scheduled:.2f}" if manual_scheduled > 0 else "N/A",
            'ML Scheduled Accuracy (R2)': f"{block_accuracy_ml:.4f}",
            'Manual Scheduled Accuracy (R2)': f"{block_accuracy_manual:.4f}" if manual_scheduled > 0 else "N/A",
            'ML Scheduled Delta (MW)': f"{ml_scheduled_delta:+.2f}",
            'Manual Delta (MW)': f"{manual_delta:+.2f}" if manual_scheduled > 0 else "N/A",
            'Improvement % (vs Manual)': f"{block_improvement_pct:.2f}%" if not np.isinf(block_improvement_pct) else "N/A"
        })

    block_table_df = pd.DataFrame(block_table_data)

    # Display table with pagination
    st.dataframe(
        block_table_df,
        use_container_width=True,
        height=400,
        hide_index=True
    )

    # Download button
    csv = block_table_df.to_csv(index=False)
    st.download_button(
        " Download Block-wise Deltas (CSV)",
        data=csv,
        file_name=f"block_wise_deltas_{plot_df['date_key'].iloc[0] if 'date_key' in plot_df.columns else 'daily'}.csv",
        mime="text/csv"
    )

    # Add block-wise line plot: ML Scheduled vs Manual Scheduled vs Actual vs Each FSP
    st.markdown("---")
    st.markdown("##### Block-wise Power Comparison")
    st.caption("ML Scheduled, Manual Scheduled, Actual Power, and FSP forecasts by 15-min block (196) for the selected day.")

    # Get FSP forecast columns from test_df
    test_df = st.session_state.test_df
    fsp_cols = get_fsp_forecast_columns(test_df)

    # Merge FSP data if needed
    if fsp_cols:
        time_col = None
        for cand in ["timestamp", "date"]:
            if cand in plot_df.columns and cand in test_df.columns:
                time_col = cand
                break

        if time_col:
            fsp_merge = test_df[[time_col] + fsp_cols].copy()
            fsp_merge[time_col] = pd.to_datetime(fsp_merge[time_col])
            plot_df[time_col] = pd.to_datetime(plot_df[time_col])
            plot_df = plot_df.merge(fsp_merge, on=time_col, how='left')
        else:
            # Merge by index
            plot_df = plot_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
            if len(plot_df) <= len(test_df):
                for col in fsp_cols:
                    if col in test_df.columns:
                        plot_df[col] = test_df[col].values[:len(plot_df)]

    # Create block-wise line plot
    fig_lines = go.Figure()

    # Actual power
    fig_lines.add_trace(go.Scatter(
        x=block_axis,
        y=plot_df['actual_power'].values,
        mode='lines',
        name='Actual Power',
        line=dict(color='black', width=3)
    ))

    # Manual scheduled
    if 'manual_scheduled_power' in plot_df.columns and plot_df['manual_scheduled_power'].notna().any():
        fig_lines.add_trace(go.Scatter(
            x=block_axis,
            y=plot_df['manual_scheduled_power'].values,
            mode='lines',
            name='Manual Scheduled',
            line=dict(color='red', width=2)
        ))

    # ML scheduled
    fig_lines.add_trace(go.Scatter(
        x=block_axis,
        y=plot_df['ml_scheduled_power'].values,
        mode='lines',
        name='ML Scheduled',
        line=dict(color='green', width=3)
    ))

    # Add each FSP forecast
    fsp_colors_map = build_fsp_palette([col.replace('forecast_power_', '').upper() for col in fsp_cols])
    for col in fsp_cols:
        fsp_name = col.replace('forecast_power_', '').upper()
        if col in plot_df.columns:
            fig_lines.add_trace(go.Scatter(
                x=block_axis,
                y=plot_df[col].values,
                mode='lines',
                name=f'FSP: {fsp_name}',
                line=dict(color=fsp_colors_map.get(fsp_name, '#888888'), width=1.5),
                opacity=0.6,
                visible='legendonly'  # Hidden by default, can be toggled
            ))

    fig_lines.update_layout(
        title=f'Block-wise Power Comparison - {plot_df["date_key"].iloc[0] if "date_key" in plot_df.columns else "Daily"}',
        xaxis_title='Block Number (15-min intervals)',
        yaxis_title='Power (MW)',
        height=500,
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        plot_bgcolor='white', paper_bgcolor='white'
    )

    st.plotly_chart(fig_lines, use_container_width=True)
    render_section_insight("block_wise_power", plot_df=plot_df, date=plot_df["date_key"].iloc[0] if "date_key" in plot_df.columns else None)


def render_daily_forecast_heatmap(plot_df, block_axis, fsp_cols, fsp_colors):
    """Render a 96-block heatmap of forecasted power for every selected FSP - Fixed version with numbers."""
    st.markdown("##### Forecast Heatmap (96 Blocks)")
    st.caption("Cell shading encodes forecasted MW; colored squares mark the scheduled FSP for each 15-minute block.")

    if plot_df is None or plot_df.empty:
        st.info("No forecast data available for the selected day.")
        return

    if not fsp_cols:
        st.info("FSP forecast columns unavailable for the selected timeframe.")
        return

    if block_axis is None:
        block_axis = np.arange(1, len(plot_df) + 1)

    fsp_pairs = []
    for col in fsp_cols:
        if col in plot_df.columns:
            name = col.replace('forecast_power_', '').upper()
            fsp_pairs.append((name, col))

    if not fsp_pairs:
        st.info("Unable to locate forecast columns for the selected FSPs.")
        return

    selected_order = [fsp.upper() for fsp in st.session_state.get('selected_fsps', [])]
    ordered_names = []
    for name in selected_order:
        if any(pair[0] == name for pair in fsp_pairs) and name not in ordered_names:
            ordered_names.append(name)
    for name, _ in fsp_pairs:
        if name not in ordered_names:
            ordered_names.append(name)

    name_to_col = {name: col for name, col in fsp_pairs}
    block_axis = np.array(block_axis[:len(plot_df)], dtype=int)
    x_labels = [f"B{int(val):02d}" for val in block_axis]

    heatmap_rows = []
    value_matrix = []
    for name in ordered_names:
        col = name_to_col.get(name)
        if not col:
            continue
        series = plot_df[col].astype(float).to_numpy()[:len(x_labels)]
        value_matrix.append(series)
        heatmap_rows.append(name)

    if not value_matrix:
        st.info("Forecast values are missing for the selected configuration.")
        return

    values = np.array(value_matrix, dtype=float)
    valid_values = values[~np.isnan(values)]
    if valid_values.size:
        zmin, zmax = float(valid_values.min()), float(valid_values.max())
    else:
        zmin, zmax = 0.0, 1.0

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=x_labels,
        y=heatmap_rows,
        z=values,
        colorscale=[
            [0.0, "#003049"],
            [0.35, "#468faf"],
            [0.65, "#f9c74f"],
            [1.0, "#f9844a"]
        ],
        colorbar=dict(title="Forecast (MW)"),
        hovertemplate="FSP %{y}<br>Block %{x}<br>Forecast %{z:.2f} MW<extra></extra>",
        zmin=zmin,
        zmax=zmax,
        showscale=True,
        xgap=1,
        ygap=1
    ))

    scheduled_series = plot_df.get('ml_selected_fsp')
    highlight_x, highlight_y = [], []
    highlight_vals, highlight_colors = [], []
    scheduled_marker_color = '#d81b60'

    if scheduled_series is not None:
        scheduled_upper = scheduled_series.astype(str).str.upper().tolist()

        for idx, fsp in enumerate(scheduled_upper):
            if idx >= len(x_labels):
                break
            if fsp not in heatmap_rows:
                continue

            highlight_x.append(x_labels[idx])
            highlight_y.append(fsp)

            col = name_to_col.get(fsp)
            if col and col in plot_df.columns:
                cell_value = plot_df.iloc[idx][col]
            else:
                cell_value = np.nan
            highlight_vals.append(cell_value)
            highlight_colors.append(scheduled_marker_color)

        if highlight_x:
            fig.add_trace(
                go.Scatter(
                    x=highlight_x,
                    y=highlight_y,
                    mode='markers',
                    name='Scheduled FSP',
                    marker=dict(
                        symbol='square',
                        size=24,
                        color=highlight_colors,
                        line=dict(color='rgba(255,255,255,0.9)', width=1.2)
                    ),
                    customdata=highlight_vals,
                    hovertemplate=(
                        "Block %{x}<br>"
                        "Scheduled %{y}<br>"
                        "Forecast %{customdata:.2f} MW"
                        "<extra></extra>"
                    ),
                    showlegend=True
                )
            )

    # Add text annotations with numbers in each cell
    text_x, text_y, text_vals = [], [], []
    for row_name, row_values in zip(heatmap_rows, values):
        for block_label, cell_val in zip(x_labels, row_values):
            text_x.append(block_label)
            text_y.append(row_name)
            text_vals.append("-" if np.isnan(cell_val) else f"{cell_val:.1f}")

    fig.add_trace(go.Scatter(
        x=text_x,
        y=text_y,
        mode='text',
        text=text_vals,
        textfont=dict(color='#ffffff', size=10, family='Arial'),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        height=max(480, 140 + 28 * len(heatmap_rows)),
        margin=dict(l=0, r=0, t=35, b=0),
        yaxis=dict(
            title='FSP',
            autorange='reversed',
            tickfont=dict(size=11),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.25)'
        ),
        xaxis=dict(
            title='Block (15-min)',
            tickangle=-45,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.25)',
            dtick=1
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        plot_bgcolor='white', paper_bgcolor='white'
    )

    if len(block_axis) != 96:
        st.caption(" Fewer than 96 records found for this day; heatmap reflects available blocks only.")

    st.plotly_chart(fig, key="forecast_heatmap_plot", width="stretch")


def visualize_fsp_selection():
    """Visualize FSP selection patterns - Full version from predictions_viz."""
    st.markdown("#### FSP Selection Analysis")
    prediction_dfs = st.session_state.prediction_dfs
    selected_fsps_list = st.session_state.get('selected_fsps', [])

    if not prediction_dfs:
        st.warning("No predictions available")
        return

    model_name = list(prediction_dfs.keys())[0]
    pred_df = prediction_dfs[model_name].copy()

    # Filter to only selected FSPs
    if selected_fsps_list:
        pred_df_filtered = pred_df[pred_df['ml_selected_fsp'].isin(selected_fsps_list)].copy()
        st.info(f" Analyzing {len(selected_fsps_list)} selected FSPs: {', '.join(selected_fsps_list)}")
    else:
        pred_df_filtered = pred_df.copy()
        st.info(" Showing all FSPs (no selection filter applied)")

    # Selection confidence
    st.markdown("##### Selection Confidence Distribution")

    fig = px.histogram(
        pred_df_filtered,
        x='selection_confidence',
        nbins=50,
        title='Distribution of Selection Confidence',
        labels={'selection_confidence': 'Confidence Score'}
    )
    fig.add_vline(
        x=pred_df_filtered['selection_confidence'].mean(),
        line_dash="solid",
        line_color="red",
        annotation_text=f"Mean: {pred_df_filtered['selection_confidence'].mean():.3f}"
    )
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig, width='stretch')

    # FSP performance comparison
    st.markdown("##### FSP Performance After Selection")

    fsp_performance = []
    for fsp in pred_df_filtered['ml_selected_fsp'].unique():
        if fsp != 'UNKNOWN':
            fsp_data = pred_df_filtered[pred_df_filtered['ml_selected_fsp'] == fsp]
            avg_error = fsp_data['ml_error'].mean()
            count = len(fsp_data)
            fsp_performance.append({
                'FSP': fsp,
                'Selections': count,
                'Avg Error (MW)': avg_error
            })

    perf_df = pd.DataFrame(fsp_performance).sort_values('Avg Error (MW)')

    fig = px.bar(
        perf_df,
        x='Avg Error (MW)',
        y='FSP',
        orientation='h',
        title='Average Error by Selected FSP (Selected FSPs Only)',
        color='Selections',
        text='Avg Error (MW)'
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig, width='stretch')


def visualize_test_set_aggregates():
    """Display comprehensive test set aggregates - Full version from predictions_viz."""
    st.markdown("#### Test Set Aggregates Summary")
    st.info("Comprehensive comparison of total power values across actual, ML predictions, ML scheduling, and manual scheduling")

    prediction_dfs = st.session_state.prediction_dfs

    if not prediction_dfs:
        st.warning("No predictions available")
        return

    model_name = list(prediction_dfs.keys())[0]
    pred_df = prediction_dfs[model_name].copy()

    # Calculate aggregates
    total_actual = pred_df['actual_power'].sum()
    total_ml_predicted = pred_df['ml_predicted_power'].sum()
    total_ml_scheduled = pred_df['ml_scheduled_power'].sum()
    total_manual_scheduled = pred_df['manual_scheduled_power'].sum()

    # Calculate percentages
    pct_ml_predicted = (total_ml_predicted / total_actual * 100) if total_actual > 0 else 0
    pct_ml_scheduled = (total_ml_scheduled / total_actual * 100) if total_actual > 0 else 0
    pct_manual_scheduled = (total_manual_scheduled / total_actual * 100) if total_actual > 0 else 0

    # Calculate differences
    diff_ml_predicted = total_ml_predicted - total_actual
    diff_ml_scheduled = total_ml_scheduled - total_actual
    diff_manual_scheduled = total_manual_scheduled - total_actual

    # Create aggregates table
    aggregates_data = {
        'Metric': [
            'Total Actual Power',
            'Total ML-Predicted Power',
            'Total ML-Scheduled Power',
            'Total Manual-Scheduled Power'
        ],
        'Value (MW)': [
            f"{total_actual:,.2f}",
            f"{total_ml_predicted:,.2f}",
            f"{total_ml_scheduled:,.2f}",
            f"{total_manual_scheduled:,.2f}"
        ],
        'Percentage (%)': [
            "100.00",
            f"{pct_ml_predicted:.2f}",
            f"{pct_ml_scheduled:.2f}",
            f"{pct_manual_scheduled:.2f}"
        ],
        'Difference from Actual (MW)': [
            "-",
            f"{diff_ml_predicted:+,.2f}",
            f"{diff_ml_scheduled:+,.2f}",
            f"{diff_manual_scheduled:+,.2f}"
        ]
    }

    aggregates_df = pd.DataFrame(aggregates_data)

    st.markdown("##### Aggregate Power Summary")
    st.dataframe(aggregates_df, use_container_width=True, hide_index=True)

    # Performance metrics
    st.markdown("---")
    st.markdown("##### Performance Metrics (Accuracy - R2 Score)")
    accuracy_ml = calculate_accuracy(pred_df['actual_power'].values, pred_df['ml_scheduled_power'].values)
    accuracy_manual = calculate_accuracy(pred_df['actual_power'].values, pred_df['manual_scheduled_power'].values)
    improvement = ((accuracy_ml - accuracy_manual) / abs(accuracy_manual) * 100) if accuracy_manual != 0 else (0.0 if accuracy_ml == 0 else float('inf'))
    better_pct = (pred_df['ml_scheduled_error'] < pred_df['manual_error']).sum() / len(pred_df) * 100 if len(pred_df) > 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("ML Scheduled Accuracy (R2)", f"{accuracy_ml:.4f}",
                 help="R2 accuracy for ML-scheduled power")

    with col2:
        st.metric("Manual Scheduled Accuracy (R2)", f"{accuracy_manual:.4f}",
                 help="R2 accuracy for manual-scheduled power")

    with col3:
        st.metric("ML Improvement", f"{improvement:.2f}%", delta=f"{improvement:.2f}%",
                 help="Percentage improvement of ML accuracy over manual scheduling accuracy")

    with col4:
        st.metric("ML Better Cases", f"{better_pct:.1f}%",
                 help="Percentage of cases where ML performed better than manual")

    # Download aggregates
    st.markdown("---")
    st.markdown("##### Export Aggregates")

    plant_name = st.session_state.get('plant_selected', 'SAMPLE_PSS')
    csv_data = aggregates_df.to_csv(index=False)
    st.download_button(
        label=" Download Aggregates Table (CSV)",
        data=csv_data,
        file_name=f"test_set_aggregates_{plant_name}_{model_name}.csv",
        mime="text/csv"
    )


def visualize_time_series():
    """Interactive time series comparison visualization - Full version from predictions_viz."""
    st.markdown("#### Time Series Comparison")
    st.info("Compare actual power, manual schedule, ML predictions, and FSP forecasts")

    prediction_dfs = st.session_state.prediction_dfs

    if not prediction_dfs:
        st.warning("No predictions available")
        return

    model_name = list(prediction_dfs.keys())[0]
    pred_df = prediction_dfs[model_name].copy()

    # Plot range selection
    has_timestamp = 'timestamp' in pred_df.columns and not pred_df['timestamp'].isna().all()
    plot_mode = None
    start_idx = None
    sample_size = None

    if has_timestamp:
        ts_series = pd.to_datetime(pred_df['timestamp'], errors='coerce')
        has_valid_ts = ts_series.notna().any()
    else:
        ts_series = None
        has_valid_ts = False

    if has_valid_ts:
        plot_mode = st.radio(
            "Plot Range",
            ["Full Test Set", "Date Range", "Single Day", "Single Month"],
            horizontal=True
        )
    else:
        st.info("Timestamp not available. Using sample-based plotting.")
        plot_mode = "Sample Range"

    if plot_mode == "Sample Range":
        col1, col2 = st.columns(2)
        with col1:
            start_idx = st.slider("Start Sample", 0, len(pred_df) - 100, 0, step=100)
        with col2:
            sample_size = st.slider("Number of Samples", 100, min(1000, len(pred_df)), 500, step=100)

    # Ensure error columns exist
    if 'ml_predicted_error' not in pred_df.columns:
        pred_df['ml_predicted_error'] = np.abs(pred_df['actual_power'] - pred_df['ml_predicted_power'])
    if 'ml_scheduled_error' not in pred_df.columns:
        pred_df['ml_scheduled_error'] = np.abs(pred_df['actual_power'] - pred_df['ml_scheduled_power'])
    if 'manual_error' not in pred_df.columns:
        pred_df['manual_error'] = np.abs(pred_df['actual_power'] - pred_df['manual_scheduled_power'])

    full_test_df = pred_df.copy()

    # Filter data for plotting
    if plot_mode == "Sample Range":
        plot_df = pred_df.iloc[start_idx:start_idx + sample_size].copy()
    elif plot_mode == "Full Test Set":
        plot_df = pred_df.copy()
    else:
        pred_df = pred_df.copy()
        pred_df['_timestamp'] = ts_series
        pred_df = pred_df[pred_df['_timestamp'].notna()]

        if plot_mode == "Date Range":
            min_date = pred_df['_timestamp'].min().date()
            max_date = pred_df['_timestamp'].max().date()
            date_range = st.date_input("Select Date Range", value=(min_date, max_date))
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date, end_date = min_date, max_date
            mask = (pred_df['_timestamp'].dt.date >= start_date) & (pred_df['_timestamp'].dt.date <= end_date)
            plot_df = pred_df[mask].copy()
        elif plot_mode == "Single Day":
            min_date = pred_df['_timestamp'].min().date()
            max_date = pred_df['_timestamp'].max().date()
            selected_day = st.date_input("Select Day", value=min_date, min_value=min_date, max_value=max_date)
            mask = pred_df['_timestamp'].dt.date == selected_day
            plot_df = pred_df[mask].copy()
        else:  # Single Month
            month_options = pred_df['_timestamp'].dt.to_period('M').sort_values().unique()
            month_labels = [str(m) for m in month_options]
            selected_month = st.selectbox("Select Month (YYYY-MM)", month_labels)
            month_period = pd.Period(selected_month, freq='M')
            mask = pred_df['_timestamp'].dt.to_period('M') == month_period
            plot_df = pred_df[mask].copy()

        if plot_df.empty:
            st.warning("No data found for selected range. Showing full test set.")
            plot_df = pred_df.copy()

    # Create x-axis
    if '_timestamp' in plot_df.columns and not plot_df['_timestamp'].isna().all():
        x_axis = plot_df['_timestamp']
        x_label = 'Timestamp'
    elif 'timestamp' in plot_df.columns and not plot_df['timestamp'].isna().all():
        x_axis = plot_df['timestamp']
        x_label = 'Timestamp'
    elif 'block' in plot_df.columns:
        x_axis = plot_df['block']
        x_label = 'Block Number'
    else:
        x_axis = plot_df.index
        x_label = 'Index'

    # Create plot - Actual vs Scheduled vs ML Scheduled vs All FSPs
    fig = go.Figure()

    # Actual power
    fig.add_trace(go.Scatter(x=x_axis, y=plot_df['actual_power'], mode='lines', name='Actual Power',
                           line=dict(color='black', width=3)))

    # Manual scheduled
    fig.add_trace(go.Scatter(x=x_axis, y=plot_df['manual_scheduled_power'], mode='lines', name='Manual Scheduled',
                           line=dict(color='red', width=2)))

    # ML Scheduled - highlighted with thicker line
    fig.add_trace(go.Scatter(x=x_axis, y=plot_df['ml_scheduled_power'], mode='lines', name='ML Scheduled (Selected FSP)',
                           line=dict(color='green', width=3)))

    # Add FSP forecasts - show all by default
    test_df = st.session_state.test_df
    fsp_cols = get_fsp_forecast_columns(test_df)
    selected_fsps = st.session_state.get('selected_fsps', [])
    available_fsp_names = [col.replace('forecast_power_', '').upper() for col in fsp_cols]
    fsp_colors = build_fsp_palette(available_fsp_names)
    render_fsp_color_legend(fsp_colors)

    # Merge FSP data if not already present
    # Determine time column from x_axis or plot_df
    time_col_merge = None
    if '_timestamp' in plot_df.columns:
        time_col_merge = '_timestamp'
    elif 'timestamp' in plot_df.columns:
        time_col_merge = 'timestamp'
    elif 'date' in plot_df.columns:
        time_col_merge = 'date'

    if time_col_merge and time_col_merge in test_df.columns:
        fsp_merge = test_df[[time_col_merge] + fsp_cols].copy()
        fsp_merge[time_col_merge] = pd.to_datetime(fsp_merge[time_col_merge])
        if time_col_merge in plot_df.columns:
            plot_df[time_col_merge] = pd.to_datetime(plot_df[time_col_merge])
            plot_df = plot_df.merge(fsp_merge, on=time_col_merge, how='left')
    elif fsp_cols:
        # Try to merge by index if time column not available
        plot_df = plot_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        if len(plot_df) == len(test_df):
            for col in fsp_cols:
                if col in test_df.columns:
                    plot_df[col] = test_df[col].values[:len(plot_df)]

    for idx, fsp_col in enumerate(fsp_cols):
        fsp_name = fsp_col.replace('forecast_power_', '').upper()
        if selected_fsps and fsp_name not in selected_fsps:
            continue
        if fsp_col in plot_df.columns:
            color = fsp_colors.get(fsp_name, px.colors.qualitative.Set3[idx % len(px.colors.qualitative.Set3)])
            fig.add_trace(go.Scatter(x=x_axis, y=plot_df[fsp_col], mode='lines', name=f'FSP: {fsp_name}',
                                   line=dict(color=color, width=1.5), opacity=0.6, visible='legendonly'))

    fig.update_layout(
        title=f'Time Series Comparison - {model_name.replace("_", " ").title()}',
        xaxis_title=x_label,
        yaxis_title='Power (MW)',
        height=600,
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        plot_bgcolor='white', paper_bgcolor='white'
    )

    st.plotly_chart(fig, width='stretch')

    # Statistics - Show all Accuracy metrics for entire test set
    st.markdown("####  Performance Metrics (Accuracy - R2 Score)")
    accuracy_ml_scheduled = calculate_accuracy(full_test_df['actual_power'].values, full_test_df['ml_scheduled_power'].values)
    accuracy_manual = calculate_accuracy(full_test_df['actual_power'].values, full_test_df['manual_scheduled_power'].values)
    improvement_ts = ((accuracy_ml_scheduled - accuracy_manual) / abs(accuracy_manual) * 100) if accuracy_manual != 0 else 0
    st.caption(" These metrics reflect the **entire test set**, not just the filtered plot range above.")

    # Calculate accuracies (removed ML predicted)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("ML Scheduled Accuracy (R2)", f"{accuracy_ml_scheduled:.4f}",
                  help="R2 accuracy between Actual Power and ML Scheduled Power (selected FSP) (Entire Test Set)")

    with col2:
        st.metric("Manual Scheduled Accuracy (R2)", f"{accuracy_manual:.4f}",
                  help="R2 accuracy between Actual Power and Manual Scheduled Power (Entire Test Set)")

    with col3:
        # Improvement: (ML_accuracy - Manual_accuracy) / |Manual_accuracy| * 100
        if accuracy_manual != 0:
            improvement = ((accuracy_ml_scheduled - accuracy_manual) / abs(accuracy_manual)) * 100
        else:
            improvement = 0.0 if accuracy_ml_scheduled == 0 else float('inf')
        st.metric("Total Improvement", f"{improvement:.2f}%",
                  help="Improvement of ML Scheduled accuracy over Manual Schedule accuracy (Entire Test Set)")


def visualize_error_analysis():
    """Error analysis and distribution - Full version from predictions_viz."""
    st.markdown("#### Error Analysis")
    prediction_dfs = st.session_state.prediction_dfs

    if not prediction_dfs:
        st.warning("No predictions available")
        return

    model_name = list(prediction_dfs.keys())[0]
    pred_df = prediction_dfs[model_name].copy()

    # Error distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### ML Error Distribution")
        fig = px.histogram(
            pred_df,
            x='ml_error',
            nbins=50,
            title='ML Prediction Error Distribution'
        )
        fig.add_vline(
            x=pred_df['ml_error'].mean(),
            line_dash="solid",
            line_color="red",
            annotation_text=f"Mean: {pred_df['ml_error'].mean():.3f}"
        )
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("##### Manual Error Distribution")
        fig = px.histogram(
            pred_df,
            x='manual_error',
            nbins=50,
            title='Manual Schedule Error Distribution'
        )
        fig.add_vline(
            x=pred_df['manual_error'].mean(),
            line_dash="solid",
            line_color="red",
            annotation_text=f"Mean: {pred_df['manual_error'].mean():.3f}"
        )
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, width='stretch')

    # Error comparison scatter
    st.markdown("##### ML vs Manual Error Comparison")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pred_df['manual_error'],
        y=pred_df['ml_error'],
        mode='markers',
        marker=dict(
            size=5,
            color=pred_df['selection_confidence'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Confidence")
        ),
        text=pred_df['ml_selected_fsp'],
        hovertemplate='Manual Error: %{x:.2f}<br>ML Error: %{y:.2f}<br>FSP: %{text}'
    ))

    # Add diagonal line
    max_error = max(pred_df['manual_error'].max(), pred_df['ml_error'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_error],
        y=[0, max_error],
        mode='lines',
        line=dict(color='red'),
        name='Equal Error Line'
    ))

    fig.update_layout(
        title='ML Error vs Manual Error',
        xaxis_title='Manual Error (MW)',
        yaxis_title='ML Error (MW)',
        height=500,
        plot_bgcolor='white', paper_bgcolor='white'
    )

    st.plotly_chart(fig, width='stretch')

    # Statistics
    better_count = (pred_df['ml_error'] < pred_df['manual_error']).sum()
    better_pct = (better_count / len(pred_df)) * 100

    st.info(f" ML performed better than manual in **{better_count:,}** cases ({better_pct:.1f}%)")


def calculate_block_wise_metrics(df, actual_col, pred_col, block_col='block'):
    """Calculate block-wise accuracy (R2) and deltas."""
    if block_col not in df.columns:
        if 'timestamp' in df.columns or 'date' in df.columns:
            time_col = 'timestamp' if 'timestamp' in df.columns else 'date'
            df[time_col] = pd.to_datetime(df[time_col])
            df[block_col] = (df[time_col].dt.hour * 4 + (df[time_col].dt.minute // 15) + 1)
        else:
            df[block_col] = (df.index % 96) + 1

    df[block_col] = df[block_col].clip(1, 96)
    df['error'] = df[actual_col] - df[pred_col]
    df['abs_error'] = np.abs(df['error'])

    # Calculate accuracy per block
    block_metrics_list = []
    for block_num in range(1, 97):
        block_data = df[df[block_col] == block_num]
        if len(block_data) > 0:
            accuracy = calculate_accuracy(block_data[actual_col].values, block_data[pred_col].values)
            block_metrics_list.append({
                'block': block_num,
                'accuracy': accuracy,
                'count': len(block_data),
                'mean_delta': block_data['error'].mean(),
                'delta_std': block_data['error'].std(),
                'mean_actual': block_data[actual_col].mean(),
                'mean_predicted': block_data[pred_col].mean()
            })
        else:
            block_metrics_list.append({
                'block': block_num,
                'accuracy': 0.0,
                'count': 0,
                'mean_delta': 0.0,
                'delta_std': 0.0,
                'mean_actual': 0.0,
                'mean_predicted': 0.0
            })

    block_metrics = pd.DataFrame(block_metrics_list)

    return block_metrics


def visualize_block_wise_analysis():
    """Block-wise accuracy (R2) and delta analysis."""
    st.markdown("####  Block-wise Analysis")
    st.info("Accuracy (R2 score) and deltas for each 15-minute time block (1-96 blocks per day)")

    prediction_dfs = st.session_state.prediction_dfs
    if not prediction_dfs:
        st.warning("No predictions available")
        return

    pred_df = list(prediction_dfs.values())[0].copy()

    # Calculate block-wise metrics for ML scheduled and manual scheduled
    block_metrics_ml = calculate_block_wise_metrics(
        pred_df, 'actual_power', 'ml_scheduled_power'
    )
    block_metrics_manual = calculate_block_wise_metrics(
        pred_df, 'actual_power', 'manual_scheduled_power'
    )

    # Merge metrics
    block_metrics = block_metrics_ml.merge(
        block_metrics_manual[['block', 'accuracy']],
        on='block',
        suffixes=('_ml', '_manual')
    )
    block_metrics = block_metrics.rename(columns={'accuracy_ml': 'accuracy_ml_scheduled', 'accuracy_manual': 'accuracy_manual_scheduled'})

    # Calculate improvement per block
    block_metrics['improvement_pct'] = block_metrics.apply(
        lambda row: ((row['accuracy_ml_scheduled'] - row['accuracy_manual_scheduled']) / abs(row['accuracy_manual_scheduled']) * 100)
        if row['accuracy_manual_scheduled'] != 0 else (0.0 if row['accuracy_ml_scheduled'] == 0 else float('inf')),
        axis=1
    )

    col1, col2 = st.columns(2)

    with col1:
        # Block-wise Accuracy - ML Scheduled
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=block_metrics['block'], y=block_metrics['accuracy_ml_scheduled'],
            mode='lines+markers', name='ML Scheduled Accuracy (R2)',
            line=dict(width=2, color='#2ca02c')
        ))
        fig.add_trace(go.Scatter(
            x=block_metrics['block'], y=block_metrics['accuracy_manual_scheduled'],
            mode='lines+markers', name='Manual Scheduled Accuracy (R2)',
            line=dict(width=2, color='#d62728')
        ))
        fig.add_hline(y=0, line_dash="solid", line_color="gray", annotation_text="Baseline")
        fig.update_layout(
            title="Accuracy (R2) by Time Block",
            xaxis_title="Block Number (1-96)",
            yaxis_title="Accuracy (R2 Score)",
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            plot_bgcolor='white', paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Block-wise Improvement
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=block_metrics['block'], y=block_metrics['improvement_pct'],
            mode='lines+markers', name='Improvement %',
            line=dict(width=2, color='#1f77b4'),
            fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.2)'
        ))
        fig.add_hline(y=0, line_dash="solid", line_color="gray")
        fig.update_layout(
            title="Improvement % by Time Block (ML vs Manual)",
            xaxis_title="Block Number (1-96)",
            yaxis_title="Improvement %",
            height=400,
            plot_bgcolor='white', paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Overall day accuracy
    st.markdown("---")
    st.markdown("##### Overall Day Accuracy")
    overall_accuracy_ml = calculate_accuracy(pred_df['actual_power'].values, pred_df['ml_scheduled_power'].values)
    overall_accuracy_manual = calculate_accuracy(pred_df['actual_power'].values, pred_df['manual_scheduled_power'].values)
    overall_improvement = ((overall_accuracy_ml - overall_accuracy_manual) / abs(overall_accuracy_manual) * 100) if overall_accuracy_manual != 0 else 0.0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall ML Scheduled Accuracy (R2)", f"{overall_accuracy_ml:.4f}")
    with col2:
        st.metric("Overall Manual Scheduled Accuracy (R2)", f"{overall_accuracy_manual:.4f}")
    with col3:
        st.metric("Overall Improvement", f"{overall_improvement:.2f}%", delta=f"{overall_improvement:.2f}%")

    # Summary table
    st.markdown("---")
    st.markdown("##### Block-wise Summary")
    summary_data = {
        'Metric': [
            'Avg ML Scheduled Accuracy (R2)', 'Max ML Scheduled Accuracy (R2)', 'Min ML Scheduled Accuracy (R2)',
            'Avg Manual Scheduled Accuracy (R2)', 'Max Manual Scheduled Accuracy (R2)', 'Min Manual Scheduled Accuracy (R2)',
            'Avg Improvement %', 'Max Improvement %', 'Min Improvement %',
            'Avg Delta (MW)', 'Max Delta (MW)', 'Min Delta (MW)'
        ],
        'Value': [
            f"{block_metrics['accuracy_ml_scheduled'].mean():.4f}",
            f"{block_metrics['accuracy_ml_scheduled'].max():.4f}",
            f"{block_metrics['accuracy_ml_scheduled'].min():.4f}",
            f"{block_metrics['accuracy_manual_scheduled'].mean():.4f}",
            f"{block_metrics['accuracy_manual_scheduled'].max():.4f}",
            f"{block_metrics['accuracy_manual_scheduled'].min():.4f}",
            f"{block_metrics['improvement_pct'].mean():.2f}",
            f"{block_metrics['improvement_pct'].max():.2f}",
            f"{block_metrics['improvement_pct'].min():.2f}",
            f"{block_metrics['mean_delta'].mean():.2f}",
            f"{block_metrics['mean_delta'].max():.2f}",
            f"{block_metrics['mean_delta'].min():.2f}"
        ]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    # Download
    csv = block_metrics.to_csv(index=False)
    st.download_button(
        " Download Block-wise Metrics (CSV)",
        data=csv,
        file_name="block_wise_metrics.csv",
        mime="text/csv"
    )


def export_predictions():
    """Provide export options for predictions - Full version from predictions_viz."""
    prediction_dfs = st.session_state.prediction_dfs
    plant_name = st.session_state.get('plant_selected', 'SAMPLE_PSS')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Models", len(prediction_dfs))

    with col2:
        total_predictions = sum(len(df) for df in prediction_dfs.values())
        st.metric("Total Predictions", f"{total_predictions:,}")

    with col3:
        st.metric("Export Formats", "CSV, ZIP")

    # Individual exports
    st.markdown("#### Individual Model Exports")

    for model_name, pred_df in prediction_dfs.items():
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.text(f" {model_name.replace('_', ' ').title()}")

        with col2:
            csv = pred_df.to_csv(index=False)
            st.download_button(
                label=" CSV",
                data=csv,
                file_name=f"predictions_{plant_name}_{model_name}.csv",
                mime="text/csv",
                key=f"download_{model_name}"
            )

        with col3:
            view_clicked = st.button(" View", key=f"view_{model_name}")

        if view_clicked:
            st.session_state[f"show_preview_{model_name}"] = not st.session_state.get(f"show_preview_{model_name}", False)

        if st.session_state.get(f"show_preview_{model_name}", False):
            with st.expander(f" Preview: {model_name.replace('_', ' ').title()}", expanded=True):
                view_mode = st.radio("Select rows to display", ["First 50", "Last 50", "Random 50"],
                                    horizontal=True, key=f"view_mode_{model_name}")

                if view_mode == "First 50":
                    display_df = pred_df.head(50)
                elif view_mode == "Last 50":
                    display_df = pred_df.tail(50)
                else:
                    sample_size = min(50, len(pred_df))
                    display_df = pred_df.sample(n=sample_size, random_state=42)

                st.caption(f"Showing {len(display_df)} of {len(pred_df)} rows")
                st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Combined export
    st.markdown("#### Combined Export (All Models)")

    if st.button(" Prepare ZIP Download"):
        zip_buffer = create_zip_export(prediction_dfs, plant_name)

        st.download_button(
            label=" Download All Predictions (ZIP)",
            data=zip_buffer,
            file_name=f"all_predictions_{plant_name}.zip",
            mime="application/zip"
        )


def create_zip_export(prediction_dfs, plant_name='unknown_plant'):
    """Create ZIP file containing all predictions."""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for model_name, pred_df in prediction_dfs.items():
            csv_data = pred_df.to_csv(index=False)
            zip_file.writestr(f"predictions_{plant_name}_{model_name}.csv", csv_data)

    zip_buffer.seek(0)
    return zip_buffer


def show_model_comparison():
    """Show Ridge vs LightGBM vs Ensemble comparison."""
    st.markdown("####  Model Comparison")

    if not (st.session_state.model_trained or st.session_state.model_loaded):
        st.warning("Model not trained or loaded yet")
        return

    model = st.session_state.model
    test_df = st.session_state.test_df
    feature_cols = st.session_state.feature_columns

    # Get component predictions
    X_test = test_df[feature_cols].values
    if model.imputer is not None:
        _ensure_imputer_compat(model.imputer)
        X_test_imputed = model.imputer.transform(X_test)
    else:
        X_test_imputed = X_test

    ridge_pred, lgb_pred = model.get_component_predictions(X_test_imputed)
    ensemble_pred = model.predict(X_test_imputed)

    ridge_pred = np.maximum(ridge_pred, 0)
    lgb_pred = np.maximum(lgb_pred, 0)
    ensemble_pred = np.maximum(ensemble_pred, 0)

    y_test = test_df[TARGET_HORIZON].values

    # Calculate metrics
    def calc_metrics(y_true, y_pred):
        accuracy = calculate_accuracy(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        return {'Accuracy': accuracy, 'RMSE': rmse}

    metrics_data = []
    for name, pred in [('Ridge', ridge_pred), ('LightGBM', lgb_pred), ('Ensemble', ensemble_pred)]:
        metrics = calc_metrics(y_test, pred)
        metrics_data.append({
            'Model': name,
            'Accuracy (R2)': metrics['Accuracy'],
            'RMSE (MW)': metrics['RMSE']
        })

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)

    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Accuracy (R2)', x=metrics_df['Model'], y=metrics_df['Accuracy (R2)'],
        marker_color='#2E86AB', text=[f'{v:.4f}' for v in metrics_df['Accuracy (R2)']],
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        name='RMSE', x=metrics_df['Model'], y=metrics_df['RMSE (MW)'],
        marker_color='#F18F01', text=[f'{v:.2f}' for v in metrics_df['RMSE (MW)']],
        textposition='outside'
    ))
    fig.update_layout(
        title="Model Comparison - Accuracy (R2) & RMSE",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode='group',
        height=400,
        plot_bgcolor='white', paper_bgcolor='white'
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_split_timeline():
    """Render a bar/timeline showing trainvalidationtest split time ranges (from when to when)."""
    train_df = st.session_state.get("train_df")
    val_df = st.session_state.get("val_df")
    test_df = st.session_state.get("test_df")
    if train_df is None or val_df is None or test_df is None or (len(train_df) == 0 and len(val_df) == 0 and len(test_df) == 0):
        return
    date_col = "timestamp" if "timestamp" in train_df.columns else "date"
    for df in (train_df, val_df, test_df):
        if date_col not in df.columns or len(df) == 0:
            return
    splits_info = []
    for name, df_split in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        if len(df_split) == 0:
            continue
        d = pd.to_datetime(df_split[date_col])
        splits_info.append({
            "Split": name,
            "Start Date": d.min().strftime("%Y-%m-%d"),
            "End Date": d.max().strftime("%Y-%m-%d"),
            "Rows": len(df_split),
        })
    if not splits_info:
        return
    splits_df = pd.DataFrame(splits_info)
    colors = {"Train": "#2E86AB", "Validation": "#A23B72", "Test": "#F18F01"}
    fig = go.Figure()
    for _, row in splits_df.iterrows():
        start_ts = pd.to_datetime(row["Start Date"])
        end_ts = pd.to_datetime(row["End Date"])
        fig.add_trace(go.Scatter(
            x=[start_ts, end_ts],
            y=[row["Split"], row["Split"]],
            mode="lines+markers+text",
            name=row["Split"],
            line=dict(color=colors.get(row["Split"], "#888888"), width=28),
            marker=dict(size=12),
            text=[row["Start Date"], row["End Date"]],
            textposition="top center",
            textfont=dict(size=12),
        ))
    fig.update_layout(
        title="TrainValidationTest Split (Time Ranges)",
        xaxis_title="Date",
        yaxis_title="",
        height=320,
        margin=dict(t=60, b=50, l=120, r=40),
        showlegend=True,
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(
            categoryorder="array",
            categoryarray=["Test", "Validation", "Train"],
            tickfont=dict(size=14),
            showgrid=False,
        ),
        xaxis=dict(tickfont=dict(size=12), showgrid=True, gridcolor="lightgray"),
        title_font=dict(size=16),
    )
    st.markdown("##### TrainValidationTest Split (Time Ranges)")
    st.plotly_chart(fig, use_container_width=True)


def render_penalty_comparison():
    """DSM penalty: daily comparison (uses top date from visualizations, no date toggle) + Nov & Dec monthly."""
    prediction_dfs = st.session_state.get('prediction_dfs') or {}
    if not prediction_dfs:
        return
    model_name = list(prediction_dfs.keys())[0]
    pred_df = prediction_dfs[model_name].copy()
    if pred_df is None or len(pred_df) == 0:
        return
    if 'actual_power' not in pred_df.columns or 'ml_scheduled_power' not in pred_df.columns:
        return
    time_col = None
    for cand in ("timestamp", "date"):
        if cand in pred_df.columns:
            time_col = cand
            break
    if time_col:
        pred_df[time_col] = pd.to_datetime(pred_df[time_col], errors="coerce")
        pred_df["date_key"] = pred_df[time_col].dt.date
        pred_df["month_num"] = pred_df[time_col].dt.month
        pred_df["month_name"] = pred_df[time_col].dt.month_name()
    else:
        pred_df["date_key"] = None
        pred_df["month_num"] = 0
        pred_df["month_name"] = "Unknown"
    date_options = sorted([d for d in pred_df["date_key"].dropna().unique() if isinstance(d, date_type)]) if pred_df["date_key"].notna().any() else []
    # Use the same date as the top visualizations (no extra date toggle here)
    selected_date = st.session_state.get("quantile_single_date_calendar")
    if selected_date is None or (date_options and selected_date not in date_options):
        selected_date = date_options[0] if date_options else None
    plant_name = st.session_state.get("plant_selected", "Sample Plant")
    sscode = get_plant_sscode(plant_name)
    band_config = load_deviation_band_config(sscode)
    if not band_config or not band_config.get("bands"):
        st.markdown("---")
        st.markdown("###  DSM Penalty Comparison (Manual vs ML Scheduled)")
        st.info(f"Deviation band config not found for **{plant_name}** ({sscode}). Add `deviation_band_configs.json` in project root to see penalty comparison.")
        return
    bands = band_config["bands"]
    st.markdown("---")
    st.markdown("###  DSM Penalty Comparison (Manual vs ML Scheduled)")

    # --- Daily penalty (uses date from top visualizations) ---
    if date_options and selected_date is not None:
        plot_df = pred_df[pred_df["date_key"] == selected_date]
        if not plot_df.empty:
            has_manual = "manual_scheduled_power" in plot_df.columns and plot_df["manual_scheduled_power"].notna().any()
            actual = plot_df["actual_power"].values
            ml_sched = plot_df["ml_scheduled_power"].values
            _, penalty_ml = compute_penalty_rs(actual, ml_sched, bands)
            total_penalty_ml = float(np.nansum(penalty_ml))
            if has_manual:
                manual_sched = np.nan_to_num(plot_df["manual_scheduled_power"].values, nan=0.0)
                _, penalty_manual = compute_penalty_rs(actual, manual_sched, bands)
                total_penalty_manual = float(np.nansum(penalty_manual))
                penalty_reduction_rs = total_penalty_manual - total_penalty_ml
                penalty_reduction_pct = (penalty_reduction_rs / total_penalty_manual * 100) if total_penalty_manual > 0 else 0.0
            else:
                total_penalty_manual = penalty_reduction_rs = penalty_reduction_pct = None
            date_label = str(selected_date)
            st.caption(f"Plant: **{plant_name}** ({sscode})  **{date_label}** (same date as in visualizations)  Penalty = band rate  |actual  scheduled| (MW)  0.25 h per 15-min block (Rs)")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Estimated Penalty (ML Scheduled)", f" {total_penalty_ml:,.2f}", help=f"DSM penalty for {date_label} using ML schedule")
            with col2:
                st.metric("Estimated Penalty (Manual Scheduled)", f" {total_penalty_manual:,.2f}" if has_manual and total_penalty_manual is not None else "", help="Manual schedule penalty for this day" if has_manual else "Manual schedule not available")
            with col3:
                st.metric("Penalty Reduction (Rs)", f" {penalty_reduction_rs:,.2f}" if has_manual and penalty_reduction_rs is not None else "", delta=f" {penalty_reduction_rs:,.2f}" if has_manual and penalty_reduction_rs is not None else None, help="Manual  ML for this day")
            with col4:
                st.metric("Penalty Reduction (%)", f"{penalty_reduction_pct:.1f}%" if has_manual and penalty_reduction_pct is not None else "", delta=f"{penalty_reduction_pct:.1f}%" if has_manual and penalty_reduction_pct is not None else None, help="(Manual  ML) / Manual  100")
            if has_manual and total_penalty_manual is not None and (total_penalty_ml > 0 or total_penalty_manual > 0):
                fig_daily = go.Figure()
                fig_daily.add_trace(go.Bar(name="ML Scheduled", x=["Penalty (Rs)"], y=[total_penalty_ml], marker_color="#2ca02c", text=[f" {total_penalty_ml:,.0f}"], textposition="outside"))
                fig_daily.add_trace(go.Bar(name="Manual Scheduled", x=["Penalty (Rs)"], y=[total_penalty_manual], marker_color="#d62728", text=[f" {total_penalty_manual:,.0f}"], textposition="outside"))
                fig_daily.update_layout(title=f"Daily DSM Penalty: ML vs Manual ({date_label})", barmode="group", height=300, plot_bgcolor="white", paper_bgcolor="white", showlegend=True)
                st.plotly_chart(fig_daily, use_container_width=True)
            render_section_insight("dsm_penalty_comparison", total_penalty_ml=total_penalty_ml, total_penalty_manual=total_penalty_manual if has_manual else None, penalty_reduction_rs=penalty_reduction_rs if has_manual else None, penalty_reduction_pct=penalty_reduction_pct if has_manual else None, date=selected_date)
            st.markdown("##### Within band (out of 96 points per day)")
            n_within_ml = int(np.sum(penalty_ml <= 1e-9))
            pct_within_ml = (n_within_ml / 96.0) * 100.0
            n_within_manual = int(np.sum(penalty_manual <= 1e-9)) if has_manual else None
            pct_within_manual = (n_within_manual / 96.0) * 100.0 if has_manual and n_within_manual is not None else None
            wb1, wb2 = st.columns(2)
            with wb1:
                st.metric("ML scheduled  within band", f"{pct_within_ml:.1f}%", help="Share of 96 blocks with no penalty")
            with wb2:
                if has_manual:
                    st.metric("Manual scheduled  within band", f"{(n_within_manual / 96.0) * 100.0:.1f}%", help="Share of 96 blocks with no penalty")
                else:
                    st.metric("Manual scheduled  within band", "", help="Manual schedule not available")
        else:
            st.info("No data for the selected date in predictions.")
    else:
        st.info("No dates available in predictions for daily penalty.")

    # --- Nov & Dec monthly comparison ---
    nov_dec = pred_df[pred_df["month_num"].isin([11, 12])].copy()
    if not nov_dec.empty:
        st.markdown("##### November & December  Monthly Penalty Comparison")
        has_manual = "manual_scheduled_power" in nov_dec.columns and nov_dec["manual_scheduled_power"].notna().any()
        actual = nov_dec["actual_power"].values
        ml_sched = nov_dec["ml_scheduled_power"].values
        _, penalty_ml_arr = compute_penalty_rs(actual, ml_sched, bands)
        nov_dec["penalty_ml"] = penalty_ml_arr
        if has_manual:
            manual_sched = np.nan_to_num(nov_dec["manual_scheduled_power"].values, nan=0.0)
            _, penalty_manual_arr = compute_penalty_rs(actual, manual_sched, bands)
            nov_dec["penalty_manual"] = penalty_manual_arr
        else:
            nov_dec["penalty_manual"] = 0.0
        monthly = nov_dec.groupby("month_name", sort=False).agg(
            penalty_ml=("penalty_ml", "sum"),
            penalty_manual=("penalty_manual", "sum"),
        ).reset_index()
        month_order = [m for m in ["November", "December"] if m in monthly["month_name"].tolist()]
        if not month_order:
            month_order = monthly["month_name"].tolist()
        monthly = monthly.set_index("month_name").loc[month_order].reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(name="ML Scheduled", x=monthly["month_name"], y=monthly["penalty_ml"], marker_color="#2ca02c", text=[f" {v:,.0f}" for v in monthly["penalty_ml"]], textposition="outside"))
        fig.add_trace(go.Bar(name="Manual Scheduled", x=monthly["month_name"], y=monthly["penalty_manual"], marker_color="#d62728", text=[f" {v:,.0f}" for v in monthly["penalty_manual"]], textposition="outside"))
        fig.update_layout(title="DSM Penalty: Nov & Dec  Manual vs ML Scheduled (Monthly)", barmode="group", height=350, plot_bgcolor="white", paper_bgcolor="white", showlegend=True, xaxis_title="Month", yaxis_title="Penalty (Rs)")
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application."""
    initialize_session_state()

    # Main content - Project Title
    st.title(" GKFS Operational Dashboard")
    st.markdown("### Automated Training & Predictions")
    st.markdown("---")

    # Plant selection  only control; data, model, and predictions load automatically
    available_plants = discover_available_plants()

    if not available_plants:
        st.error("No plant data files found. Please ensure data files are in data/processed/parquet/")
        return

    if 'plant_selected' not in st.session_state or st.session_state.plant_selected is None:
        st.session_state.plant_selected = available_plants[0] if available_plants else None

    selected_plant = st.selectbox(
        "Select Plant",
        available_plants,
        index=available_plants.index(st.session_state.plant_selected) if st.session_state.plant_selected in available_plants else 0,
        key="plant_selector"
    )

    plant_changed = selected_plant != st.session_state.plant_selected
    if plant_changed:
        st.session_state.plant_selected = selected_plant
        st.session_state.data_loaded = False
        st.session_state.model_trained = False
        st.session_state.model_loaded = False
        st.session_state.predictions_generated = False

    # Auto-run pipeline when plant is selected and state is incomplete (no extra buttons)
    need_data = not st.session_state.data_loaded
    need_model = not (st.session_state.model_trained or st.session_state.model_loaded)
    need_predictions = not st.session_state.predictions_generated

    if need_data or need_model or need_predictions:
        with st.spinner(f"Loading data and model for **{selected_plant}**..."):
            if need_data:
                if not load_and_prepare_data(selected_plant, silent=True):
                    st.error(f"Failed to load data for {selected_plant}.")
                    return
            if need_model:
                if not load_trained_model(selected_plant, use_model_savesss=True, silent=True):
                    st.error(f"Failed to load trained model for {selected_plant}. Ensure a model exists in model_savesss.")
                    return
            if need_predictions:
                if not generate_predictions(silent=True):
                    st.error("Failed to generate predictions.")
                    return

    st.success(f"**{selected_plant}**  data, model, and predictions ready.")
    st.markdown("---")

    st.markdown("###  Interactive Visualizations")

    tab1, tab2, tab3 = st.tabs([
        " Quantile Forecasts",
        " Block-wise Analysis",
        " Test Set Aggregates"
    ])

    with tab1:
        visualize_quantile_forecasts()

    with tab2:
        visualize_block_wise_analysis()

    with tab3:
        visualize_test_set_aggregates()

    render_penalty_comparison()

    # Train-Valid-Test split at the end (before Summary)
    st.markdown("---")
    _render_split_timeline()

    # Data Summary & Model Summary as cards at the end, side by side
    st.markdown("---")
    st.markdown("###  Summary")
    sum_col1, sum_col2 = st.columns(2)
    with sum_col1:
        st.markdown("####  Data Summary")
        st.metric("Total Rows", f"{len(st.session_state.df_pivoted):,}")
        st.metric("Train Set", f"{len(st.session_state.train_df):,}")
        st.metric("Validation Set", f"{len(st.session_state.val_df):,}")
        st.metric("Test Set", f"{len(st.session_state.test_df):,}")
        fc = len(st.session_state.feature_columns) if st.session_state.feature_columns else 0
        st.metric("Input Features", f"{fc:,}")
    with sum_col2:
        st.markdown("####  Model Summary")
        model_stats = st.session_state.get("model_stats") or {}
        st.metric("Model Type", "Ridge-LightGBM Ensemble")
        st.metric("Ridge Weight", f"{ENSEMBLE_RIDGE_WEIGHT:.1%}")
        fc = len(st.session_state.model_feature_columns or st.session_state.feature_columns or [])
        st.metric("Features", f"{fc:,}" if fc else "N/A")
        st.metric("Test Accuracy (R2)", f"{model_stats.get('test_accuracy', 0):.4f}")
        st.metric("Test RMSE", f"{model_stats.get('test_rmse', 0):.2f} MW")
        if model_stats:
            st.caption(f"Train: {model_stats.get('train_size', 0):,} | Val: {model_stats.get('val_size', 0):,} | Test: {model_stats.get('test_size', 0):,}")


if __name__ == "__main__":
    main()
