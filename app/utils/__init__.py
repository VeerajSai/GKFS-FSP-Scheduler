"""
Utility Functions
=================

Helper functions for the Streamlit application.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


def format_number(num: float, decimals: int = 2) -> str:
    """Format number with commas and decimals."""
    return f"{num:,.{decimals}f}"


def calculate_percentage_improvement(old_value: float, new_value: float) -> float:
    """Calculate percentage improvement."""
    if old_value == 0:
        return 0.0
    return ((old_value - new_value) / old_value) * 100


def create_color_scale(values: List[float], reverse: bool = False) -> List[str]:
    """Create color scale for values."""
    import plotly.express as px

    normalized = (np.array(values) - min(values)) / (max(values) - min(values) + 1e-8)

    if reverse:
        normalized = 1 - normalized

    colors = px.colors.sample_colorscale('RdYlGn', normalized)
    return colors


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division handling zero denominator."""
    return numerator / denominator if denominator != 0 else default


def get_date_range_summary(df: pd.DataFrame, date_col: str = 'timestamp') -> str:
    """Get date range summary string."""
    df[date_col] = pd.to_datetime(df[date_col])
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    days = (max_date - min_date).days

    return f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({days} days)"


def detect_outliers(series: pd.Series, n_std: float = 3.0) -> pd.Series:
    """Detect outliers using standard deviation method."""
    mean = series.mean()
    std = series.std()

    lower_bound = mean - n_std * std
    upper_bound = mean + n_std * std

    return (series < lower_bound) | (series > upper_bound)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    safe_denominator = np.where(denominator == 0, 1, denominator)
    return np.mean(np.abs(y_true - y_pred) / safe_denominator) * 100


def create_summary_stats(df: pd.DataFrame, column: str) -> dict:
    """Create summary statistics for a column."""
    return {
        'count': len(df),
        'mean': df[column].mean(),
        'std': df[column].std(),
        'min': df[column].min(),
        'q25': df[column].quantile(0.25),
        'median': df[column].median(),
        'q75': df[column].quantile(0.75),
        'max': df[column].max()
    }
