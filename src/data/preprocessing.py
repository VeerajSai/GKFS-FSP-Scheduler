"""
Data Preprocessing for FSP Selection
=====================================

Pivots FSP data from multiple rows per time block to a single row format.

Input format: 1 row per (date, block, FSP) = 5 rows per time block
Output format: 1 row per (date, block) with separate columns for each FSP

Maintainer: Project Team
Date: January 2026
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from pathlib import Path
import sys

# Add src to path for config loading
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_loader import load_config

# Load FSP providers from config (single source of truth)
_config = load_config()
FSP_PROVIDERS = _config.get('fsp_providers', [
    'FA_PROVIDER_A',
    'FA_PROVIDER_B',
    'FA_PROVIDER_C',
    'FA_PROVIDER_D'
])


def pivot_fsp_data(df: pd.DataFrame, fsp_col: str = 'forecast_facode') -> pd.DataFrame:
    """
    Pivot FSP data from multiple rows per time block to single row format.

    Input: DataFrame with 1 row per (date, block, FSP)
    Output: DataFrame with 1 row per (date, block), separate columns for each FSP forecast

    This function handles the raw data and pivots forecast/schedule data by FSP provider.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with multiple FSP rows per time block
    fsp_col : str
        Column containing FSP provider names

    Returns:
    --------
    pd.DataFrame : Pivoted dataframe with 1 row per time block
    """
    # Identify key columns for grouping
    group_cols = ['date', 'block']
    if 'timestamp' in df.columns:
        group_cols.insert(2, 'timestamp')  # Keep order: date, block, timestamp
    if 'sscode' in df.columns:
        group_cols.append('sscode')

    # Get unique FSP providers from data
    unique_fsps = df[fsp_col].unique()
    print(f"  FSP providers found in data: {sorted(unique_fsps)}")

    # Start with groupby to get 1 row per time block (taking first value of common columns)
    # These columns are identical for all FSPs in a block
    common_cols = [
        'actual_power', 'actual_avc', 'actual_windspeed', 'actual_ghirr',
        'actual_flowrate', 'actual_time', 'actual_source',
        'schedule_power', 'schedule_avc', 'schedule_windspeed', 'schedule_ghirr',
        'schedule_flowrate', 'schedule_time', 'schedule_source', 'schedule_revno'
    ]
    existing_common = [c for c in common_cols if c in df.columns]

    # Create base with common columns
    base_df = df.groupby(group_cols)[existing_common].first().reset_index()

    # For each FSP-specific column, pivot and merge
    fsp_cols_to_pivot = [
        'forecast_power', 'forecast_avc', 'forecast_windspeed',
        'forecast_ghirr', 'forecast_flowrate', 'forecast_revno', 'forecast_source'
    ]

    for col in fsp_cols_to_pivot:
        if col in df.columns:
            # Pivot this column by FSP
            pivot_data = df.pivot_table(
                index=group_cols,
                columns=fsp_col,
                values=col,
                aggfunc='first'
            )

            # Rename columns to indicate FSP
            for fsp in pivot_data.columns:
                if fsp != fsp_col:  # Skip the column name if it got included
                    new_col_name = f'{col}_{fsp.lower()}'
                    pivot_data = pivot_data.rename(columns={fsp: new_col_name})

            # Reset index and merge with base
            pivot_data = pivot_data.reset_index(drop=False)

            # Drop the index columns (they're in base_df)
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
    if rows_after > 0:
        try:
            print(f" Pivoted data: {rows_before:,} rows  {rows_after:,} rows")
            print(f"  Compression: {len(unique_fsps)} FSP providers  1 row per time block")
        except UnicodeEncodeError:
            print(f"[OK] Pivoted data: {rows_before:,} rows -> {rows_after:,} rows")
            print(f"  Compression: {len(unique_fsps)} FSP providers -> 1 row per time block")

    return base_df


def calculate_fsp_errors(df: pd.DataFrame, target_col: str = 'actual_power') -> pd.DataFrame:
    """
    Calculate absolute error for each FSP forecast compared to actual power.

    Parameters:
    -----------
    df : pd.DataFrame
        Pivoted dataframe with separate FSP forecast columns
    target_col : str
        Target column name (actual power)

    Returns:
    --------
    pd.DataFrame : DataFrame with error columns added
    """
    df_out = df.copy()

    # Dynamically detect FSP forecast columns
    fsp_forecast_cols = get_fsp_forecast_columns(df_out)

    for fsp_col in fsp_forecast_cols:
        # Extract FSP name from column (e.g., 'forecast_power_fa_provider_a' -> 'fa_provider_a')
        fsp_name = fsp_col.replace('forecast_power_', '')
        error_col = f'error_{fsp_name}'

        if target_col in df_out.columns:
            df_out[error_col] = np.abs(df_out[fsp_col] - df_out[target_col])

    return df_out


def select_best_fsp_oracle(df: pd.DataFrame, target_col: str = 'actual_power') -> pd.DataFrame:
    """
    Select the best FSP for each time block based on actual error (oracle - for analysis only).

    This uses actual data and should only be used for comparison/analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with FSP forecasts and actual power
    target_col : str
        Target column name

    Returns:
    --------
    pd.DataFrame : DataFrame with oracle_best_fsp column
    """
    df_out = df.copy()

    # Dynamically find error columns
    error_cols = [col for col in df_out.columns if col.startswith('error_')]

    if len(error_cols) == 0:
        df_out = calculate_fsp_errors(df_out, target_col)
        error_cols = [col for col in df_out.columns if col.startswith('error_')]

    if len(error_cols) > 0:
        error_matrix = df_out[error_cols].values
        min_indices = np.argmin(error_matrix, axis=1)

        fsp_names = [col.replace('error_', '').upper() for col in error_cols]
        df_out['oracle_best_fsp'] = [fsp_names[i] for i in min_indices]
        df_out['oracle_best_error'] = np.min(error_matrix, axis=1)

    return df_out


def get_fsp_forecast_columns(df: pd.DataFrame) -> List[str]:
    """Get list of FSP forecast power columns dynamically from data."""
    # Find all forecast_power_* columns in the dataframe
    forecast_cols = [col for col in df.columns if col.startswith('forecast_power_')]
    return forecast_cols


def get_available_fsps(df: pd.DataFrame) -> List[str]:
    """Get list of available FSP names from the dataframe."""
    forecast_cols = get_fsp_forecast_columns(df)
    fsp_names = [col.replace('forecast_power_', '').upper() for col in forecast_cols]
    return sorted(fsp_names)


def format_output_csv(
    df: pd.DataFrame,
    predictions: np.ndarray,
    selected_fsps: np.ndarray,
    confidence: np.ndarray,
    model_name: str,
    model_version: str = '1.0.0'
) -> pd.DataFrame:
    """
    Format output CSV with all required columns.

    Structure:
    - timestamp, date, block
    - All FSP forecasts (fa_provider_a_power, fa_provider_c_power, etc.)
    - actual_power, manual_scheduled_power
    - ml_selected_fsp, ml_scheduled_power
    - selection_confidence, model_version, model_name
    - ml_error, manual_error

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (pivoted format)
    predictions : np.ndarray
        Model predictions (predicted actual power)
    selected_fsps : np.ndarray
        Selected FSP name for each row
    confidence : np.ndarray
        Selection confidence scores
    model_name : str
        Name of the model
    model_version : str
        Model version string

    Returns:
    --------
    pd.DataFrame : Formatted output DataFrame
    """
    output = pd.DataFrame()

    # Time identifiers
    output['timestamp'] = df.get('timestamp', pd.NaT)
    output['date'] = df.get('date', '')
    output['block'] = df.get('block', 0)

    # FSP forecasts - all providers
    fsp_cols = get_fsp_forecast_columns(df)
    for fsp in FSP_PROVIDERS:
        col_name = f'forecast_power_{fsp.lower()}'
        if col_name in df.columns:
            output[f'{fsp.lower()}_power'] = df[col_name]
        else:
            output[f'{fsp.lower()}_power'] = np.nan

    # Actual and manual scheduled
    output['actual_power'] = df.get('actual_power', np.nan)
    output['manual_scheduled_power'] = df.get('schedule_power', np.nan)

    # ML outputs
    output['ml_predicted_power'] = predictions
    output['ml_selected_fsp'] = selected_fsps

    # Get scheduled power based on selected FSP
    ml_scheduled = []
    for i, fsp in enumerate(selected_fsps):
        fsp_lower = fsp.lower() if isinstance(fsp, str) else 'unknown'
        fsp_col = f'forecast_power_{fsp_lower}'
        if fsp_col in df.columns:
            ml_scheduled.append(df.iloc[i][fsp_col])
        else:
            ml_scheduled.append(np.nan)
    output['ml_scheduled_power'] = ml_scheduled

    output['selection_confidence'] = confidence
    output['model_version'] = model_version
    output['model_name'] = model_name

    # Error calculations
    output['ml_error'] = np.abs(output['actual_power'] - output['ml_scheduled_power'])
    output['manual_error'] = np.abs(output['actual_power'] - output['manual_scheduled_power'])

    return output


if __name__ == '__main__':
    # Test pivoting
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    df = pd.read_parquet('data/interim/eda_processed_data.parquet')

    print("Before pivoting:")
    print(f"  Shape: {df.shape}")
    print(f"  Unique (date, block) pairs: {df.groupby(['date', 'block']).ngroups}")

    df_pivoted = pivot_fsp_data(df)

    print("\nAfter pivoting:")
    print(f"  Shape: {df_pivoted.shape}")
    print(f"  Columns: {list(df_pivoted.columns)}")
