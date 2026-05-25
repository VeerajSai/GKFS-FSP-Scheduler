"""
Feature Engineering Module
===========================

Functions for creating features for FSP scheduling models.
All rolling features computed AFTER train/val/test split to prevent data leakage.

Maintainer: Project Team
Date: January 2026
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress only specific warnings, not all
warnings.filterwarnings('ignore', message='.*divide by zero.*')
warnings.filterwarnings('ignore', message='.*invalid value encountered.*')


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: List[int] = [1, 6, 24, 96],
    fsp_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create rolling window features AFTER split to prevent data leakage.

    Uses closed='left' to exclude current observation from rolling calculations.
    This ensures we only use past data for predictions.

    IMPORTANT - Rolling Feature Warm-up Period:
    The first N rows (where N is the largest window size) will have incomplete
    rolling statistics. In production inference, you should maintain a buffer
    of historical data from training to ensure accurate rolling features.
    For a window of 96 blocks, you need at least 96 prior observations.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (should be a single split: train, val, or test)
    target_col : str
        Target column name for rolling stats
    windows : List[int]
        Window sizes in number of timesteps
    fsp_cols : List[str], optional
        FSP prediction columns for rolling error calculations

    Returns:
    --------
    pd.DataFrame : DataFrame with rolling features added
    """
    df_out = df.copy()

    for window in windows:
        # Rolling stats on target (using only past data)
        df_out[f'rolling_mean_{window}'] = df_out[target_col].rolling(
            window, min_periods=1, closed='left'
        ).mean()

        df_out[f'rolling_std_{window}'] = df_out[target_col].rolling(
            window, min_periods=1, closed='left'
        ).std().fillna(0)

        # Rolling errors for each FSP if columns provided
        if fsp_cols:
            for fsp_col in fsp_cols:
                if fsp_col in df_out.columns:
                    error = (df_out[fsp_col] - df_out[target_col]).abs()
                    df_out[f'rolling_mae_{fsp_col}_{window}'] = error.rolling(
                        window, min_periods=1, closed='left'
                    ).mean()

    return df_out


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from timestamp or block number.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'timestamp' or 'block' column

    Returns:
    --------
    pd.DataFrame : DataFrame with time features added
    """
    df_out = df.copy()

    if 'timestamp' in df_out.columns:
        df_out['timestamp'] = pd.to_datetime(df_out['timestamp'])
        df_out['hour'] = df_out['timestamp'].dt.hour
        df_out['dayofweek'] = df_out['timestamp'].dt.dayofweek
        df_out['month'] = df_out['timestamp'].dt.month
        df_out['is_weekend'] = (df_out['dayofweek'] >= 5).astype(int)
    elif 'block' in df_out.columns:
        # Derive hour from block number (15-min intervals, 96 blocks per day)
        df_out['hour'] = ((df_out['block'] - 1) * 15 // 60).astype(int)

    # Cyclical encoding for time features
    if 'hour' in df_out.columns:
        df_out['hour_sin'] = np.sin(2 * np.pi * df_out['hour'] / 24)
        df_out['hour_cos'] = np.cos(2 * np.pi * df_out['hour'] / 24)

    if 'dayofweek' in df_out.columns:
        df_out['dow_sin'] = np.sin(2 * np.pi * df_out['dayofweek'] / 7)
        df_out['dow_cos'] = np.cos(2 * np.pi * df_out['dayofweek'] / 7)

    if 'month' in df_out.columns:
        df_out['month_sin'] = np.sin(2 * np.pi * df_out['month'] / 12)
        df_out['month_cos'] = np.cos(2 * np.pi * df_out['month'] / 12)

    return df_out


def encode_categorical_features(
    df: pd.DataFrame,
    exclude_cols: List[str],
    label_encoders: Optional[dict] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical columns using pandas Categorical codes (fast, unseen  -1).

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    exclude_cols : List[str]
        Columns to exclude from encoding
    label_encoders : dict, optional
        Pre-fit encoders for transform-only mode (for val/test sets)

    Returns:
    --------
    Tuple[pd.DataFrame, dict]
        Encoded dataframe and dictionary of label encoders
    """
    df_out = df.copy()
    encoders = label_encoders if label_encoders else {}

    cat_cols = df_out.select_dtypes(include=['object']).columns.tolist()

    for col in cat_cols:
        if col not in exclude_cols:
            series = df_out[col].fillna('unknown').astype(str)
            if label_encoders is None:
                # Fit categories and use fast categorical codes
                cats = series.astype('category').cat.categories.tolist()
                df_out[f'{col}_encoded'] = series.astype(CategoricalDtype(categories=cats)).cat.codes
                encoders[col] = cats
            else:
                if col in encoders:
                    cats = encoders[col]
                    dtype = CategoricalDtype(categories=cats)
                    df_out[f'{col}_encoded'] = series.astype(dtype).cat.codes

    # Drop original categorical columns
    df_out = df_out.drop(columns=cat_cols, errors='ignore')

    return df_out, encoders


def get_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    exclude_patterns: List[str] = ['date', 'timestamp', 'index', 'actual_', 'schedule_']
) -> List[str]:
    """
    Get list of feature columns, excluding target and metadata.

    IMPORTANT: Excludes all 'actual_*' columns to prevent target leakage.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Target column name
    exclude_patterns : List[str]
        Patterns to exclude from features

    Returns:
    --------
    List[str] : List of feature column names
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_cols = [
        c for c in numeric_cols
        if c != target_col and not any(ex in c.lower() for ex in exclude_patterns)
    ]

    return feature_cols


def create_temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create temporal train/val/test split to avoid data leakage.

    Data is sorted chronologically and split into non-overlapping segments.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    train_ratio : float
        Proportion for training set
    val_ratio : float
        Proportion for validation set
    test_ratio : float
        Proportion for test set

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        train_df, val_df, test_df
    """
    # Validate ratios sum to 1.0
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    # Sort by time
    if 'timestamp' in df.columns:
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    elif 'date' in df.columns and 'block' in df.columns:
        df_sorted = df.sort_values(['date', 'block']).reset_index(drop=True)
    else:
        df_sorted = df.reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    return train_df, val_df, test_df


def drop_missing_data(
    df: pd.DataFrame,
    required_cols: List[str],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Drop rows with missing values in required columns.

    NO IMPUTATION - as per requirement to avoid biasing FSP predictions.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    required_cols : List[str]
        Columns that must have valid values
    verbose : bool
        Print statistics about dropped rows

    Returns:
    --------
    pd.DataFrame : Dataframe with missing rows dropped
    """
    initial_len = len(df)

    # Check which required columns exist
    existing_cols = [c for c in required_cols if c in df.columns]

    df_clean = df.dropna(subset=existing_cols).copy()

    if verbose:
        dropped = initial_len - len(df_clean)
        pct = (dropped / initial_len) * 100 if initial_len > 0 else 0
        print(f"Dropped {dropped:,} rows ({pct:.1f}%) with missing values in required columns")

    return df_clean


def check_distribution_shift(
    train_values: np.ndarray,
    test_values: np.ndarray,
    alpha: float = 0.05
) -> Tuple[bool, float]:
    """
    Check for distribution shift between train and test sets using KS test.

    Parameters:
    -----------
    train_values : np.ndarray
        Training set values
    test_values : np.ndarray
        Test set values
    alpha : float
        Significance level

    Returns:
    --------
    Tuple[bool, float]
        (has_significant_shift, p_value)
    """
    from scipy.stats import ks_2samp

    stat, pval = ks_2samp(train_values, test_values)
    has_shift = pval < alpha

    return has_shift, pval


def prepare_features_for_training(
    df: pd.DataFrame,
    target_col: str = 'actual_power',
    fsp_cols: Optional[List[str]] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    rolling_windows: List[int] = [1, 6, 24, 96]
) -> dict:
    """
    Complete feature preparation pipeline.

    This function:
    1. Drops rows with missing target or FSP predictions
    2. Creates temporal split (70-15-15)
    3. Creates time features
    4. Encodes categorical columns
    5. Creates rolling features AFTER split (no data leakage)
    6. Checks for distribution shift
    7. Scales features

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Target column name
    fsp_cols : List[str], optional
        FSP prediction column names
    train_ratio, val_ratio, test_ratio : float
        Split ratios (must sum to 1.0)
    rolling_windows : List[int]
        Windows for rolling features

    Returns:
    --------
    dict : Dictionary containing all prepared data and artifacts
    """
    print("=" * 70)
    print("FEATURE PREPARATION PIPELINE")
    print("=" * 70)

    # Step 1: Drop rows with missing required data
    print("\n1 Dropping rows with missing data (no imputation)...")
    required_cols = [target_col]
    if fsp_cols:
        required_cols.extend(fsp_cols)

    df_clean = drop_missing_data(df, required_cols)

    # Step 2: Create temporal split
    print(f"\n2 Creating temporal split ({train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%})...")
    train_df, val_df, test_df = create_temporal_split(
        df_clean, train_ratio, val_ratio, test_ratio
    )
    print(f"   Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Step 3: Create time features
    print("\n3 Creating time features...")
    train_df = create_time_features(train_df)
    val_df = create_time_features(val_df)
    test_df = create_time_features(test_df)

    # Step 4: Encode categorical features
    print("\n4 Encoding categorical features...")
    exclude_cols = [target_col, 'date', 'timestamp', 'sscode']
    train_df, encoders = encode_categorical_features(train_df, exclude_cols)
    val_df, _ = encode_categorical_features(val_df, exclude_cols, encoders)
    test_df, _ = encode_categorical_features(test_df, exclude_cols, encoders)

    # Step 5: Create rolling features AFTER split
    print("\n5 Creating rolling features (after split, no leakage)...")
    train_df = create_rolling_features(train_df, target_col, rolling_windows, fsp_cols)
    val_df = create_rolling_features(val_df, target_col, rolling_windows, fsp_cols)
    test_df = create_rolling_features(test_df, target_col, rolling_windows, fsp_cols)

    # Step 6: Get feature columns (excluding actual_* to prevent leakage)
    print("\n6 Selecting features (excluding actual_* columns)...")
    feature_cols = get_feature_columns(train_df, target_col)
    print(f"   Selected {len(feature_cols)} features")

    # Step 7: Extract X and y
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    # Step 8: Scale features
    print("\n7 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Step 9: Check distribution shift
    print("\n8 Checking for distribution shift...")
    has_shift, pval = check_distribution_shift(y_train, y_test)
    if has_shift:
        print(f"    Significant distribution shift detected (p={pval:.4f})")
    else:
        print(f"    No significant distribution shift (p={pval:.4f})")

    print("\n" + "=" * 70)
    print(" Feature preparation complete!")
    print("=" * 70)

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_cols': feature_cols,
        'scaler': scaler,
        'encoders': encoders,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'distribution_shift': {'has_shift': has_shift, 'p_value': pval}
    }
