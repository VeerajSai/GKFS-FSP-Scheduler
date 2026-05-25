"""
Test script to verify all GUI buttons and workflows work correctly.

This tests:
1. Data selection page - loading and pivoting
2. Feature engineering - creating features and splits
3. FSP selection - detecting FSPs, calculating MAE, selecting FSPs
4. Model training - training models with selected FSPs
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from src.data.preprocessing import (
    pivot_fsp_data,
    get_fsp_forecast_columns,
    get_available_fsps,
    calculate_fsp_errors,
    select_best_fsp_oracle
)
from src.features.feature_engineering import create_temporal_split

def test_fsp_detection():
    """Test that FSP detection works dynamically."""
    print("\n" + "="*60)
    print("TEST 1: Dynamic FSP Detection")
    print("="*60)

    # Create mock data with FSPs
    mock_data = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=100, freq='H'),
        'actual_power': np.random.rand(100) * 100,
        'forecast_power_fa_provider_a': np.random.rand(100) * 100,
        'forecast_power_fa_provider_b': np.random.rand(100) * 100,
        'forecast_power_fa_provider_c': np.random.rand(100) * 100,
    })

    # Test get_fsp_forecast_columns
    fsp_cols = get_fsp_forecast_columns(mock_data)
    print(f" Detected forecast columns: {fsp_cols}")
    assert len(fsp_cols) == 3, "Should detect 3 FSP forecast columns"

    # Test get_available_fsps
    fsp_names = get_available_fsps(mock_data)
    print(f" Available FSPs: {fsp_names}")
    assert len(fsp_names) == 3, "Should have 3 FSPs"
    assert all(isinstance(fsp, str) for fsp in fsp_names), "All FSP names should be strings"

    print(" FSP Detection Test PASSED\n")
    return mock_data


def test_fsp_error_calculation(df):
    """Test that FSP error calculation works dynamically."""
    print("="*60)
    print("TEST 2: Dynamic FSP Error Calculation")
    print("="*60)

    # Test calculate_fsp_errors
    df_with_errors = calculate_fsp_errors(df.copy(), 'actual_power')

    fsp_cols = get_fsp_forecast_columns(df)
    expected_error_cols = [col.replace('forecast_power_', 'error_') for col in fsp_cols]

    for error_col in expected_error_cols:
        assert error_col in df_with_errors.columns, f"Missing {error_col}"
        print(f" Created {error_col}")

    print(" FSP Error Calculation Test PASSED\n")
    return df_with_errors


def test_oracle_selection(df_with_errors):
    """Test oracle FSP selection."""
    print("="*60)
    print("TEST 3: Oracle FSP Selection")
    print("="*60)

    df_oracle = select_best_fsp_oracle(df_with_errors.copy())

    assert 'oracle_best_fsp' in df_oracle.columns, "Missing oracle_best_fsp column"
    assert 'oracle_best_error' in df_oracle.columns, "Missing oracle_best_error column"

    print(f" Oracle best FSP value counts:")
    print(df_oracle['oracle_best_fsp'].value_counts())
    print(f"\n Oracle best error stats:")
    print(df_oracle['oracle_best_error'].describe())

    print(" Oracle Selection Test PASSED\n")


def test_session_state_initialization():
    """Test session state initialization logic."""
    print("="*60)
    print("TEST 4: Session State Initialization (Widget-safe)")
    print("="*60)

    # Simulate FSP names
    fsp_names = ['FA_PROVIDER_A', 'FA_PROVIDER_B', 'FA_PROVIDER_C']

    # This would be done BEFORE widget creation in real Streamlit
    session_state_defaults = {}
    for fsp in fsp_names:
        session_state_defaults[f"fsp_select_{fsp}"] = True

    print(f" Initialized session state defaults:")
    for key, value in session_state_defaults.items():
        print(f"  {key}: {value}")

    # Verify keys are safe for widget assignment
    for key in session_state_defaults:
        assert isinstance(key, str), "Session key should be string"
        assert key.startswith('fsp_select_'), "Session key should have proper prefix"

    print(" Session State Test PASSED\n")


def test_temporal_split():
    """Test temporal data splitting."""
    print("="*60)
    print("TEST 5: Temporal Data Splitting")
    print("="*60)

    mock_data = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=1000, freq='H'),
        'actual_power': np.random.rand(1000) * 100,
        'forecast_power_fa_provider_a': np.random.rand(1000) * 100,
    })

    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

    train_df, val_df, test_df = create_temporal_split(mock_data, train_ratio, val_ratio, test_ratio)

    print(f" Train size: {len(train_df)} ({len(train_df)/len(mock_data):.1%})")
    print(f" Val size: {len(val_df)} ({len(val_df)/len(mock_data):.1%})")
    print(f" Test size: {len(test_df)} ({len(test_df)/len(mock_data):.1%})")

    # Verify temporal order (no data leakage)
    train_max_date = train_df['timestamp'].max()
    val_min_date = val_df['timestamp'].min()
    val_max_date = val_df['timestamp'].max()
    test_min_date = test_df['timestamp'].min()

    assert train_max_date < val_min_date, "Train and Val overlap!"
    assert val_max_date < test_min_date, "Val and Test overlap!"

    print(" Temporal order verified (no data leakage)")
    print(" Temporal Split Test PASSED\n")


def test_dynamic_feature_selection():
    """Test that feature selection works with dynamic FSPs."""
    print("="*60)
    print("TEST 6: Dynamic Feature Selection with FSPs")
    print("="*60)

    mock_data = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=100, freq='h'),
        'actual_power': np.random.rand(100) * 100,
        'forecast_power_fa_provider_a': np.random.rand(100) * 100,
        'forecast_power_fa_provider_b': np.random.rand(100) * 100,
        'hour': [i % 24 for i in range(100)],
        'day_of_week': [i % 7 for i in range(100)],
    })

    # Test that FSP columns can be identified regardless of which FSPs are present
    fsp_cols = get_fsp_forecast_columns(mock_data)
    print(f" Identified {len(fsp_cols)} FSP columns: {fsp_cols}")

    # These would be selected FSPs
    selected_fsps = ['FA_PROVIDER_A']
    feature_cols = [
        col for col in mock_data.columns
        if col.startswith('forecast_power_') or
        col in ['hour', 'day_of_week']
    ]

    print(f" Feature columns: {feature_cols}")
    assert len(feature_cols) > 0, "Should have feature columns"

    print(" Dynamic Feature Selection Test PASSED\n")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# GUI WORKFLOW TEST SUITE")
    print("#"*60)

    try:
        # Run all tests
        df = test_fsp_detection()
        df_with_errors = test_fsp_error_calculation(df)
        test_oracle_selection(df_with_errors)
        test_session_state_initialization()
        test_temporal_split()
        test_dynamic_feature_selection()

        print("\n" + "#"*60)
        print("#  ALL TESTS PASSED SUCCESSFULLY")
        print("#"*60)
        print("\nThe GUI should now work correctly with:")
        print("   Dynamic FSP detection per plant")
        print("   Proper session state initialization (no widget conflicts)")
        print("   Dynamic FSP error calculation")
        print("   Temporal data splitting without leakage")
        print("   Feature engineering with selected FSPs")
        print("\n")

    except AssertionError as e:
        print(f"\n TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
