"""
Unit Tests for Data Processing Functions
=========================================

Tests for feature engineering, temporal splitting, and data leakage prevention.

Maintainer: Project Team
Date: January 2026
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features.feature_engineering import (
    create_temporal_split,
    create_rolling_features,
    drop_missing_data,
    get_feature_columns,
    check_distribution_shift,
    create_time_features
)


class TestTemporalSplit:
    """Tests for temporal split function."""

    def test_split_ratios_are_correct(self):
        """Test that split ratios match expected proportions."""
        # Create sample data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='15min'),
            'value': np.random.randn(1000),
            'actual_power': np.random.rand(1000) * 100
        })

        train_df, val_df, test_df = create_temporal_split(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
        )

        total = len(train_df) + len(val_df) + len(test_df)
        assert total == len(df), "Total samples should match original"

        # Allow 1% tolerance
        assert abs(len(train_df) / len(df) - 0.70) < 0.01
        assert abs(len(val_df) / len(df) - 0.15) < 0.01
        assert abs(len(test_df) / len(df) - 0.15) < 0.01

    def test_no_overlap_between_splits(self):
        """Test that there is no data leakage between splits."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='15min'),
            'value': np.random.randn(1000)
        })

        train_df, val_df, test_df = create_temporal_split(df)

        train_max = train_df['timestamp'].max()
        val_min = val_df['timestamp'].min()
        val_max = val_df['timestamp'].max()
        test_min = test_df['timestamp'].min()

        assert train_max < val_min, "Train should end before val starts"
        assert val_max < test_min, "Val should end before test starts"

    def test_chronological_order(self):
        """Test that splits maintain chronological order."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='15min'),
            'value': np.random.randn(100)
        })

        train_df, val_df, test_df = create_temporal_split(df)

        # Each split should be sorted
        assert train_df['timestamp'].is_monotonic_increasing
        assert val_df['timestamp'].is_monotonic_increasing
        assert test_df['timestamp'].is_monotonic_increasing


class TestRollingFeatures:
    """Tests for rolling feature creation."""

    def test_rolling_uses_past_data_only(self):
        """Test that rolling features only use past data (no look-ahead)."""
        df = pd.DataFrame({
            'actual_power': [10, 20, 30, 40, 50],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='15min')
        })

        df_out = create_rolling_features(df, 'actual_power', windows=[2])

        # First value should be NaN (no past data)
        assert np.isnan(df_out['rolling_mean_2'].iloc[0]) or df_out['rolling_mean_2'].iloc[0] == 10

        # Second value should only use first value
        # With closed='left', it won't include current observation
        # So rolling_mean_2 at index 1 should be based on [10] only
        assert df_out['rolling_mean_2'].iloc[1] == 10  # Only index 0

        # Third value should use indices 0 and 1
        assert df_out['rolling_mean_2'].iloc[2] == 15  # (10 + 20) / 2

    def test_rolling_windows_are_created(self):
        """Test that all specified window sizes create features."""
        df = pd.DataFrame({
            'actual_power': np.random.rand(100),
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='15min')
        })

        windows = [1, 6, 24]
        df_out = create_rolling_features(df, 'actual_power', windows=windows)

        for w in windows:
            assert f'rolling_mean_{w}' in df_out.columns
            assert f'rolling_std_{w}' in df_out.columns


class TestMissingDataHandling:
    """Tests for missing data handling."""

    def test_drops_rows_with_missing_target(self):
        """Test that rows with missing target are dropped."""
        df = pd.DataFrame({
            'actual_power': [10, np.nan, 30, np.nan, 50],
            'feature': [1, 2, 3, 4, 5]
        })

        df_clean = drop_missing_data(df, ['actual_power'], verbose=False)

        assert len(df_clean) == 3
        assert not df_clean['actual_power'].isna().any()

    def test_drops_rows_with_missing_required_cols(self):
        """Test that rows with missing required columns are dropped."""
        df = pd.DataFrame({
            'actual_power': [10, 20, 30, 40, 50],
            'fsp_prediction': [1, np.nan, 3, np.nan, 5]
        })

        df_clean = drop_missing_data(
            df,
            ['actual_power', 'fsp_prediction'],
            verbose=False
        )

        assert len(df_clean) == 3


class TestFeatureColumnSelection:
    """Tests for feature column selection."""

    def test_excludes_actual_columns(self):
        """Test that actual_* columns are excluded (prevents target leakage)."""
        df = pd.DataFrame({
            'actual_power': [1, 2, 3],
            'actual_windspeed': [4, 5, 6],
            'forecast_power': [1.1, 2.1, 3.1],
            'block': [1, 2, 3]
        })

        feature_cols = get_feature_columns(df, 'actual_power')

        assert 'actual_power' not in feature_cols
        assert 'actual_windspeed' not in feature_cols
        assert 'forecast_power' in feature_cols
        assert 'block' in feature_cols

    def test_excludes_metadata_columns(self):
        """Test that metadata columns are excluded."""
        df = pd.DataFrame({
            'actual_power': [1, 2, 3],
            'timestamp': pd.date_range('2024-01-01', periods=3),
            'date': ['2024-01-01', '2024-01-01', '2024-01-01'],
            'feature1': [0.1, 0.2, 0.3]
        })

        feature_cols = get_feature_columns(df, 'actual_power')

        assert 'timestamp' not in feature_cols
        assert 'date' not in feature_cols
        assert 'feature1' in feature_cols


class TestDistributionShift:
    """Tests for distribution shift detection."""

    def test_detects_significant_shift(self):
        """Test detection of significant distribution shift."""
        train = np.random.normal(0, 1, 1000)
        test = np.random.normal(5, 1, 1000)  # Very different distribution

        has_shift, pval = check_distribution_shift(train, test)

        assert has_shift == True  # Use == to handle numpy bool
        assert pval < 0.05

    def test_no_shift_for_same_distribution(self):
        """Test no shift detected for similar distributions."""
        np.random.seed(42)
        train = np.random.normal(0, 1, 1000)
        test = np.random.normal(0, 1, 1000)

        has_shift, pval = check_distribution_shift(train, test)

        # p-value should be high for similar distributions
        assert pval > 0.01 or has_shift is False


class TestTimeFeatures:
    """Tests for time feature creation."""

    def test_creates_hour_from_timestamp(self):
        """Test hour extraction from timestamp."""
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01 08:00', '2024-01-01 16:00'])
        })

        df_out = create_time_features(df)

        assert 'hour' in df_out.columns
        assert df_out['hour'].iloc[0] == 8
        assert df_out['hour'].iloc[1] == 16

    def test_creates_cyclical_encoding(self):
        """Test cyclical encoding of time features."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=24, freq='h')  # lowercase for pandas 3.0
        })

        df_out = create_time_features(df)

        assert 'hour_sin' in df_out.columns
        assert 'hour_cos' in df_out.columns

        # Values should be between -1 and 1
        assert df_out['hour_sin'].min() >= -1
        assert df_out['hour_sin'].max() <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
