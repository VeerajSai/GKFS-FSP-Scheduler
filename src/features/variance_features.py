"""
Variance-Based Feature Engineering Module - V4
=================================================

Enhanced feature engineering with variance-specific features for wind power prediction.
Includes wind variance features, power-wind relationship features, and temporal features.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class VarianceFeatureConfig:
    """Configuration for variance-based feature engineering."""
    wind_col: str = "actual_windspeed"
    power_col: str = "actual_power"
    target_col: str = "actual_power"

    # Rolling windows (in blocks, 1 block = 15 minutes)
    rolling_windows: List[int] = None

    # Feature flags
    include_rolling_stats: bool = True
    include_wind_gust: bool = True
    include_wind_acceleration: bool = True
    include_power_wind_ratio: bool = True
    include_wind_efficiency: bool = True
    include_rolling_correlation: bool = True
    include_temporal_features: bool = True
    include_cyclical_features: bool = True

    def __post_init__(self):
        if self.rolling_windows is None:
            # Default windows: 1 hour (4 blocks), 6 hours (24 blocks), 24 hours (96 blocks)
            self.rolling_windows = [4, 24, 96]


class VarianceFeatureEngineer:
    """Engineers variance-specific features for wind power prediction."""

    def __init__(self, config: VarianceFeatureConfig):
        self.config = config
        self.feature_columns: List[str] = []

    def add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics for wind speed."""
        print("  Adding rolling statistics...")

        for window in self.config.rolling_windows:
            # Rolling mean
            df[f'wind_rolling_mean_{window}'] = df[self.config.wind_col].rolling(
                window=window, min_periods=1
            ).mean()

            # Rolling std
            df[f'wind_rolling_std_{window}'] = df[self.config.wind_col].rolling(
                window=window, min_periods=1
            ).std().fillna(0)

            # Rolling variance
            df[f'wind_rolling_var_{window}'] = df[self.config.wind_col].rolling(
                window=window, min_periods=1
            ).var().fillna(0)

            # Rolling min/max
            df[f'wind_rolling_min_{window}'] = df[self.config.wind_col].rolling(
                window=window, min_periods=1
            ).min()

            df[f'wind_rolling_max_{window}'] = df[self.config.wind_col].rolling(
                window=window, min_periods=1
            ).max()

            # Rolling range (gust index)
            df[f'wind_rolling_range_{window}'] = (
                df[f'wind_rolling_max_{window}'] - df[f'wind_rolling_min_{window}']
            )

        return df

    def add_wind_gust_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add wind gust features."""
        print("  Adding wind gust features...")

        # Wind gust index (max - min in rolling window)
        for window in self.config.rolling_windows:
            df[f'wind_gust_index_{window}'] = (
                df[f'wind_rolling_max_{window}'] - df[f'wind_rolling_min_{window}']
            )

        # Gust ratio (current / rolling mean)
        for window in self.config.rolling_windows:
            df[f'wind_gust_ratio_{window}'] = (
                df[self.config.wind_col] / (df[f'wind_rolling_mean_{window}'] + 1e-8)
            )

        return df

    def add_wind_acceleration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add wind acceleration features."""
        print("  Adding wind acceleration features...")

        # First derivative (acceleration)
        df['wind_acceleration_1'] = df[self.config.wind_col].diff(1).fillna(0)

        # Second derivative (jerk)
        df['wind_acceleration_2'] = df['wind_acceleration_1'].diff(1).fillna(0)

        # Rolling acceleration
        for window in [4, 24]:
            df[f'wind_accel_rolling_mean_{window}'] = df['wind_acceleration_1'].rolling(
                window=window, min_periods=1
            ).mean().fillna(0)

            df[f'wind_accel_rolling_std_{window}'] = df['wind_acceleration_1'].rolling(
                window=window, min_periods=1
            ).std().fillna(0)

        return df

    def add_power_wind_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add power-to-wind ratio features."""
        print("  Adding power-wind ratio features...")

        epsilon = 1e-8

        # Power-to-wind ratio (with clipping to avoid infinity)
        df['power_wind_ratio'] = np.clip(
            df[self.config.power_col] / (df[self.config.wind_col] + epsilon),
            -1e6, 1e6
        )

        # Rolling power-wind ratio
        for window in [4, 24, 96]:
            df[f'power_wind_ratio_rolling_mean_{window}'] = df['power_wind_ratio'].rolling(
                window=window, min_periods=1
            ).mean()

        return df

    def add_wind_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add wind efficiency features (power / wind^3)."""
        print("  Adding wind efficiency features...")

        epsilon = 1e-8

        # Wind efficiency (power / wind^3) with clipping to avoid infinity
        df['wind_efficiency'] = np.clip(
            df[self.config.power_col] / (df[self.config.wind_col]**3 + epsilon),
            -1e6, 1e6
        )

        # Rolling wind efficiency
        for window in [4, 24, 96]:
            df[f'wind_efficiency_rolling_mean_{window}'] = df['wind_efficiency'].rolling(
                window=window, min_periods=1
            ).mean()

        return df

    def add_rolling_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling correlation between wind and power."""
        print("  Adding rolling correlation features...")

        # Rolling correlation (computationally expensive, use smaller windows)
        for window in [24, 96]:
            rolling_corr = df[[self.config.wind_col, self.config.power_col]].rolling(
                window=window, min_periods=10
            ).corr().iloc[0::2, -1].reset_index(drop=True)
            df[f'wind_power_corr_{window}'] = rolling_corr.fillna(0)

        return df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features."""
        print("  Adding temporal features...")

        # Ensure date column exists
        if 'date' not in df.columns:
            if 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp'])
            else:
                df['date'] = pd.to_datetime(df.index)

        # Extract temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofyear'] = df['date'].dt.dayofyear
        df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)

        # Time since last high wind event (> 10 m/s)
        high_wind_mask = df[self.config.wind_col] > 10
        df['time_since_high_wind'] = high_wind_mask[::-1].cumsum()[::-1]

        # Time since last low wind event (< 3 m/s)
        low_wind_mask = df[self.config.wind_col] < 3
        df['time_since_low_wind'] = low_wind_mask[::-1].cumsum()[::-1]

        return df

    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical encoding for temporal features."""
        print("  Adding cyclical features...")

        # Cyclical encoding for hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Cyclical encoding for day of week
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all features for the dataframe.

        Args:
            df: Input dataframe with wind and power columns

        Returns:
            Dataframe with engineered features
        """
        print("\nEngineering variance-based features...")
        print("=" * 60)

        df = df.copy()

        # Add features based on configuration
        if self.config.include_rolling_stats:
            df = self.add_rolling_stats(df)

        if self.config.include_wind_gust:
            df = self.add_wind_gust_features(df)

        if self.config.include_wind_acceleration:
            df = self.add_wind_acceleration(df)

        if self.config.include_power_wind_ratio:
            df = self.add_power_wind_ratio(df)

        if self.config.include_wind_efficiency:
            df = self.add_wind_efficiency(df)

        if self.config.include_rolling_correlation:
            df = self.add_rolling_correlation(df)

        if self.config.include_temporal_features:
            df = self.add_temporal_features(df)

        if self.config.include_cyclical_features:
            df = self.add_cyclical_features(df)

        # Identify feature columns (exclude target and date columns)
        exclude_cols = [self.config.target_col, 'date', 'timestamp']
        self.feature_columns = [
            col for col in df.columns
            if col not in exclude_cols and not col.startswith('actual_')
        ]

        print("=" * 60)
        print(f"Total features engineered: {len(self.feature_columns)}")

        return df

    def get_feature_columns(self) -> List[str]:
        """Get list of engineered feature columns."""
        return self.feature_columns


def main():
    """Main function to test feature engineering."""
    # Load sample data
    df = pd.read_parquet("data/processed/sample_pss_dataset.parquet")
    df = df.dropna(subset=['actual_windspeed', 'actual_power']).head(10000)

    # Engineer features
    config = VarianceFeatureConfig()
    engineer = VarianceFeatureEngineer(config)
    df_features = engineer.engineer_features(df)

    print(f"\nFeature columns: {len(engineer.get_feature_columns())}")
    print(f"Shape: {df_features.shape}")


if __name__ == "__main__":
    main()
