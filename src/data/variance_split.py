"""
Variance-Based Data Split Module - V4
======================================

Creates train/validation/test splits based on wind variance classification.
Uses temporal splitting: train on 2024 data, val/test on 2025 data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class VarianceSplitConfig:
    """Configuration for variance-based splitting."""
    data_path: str = "data/processed/sample_pss_dataset.parquet"
    classification_path: str = "outputs/wind_variance_analysis/variance_classification.json"
    train_year: int = 2024
    val_test_year: int = 2025
    val_ratio: float = 0.5
    wind_col: str = "actual_windspeed"
    power_col: str = "actual_power"
    target_col: str = "actual_power"


class VarianceSplitter:
    """Creates variance-based train/val/test splits."""

    def __init__(self, config: VarianceSplitConfig):
        self.config = config
        self.low_variance_months: List[int] = []
        self.high_variance_months: List[int] = []
        self._load_classification()

    def _load_classification(self):
        """Load variance classification from file."""
        classification_path = Path(self.config.classification_path)
        if classification_path.exists():
            with open(classification_path, 'r') as f:
                classification = json.load(f)
            self.low_variance_months = classification['low_variance_months']
            self.high_variance_months = classification['high_variance_months']
            print(f"Loaded variance classification:")
            print(f"  LOW_VARIANCE: {self.low_variance_months}")
            print(f"  HIGH_VARIANCE: {self.high_variance_months}")
        else:
            raise FileNotFoundError(f"Classification file not found: {classification_path}")

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess data."""
        print(f"\nLoading data from {self.config.data_path}")
        df = pd.read_parquet(self.config.data_path)

        # Filter valid data
        df = df.dropna(subset=[self.config.wind_col, self.config.power_col])

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

    def create_variance_split(
        self,
        df: pd.DataFrame,
        variance_type: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/val/test split for a specific variance type.

        Args:
            df: Input dataframe
            variance_type: 'low_variance' or 'high_variance'

        Returns:
            train_df, val_df, test_df
        """
        if variance_type == 'low_variance':
            months = self.low_variance_months
        elif variance_type == 'high_variance':
            months = self.high_variance_months
        else:
            raise ValueError(f"Unknown variance_type: {variance_type}")

        print(f"\nCreating {variance_type.upper()} split...")
        print(f"  Months: {months}")

        # Train: All data for variance type in train_year
        train_mask = (df['year'] == self.config.train_year) & (df['month'].isin(months))
        train_df = df[train_mask].copy()
        print(f"  Train ({self.config.train_year}): {len(train_df)} records")

        # Val/Test: Split data for variance type in val_test_year
        val_test_mask = (df['year'] == self.config.val_test_year) & (df['month'].isin(months))
        val_test_df = df[val_test_mask].copy().sort_values('date')

        # Split val_test into val and test
        val_size = int(len(val_test_df) * self.config.val_ratio)
        val_df = val_test_df.iloc[:val_size].copy()
        test_df = val_test_df.iloc[val_size:].copy()

        print(f"  Val ({self.config.val_test_year}): {len(val_df)} records")
        print(f"  Test ({self.config.val_test_year}): {len(test_df)} records")

        return train_df, val_df, test_df

    def create_all_splits(self, df: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Create splits for both variance types.

        Returns:
            Dictionary with 'low_variance' and 'high_variance' keys,
            each containing 'train', 'val', 'test' dataframes
        """
        print("\n" + "=" * 80)
        print("CREATING VARIANCE-BASED SPLITS")
        print("=" * 80)

        splits = {}

        # Low variance split
        splits['low_variance'] = {
            'train': None,
            'val': None,
            'test': None,
        }
        train_df, val_df, test_df = self.create_variance_split(df, 'low_variance')
        splits['low_variance']['train'] = train_df
        splits['low_variance']['val'] = val_df
        splits['low_variance']['test'] = test_df

        # High variance split
        splits['high_variance'] = {
            'train': None,
            'val': None,
            'test': None,
        }
        train_df, val_df, test_df = self.create_variance_split(df, 'high_variance')
        splits['high_variance']['train'] = train_df
        splits['high_variance']['val'] = val_df
        splits['high_variance']['test'] = test_df

        # Print summary
        print("\n" + "=" * 80)
        print("SPLIT SUMMARY")
        print("=" * 80)
        for var_type in ['low_variance', 'high_variance']:
            print(f"\n{var_type.upper()}:")
            print(f"  Train: {len(splits[var_type]['train'])} records")
            print(f"  Val:   {len(splits[var_type]['val'])} records")
            print(f"  Test:  {len(splits[var_type]['test'])} records")
            print(f"  Total: {len(splits[var_type]['train']) + len(splits[var_type]['val']) + len(splits[var_type]['test'])} records")

        print("=" * 80)

        return splits

    def get_split_info(self, splits: Dict[str, Dict[str, pd.DataFrame]]) -> Dict:
        """Get summary information about splits."""
        info = {}
        for var_type in ['low_variance', 'high_variance']:
            info[var_type] = {
                'train_size': len(splits[var_type]['train']),
                'val_size': len(splits[var_type]['val']),
                'test_size': len(splits[var_type]['test']),
                'total_size': len(splits[var_type]['train']) + len(splits[var_type]['val']) + len(splits[var_type]['test']),
            }
        return info


def main():
    """Main function to test variance splitting."""
    config = VarianceSplitConfig()
    splitter = VarianceSplitter(config)

    # Load data
    df = splitter.load_data()

    # Create splits
    splits = splitter.create_all_splits(df)

    # Get split info
    info = splitter.get_split_info(splits)
    print("\nSplit Info:")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
