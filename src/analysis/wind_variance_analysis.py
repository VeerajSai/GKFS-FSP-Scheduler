"""
Wind Variance Analysis Module - V4
===================================

Analyzes wind variance patterns to determine optimal variance-based splitting strategy.
Calculates monthly and seasonal variance statistics and classifies months into
HIGH_VARIANCE and LOW_VARIANCE categories.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


@dataclass
class VarianceAnalysisConfig:
    """Configuration for variance analysis."""
    data_path: str = "data/processed/sample_pss_dataset.parquet"
    output_dir: str = "outputs/wind_variance_analysis"
    wind_col: str = "actual_windspeed"
    power_col: str = "actual_power"
    threshold_method: str = "median"  # 'median', 'percentile_75', 'percentile_25'
    percentile_value: float = 75.0


@dataclass
class VarianceResults:
    """Results from variance analysis."""
    monthly_variance: pd.DataFrame
    seasonal_variance: pd.DataFrame
    threshold: float
    low_variance_months: List[int]
    high_variance_months: List[int]
    power_wind_correlation: float


class WindVarianceAnalyzer:
    """Analyzes wind variance patterns for model splitting."""

    def __init__(self, config: VarianceAnalysisConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess data."""
        print(f"Loading data from {self.config.data_path}")
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

    def calculate_monthly_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate variance statistics by month."""
        print("\nCalculating monthly variance statistics...")

        monthly_stats = []
        for month in range(1, 13):
            month_df = df[df['month'] == month]
            if len(month_df) == 0:
                continue

            wind_speed = month_df[self.config.wind_col]
            power = month_df[self.config.power_col]

            stats = {
                'month': month,
                'month_name': pd.to_datetime(month, format='%m').strftime('%B'),
                'count': len(month_df),
                'wind_mean': wind_speed.mean(),
                'wind_std': wind_speed.std(),
                'wind_variance': wind_speed.var(),
                'wind_min': wind_speed.min(),
                'wind_max': wind_speed.max(),
                'wind_range': wind_speed.max() - wind_speed.min(),
                'wind_cv': (wind_speed.std() / (wind_speed.mean() + 1e-8)) * 100,
                'power_mean': power.mean(),
                'power_std': power.std(),
                'power_variance': power.var(),
            }
            monthly_stats.append(stats)

        monthly_df = pd.DataFrame(monthly_stats)
        monthly_df = monthly_df.sort_values('month')

        return monthly_df

    def calculate_seasonal_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate variance statistics by season."""
        print("\nCalculating seasonal variance statistics...")

        seasons = {
            'WINTER': [12, 1, 2],
            'SPRING': [3, 4, 5],
            'SUMMER': [6, 7, 8],
            'FALL': [9, 10, 11],
        }

        seasonal_stats = []
        for season_name, months in seasons.items():
            season_df = df[df['month'].isin(months)]
            if len(season_df) == 0:
                continue

            wind_speed = season_df[self.config.wind_col]
            power = season_df[self.config.power_col]

            stats = {
                'season': season_name,
                'months': ', '.join([str(m) for m in months]),
                'count': len(season_df),
                'wind_mean': wind_speed.mean(),
                'wind_std': wind_speed.std(),
                'wind_variance': wind_speed.var(),
                'wind_min': wind_speed.min(),
                'wind_max': wind_speed.max(),
                'wind_range': wind_speed.max() - wind_speed.min(),
                'wind_cv': (wind_speed.std() / (wind_speed.mean() + 1e-8)) * 100,
                'power_mean': power.mean(),
                'power_std': power.std(),
                'power_variance': power.var(),
            }
            seasonal_stats.append(stats)

        seasonal_df = pd.DataFrame(seasonal_stats)

        return seasonal_df

    def calculate_power_wind_correlation(self, df: pd.DataFrame) -> float:
        """Calculate correlation between wind speed and power."""
        correlation = df[[self.config.wind_col, self.config.power_col]].corr().iloc[0, 1]
        return correlation

    def determine_threshold(self, monthly_variance: pd.DataFrame) -> float:
        """Determine variance threshold based on method."""
        if self.config.threshold_method == 'median':
            threshold = monthly_variance['wind_variance'].median()
        elif self.config.threshold_method == 'percentile_75':
            threshold = monthly_variance['wind_variance'].quantile(0.75)
        elif self.config.threshold_method == 'percentile_25':
            threshold = monthly_variance['wind_variance'].quantile(0.25)
        else:
            threshold = monthly_variance['wind_variance'].median()

        return threshold

    def classify_months(self, monthly_variance: pd.DataFrame, threshold: float) -> Tuple[List[int], List[int]]:
        """Classify months into HIGH_VARIANCE and LOW_VARIANCE."""
        low_variance_months = monthly_variance[
            monthly_variance['wind_variance'] <= threshold
        ]['month'].tolist()

        high_variance_months = monthly_variance[
            monthly_variance['wind_variance'] > threshold
        ]['month'].tolist()

        return low_variance_months, high_variance_months

    def visualize_variance(self, monthly_variance: pd.DataFrame, seasonal_variance: pd.DataFrame):
        """Create visualizations for variance analysis."""
        print("\nGenerating visualizations...")

        # 1. Monthly Wind Variance Bar Chart
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = ['green' if v <= monthly_variance['wind_variance'].median() else 'red'
                  for v in monthly_variance['wind_variance']]
        bars = ax.bar(monthly_variance['month_name'], monthly_variance['wind_variance'], color=colors)
        ax.axhline(y=monthly_variance['wind_variance'].median(), color='blue', linestyle='--',
                   label='Median Threshold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Wind Variance (m/s)2', fontsize=12)
        ax.set_title('Monthly Wind Variance', fontsize=14, fontweight='bold')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'monthly_wind_variance.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Monthly Wind Coefficient of Variation
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = ['green' if v <= monthly_variance['wind_cv'].median() else 'red'
                  for v in monthly_variance['wind_cv']]
        bars = ax.bar(monthly_variance['month_name'], monthly_variance['wind_cv'], color=colors)
        ax.axhline(y=monthly_variance['wind_cv'].median(), color='blue', linestyle='--',
                   label='Median CV')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Coefficient of Variation (%)', fontsize=12)
        ax.set_title('Monthly Wind Coefficient of Variation', fontsize=14, fontweight='bold')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'monthly_wind_cv.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Seasonal Variance Comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Wind variance by season
        axes[0].bar(seasonal_variance['season'], seasonal_variance['wind_variance'],
                   color=['green', 'red', 'red', 'green'])
        axes[0].set_xlabel('Season', fontsize=12)
        axes[0].set_ylabel('Wind Variance (m/s)2', fontsize=12)
        axes[0].set_title('Wind Variance by Season', fontsize=14, fontweight='bold')

        # Wind CV by season
        axes[1].bar(seasonal_variance['season'], seasonal_variance['wind_cv'],
                   color=['green', 'red', 'red', 'green'])
        axes[1].set_xlabel('Season', fontsize=12)
        axes[1].set_ylabel('Coefficient of Variation (%)', fontsize=12)
        axes[1].set_title('Wind Coefficient of Variation by Season', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'seasonal_variance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Wind Speed Distribution by Variance Type
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Low variance months
        low_months = monthly_variance[monthly_variance['wind_variance'] <= monthly_variance['wind_variance'].median()]['month'].tolist()
        high_months = monthly_variance[monthly_variance['wind_variance'] > monthly_variance['wind_variance'].median()]['month'].tolist()

        axes[0].bar(monthly_variance['month_name'], monthly_variance['wind_mean'],
                   color=['green' if m in low_months else 'red' for m in monthly_variance['month']])
        axes[0].set_xlabel('Month', fontsize=12)
        axes[0].set_ylabel('Mean Wind Speed (m/s)', fontsize=12)
        axes[0].set_title('Mean Wind Speed by Month', fontsize=14, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)

        axes[1].bar(monthly_variance['month_name'], monthly_variance['wind_range'],
                   color=['green' if m in low_months else 'red' for m in monthly_variance['month']])
        axes[1].set_xlabel('Month', fontsize=12)
        axes[1].set_ylabel('Wind Speed Range (m/s)', fontsize=12)
        axes[1].set_title('Wind Speed Range by Month', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'wind_speed_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualizations saved to {self.output_dir}")

    def save_results(self, results: VarianceResults):
        """Save analysis results to files."""
        print("\nSaving analysis results...")

        # Save monthly variance
        monthly_variance_path = self.output_dir / 'monthly_variance.csv'
        results.monthly_variance.to_csv(monthly_variance_path, index=False)
        print(f"  - Monthly variance: {monthly_variance_path}")

        # Save seasonal variance
        seasonal_variance_path = self.output_dir / 'seasonal_variance.csv'
        results.seasonal_variance.to_csv(seasonal_variance_path, index=False)
        print(f"  - Seasonal variance: {seasonal_variance_path}")

        # Save variance classification
        classification = {
            'threshold_method': self.config.threshold_method,
            'threshold_value': results.threshold,
            'low_variance_months': results.low_variance_months,
            'high_variance_months': results.high_variance_months,
            'low_variance_month_names': [pd.to_datetime(m, format='%m').strftime('%B')
                                          for m in results.low_variance_months],
            'high_variance_month_names': [pd.to_datetime(m, format='%m').strftime('%B')
                                           for m in results.high_variance_months],
            'power_wind_correlation': results.power_wind_correlation,
        }

        classification_path = self.output_dir / 'variance_classification.json'
        with open(classification_path, 'w') as f:
            json.dump(classification, f, indent=2)
        print(f"  - Variance classification: {classification_path}")

        # Save as CSV for easy reading
        classification_df = pd.DataFrame([{
            'threshold_method': self.config.threshold_method,
            'threshold_value': results.threshold,
            'low_variance_months': ', '.join(map(str, results.low_variance_months)),
            'high_variance_months': ', '.join(map(str, results.high_variance_months)),
            'low_variance_month_names': ', '.join([pd.to_datetime(m, format='%m').strftime('%B')
                                                     for m in results.low_variance_months]),
            'high_variance_month_names': ', '.join([pd.to_datetime(m, format='%m').strftime('%B')
                                                      for m in results.high_variance_months]),
            'power_wind_correlation': results.power_wind_correlation,
        }])
        classification_csv_path = self.output_dir / 'variance_classification.csv'
        classification_df.to_csv(classification_csv_path, index=False)
        print(f"  - Variance classification CSV: {classification_csv_path}")

    def analyze(self) -> VarianceResults:
        """Run complete variance analysis."""
        print("=" * 80)
        print("WIND VARIANCE ANALYSIS - V4")
        print("=" * 80)

        # Load data
        df = self.load_data()

        # Calculate variance statistics
        monthly_variance = self.calculate_monthly_variance(df)
        seasonal_variance = self.calculate_seasonal_variance(df)

        # Calculate power-wind correlation
        power_wind_correlation = self.calculate_power_wind_correlation(df)
        print(f"\nPower-Wind Correlation: {power_wind_correlation:.4f}")

        # Determine threshold
        threshold = self.determine_threshold(monthly_variance)
        print(f"\nVariance Threshold ({self.config.threshold_method}): {threshold:.4f}")

        # Classify months
        low_variance_months, high_variance_months = self.classify_months(monthly_variance, threshold)

        print(f"\nLOW_VARIANCE Months: {low_variance_months}")
        print(f"  -> {[pd.to_datetime(m, format='%m').strftime('%B') for m in low_variance_months]}")
        print(f"\nHIGH_VARIANCE Months: {high_variance_months}")
        print(f"  -> {[pd.to_datetime(m, format='%m').strftime('%B') for m in high_variance_months]}")

        # Create results object
        results = VarianceResults(
            monthly_variance=monthly_variance,
            seasonal_variance=seasonal_variance,
            threshold=threshold,
            low_variance_months=low_variance_months,
            high_variance_months=high_variance_months,
            power_wind_correlation=power_wind_correlation,
        )

        # Generate visualizations
        self.visualize_variance(monthly_variance, seasonal_variance)

        # Save results
        self.save_results(results)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)

        return results


def main():
    """Main function to run variance analysis."""
    config = VarianceAnalysisConfig()
    analyzer = WindVarianceAnalyzer(config)
    results = analyzer.analyze()

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Threshold Method: {config.threshold_method}")
    print(f"Threshold Value: {results.threshold:.4f}")
    print(f"Low Variance Months: {results.low_variance_months}")
    print(f"High Variance Months: {results.high_variance_months}")
    print(f"Power-Wind Correlation: {results.power_wind_correlation:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
