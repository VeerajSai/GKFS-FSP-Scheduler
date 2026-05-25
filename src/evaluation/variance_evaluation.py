"""
Variance-Based Model Evaluation Module - V4
============================================

Comprehensive evaluation of variance-based models including:
- Performance metrics calculation
- Comparison with baseline and V3 results
- Visualization generation
- Summary report generation
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    predictions_dir: str = "outputs/predictions_variance_v4"
    reports_dir: str = "outputs/reports_variance_v4"
    models_dir: str = "outputs/models_variance_v4"
    v3_results_path: Optional[str] = None
    baseline_mae: float = 9.23  # From V3 baseline


class VarianceModelEvaluator:
    """Evaluates variance-based models."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.predictions_dir = Path(config.predictions_dir)
        self.reports_dir = Path(config.reports_dir)
        self.models_dir = Path(config.models_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.comparisons: Dict[str, pd.DataFrame] = {}

    def load_predictions(self, var_type: str, split: str = 'test') -> pd.DataFrame:
        """Load predictions for a variance type."""
        pred_path = self.predictions_dir / var_type / f"{split}_predictions.csv"
        if pred_path.exists():
            return pd.read_csv(pred_path)
        else:
            raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # sMAPE (symmetric Mean Absolute Percentage Error)
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'smape': smape,
            'mape': mape,
        }

    def evaluate_variance_type(self, var_type: str) -> Dict[str, Dict[str, float]]:
        """Evaluate all models for a variance type."""
        print(f"\nEvaluating {var_type.upper()} models...")

        # Load test predictions
        test_df = self.load_predictions(var_type, 'test')

        # Get actual values
        y_true = test_df['actual_power'].values

        # Get prediction columns
        pred_cols = [col for col in test_df.columns if col.startswith('pred_')]

        # Calculate metrics for each model
        metrics = {}
        for pred_col in pred_cols:
            model_name = pred_col.replace('pred_', '')
            y_pred = test_df[pred_col].values
            metrics[model_name] = self.calculate_metrics(y_true, y_pred)

        self.metrics[var_type] = metrics
        return metrics

    def compare_with_baseline(self, var_type: str) -> pd.DataFrame:
        """Compare model performance with baseline."""
        metrics = self.metrics.get(var_type, {})

        comparison_data = []
        for model_name, model_metrics in metrics.items():
            mae = model_metrics['mae']
            improvement = ((self.config.baseline_mae - mae) / self.config.baseline_mae) * 100

            comparison_data.append({
                'model': model_name,
                'mae': mae,
                'rmse': model_metrics['rmse'],
                'r2': model_metrics['r2'],
                'smape': model_metrics['smape'],
                'improvement_vs_baseline': improvement,
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('mae')

        self.comparisons[f'{var_type}_vs_baseline'] = comparison_df
        return comparison_df

    def compare_with_v3(self) -> pd.DataFrame:
        """Compare V4 results with V3 seasonal results."""
        # V3 results from the plan
        v3_results = {
            'WINTER': {'mae': 2.88, 'best_model': 'ridge'},
            'SPRING': {'mae': 8.36, 'best_model': 'random_forest'},
            'SUMMER': {'mae': 7.21, 'best_model': 'ridge'},
            'FALL': {'mae': 3.74, 'best_model': 'ridge'},
        }

        # Map seasons to variance types
        # LOW_VARIANCE: WINTER (Dec, Jan, Feb), FALL (Sep, Oct, Nov)
        # HIGH_VARIANCE: SPRING (Mar, Apr, May), SUMMER (Jun, Jul, Aug)

        comparison_data = []

        # Low variance comparison
        low_metrics = self.metrics.get('low_variance', {})
        if low_metrics:
            best_low_model = min(low_metrics.items(), key=lambda x: x[1]['mae'])
            low_mae = best_low_model[1]['mae']

            # Average of WINTER and FALL
            v3_low_avg = (v3_results['WINTER']['mae'] + v3_results['FALL']['mae']) / 2
            improvement = ((v3_low_avg - low_mae) / v3_low_avg) * 100

            comparison_data.append({
                'variance_type': 'low_variance',
                'v4_mae': low_mae,
                'v4_best_model': best_low_model[0],
                'v3_avg_mae': v3_low_avg,
                'v3_seasons': 'WINTER + FALL',
                'improvement_vs_v3': improvement,
            })

        # High variance comparison
        high_metrics = self.metrics.get('high_variance', {})
        if high_metrics:
            best_high_model = min(high_metrics.items(), key=lambda x: x[1]['mae'])
            high_mae = best_high_model[1]['mae']

            # Average of SPRING and SUMMER
            v3_high_avg = (v3_results['SPRING']['mae'] + v3_results['SUMMER']['mae']) / 2
            improvement = ((v3_high_avg - high_mae) / v3_high_avg) * 100

            comparison_data.append({
                'variance_type': 'high_variance',
                'v4_mae': high_mae,
                'v4_best_model': best_high_model[0],
                'v3_avg_mae': v3_high_avg,
                'v3_seasons': 'SPRING + SUMMER',
                'improvement_vs_v3': improvement,
            })

        comparison_df = pd.DataFrame(comparison_data)
        self.comparisons['v4_vs_v3'] = comparison_df
        return comparison_df

    def visualize_performance(self):
        """Generate performance visualizations."""
        print("\nGenerating visualizations...")

        # 1. Model Performance Comparison (MAE)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for idx, var_type in enumerate(['low_variance', 'high_variance']):
            metrics = self.metrics.get(var_type, {})
            if not metrics:
                continue

            models = list(metrics.keys())
            maes = [metrics[m]['mae'] for m in models]

            colors = ['green' if mae <= self.config.baseline_mae else 'red' for mae in maes]
            axes[idx].barh(models, maes, color=colors)
            axes[idx].axvline(x=self.config.baseline_mae, color='blue', linestyle='--',
                            label=f'Baseline ({self.config.baseline_mae:.2f})')
            axes[idx].set_xlabel('MAE (MW)', fontsize=12)
            axes[idx].set_title(f'{var_type.upper()} - Model MAE', fontsize=14, fontweight='bold')
            axes[idx].legend()

        plt.tight_layout()
        plt.savefig(self.reports_dir / 'model_mae_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Comprehensive Metrics Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        metrics_names = ['mae', 'rmse', 'r2', 'smape']
        for idx, metric_name in enumerate(metrics_names):
            ax = axes[idx // 2, idx % 2]

            x_pos = np.arange(len(['low_variance', 'high_variance']))
            width = 0.15

            for var_idx, var_type in enumerate(['low_variance', 'high_variance']):
                metrics = self.metrics.get(var_type, {})
                if not metrics:
                    continue

                models = list(metrics.keys())
                values = [metrics[m][metric_name] for m in models]

                offset = (np.arange(len(models)) - len(models)/2 + 0.5) * width
                ax.bar(x_pos[var_idx] + offset, values, width, label=models)

            ax.set_xlabel('Variance Type', fontsize=12)
            ax.set_ylabel(metric_name.upper(), fontsize=12)
            ax.set_title(f'{metric_name.upper()} by Model', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(['LOW', 'HIGH'])
            ax.legend()

        plt.tight_layout()
        plt.savefig(self.reports_dir / 'comprehensive_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Improvement vs Baseline
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for idx, var_type in enumerate(['low_variance', 'high_variance']):
            comparison = self.comparisons.get(f'{var_type}_vs_baseline')
            if comparison is None:
                comparison = self.compare_with_baseline(var_type)

            models = comparison['model'].values
            improvements = comparison['improvement_vs_baseline'].values

            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            axes[idx].barh(models, improvements, color=colors)
            axes[idx].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            axes[idx].set_xlabel('Improvement vs Baseline (%)', fontsize=12)
            axes[idx].set_title(f'{var_type.upper()} - Improvement vs Baseline', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.reports_dir / 'improvement_vs_baseline.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. V4 vs V3 Comparison
        comparison_v3 = self.compare_with_v3()
        if not comparison_v3.empty:
            fig, ax = plt.subplots(figsize=(10, 6))

            x_pos = np.arange(len(comparison_v3))
            width = 0.35

            ax.bar(x_pos - width/2, comparison_v3['v3_avg_mae'], width, label='V3 (Seasonal)', color='orange')
            ax.bar(x_pos + width/2, comparison_v3['v4_mae'], width, label='V4 (Variance)', color='green')

            ax.set_xlabel('Variance Type', fontsize=12)
            ax.set_ylabel('MAE (MW)', fontsize=12)
            ax.set_title('V4 vs V3 Performance Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([vt.upper() for vt in comparison_v3['variance_type']])
            ax.legend()

            # Add improvement annotations
            for i, row in comparison_v3.iterrows():
                ax.annotate(f"{row['improvement_vs_v3']:.1f}%",
                           xy=(i, min(row['v3_avg_mae'], row['v4_mae'])),
                           xytext=(0, -20), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold')

            plt.tight_layout()
            plt.savefig(self.reports_dir / 'v4_vs_v3_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Visualizations saved to {self.reports_dir}")

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\nGenerating summary report...")

        # Load metrics from models directory if not already loaded
        for var_type in ['low_variance', 'high_variance']:
            if var_type not in self.metrics:
                metrics_path = self.models_dir / var_type / "metrics.json"
                if metrics_path.exists():
                    with open(metrics_path, 'r') as f:
                        self.metrics[var_type] = json.load(f)

        # Generate comparisons
        for var_type in ['low_variance', 'high_variance']:
            self.compare_with_baseline(var_type)

        v3_comparison = self.compare_with_v3()

        # Create summary
        summary = {
            'version': '4.0.0',
            'baseline_mae': self.config.baseline_mae,
            'low_variance_months': [1, 2, 3, 4, 10, 11],
            'high_variance_months': [5, 6, 7, 8, 9, 12],
            'low_variance_metrics': self.metrics.get('low_variance', {}),
            'high_variance_metrics': self.metrics.get('high_variance', {}),
            'v3_comparison': v3_comparison.to_dict('records') if not v3_comparison.empty else [],
        }

        # Save summary JSON
        summary_path = self.reports_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {summary_path}")

        # Create summary CSV
        summary_data = []
        for var_type in ['low_variance', 'high_variance']:
            for model_name, metrics in self.metrics.get(var_type, {}).items():
                mae = metrics['mae']
                improvement = ((self.config.baseline_mae - mae) / self.config.baseline_mae) * 100

                summary_data.append({
                    'variance_type': var_type,
                    'model': model_name,
                    'mae': mae,
                    'rmse': metrics['rmse'],
                    'r2': metrics['r2'],
                    'smape': metrics['smape'],
                    'improvement_vs_baseline': improvement,
                })

        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = self.reports_dir / "evaluation_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"  Saved: {summary_csv_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("=" * 80)

        # Print V3 comparison
        if not v3_comparison.empty:
            print("\nV4 vs V3 Comparison:")
            print(v3_comparison.to_string(index=False))
            print("=" * 80)

    def run(self):
        """Run complete evaluation pipeline."""
        print("=" * 80)
        print("VARIANCE-BASED MODEL EVALUATION - V4")
        print("=" * 80)

        # Evaluate each variance type
        for var_type in ['low_variance', 'high_variance']:
            self.evaluate_variance_type(var_type)

        # Generate comparisons
        for var_type in ['low_variance', 'high_variance']:
            self.compare_with_baseline(var_type)

        # Compare with V3
        self.compare_with_v3()

        # Generate visualizations
        self.visualize_performance()

        # Generate summary report
        self.generate_summary_report()

        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)


def main():
    """Main function."""
    config = EvaluationConfig()
    evaluator = VarianceModelEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
