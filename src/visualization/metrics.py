from pathlib import Path
import numpy as np
from config import RESULTS_DIR

class MetricsManager:
    def __init__(self):
        self.results_dir = RESULTS_DIR

    def save_fold_metrics(self, val_results, fold_num):
        """Save metrics for current fold"""
        metrics_dir = self.results_dir / f'fold_{fold_num}' / 'metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Get metrics
        metrics = {
            'precision': val_results.results_dict['metrics/precision(B)'],
            'recall': val_results.results_dict['metrics/recall(B)'],
            'mAP50': val_results.results_dict['metrics/mAP50(B)'],
            'mAP50-95': val_results.results_dict['metrics/mAP50-95(B)']
        }
        
        # Save metrics to text file
        metrics_file = metrics_dir / 'metrics.txt'
        with open(metrics_file, 'w') as f:
            f.write(f"Metrics for Fold {fold_num}\n")
            f.write("=" * 30 + "\n")
            for name, value in metrics.items():
                f.write(f"{name}: {value:.4f}\n")
        
        return metrics

    def summarize_results(self, all_metrics):
        """Summarize results across all folds"""
        summary_dir = self.results_dir / 'summary'
        summary_dir.mkdir(exist_ok=True)
        
        # Calculate statistics
        stats = {}
        for metric in ['precision', 'recall', 'mAP50', 'mAP50-95']:
            values = [m[metric] for m in all_metrics]
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        # Save numerical results
        with open(summary_dir / 'results.txt', 'w') as f:
            f.write("Cross-Validation Results Summary\n")
            f.write("=" * 30 + "\n\n")
            for metric, data in stats.items():
                f.write(f"{metric}:\n")
                f.write(f"  Mean: {data['mean']:.4f}\n")
                f.write(f"  Std:  {data['std']:.4f}\n")
                f.write("  Values: " + ", ".join(f"{v:.4f}" for v in data['values']) + "\n\n")
        
        return stats