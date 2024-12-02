import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from config import RESULTS_DIR

class Plotter:
    def __init__(self, results_dir=None):
        self.results_dir = Path(results_dir) if results_dir else RESULTS_DIR

    def plot_fold_metrics(self, metrics, fold_num):
        """Plot detailed metrics for current fold"""
        metrics_dir = self.results_dir / f'fold_{fold_num}' / 'metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # 1. Bar plot of main metrics
        plt.figure(figsize=(12, 6))
        names = ['Precision', 'Recall', 'mAP50', 'mAP50-95']
        values = [metrics[k] for k in ['precision', 'recall', 'mAP50', 'mAP50-95']]
        
        bars = plt.bar(names, values, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
        plt.title(f'Fold {fold_num} Performance Metrics', fontsize=14, pad=20)
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, max(values) + 0.1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.savefig(metrics_dir / 'metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_training_history(self, results, fold_num):
        """Plot training history metrics"""
        history_dir = self.results_dir / f'fold_{fold_num}' / 'history'
        history_dir.mkdir(parents=True, exist_ok=True)

        # Extract metrics from results
        try:
            # Get the results dictionary
            results_dict = results.results_dict
            
            # Plot losses
            plt.figure(figsize=(15, 5))
            
            # Plot box_loss
            plt.subplot(1, 3, 1)
            if 'metrics/box_loss' in results_dict:
                plt.plot(results_dict['metrics/box_loss'], label='box_loss')
                plt.title('Box Loss Over Time')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.legend()
            
            # Plot cls_loss
            plt.subplot(1, 3, 2)
            if 'metrics/cls_loss' in results_dict:
                plt.plot(results_dict['metrics/cls_loss'], label='cls_loss')
                plt.title('Classification Loss Over Time')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.legend()
            
            # Plot dfl_loss
            plt.subplot(1, 3, 3)
            if 'metrics/dfl_loss' in results_dict:
                plt.plot(results_dict['metrics/dfl_loss'], label='dfl_loss')
                plt.title('DFL Loss Over Time')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(history_dir / 'training_history.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not plot training history for fold {fold_num}: {str(e)}")
            # Continue execution even if plotting fails

    def plot_summary(self, stats):
        """Plot comprehensive summary of all folds"""
        summary_dir = self.results_dir / 'summary'
        summary_dir.mkdir(parents=True, exist_ok=True)

        # 1. Box plot of metrics across folds
        plt.figure(figsize=(12, 6))
        data = []
        labels = []
        for metric, values in stats.items():
            data.append(values['values'])
            labels.append(metric)

        plt.boxplot(data, labels=labels, patch_artist=True)
        plt.title('Distribution of Metrics Across Folds', fontsize=14, pad=20)
        plt.ylabel('Score', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add mean values on top of each box
        means = [np.mean(d) for d in data]
        for i, mean in enumerate(means, 1):
            plt.text(i, mean, f'μ={mean:.3f}', ha='center', va='bottom')
        
        plt.savefig(summary_dir / 'metrics_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Heatmap of metric correlations
        plt.figure(figsize=(10, 8))
        corr_data = np.zeros((len(stats), len(stats)))
        for i, (metric1, data1) in enumerate(stats.items()):
            for j, (metric2, data2) in enumerate(stats.items()):
                corr = np.corrcoef(data1['values'], data2['values'])[0, 1]
                corr_data[i, j] = corr

        sns.heatmap(corr_data, annot=True, fmt='.2f', 
                   xticklabels=list(stats.keys()),
                   yticklabels=list(stats.keys()),
                   cmap='RdYlBu')
        plt.title('Metric Correlations Across Folds')
        plt.tight_layout()
        plt.savefig(summary_dir / 'metric_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_predictions(self, results, fold_num):
        """Create visualization of model predictions"""
        viz_dir = self.results_dir / f'fold_{fold_num}' / 'predictions'
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Save prediction summary
        summary_file = viz_dir / 'prediction_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Prediction Summary for Fold {fold_num}\n")
            f.write("=" * 50 + "\n\n")
            
            total_detections = 0
            all_confidences = []
            
            for i, result in enumerate(results, 1):
                boxes = result.boxes
                f.write(f"Image {i}:\n")
                f.write(f"  Total detections: {len(boxes)}\n")
                
                if len(boxes) > 0:
                    confidences = boxes.conf.cpu().numpy()
                    all_confidences.extend(confidences)
                    
                    f.write(f"  Average confidence: {np.mean(confidences):.4f}\n")
                    f.write(f"  Confidence range: {np.min(confidences):.4f} - {np.max(confidences):.4f}\n")
                    
                    # Per-detection details
                    f.write("  Detections:\n")
                    for j, conf in enumerate(confidences, 1):
                        f.write(f"    {j}. Confidence: {conf:.4f}\n")
                
                total_detections += len(boxes)
                f.write("\n")
            
            # Overall summary
            f.write("\nOverall Summary:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total images processed: {len(results)}\n")
            f.write(f"Total detections: {total_detections}\n")
            if all_confidences:
                f.write(f"Average confidence: {np.mean(all_confidences):.4f}\n")
                f.write(f"Confidence std dev: {np.std(all_confidences):.4f}\n")

        return {
            'total_detections': total_detections,
            'confidences': all_confidences if all_confidences else None
        }