import torch
import numpy as np
from pathlib import Path
import shutil
from ultralytics import YOLO

class ModelUtils:
    @staticmethod
    def get_device():
        """Get the best available device"""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @staticmethod
    def get_gpu_memory():
        """Get available GPU memory"""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device).total_memory / 1024**2  # MB
            allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
            reserved = torch.cuda.memory_reserved(device) / 1024**2  # MB
            free = total - reserved
            
            return {
                'total': total,
                'allocated': allocated,
                'reserved': reserved,
                'free': free
            }
        return None

    @staticmethod
    def save_model(model, path, fold=None):
        """Save model with metadata"""
        save_path = Path(path)
        if fold is not None:
            save_path = save_path / f'fold_{fold}'
        save_path.mkdir(parents=True, exist_ok=True)
        
        model_path = save_path / 'best.pt'
        model.save(str(model_path))
        return model_path

    @staticmethod
    def load_model(path, device=None):
        """Load model with specific device"""
        if device is None:
            device = ModelUtils.get_device()
        return YOLO(path)

class DataUtils:
    @staticmethod
    def calculate_dataset_stats(dataset_path):
        """Calculate dataset statistics"""
        dataset_path = Path(dataset_path)
        image_files = list(dataset_path.rglob('*.jpg'))
        label_files = list(dataset_path.rglob('*.txt'))
        
        stats = {
            'total_images': len(image_files),
            'total_labels': len(label_files),
            'boxes_per_image': [],
            'image_sizes': []
        }
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                boxes = f.readlines()
                stats['boxes_per_image'].append(len(boxes))
        
        if stats['boxes_per_image']:
            stats.update({
                'avg_boxes': np.mean(stats['boxes_per_image']),
                'std_boxes': np.std(stats['boxes_per_image']),
                'min_boxes': min(stats['boxes_per_image']),
                'max_boxes': max(stats['boxes_per_image'])
            })
        
        return stats

    @staticmethod
    def check_dataset_balance(dataset_path):
        """Check dataset balance across splits"""
        dataset_path = Path(dataset_path)
        splits = ['train', 'valid', 'test']
        balance = {}
        
        for split in splits:
            split_path = dataset_path / split
            if split_path.exists():
                images = list((split_path / 'images').glob('*.jpg'))
                labels = list((split_path / 'labels').glob('*.txt'))
                
                total_boxes = 0
                for label in labels:
                    with open(label, 'r') as f:
                        total_boxes += len(f.readlines())
                
                balance[split] = {
                    'images': len(images),
                    'labels': len(labels),
                    'total_boxes': total_boxes,
                    'boxes_per_image': total_boxes / len(images) if images else 0
                }
        
        return balance

class MetricsUtils:
    @staticmethod
    def calculate_metrics(predictions, targets):
        """Calculate detailed metrics"""
        metrics = {
            'total_predictions': len(predictions),
            'total_targets': len(targets),
            'confidences': [],
            'ious': []
        }
        
        # Calculate confidence scores and IoUs
        for pred, target in zip(predictions, targets):
            if len(pred) > 0:
                metrics['confidences'].extend(pred.conf.cpu().numpy())
                if len(target) > 0:
                    iou = MetricsUtils.calculate_box_iou(pred.boxes.xyxy, target.boxes.xyxy)
                    metrics['ious'].extend(iou.cpu().numpy())
        
        # Calculate summary statistics
        if metrics['confidences']:
            metrics.update({
                'mean_confidence': np.mean(metrics['confidences']),
                'std_confidence': np.std(metrics['confidences'])
            })
        
        if metrics['ious']:
            metrics.update({
                'mean_iou': np.mean(metrics['ious']),
                'std_iou': np.std(metrics['ious'])
            })
        
        return metrics

    @staticmethod
    def calculate_box_iou(boxes1, boxes2):
        """
        Calculate IoU between two sets of bounding boxes.
        
        Args:
            boxes1: tensor of shape (N, 4) containing N boxes first set
            boxes2: tensor of shape (M, 4) containing M boxes second set
            
        Returns:
            IoU: tensor of shape (N, M) containing NxM IoU values for every possible box pair
        """
        area1 = MetricsUtils.box_area(boxes1)
        area2 = MetricsUtils.box_area(boxes2)
        
        # Get the coordinates of intersecting boxes
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top coordinates
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom coordinates
        
        # Calculate intersection area
        wh = (rb - lt).clamp(min=0)  # widths and heights of intersection
        inter = wh[:, :, 0] * wh[:, :, 1]  # intersection areas
        
        # Calculate union area
        union = area1[:, None] + area2 - inter
        
        # Calculate IoU
        iou = inter / (union + 1e-6)  # add small epsilon to avoid division by zero
        return iou
    
    @staticmethod
    def box_area(boxes):
        """
        Calculate area of bounding boxes.
        
        Args:
            boxes: tensor of shape (N, 4) containing N boxes coordinates in format (x1, y1, x2, y2)
            
        Returns:
            areas: tensor of shape (N,) containing areas for each box
        """
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])