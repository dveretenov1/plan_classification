import json
import numpy as np
from pathlib import Path
import cv2
import logging
from ultralytics.utils.ops import non_max_suppression

logger = logging.getLogger(__name__)

class PredictionReconstructor:
    def __init__(self, metadata_path, iou_threshold=0.45):
        """Initialize reconstructor with metadata from splitting process.
        
        Args:
            metadata_path: Path to metadata JSON file created during splitting
            iou_threshold: IoU threshold for merging overlapping predictions
        """
        self.iou_threshold = iou_threshold
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded metadata for {len(self.metadata)} images")
    
    def reconstruct_predictions(self, predictions_dir, output_dir, original_images_dir):
        """Reconstruct predictions from tiles back to original image size.
        
        Args:
            predictions_dir: Directory containing prediction files for tiles
            output_dir: Directory to save reconstructed predictions
            original_images_dir: Directory containing original images (for size reference)
        """
        predictions_dir = Path(predictions_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        reconstructed_predictions = {}
        
        for img_name, img_data in self.metadata.items():
            logger.info(f"Reconstructing predictions for {img_name}")
            
            # Original image dimensions
            orig_width = img_data['width']
            orig_height = img_data['height']
            
            all_boxes = []
            all_scores = []
            all_classes = []
            
            # Gather predictions from all tiles
            for tile_id, tile_info in img_data['tiles'].items():
                tile_name = tile_info['tile_name']
                pred_file = predictions_dir / f"{tile_name}.txt"
                
                if not pred_file.exists():
                    continue
                
                # Read predictions for this tile
                with open(pred_file, 'r') as f:
                    tile_preds = [line.strip().split() for line in f.readlines()]
                
                for pred in tile_preds:
                    class_id = int(pred[0])
                    conf = float(pred[5]) if len(pred) > 5 else 1.0
                    
                    # Convert relative tile coordinates to absolute image coordinates
                    tile_width = tile_info['x2'] - tile_info['x1']
                    tile_height = tile_info['y2'] - tile_info['y1']
                    
                    # Center coordinates and dimensions from YOLO format
                    rel_x_center = float(pred[1])
                    rel_y_center = float(pred[2])
                    rel_width = float(pred[3])
                    rel_height = float(pred[4])
                    
                    # Convert to absolute pixel coordinates
                    abs_x_center = rel_x_center * tile_width + tile_info['x1']
                    abs_y_center = rel_y_center * tile_height + tile_info['y1']
                    abs_width = rel_width * tile_width
                    abs_height = rel_height * tile_height
                    
                    # Convert to corners format
                    x1 = abs_x_center - abs_width / 2
                    y1 = abs_y_center - abs_height / 2
                    x2 = abs_x_center + abs_width / 2
                    y2 = abs_y_center + abs_height / 2
                    
                    # Add to collection
                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(conf)
                    all_classes.append(class_id)
            
            if all_boxes:
                # Convert to numpy arrays
                boxes = np.array(all_boxes)
                scores = np.array(all_scores)
                classes = np.array(all_classes)
                
                # Apply NMS
                indices = non_max_suppression(
                    boxes,
                    scores,
                    self.iou_threshold
                )
                
                # Get final predictions
                final_boxes = boxes[indices]
                final_scores = scores[indices]
                final_classes = classes[indices]
                
                # Convert back to YOLO format and save
                output_file = output_dir / f"{img_name}.txt"
                with open(output_file, 'w') as f:
                    for box, score, class_id in zip(final_boxes, final_scores, final_classes):
                        # Convert to relative coordinates
                        x_center = (box[0] + box[2]) / 2 / orig_width
                        y_center = (box[1] + box[3]) / 2 / orig_height
                        width = (box[2] - box[0]) / orig_width
                        height = (box[3] - box[1]) / orig_height
                        
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}\n")
                
                reconstructed_predictions[img_name] = {
                    'boxes': final_boxes.tolist(),
                    'scores': final_scores.tolist(),
                    'classes': final_classes.tolist()
                }
            
            logger.info(f"Completed reconstruction for {img_name}")
        
        return reconstructed_predictions

    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate IoU between two boxes in [x1, y1, x2, y2] format."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = box1_area + box2_area - intersection
        
        return intersection / (union + 1e-6)