import json
import numpy as np
from pathlib import Path
import cv2
import logging

logger = logging.getLogger(__name__)

class PredictionReconstructor:
    def __init__(self, metadata_path, iou_threshold=0.45):
        """Initialize reconstructor with metadata from splitting process."""
        self.iou_threshold = iou_threshold
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded metadata for {len(self.metadata)} images")
    
    def reconstruct_predictions(self, predictions_dir, output_dir, original_images_dir):
        """Reconstruct predictions from tiles back to original image size."""
        predictions_dir = Path(predictions_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for labels and visualizations
        labels_dir = output_dir / 'labels'
        viz_dir = output_dir / 'visualizations'
        labels_dir.mkdir(exist_ok=True)
        viz_dir.mkdir(exist_ok=True)
        
        reconstructed_predictions = {}
        
        original_images_dir = Path(original_images_dir)
        
        for img_name, img_data in self.metadata.items():
            logger.info(f"Reconstructing predictions for {img_name}")
            
            # Original image dimensions
            orig_width = img_data['width']
            orig_height = img_data['height']
            
            # Load original image for visualization
            img_path = original_images_dir / f"{img_name}.jpg"
            if not img_path.exists():
                logger.warning(f"Original image not found: {img_path}")
                continue
            
            original_image = cv2.imread(str(img_path))
            if original_image is None:
                logger.error(f"Failed to read image: {img_path}")
                continue
            
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
                
                # Perform NMS
                keep = []
                for class_id in np.unique(classes):
                    class_mask = classes == class_id
                    class_boxes = boxes[class_mask]
                    class_scores = scores[class_mask]
                    
                    # Sort by confidence
                    sorted_idx = np.argsort(-class_scores)
                    class_boxes = class_boxes[sorted_idx]
                    class_scores = class_scores[sorted_idx]
                    
                    keep_class = []
                    while len(class_boxes) > 0:
                        keep_class.append(sorted_idx[0])
                        
                        if len(class_boxes) == 1:
                            break
                            
                        ious = self.calculate_iou(class_boxes[0], class_boxes[1:])
                        mask = ious < self.iou_threshold
                        class_boxes = class_boxes[1:][mask]
                        class_scores = class_scores[1:][mask]
                        sorted_idx = sorted_idx[1:][mask]
                    
                    keep.extend([idx for idx in keep_class])
                
                # Get final predictions
                final_boxes = boxes[keep]
                final_scores = scores[keep]
                final_classes = classes[keep]
                
                # Save YOLO format labels
                label_file = labels_dir / f"{img_name}.txt"
                with open(label_file, 'w') as f:
                    for box, score, class_id in zip(final_boxes, final_scores, final_classes):
                        # Convert to relative coordinates
                        x_center = (box[0] + box[2]) / 2 / orig_width
                        y_center = (box[1] + box[3]) / 2 / orig_height
                        width = (box[2] - box[0]) / orig_width
                        height = (box[3] - box[1]) / orig_height
                        
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}\n")
                
                # Create visualization
                vis_image = original_image.copy()
                for box, score, class_id in zip(final_boxes, final_scores, final_classes):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Draw box
                    color = (0, 255, 0)  # Green color for boxes
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"Box {score:.2f}"
                    labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    y1_label = max(y1, labelSize[1] + 10)
                    cv2.rectangle(vis_image, (x1, y1_label - labelSize[1] - 10), 
                                (x1 + labelSize[0], y1_label + 10), color, cv2.FILLED)
                    cv2.putText(vis_image, label, (x1, y1_label),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Save visualization
                viz_path = viz_dir / f"{img_name}_predictions.jpg"
                cv2.imwrite(str(viz_path), vis_image)
                
                reconstructed_predictions[img_name] = {
                    'boxes': final_boxes.tolist(),
                    'scores': final_scores.tolist(),
                    'classes': final_classes.tolist()
                }
            
            logger.info(f"Completed reconstruction for {img_name}")
        
        return reconstructed_predictions

    @staticmethod
    def calculate_iou(box, boxes):
        """Calculate IoU between a box and an array of boxes."""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area
        
        iou = intersection_area / (union_area + 1e-6)
        return iou