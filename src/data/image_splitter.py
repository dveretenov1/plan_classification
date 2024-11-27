import cv2
import numpy as np
from pathlib import Path
import json
import shutil
import logging

logger = logging.getLogger(__name__)

class ImageSplitter:
    def __init__(self, dataset_path, output_path, grid_size=(4, 4)):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.grid_size = grid_size
        self.metadata = {}
        
        self.output_images = self.output_path / 'images'
        self.output_labels = self.output_path / 'labels'
        
    def setup_directories(self):
        """Create necessary directories"""
        self.output_images.mkdir(parents=True, exist_ok=True)
        self.output_labels.mkdir(parents=True, exist_ok=True)

    def process_dataset(self):
        """Process all images and labels in the dataset"""
        logger.info("Starting dataset splitting process...")
        self.setup_directories()
        
        image_files = list(sorted((self.dataset_path / 'images').glob('*.jpg')))
        logger.info(f"Found {len(image_files)} images to process")
        
        for img_path in image_files:
            label_path = self.dataset_path / 'labels' / f"{img_path.stem}.txt"
            if not label_path.exists():
                logger.warning(f"No label file found for {img_path.name}")
                continue
                
            self.process_image_and_labels(img_path, label_path)
            
        metadata_path = self.output_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)
            
        logger.info(f"Dataset splitting completed. Metadata saved to {metadata_path}")
        return metadata_path

    def process_image_and_labels(self, img_path, label_path):
        """Process a single image and its labels"""
        logger.info(f"Processing {img_path.name}")
        
        image = cv2.imread(str(img_path))
        if image is None:
            logger.error(f"Could not read image {img_path}")
            return
            
        h, w = image.shape[:2]
        
        self.metadata[img_path.stem] = {
            'width': w,
            'height': h,
            'grid_size': self.grid_size,
            'tiles': {}
        }
        
        tile_w = w // self.grid_size[1]
        tile_h = h // self.grid_size[0]
        
        # Read and parse labels
        with open(label_path, 'r') as f:
            labels = [line.strip().split() for line in f.readlines()]
        
        boxes = []
        for label in labels:
            class_id = int(label[0])
            x_center, y_center = float(label[1]) * w, float(label[2]) * h
            width, height = float(label[3]) * w, float(label[4]) * h
            boxes.append({
                'class': class_id,
                'x1': x_center - width/2,
                'y1': y_center - height/2,
                'x2': x_center + width/2,
                'y2': y_center + height/2
            })
        
        # Process each tile
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                x1 = j * tile_w
                y1 = i * tile_h
                x2 = (j + 1) * tile_w
                y2 = (i + 1) * tile_h
                
                tile_name = f"{img_path.stem}_r{i}c{j}_g{self.grid_size[0]}x{self.grid_size[1]}"
                
                self.metadata[img_path.stem]['tiles'][f"r{i}c{j}"] = {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'tile_name': tile_name
                }
                
                tile = image[y1:y2, x1:x2]
                
                # Process boxes for this tile
                tile_boxes = []
                for box in boxes:
                    intersect_x1 = max(box['x1'], x1)
                    intersect_y1 = max(box['y1'], y1)
                    intersect_x2 = min(box['x2'], x2)
                    intersect_y2 = min(box['y2'], y2)
                    
                    if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
                        rel_x1 = (intersect_x1 - x1) / tile_w
                        rel_y1 = (intersect_y1 - y1) / tile_h
                        rel_x2 = (intersect_x2 - x1) / tile_w
                        rel_y2 = (intersect_y2 - y1) / tile_h
                        
                        center_x = (rel_x1 + rel_x2) / 2
                        center_y = (rel_y1 + rel_y2) / 2
                        width = rel_x2 - rel_x1
                        height = rel_y2 - rel_y1
                        
                        tile_boxes.append({
                            'class': box['class'],
                            'x': center_x,
                            'y': center_y,
                            'w': width,
                            'h': height
                        })
                
                if tile_boxes:  # Only save tiles that contain objects
                    cv2.imwrite(str(self.output_images / f"{tile_name}.jpg"), tile)
                    
                    with open(self.output_labels / f"{tile_name}.txt", 'w') as f:
                        for box in tile_boxes:
                            f.write(f"{box['class']} {box['x']:.6f} {box['y']:.6f} {box['w']:.6f} {box['h']:.6f}\n")
                            
        logger.info(f"Completed processing {img_path.name}")

class PredictionReconstructor:
    def __init__(self, metadata_path):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

    def reconstruct_predictions(self, predictions_dir, output_dir, original_images_dir):
        """Reconstruct predictions from split images"""
        logger.info("Starting prediction reconstruction")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir = Path(predictions_dir)
        
        reconstructed_predictions = {}
        
        for img_name, img_data in self.metadata.items():
            logger.info(f"Reconstructing predictions for {img_name}")
            
            reconstructed_boxes = []
            
            for tile_id, tile_info in img_data['tiles'].items():
                tile_name = tile_info['tile_name']
                pred_file = predictions_dir / f"{tile_name}.txt"
                
                if not pred_file.exists():
                    continue
                
                with open(pred_file, 'r') as f:
                    tile_preds = [line.strip().split() for line in f.readlines()]
                
                tile_w = tile_info['x2'] - tile_info['x1']
                tile_h = tile_info['y2'] - tile_info['y1']
                
                for pred in tile_preds:
                    class_id = int(pred[0])
                    conf = float(pred[5]) if len(pred) > 5 else 1.0
                    
                    x_center = float(pred[1]) * tile_w + tile_info['x1']
                    y_center = float(pred[2]) * tile_h + tile_info['y1']
                    width = float(pred[3]) * tile_w
                    height = float(pred[4]) * tile_h
                    
                    reconstructed_boxes.append({
                        'class': class_id,
                        'x': x_center / img_data['width'],
                        'y': y_center / img_data['height'],
                        'w': width / img_data['width'],
                        'h': height / img_data['height'],
                        'conf': conf
                    })
            
            # Save reconstructed predictions
            output_file = output_dir / f"{img_name}.txt"
            with open(output_file, 'w') as f:
                for box in reconstructed_boxes:
                    f.write(f"{box['class']} {box['x']:.6f} {box['y']:.6f} {box['w']:.6f} {box['h']:.6f}")
                    if 'conf' in box:
                        f.write(f" {box['conf']:.6f}")
                    f.write("\n")
            
            reconstructed_predictions[img_name] = reconstructed_boxes
        
        logger.info("Prediction reconstruction completed")
        return reconstructed_predictions