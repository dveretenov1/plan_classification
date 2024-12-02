import cv2
import numpy as np
from pathlib import Path
import json
import shutil
import logging
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)

class ImageSplitter:
    def __init__(self, dataset_path, output_path, grid_size=(4, 4), overlap=0.1):
        """
        Initialize the ImageSplitter for processing large architectural drawings.
        
        Args:
            dataset_path (Path or str): Path to the dataset containing images and labels
            output_path (Path or str): Path where split images and labels will be saved
            grid_size (tuple): Number of rows and columns for splitting (e.g., (4,4) for 16 tiles)
            overlap (float): Overlap between tiles as a fraction (0.1 = 10% overlap)
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.grid_size = grid_size
        self.overlap = overlap
        self.metadata = {}
        
        # Setup output directories
        self.output_images = self.output_path / 'images'
        self.output_labels = self.output_path / 'labels'
        
        logger.info(f"Initialized ImageSplitter with grid size {grid_size} and {overlap*100}% overlap")

    def setup_directories(self):
        """Create necessary output directories."""
        self.output_images.mkdir(parents=True, exist_ok=True)
        self.output_labels.mkdir(parents=True, exist_ok=True)

    def _calculate_tile_dimensions(self, image_size, grid_dimension, overlap):
        """Calculate dimensions and positions of tiles with overlap."""
        base_size = image_size // grid_dimension
        overlap_size = int(base_size * overlap)
        
        # Calculate positions including overlap
        positions = []
        for i in range(grid_dimension):
            start = max(0, i * base_size - overlap_size)
            end = min(image_size, (i + 1) * base_size + overlap_size)
            positions.append((start, end))
            
        return positions

    def process_dataset(self):
        """
        Process all images in the dataset, splitting them into tiles.
        
        Returns:
            Path: Path to the metadata JSON file
        """
        logger.info("Starting dataset splitting process...")
        self.setup_directories()
        
        # Get all image files
        image_files = list(sorted((self.dataset_path / 'images').glob('*.jpg')))
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        for img_path in tqdm(image_files, desc="Processing images"):
            label_path = self.dataset_path / 'labels' / f"{img_path.stem}.txt"
            if not label_path.exists():
                logger.warning(f"No label file found for {img_path.name}")
                continue
            
            try:
                self._process_single_image(img_path, label_path)
            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {str(e)}")
                continue
        
        # Save metadata
        metadata_path = self.output_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        
        logger.info(f"Dataset splitting completed. Metadata saved to {metadata_path}")
        return metadata_path

    def _process_single_image(self, img_path, label_path):
        """Process a single image and its corresponding label file."""
        logger.info(f"Processing {img_path.name}")
        
        # Read image and check dimensions
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not read image {img_path}")
            
        h, w = image.shape[:2]
        logger.debug(f"Image dimensions: {w}x{h}")
        
        # Store original image info
        self.metadata[img_path.stem] = {
            'width': w,
            'height': h,
            'grid_size': self.grid_size,
            'overlap': self.overlap,
            'tiles': {}
        }
        
        # Calculate tile positions with overlap
        x_positions = self._calculate_tile_dimensions(w, self.grid_size[1], self.overlap)
        y_positions = self._calculate_tile_dimensions(h, self.grid_size[0], self.overlap)
        
        # Read labels
        with open(label_path, 'r') as f:
            labels = [line.strip().split() for line in f.readlines()]
        
        # Convert YOLO format to absolute coordinates
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
        for i, (y1, y2) in enumerate(y_positions):
            for j, (x1, x2) in enumerate(x_positions):
                tile_name = f"{img_path.stem}_r{i}c{j}_g{self.grid_size[0]}x{self.grid_size[1]}"
                
                # Store tile metadata
                self.metadata[img_path.stem]['tiles'][f"r{i}c{j}"] = {
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'tile_name': tile_name
                }
                
                # Extract tile
                tile = image[y1:y2, x1:x2].copy()
                tile_w = x2 - x1
                tile_h = y2 - y1
                
                # Find boxes that intersect with this tile
                tile_boxes = []
                for box in boxes:
                    # Calculate intersection
                    intersect_x1 = max(box['x1'], x1)
                    intersect_y1 = max(box['y1'], y1)
                    intersect_x2 = min(box['x2'], x2)
                    intersect_y2 = min(box['y2'], y2)
                    
                    if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
                        # Convert to tile-relative coordinates
                        rel_x1 = (intersect_x1 - x1) / tile_w
                        rel_y1 = (intersect_y1 - y1) / tile_h
                        rel_x2 = (intersect_x2 - x1) / tile_w
                        rel_y2 = (intersect_y2 - y1) / tile_h
                        
                        # Convert to YOLO format
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
                
                # Only save tiles that contain objects
                if tile_boxes:
                    # Save tile image
                    cv2.imwrite(str(self.output_images / f"{tile_name}.jpg"), tile)
                    
                    # Save tile labels
                    with open(self.output_labels / f"{tile_name}.txt", 'w') as f:
                        for box in tile_boxes:
                            f.write(f"{box['class']} {box['x']:.6f} {box['y']:.6f} {box['w']:.6f} {box['h']:.6f}\n")
        
        logger.info(f"Completed processing {img_path.name}")

    def cleanup_temp_files(self):
        """Clean up any temporary files created during processing."""
        temp_files = []  # Add any temp file patterns here
        for pattern in temp_files:
            for file_path in Path('.').glob(pattern):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Could not remove temp file {file_path}: {e}")