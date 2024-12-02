from pathlib import Path

# Base paths
BASE_DIR = Path('.')  # Current directory
RESULTS_DIR = BASE_DIR / 'results'
DATASET_DIR = BASE_DIR / 'dataset'
TEMP_DIR = BASE_DIR / 'temp'

# Create necessary directories
for d in [RESULTS_DIR, DATASET_DIR, TEMP_DIR]:
    d.mkdir(exist_ok=True)

# Image Processing Configuration
IMAGE_CONFIG = {
    'original_sizes': {
        'max_width': 14400,   # Maximum width in dataset
        'max_height': 10800,  # Maximum height in dataset
        'typical_width': 12600,  # Another common width
        'typical_height': 9000   # Another common height
    },
    'grid_size': (4, 4),  # Splits into 16 tiles
    'target_size': 960,   # Reduced from 1280
    
    'max_tile_size': {
        'width': 14400 // 4,   # ~3600 pixels
        'height': 10800 // 4   # ~2700 pixels
    }
}

# Training configuration
TRAIN_CONFIG = {
    'epochs': 50,             # Training epochs
    'imgsz': 960,            # Reduced image size
    'batch': 1,              # Minimal batch size for memory
    'patience': 15,          # Reduced early stopping patience
    'workers': 4,            # Reduced worker threads
    
    # Augmentation parameters tuned for technical drawings
    'hsv_h': 0.01,   # Very minimal hue change
    'hsv_s': 0.2,    # Minimal saturation change
    'hsv_v': 0.3,    # Moderate brightness augmentation
    'degrees': 0.0,  # No rotation for technical drawings
    'translate': 0.15,  # Slight translation
    'scale': 0.4,    # Scale variation
    'flipud': 0.0,   # No vertical flip
    'fliplr': 0.0,   # No horizontal flip
    'mosaic': 0.0,   # Disabled for technical drawings
    'mixup': 0.0,    # Disabled for technical drawings
    
    # Optimization parameters
    'optimizer': 'AdamW',
    'lr0': 0.0005,    # Initial learning rate
    'lrf': 0.005,     # Final learning rate
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 5,  # Increased warmup epochs
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    
    # Loss parameters optimized for technical drawings
    'box': 8.0,      # Box loss weight
    'cls': 0.5,      # Classification loss weight
    'dfl': 2.0,      # DFL loss weight
    
    # Memory optimization parameters
    'mask_ratio': 4,
    'dropout': 0.1,
    'amp': True,          # Mixed precision training
    'cache': False,       # No caching to save memory
    
    # Additional training stability
    'nbs': 64,           # Nominal batch size
    'close_mosaic': 0,
    'rect': True,        # Enable rectangular training
    'plots': True,       # Generate training plots
    
    # Device settings
    'deterministic': True  # Enable deterministic training
}

# Model configuration
MODEL_CONFIG = {
    'base_model': 'yolov8s.pt',  # Using smaller model
    'conf': 0.45,      # Lower confidence threshold
    'iou': 0.4,       # Adjusted for overlapping boxes
    'max_det': 300,             # Reduced max detections
    'agnostic_nms': True,       # Class-agnostic NMS
    'device': 0,               # Use first GPU
    'save': True,           # Save results
    'name': 'exp',          # Name of results folder
    'exist_ok': True,       # Overwrite existing folder
}

# Dataset configuration
DATASET_CONFIG = {
    'n_splits': 5,
    'random_state': 42,
    'test_size': 0.2,
    'grid_size': IMAGE_CONFIG['grid_size'],
    'overlap': 0.10  # Reduced overlap
}