from pathlib import Path

# Base paths - modify for Discovery
BASE_DIR = Path('.')  # Adjust path
RESULTS_DIR = BASE_DIR / 'results'
DATASET_DIR = BASE_DIR / 'dataset'
TEMP_DIR = BASE_DIR / 'temp'

# Training configuration
TRAIN_CONFIG = {
    'epochs': 100,
    'imgsz': 1280,
    'batch': 8,
    'patience': 50,
    
    # Augmentation parameters
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'translate': 0.2,
    'scale': 0.5,
    'flipud': 0.5,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.1,
    
    # Optimization parameters
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    
    # Loss parameters
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    
    # Additional parameters
    'overlap_mask': True,
    'mask_ratio': 4,
    'dropout': 0.2
}

# Model configuration
MODEL_CONFIG = {
    'base_model': 'yolov8n.pt',
    'conf_threshold': 0.25
}

# Dataset configuration
DATASET_CONFIG = {
    'n_splits': 5,
    'random_state': 42,
    'test_size': 0.2,  # 20% of data for testing
    'grid_size': (4, 4)  # Default grid size for splitting images
}