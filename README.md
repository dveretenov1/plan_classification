# Technical Drawing Box Detection with YOLOv8

System for detecting boxes in large technical drawings using YOLOv8 with image tiling and reconstruction.

## Requirements
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.7 or higher
- Python 3.10+

## Installation
```bash
pip install -r requirements.txt
```

## Setup
```bash
# Directory structure
├── dataset/
│   ├── images/
│   └── labels/
├── results/
└── temp/
```

## Usage
```python
python main.py
```

## Configuration
Key settings in `config.py`:

```python
IMAGE_CONFIG = {
    'original_sizes': {
        'max_width': 14400,
        'max_height': 10800,
        'typical_width': 12600,
        'typical_height': 9000
    },
    'grid_size': (4, 4),
    'target_size': 1280
}

TRAIN_CONFIG = {
    'epochs': 50,
    'batch': 2,
    'hsv_h': 0.01,
    'hsv_s': 0.2,
    'hsv_v': 0.3,
    'degrees': 0.0,
    'translate': 0.15,
    'scale': 0.4,
    'mosaic': 0.0,
    'mixup': 0.0
}

MODEL_CONFIG = {
    'base_model': 'yolov8m.pt',
    'conf_threshold': 0.2,
    'iou_threshold': 0.4,
    'max_det': 400,
    'agnostic_nms': True
}
```

## Components
- `ImageSplitter`: Large image tiling
- `DatasetManager`: Data and CV splits
- `YOLOTrainer`: Training pipeline
- `PredictionReconstructor`: Full-size reconstruction
- `MetricsManager`: Performance tracking

## Data Format
- Images: JPG format in dataset/images/
- Labels: YOLO format in dataset/labels/

## License
MIT