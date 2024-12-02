Here's a README file based on your content:

```markdown
# Technical Drawing Box Detection with YOLOv8

This project implements a system for detecting boxes in large technical drawings using YOLOv8. It includes image splitting, model training, and prediction reconstruction capabilities.

---

## Core Features

- Splits large technical drawings into overlapping tiles.
- Cross-validation training with 5 folds.
- Automatic model selection based on validation metrics.
- Prediction reconstruction from tiles to original image size.
- Performance visualization and metrics tracking.

---

## Setup

### Install dependencies
```bash
pip install -r requirements.txt
```

### Directory structure
```
├── dataset/
│   ├── images/
│   └── labels/
├── results/
└── temp/
```

---

## Usage

### Run training pipeline
```python
python main.py
```

### Make predictions on new images
```python
python predict.py
```

---

## Configuration

Key settings in `config.py`:

```python
IMAGE_CONFIG = {
    'grid_size': (4, 4),
    'target_size': 1280
}

TRAIN_CONFIG = {
    'epochs': 50,
    'batch': 2,
    'imgsz': 1280
}

MODEL_CONFIG = {
    'base_model': 'yolov8m.pt',
    'conf_threshold': 0.2,
    'iou_threshold': 0.4
}
```

---

## Components

- **ImageSplitter**: Splits large images into overlapping tiles.
- **DatasetManager**: Handles data organization and cross-validation splits.
- **YOLOTrainer**: Manages model training and validation.
- **PredictionReconstructor**: Reconstructs predictions on original images.
- **MetricsManager**: Tracks and visualizes performance metrics.

---

## Data Format

- **Images**: JPG format in `dataset/images/`
- **Labels**: YOLO format text files in `dataset/labels/`

---

## License

This project is licensed under the MIT License.
```
