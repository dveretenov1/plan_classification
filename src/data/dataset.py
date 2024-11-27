from pathlib import Path
import shutil
import yaml
from config import BASE_DIR, TEMP_DIR, DATASET_DIR

class DatasetManager:
    def __init__(self):
        self.base_dir = BASE_DIR
        self.temp_dir = TEMP_DIR
        self.dataset_dir = DATASET_DIR
        self.images_dir = self.dataset_dir / 'images'
        self.labels_dir = self.dataset_dir / 'labels'
        self.all_images = []
        self.all_labels = []

    def gather_data(self):
        """Gather all images and corresponding labels from the unified directory"""
        print("Gathering all data...")
        
        self.all_images = []
        self.all_labels = []
        
        # Ensure directories exist
        if not self.images_dir.exists() or not self.labels_dir.exists():
            raise FileNotFoundError(f"Dataset directories not found at {self.dataset_dir}")
        
        # Gather all image files and their corresponding labels
        for img_path in sorted(self.images_dir.glob('*.jpg')):  # sorted for consistency
            label_path = self.labels_dir / (img_path.stem + '.txt')
            if label_path.exists():
                self.all_images.append(img_path)
                self.all_labels.append(label_path)
            else:
                print(f"Warning: No label file found for {img_path.name}")
        
        if not self.all_images:
            raise ValueError("No valid image-label pairs found in dataset")
        
        print(f"Total valid image-label pairs found: {len(self.all_images)}")
        return self.all_images, self.all_labels

    def setup_fold(self, train_idx, val_idx, fold):
        """Setup directory structure for current fold"""
        print(f"Setting up fold {fold}...")
        
        # Create fold directory structure
        fold_dir = self.temp_dir / f'fold_{fold}'
        train_img_dir = fold_dir / 'train' / 'images'
        train_label_dir = fold_dir / 'train' / 'labels'
        val_img_dir = fold_dir / 'valid' / 'images'
        val_label_dir = fold_dir / 'valid' / 'labels'
        
        # Remove existing fold directory if it exists
        if fold_dir.exists():
            shutil.rmtree(fold_dir)
        
        # Create new directories
        for d in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Copy files for this fold
        for idx in train_idx:
            shutil.copy2(self.all_images[idx], train_img_dir)
            shutil.copy2(self.all_labels[idx], train_label_dir)
        
        for idx in val_idx:
            shutil.copy2(self.all_images[idx], val_img_dir)
            shutil.copy2(self.all_labels[idx], val_label_dir)
        
        print(f"Fold {fold} setup complete with {len(train_idx)} training and {len(val_idx)} validation samples")
        return fold_dir

    def create_yaml(self, fold_dir, fold):
        """Create YAML file for current fold"""
        yaml_content = {
            'path': str(fold_dir.absolute()),  # Use absolute path
            'train': str(Path('train/images')),  # Relative paths for train/val
            'val': str(Path('valid/images')),
            'nc': 1,  # Number of classes
            'names': ['Box']  # Class names
        }
        
        yaml_path = fold_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        return yaml_path

    def cleanup(self):
        """Clean up temporary directories"""
        if self.temp_dir.exists():
            print("Cleaning up temporary directories...")
            shutil.rmtree(self.temp_dir)
            print("Cleanup complete")