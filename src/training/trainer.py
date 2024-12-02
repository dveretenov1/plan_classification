from pathlib import Path
import shutil
from ultralytics import YOLO
from config import TRAIN_CONFIG, MODEL_CONFIG, RESULTS_DIR

class YOLOTrainer:
    def __init__(self):
        self.results_dir = RESULTS_DIR
        self.model = None
        self.results = None

    def train_fold(self, yaml_path, fold):
        """Train model for current fold"""
        print(f"\nTraining fold {fold}...")
        
        # Initialize model
        self.model = YOLO(MODEL_CONFIG['base_model'])
        
        # Create training config without invalid parameters
        train_args = TRAIN_CONFIG.copy()
        
        # Remove parameters that might cause issues
        params_to_remove = [
            'pin_memory', 'cuda_cache_disable', 'overlap_mask', 
            'deterministic', 'mask_ratio', 'dropout'
        ]
        
        for param in params_to_remove:
            if param in train_args:
                del train_args[param]
        
        # Train with specific naming pattern
        model_args = {
            'data': str(yaml_path),
            'project': str(self.results_dir),
            'name': f'fold_{fold}',
            'exist_ok': True,
        }
        
        # Combine valid arguments
        train_args.update(model_args)
        
        # Train the model
        self.results = self.model.train(**train_args)
        
        return self.model, self.results

    def validate(self, data_yaml):
        """Validate the model"""
        if self.model is None:
            raise ValueError("No model has been trained yet")
        
        val_results = self.model.val(
            data=str(data_yaml)
        )
        
        return val_results

    @staticmethod
    def load_model(weights_path):
        """Load a trained model"""
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
        return YOLO(weights_path)

    def predict(self, source, conf=None):
        """Run prediction on images"""
        if self.model is None:
            raise ValueError("No model has been trained yet")
            
        pred_args = MODEL_CONFIG.copy()
        pred_args['source'] = source
        pred_args['conf'] = conf or pred_args.get('conf', 0.2)
        
        results = self.model.predict(**pred_args)
        
        return results