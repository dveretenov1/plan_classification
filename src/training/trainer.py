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
        self.model = YOLO(MODEL_CONFIG['base_model'])
        
        self.results = self.model.train(
            data=str(yaml_path),
            project=str(self.results_dir),
            name=f'fold_{fold}',
            exist_ok=True,
            device=0,
            **TRAIN_CONFIG
        )
        
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
        return YOLO(weights_path)

    def predict(self, source, conf=None):
        """Run prediction on images"""
        if self.model is None:
            raise ValueError("No model has been trained yet")
            
        conf = conf or MODEL_CONFIG['conf_threshold']
        
        results = self.model.predict(
            source=source,
            save=True,
            conf=conf,
            show_boxes=True
        )
        
        return results