import logging
import sys
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
import shutil
import yaml
import numpy as np

# Import from src directory
from src.data.image_splitter import ImageSplitter
from src.data.dataset import DatasetManager
from src.training.trainer import YOLOTrainer
from src.visualization.metrics import MetricsManager
from src.visualization.plotting import Plotter
from src.training.utils import ModelUtils, DataUtils

# Import configurations
from config import DATASET_DIR, RESULTS_DIR, TRAIN_CONFIG, MODEL_CONFIG, DATASET_CONFIG, BASE_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RESULTS_DIR / f'run_{timestamp}'
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. First split into train and test sets
        logger.info("Step 1: Splitting dataset into train and test sets")
        data_utils = DataUtils()
        all_images = sorted(list((DATASET_DIR / 'images').glob('*.jpg')))
        
        train_images, test_images = train_test_split(
            all_images,
            test_size=DATASET_CONFIG['test_size'],
            random_state=DATASET_CONFIG['random_state']
        )
        
        # Create directories for split data
        train_dir = run_dir / 'train_data'
        test_dir = run_dir / 'test_data'
        for d in [train_dir, test_dir]:
            (d / 'images').mkdir(parents=True)
            (d / 'labels').mkdir(parents=True)
        
        # Copy files to respective directories
        for img in train_images:
            shutil.copy2(img, train_dir / 'images')
            label = DATASET_DIR / 'labels' / f"{img.stem}.txt"
            if label.exists():
                shutil.copy2(label, train_dir / 'labels')
        
        for img in test_images:
            shutil.copy2(img, test_dir / 'images')
            label = DATASET_DIR / 'labels' / f"{img.stem}.txt"
            if label.exists():
                shutil.copy2(label, test_dir / 'labels')
        
        # 2. Split training images into grid
        logger.info("Step 2: Splitting training images into grid")
        train_split_dir = run_dir / 'train_splits'
        splitter = ImageSplitter(train_dir, train_split_dir, DATASET_CONFIG['grid_size'])
        train_metadata_path = splitter.process_dataset()

        # 3. Setup Cross-Validation
        dataset_manager = DatasetManager(train_split_dir)
        dataset_manager.gather_data()
        kf = KFold(
            n_splits=DATASET_CONFIG['n_splits'],
            shuffle=True,
            random_state=DATASET_CONFIG['random_state']
        )
        
        # Store metrics for each fold
        metrics_manager = MetricsManager()
        fold_metrics = []
        trainer = YOLOTrainer()
        plotter = Plotter(run_dir)

        # 4. Train and Validate Each Fold
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_manager.all_images), 1):
            logger.info(f"\nProcessing Fold {fold}/{DATASET_CONFIG['n_splits']}")
            
            # Setup fold directory structure
            fold_dir = run_dir / f'fold_{fold}'
            yaml_path = dataset_manager.setup_fold(train_idx, val_idx, fold)

            # Train model
            model, results = trainer.train_fold(Path(BASE_DIR / 'data.yaml'), fold)
            
            # Validate model
            val_results = trainer.validate(yaml_path)
            metrics = metrics_manager.save_fold_metrics(val_results, fold)
            fold_metrics.append(metrics)
            
            # Plot metrics
            plotter.plot_fold_metrics(metrics, fold)
            plotter.plot_training_history(results, fold)
        
        # 5. Analyze Results and Select Best Model
        stats = metrics_manager.summarize_results(fold_metrics)
        plotter.plot_summary(stats)
        
        best_fold = np.argmax([m['mAP50'] for m in fold_metrics]) + 1
        logger.info(f"Best performing model was from fold {best_fold}")
        
        # 6. Process Test Set
        logger.info("\nProcessing test set")
        test_split_dir = run_dir / 'test_splits'
        test_splitter = ImageSplitter(test_dir, test_split_dir, DATASET_CONFIG['grid_size'])
        test_metadata_path = test_splitter.process_dataset()
        
        # Load best model
        best_model_path = run_dir / f'fold_{best_fold}/training/weights/best.pt'
        model = ModelUtils.load_model(best_model_path)
        
        # Run predictions on test splits
        predictions_dir = run_dir / 'predictions'
        results = model.predict(
            source=str(test_split_dir / 'images'),
            conf=MODEL_CONFIG['conf_threshold'],
            save=True,
            save_txt=True,
            project=str(predictions_dir),
            name='test_predictions'
        )
        
        # Save predictions visualization
        plotter.plot_predictions(results, 'test')
        
        logger.info(f"Pipeline completed successfully! Results saved in {run_dir}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()