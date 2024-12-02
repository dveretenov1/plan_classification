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
from src.data.prediction_reconstructor import PredictionReconstructor

# Import configurations
from config import (
    DATASET_DIR, 
    RESULTS_DIR, 
    TRAIN_CONFIG, 
    MODEL_CONFIG, 
    DATASET_CONFIG, 
    BASE_DIR,
    IMAGE_CONFIG,
    TEMP_DIR
)

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

def cleanup_temp_files():
    """Clean up temporary files and directories"""
    try:
        if TEMP_DIR.exists():
            logger.info(f"Cleaning up temporary directory: {TEMP_DIR}")
            shutil.rmtree(TEMP_DIR)
            logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.warning(f"Error during cleanup: {str(e)}")

def cleanup_failed_run(run_dir):
    """Clean up a failed run directory"""
    try:
        if run_dir.exists():
            logger.info(f"Cleaning up failed run directory: {run_dir}")
            shutil.rmtree(run_dir)
            logger.info("Failed run cleanup completed successfully")
    except Exception as e:
        logger.warning(f"Error during failed run cleanup: {str(e)}")

def find_best_model(run_dir, best_fold):
    """
    Find the best model weights by checking multiple possible locations.
    Returns the path if found, raises FileNotFoundError if not found.
    """
    possible_paths = [
        RESULTS_DIR / f'fold_{best_fold}' / 'weights' / 'best.pt',  # Primary YOLOv8 path
        run_dir / f'fold_{best_fold}' / 'weights' / 'best.pt',
        run_dir / f'fold_{best_fold}' / 'training' / 'weights' / 'best.pt',
        run_dir / f'fold_{best_fold}' / 'best.pt',
    ]
    
    for path in possible_paths:
        logger.debug(f"Checking for model at: {path}")
        if path.exists():
            logger.info(f"Found best model at: {path}")
            return path
            
    paths_tried = '\n'.join([f"- {p}" for p in possible_paths])
    raise FileNotFoundError(
        f"Best model not found. Tried the following paths:\n{paths_tried}"
    )

def main():
    run_dir = None
    success = False
    
    try:
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RESULTS_DIR / f'run_{timestamp}'
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created run directory: {run_dir}")
        
        # Clean up any existing temp files before starting
        cleanup_temp_files()
        
        # 1. First split into train and test sets
        logger.info("Step 1: Splitting dataset into train and test sets")
        logger.info(f"Processing large images up to {IMAGE_CONFIG['original_sizes']['max_width']}x{IMAGE_CONFIG['original_sizes']['max_height']} pixels")
        
        data_utils = DataUtils()
        all_images = sorted(list((DATASET_DIR / 'images').glob('*.jpg')))
        
        if not all_images:
            raise FileNotFoundError(f"No images found in {DATASET_DIR / 'images'}")
        
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
        splitter = ImageSplitter(
            train_dir, 
            train_split_dir, 
            DATASET_CONFIG['grid_size'],
            overlap=DATASET_CONFIG.get('overlap', 0.1)
        )
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
            
            try:
                # Setup fold directory structure
                yaml_path = dataset_manager.setup_fold(train_idx, val_idx, fold)
                
                # Train model
                model, results = trainer.train_fold(yaml_path, fold)
                
                # Validate model
                val_results = trainer.validate(yaml_path)
                metrics = metrics_manager.save_fold_metrics(val_results, fold)
                fold_metrics.append(metrics)
                
                # Plot metrics
                plotter.plot_fold_metrics(metrics, fold)
                if results is not None:  # Only plot if training succeeded
                    plotter.plot_training_history(results, fold)
                
                logger.info(f"Completed fold {fold} training")
                
            except Exception as e:
                logger.error(f"Error in fold {fold}: {str(e)}")
                continue
        
        if not fold_metrics:
            raise RuntimeError("No successful folds completed")
        
        # 5. Analyze Results and Select Best Model
        stats = metrics_manager.summarize_results(fold_metrics)
        plotter.plot_summary(stats)
        
        best_fold = np.argmax([m['mAP50'] for m in fold_metrics]) + 1
        logger.info(f"Best performing model was from fold {best_fold}")
        
        # 6. Process Test Set
        logger.info("\nProcessing test set")
        test_split_dir = run_dir / 'test_splits'
        test_splitter = ImageSplitter(
            test_dir, 
            test_split_dir, 
            DATASET_CONFIG['grid_size'],
            overlap=DATASET_CONFIG.get('overlap', 0.1)
        )
        test_metadata_path = test_splitter.process_dataset()
        
        # Load best model
        try:
            best_model_path = find_best_model(run_dir, best_fold)
            logger.info(f"Loading best model from {best_model_path}")
            model = ModelUtils.load_model(best_model_path)
        except FileNotFoundError as e:
            logger.error(str(e))
            raise
        
        # Run predictions on test splits
        predictions_dir = run_dir / 'predictions'
        predictions_dir.mkdir(parents=True, exist_ok=True)
        
        results = model.predict(
            source=str(test_split_dir / 'images'),
            conf=MODEL_CONFIG['conf_threshold'],
            iou=MODEL_CONFIG['iou_threshold'],
            save=True,
            save_txt=True,
            project=str(predictions_dir),
            name='test_predictions'
        )
        
        # Reconstruct predictions for full images
        reconstructor = PredictionReconstructor(
            test_metadata_path,
            iou_threshold=MODEL_CONFIG['iou_threshold']
        )
        
        final_predictions = reconstructor.reconstruct_predictions(
            predictions_dir / 'test_predictions' / 'labels',
            run_dir / 'final_predictions',
            test_dir / 'images'
        )
        
        # Save predictions visualization
        plotter.plot_predictions(results, 'test')
        
        logger.info(f"Pipeline completed successfully! Results saved in {run_dir}")
        success = True
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        if run_dir:
            cleanup_failed_run(run_dir)
        raise
    
    finally:
        # Always cleanup temp files
        cleanup_temp_files()
        # If the run failed, clean up the run directory
        if not success and run_dir:
            cleanup_failed_run(run_dir)

if __name__ == "__main__":
    main()