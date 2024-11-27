from pathlib import Path
import logging
import sys
from datetime import datetime

# Import from src directory with correct structure
from src.training.utils import ModelUtils
from src.data.image_splitter import ImageSplitter
from config import RESULTS_DIR, MODEL_CONFIG, DATASET_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('prediction.log')
    ]
)
logger = logging.getLogger(__name__)

def run_prediction():
    try:
        # Setup directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_dir = RESULTS_DIR / f'predictions_{timestamp}'
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        # Get latest model
        model_dirs = sorted(list(RESULTS_DIR.glob('run_*')))
        if not model_dirs:
            raise FileNotFoundError("No trained models found in results directory")
        
        latest_run = model_dirs[-1]
        
        # Find best model from latest run
        model_files = list(latest_run.rglob('best.pt'))
        if not model_files:
            raise FileNotFoundError(f"No model weights found in {latest_run}")
        
        model_path = model_files[0]
        logger.info(f"Using model: {model_path}")
        
        # Load the model
        model = ModelUtils.load_model(model_path)
        
        # Process each image in dataset/images
        img_dir = Path('dataset/images')
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found at {img_dir}")
        
        images = list(img_dir.glob('*.jpg'))
        logger.info(f"Found {len(images)} images to process")
        
        # Process each image
        for img_path in images:
            logger.info(f"Processing {img_path.name}")
            
            # Split image
            split_dir = pred_dir / 'splits' / img_path.stem
            splitter = ImageSplitter(img_path.parent, split_dir, DATASET_CONFIG['grid_size'])
            metadata_path = splitter.process_dataset()
            
            # Run prediction on splits
            results = model.predict(
                source=str(split_dir / 'images'),
                conf=MODEL_CONFIG['conf_threshold'],
                save=True,
                save_txt=True,
                project=str(pred_dir),
                name=img_path.stem
            )
            
            logger.info(f"Completed processing {img_path.name}")
        
        logger.info(f"\nPredictions completed! Results saved in: {pred_dir}")
        
        # List generated files
        logger.info("\nGenerated files:")
        for file in pred_dir.rglob('*'):
            if file.is_file():
                logger.info(f"- {file.relative_to(pred_dir)}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    run_prediction()