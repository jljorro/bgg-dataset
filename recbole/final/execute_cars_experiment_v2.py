import logging
import os
import time
import gc
import torch
from tqdm import tqdm
import psutil
import numpy as np

from recbole.config import Config
from recbole.data.utils import create_dataset, data_preparation
from recbole.quick_start import load_data_and_model
from recbole.trainer import Trainer

# Imports conditionally to avoid unnecessary load
# from recbole.model.context_aware_recommender import FM, DeepFM

import utils_cars_evaluation_v2 as eval_cars

# PATHS
TRAINING_DATA_PATH = '../../data/{}/{}.{}.train.inter'  # dataset_name, dataset_name, fold
TEST_DATA_PATH = '../../data/{}/{}.{}.test.inter'  # dataset_name, dataset_name, fold
PREDICTIONS_PATH = './predictions/{}_{}_f{}.tsv'  # dataset_name, model_name, fold
K = 15
USER_BATCH_SIZE = 32   # Reduced for lower memory usage
ITEM_BATCH_SIZE = 1024  # Reduced for lower memory usage

def clear_memory():
    """Aggressively cleans memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Sync to ensure the GPU has finished
        torch.cuda.synchronize()
    
    # Force Python to free unused memory
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_info = py.memory_info()
    logging.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def log_memory_usage():
    """Records current memory usage"""
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_info = py.memory_info()
    logging.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        logging.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")

def load_model(path):
    """Load the model with improved memory management"""
    # Clear memory before loading a new model
    clear_memory()
    
    # Log memory usage before loading
    logging.info(f"Memory before loading model {path}")
    log_memory_usage()
    
    # Load the model
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=path)
    
    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Registrar uso de memoria después de cargar
    logging.info(f"Memory after loading model {path}")
    log_memory_usage()
    
    return model, dataset, train_data, valid_data, test_data, config

def process_model(model_file):
    """Process a single model with proper memory management"""
    start_time = time.time()
    model_path = os.path.join('saved', model_file)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Loading model from {model_path}")
        model, dataset, train_data, valid_data, test_data, config = load_model(model_path)
        
        model_name = config['model']
        dataset_name = dataset.dataset_name
        fold = config['benchmark_filename'][0].split('.')[0]
        
        logger.info(f"Model: {model_name}, Dataset: {dataset_name}, Fold: {fold}")
        
        # Get all available item IDs
        all_item_ids = dataset.get_item_feature().numpy()['game_id']
        logger.info(f"Total number of items: {len(all_item_ids)}")
        
        # Get unique users and their test items
        users, test_items = eval_cars.get_unique_users_and_test_items(test_data)
        logger.info(f"Total number of test users: {len(users)}")
        
        # Set the model to evaluation mode
        model.eval()
        
        # Path to save predictions
        predictions_path = PREDICTIONS_PATH.format(dataset_name, model_name, fold)
        
        # Process in batches of users
        with torch.no_grad():
            for start_idx in tqdm(range(0, len(users), USER_BATCH_SIZE), desc=f"Processing {model_name}"):
                end_idx = min(start_idx + USER_BATCH_SIZE, len(users))
                
                user_batch = users[start_idx:end_idx]
                test_item_batch = test_items[start_idx:end_idx]
                
                # Get predictions for this batch of users
                batch_predictions = eval_cars.process_user_batch(
                    user_batch, 
                    test_item_batch, 
                    all_item_ids, 
                    model, 
                    dataset, 
                    k=K, 
                    item_batch_size=ITEM_BATCH_SIZE
                )
                
                # Save predictions
                eval_cars.save_predictions(predictions_path, batch_predictions)
                
                # Free memory after each user batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Periodic memory log
                if start_idx % (USER_BATCH_SIZE * 10) == 0:
                    log_memory_usage()
        
        # Calculate and record execution time
        elapsed_time = time.time() - start_time
        logger.info(f"Finished processing {model_name} on {dataset_name} fold {fold}. Time elapsed: {elapsed_time:.2f} seconds")
        
        # Explicitly delete objects to free memory
        del model, dataset, train_data, valid_data, test_data
        clear_memory()
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {model_file}: {str(e)}")
        # Still clean memory even if there is an error
        clear_memory()
        return False

def execute_cars_experiment():
    """Run the experiment with improved memory handling between models"""
    logger = init_logger()
    logger.info('Starting CARS experiment with optimized implementation and memory management')
    logger.info(f'Device: {torch.device("cuda" if torch.cuda.is_available() else "cpu")}')

    try:
        # Install psutil if not already installed
        import psutil
    except ImportError:
        logger.warning("psutil is not installed. Install it with 'pip install psutil' for memory monitoring.")
    
    # Display information about batch processing settings
    logger.info(f'User batch size: {USER_BATCH_SIZE}, Item batch size: {ITEM_BATCH_SIZE}, Top-K: {K}')
    
    # Register initial memory
    log_memory_usage()

    # List model files
    models = [f for f in os.listdir('saved') if f.endswith('.pth')]
    logger.info(f"Models found: {len(models)}")

    # Process each model independently
    for i, model_file in enumerate(models):
        logger.info(f"Starting model {i+1}/{len(models)}: {model_file}")
        success = process_model(model_file)
        
        # Force memory release after each model
        clear_memory()
        
        # Log memory usage after processing the model
        logger.info(f"Memory after processing model {model_file}")
        log_memory_usage()
        
        # Optional: Short pause between models to allow the OS to clean up memory
        time.sleep(2)

    logger.info('CARS experiment finished')

def init_logger():
    """
    Initializes the logger to record model information.
    """
    # Create a directory to save the logs if it doesn't exist
    if not os.path.exists('logs_ranks'):
        os.makedirs('logs_ranks')

    # Configure the logger
    logging.basicConfig(
        filename='logs_ranks/get_rankings_optimized.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Also show logs in console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging.getLogger(__name__)

if __name__ == '__main__':
    # Try to install psutil if it is not available
    try:
        import psutil
    except ImportError:
        print("Installing psutil for memory monitoring...")
        import subprocess
        subprocess.check_call(["pip", "install", "psutil"])
        import psutil
    
    try:
        execute_cars_experiment()
    except KeyboardInterrupt:
        logging.info("Experiment interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
    finally:
        # Ensure memory release even if there is an error
        logging.info("Freeing final memory...")
        clear_memory()
        logging.info("Experiment completed")