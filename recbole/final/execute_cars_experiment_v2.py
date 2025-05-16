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

# Importaciones condicionalmente para evitar carga innecesaria
# from recbole.model.context_aware_recommender import FM, DeepFM

import utils_cars_evaluation_v2 as eval_cars

# PATHS
TRAINING_DATA_PATH = '../../data/{}/{}.{}.train.inter'  # dataset_name, dataset_name, fold
TEST_DATA_PATH = '../../data/{}/{}.{}.test.inter'  # dataset_name, dataset_name, fold
PREDICTIONS_PATH = './predictions/{}_{}_f{}.tsv'  # dataset_name, model_name, fold
K = 15
USER_BATCH_SIZE = 32   # Reducido para menor uso de memoria
ITEM_BATCH_SIZE = 1024  # Reducido para menor uso de memoria

def clear_memory():
    """Limpia agresivamente la memoria"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Sincronizar para asegurar que la GPU ha terminado
        torch.cuda.synchronize()
    
    # Forzar Python a liberar memoria no utilizada
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_info = py.memory_info()
    logging.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def log_memory_usage():
    """Registra el uso actual de memoria"""
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_info = py.memory_info()
    logging.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        logging.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")

def load_model(path):
    """Carga el modelo con gestión de memoria mejorada"""
    # Limpiar memoria antes de cargar un nuevo modelo
    clear_memory()
    
    # Registrar uso de memoria antes de cargar
    logging.info(f"Memory before loading model {path}")
    log_memory_usage()
    
    # Cargar el modelo
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=path)
    
    # Mover el modelo a GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Registrar uso de memoria después de cargar
    logging.info(f"Memory after loading model {path}")
    log_memory_usage()
    
    return model, dataset, train_data, valid_data, test_data, config

def process_model(model_file):
    """Procesa un solo modelo con gestión adecuada de memoria"""
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
        
        # Obtener todos los IDs de items disponibles
        all_item_ids = dataset.get_item_feature().numpy()['game_id']
        logger.info(f"Total number of items: {len(all_item_ids)}")
        
        # Obtener usuarios únicos y sus items de test
        users, test_items = eval_cars.get_unique_users_and_test_items(test_data)
        logger.info(f"Total number of test users: {len(users)}")
        
        # Establecer el modelo en modo evaluación
        model.eval()
        
        # Ruta para guardar las predicciones
        predictions_path = PREDICTIONS_PATH.format(dataset_name, model_name, fold)
        
        # Procesar en lotes de usuarios
        with torch.no_grad():
            for start_idx in tqdm(range(0, len(users), USER_BATCH_SIZE), desc=f"Processing {model_name}"):
                end_idx = min(start_idx + USER_BATCH_SIZE, len(users))
                
                user_batch = users[start_idx:end_idx]
                test_item_batch = test_items[start_idx:end_idx]
                
                # Obtener predicciones para este lote de usuarios
                batch_predictions = eval_cars.process_user_batch(
                    user_batch, 
                    test_item_batch, 
                    all_item_ids, 
                    model, 
                    dataset, 
                    k=K, 
                    item_batch_size=ITEM_BATCH_SIZE
                )
                
                # Guardar predicciones
                eval_cars.save_predictions(predictions_path, batch_predictions)
                
                # Liberar memoria después de cada lote de usuario
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Registro periódico de memoria
                if start_idx % (USER_BATCH_SIZE * 10) == 0:
                    log_memory_usage()
        
        # Calcular y registrar tiempo de ejecución
        elapsed_time = time.time() - start_time
        logger.info(f"Finished processing {model_name} on {dataset_name} fold {fold}. Time elapsed: {elapsed_time:.2f} seconds")
        
        # Explícitamente eliminar objetos para liberar memoria
        del model, dataset, train_data, valid_data, test_data
        clear_memory()
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {model_file}: {str(e)}")
        # Aún así limpiar memoria incluso si hay error
        clear_memory()
        return False

def execute_cars_experiment():
    """Ejecuta el experimento con manejo mejorado de memoria entre modelos"""
    logger = init_logger()
    logger.info('Starting CARS experiment with optimized implementation and memory management')
    logger.info(f'Device: {torch.device("cuda" if torch.cuda.is_available() else "cpu")}')

    try:
        # Instalar psutil si no está ya instalado
        import psutil
    except ImportError:
        logger.warning("psutil no está instalado. Instálalo con 'pip install psutil' para monitoreo de memoria.")
    
    # Mostrar información sobre la configuración de procesamiento por lotes
    logger.info(f'User batch size: {USER_BATCH_SIZE}, Item batch size: {ITEM_BATCH_SIZE}, Top-K: {K}')
    
    # Registrar memoria inicial
    log_memory_usage()

    # Listar archivos de modelos
    models = [f for f in os.listdir('saved') if f.endswith('.pth')]
    logger.info(f"Models found: {len(models)}")

    # Procesar cada modelo de forma independiente
    for i, model_file in enumerate(models):
        logger.info(f"Starting model {i+1}/{len(models)}: {model_file}")
        success = process_model(model_file)
        
        # Forzar liberación de memoria después de cada modelo
        clear_memory()
        
        # Registrar uso de memoria después de procesar el modelo
        logger.info(f"Memory after processing model {model_file}")
        log_memory_usage()
        
        # Opcional: Pequeña pausa entre modelos para permitir que el SO limpie memoria
        time.sleep(2)

    logger.info('CARS experiment finished')

def init_logger():
    """
    Inicializa el logger para registrar la información del modelo.
    """
    # Crear un directorio para guardar los logs si no existe
    if not os.path.exists('logs_ranks'):
        os.makedirs('logs_ranks')

    # Configurar el logger
    logging.basicConfig(
        filename='logs_ranks/get_rankings_optimized.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # También mostrar logs en consola
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging.getLogger(__name__)

if __name__ == '__main__':
    # Intentar instalar psutil si no está disponible
    try:
        import psutil
    except ImportError:
        print("Instalando psutil para monitoreo de memoria...")
        import subprocess
        subprocess.check_call(["pip", "install", "psutil"])
        import psutil
    
    try:
        execute_cars_experiment()
    except KeyboardInterrupt:
        logging.info("Experimento interrumpido por el usuario")
    except Exception as e:
        logging.error(f"Error inesperado: {str(e)}", exc_info=True)
    finally:
        # Asegurar liberación de memoria incluso si hay error
        logging.info("Liberando memoria final...")
        clear_memory()
        logging.info("Experimento finalizado")