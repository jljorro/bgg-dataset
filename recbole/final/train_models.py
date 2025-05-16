from recbole.quick_start import run_recbole

# PATHS
CONFIGS_DATASET_PATH = 'configs/datasets/dataset_{}.yml' # Dataset
CONFIGS_BENCHMARK_PATH = 'configs/benchmarks/benchmark_f{}.yml' # Fold
CONFIGS_HYPERPARAMETER_PATH = 'configs/hyperparameters/{}_{}.yml' # Algorithm, Datasets
CONFIGS_MODEL_PATH = 'configs/models/{}_config.yml' # Model
CONFIGS_DATA_PATH = 'configs/data_config.yml'
CONFIGS_ENVIRONMENT_PATH = 'configs/environment_config.yml'
CONFIGS_EVALUATION_PATH = 'configs/evaluation_config.yml'

# DATASETS
DATASETS = ['continuous_metadata', 'discrete_metadata', 'continuous_reviews', 'discrete_reviews']
MODELS = ['FM', 'DeepFM']
FOLDS = 5

# Run the model for each dataset and fold
def run_model(dataset, fold, model):
    """
    Ejecuta el modelo especificado para un conjunto de datos y pliegue dados.
    
    Parámetros:
    - dataset: Nombre del conjunto de datos
    - fold: Número del pliegue
    - model: Nombre del modelo a ejecutar
    """
    # Configurar los archivos de configuración
    dataset_path = CONFIGS_DATASET_PATH.format(dataset)
    benchmark_path = CONFIGS_BENCHMARK_PATH.format(fold)
    hyperparameter_path = CONFIGS_HYPERPARAMETER_PATH.format(model, dataset)
    model_path = CONFIGS_MODEL_PATH.format(model)
    
    # Ejecutar el modelo
    run_recbole(config_file_list=[
        dataset_path, 
        benchmark_path, 
        hyperparameter_path, 
        model_path, 
        CONFIGS_DATA_PATH,
        CONFIGS_ENVIRONMENT_PATH,
        CONFIGS_EVALUATION_PATH])
    

if __name__ == "__main__":

    # Iterar sobre cada combinación de dataset, fold y modelo
    for dataset in DATASETS:
        for fold in range(0, FOLDS):
            for model in MODELS:
                run_model(dataset, fold, model)
                print(f"Ejecutado {model} para {dataset} en el fold {fold}")