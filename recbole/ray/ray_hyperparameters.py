import yaml
import os

#from recbole.quick_start import objective_function

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp

from trainer import *

CONFIG_PATHS = '../configs/{}.yml'
SEARCH_SPACE_PATHS = '{}_search.yml'
DATASETS = ['discrete_metadata', 'continuous_metadata', 'discrete_reviews', 'continuous_reviews']
ALGORITHMS = ['FM', 'DeepFM', 'NMF', 'AutoInt', 'DCN']

# Lista con los modelos ya entrenados para evitar volver a entrenarlos
TRAINED = ['discrete_metadata_FM', 'discrete_metadata_DeepFM', 'discrete_metadata_NMF']

def get_config_file_list(dataset, algorithm):
    """
    Get the list of configuration files for the RecBole model.
    """
    config_files = [
        CONFIG_PATHS.format(f'{dataset}'),
        CONFIG_PATHS.format(f'{algorithm}_config')
    ]

    config_files = [os.path.join(os.getcwd(), file) for file in config_files]

    return config_files

def load_search_space(yaml_path):
    with open(yaml_path, 'r') as f:
        search_space_raw = yaml.safe_load(f)

    search_space = {}
    for param, spec in search_space_raw.items():
        if spec['type'] == 'choice':
            search_space[param] = tune.choice(spec['values'])
        elif spec['type'] == 'uniform':
            lower, upper = map(float, spec['bounds'])
            search_space[param] = tune.uniform(lower=lower, upper=upper)
        elif spec['type'] == 'loguniform':
            lower, upper = map(float, spec['bounds'])
            search_space[param] = tune.loguniform(lower=lower, upper=upper)
        elif spec['type'] == 'randint':
            lower, upper = map(int, spec['bounds'])
            search_space[param] = tune.randint(lower=lower, upper=upper)
        else:
            raise ValueError(f"Unknown search type {spec['type']} for param {param}")
    
    return search_space

def exec_hyperparameter_search(config_file_list, config_ray):
    # ray.init()

    tune.register_trainable("train_func", objective_function)

    search_alg = HyperOptSearch(
        metric="auc", mode="max"
    )

    scheduler = ASHAScheduler(metric="auc", mode="max")

    local_dir = "./ray_results"
    result = tune.run(
        tune.with_parameters(objective_function, config_file_list=config_file_list),
        config=config_ray,
        num_samples=20,
        log_to_file='./logs',
        scheduler=scheduler,
        search_alg=search_alg,
        local_dir=local_dir,
        resources_per_trial={"cpu": 10} # Uncomment in local
        # resources_per_trial={"cpu": 40, "gpu": 1} # Uncomment in ceres
    )

    best_trial = result.get_best_trial("auc", "max", "last")
    print("best params: ", best_trial.config)
    print("best result: ", best_trial.last_result)

def main():

    ray.init()
    
    for dataset in DATASETS:
        for algorithm in ALGORITHMS:

            if f"{dataset}_{algorithm}" in TRAINED:
                print(f"Skipping hyperparameter search for {algorithm} and dataset {dataset} as it has already been trained.")
                continue
            print("=====================================")

            print(f"Running hyperparameter search for {algorithm} and dataset {dataset}...")
            config_file_list = get_config_file_list(dataset, algorithm)
            config_ray = load_search_space(SEARCH_SPACE_PATHS.format(algorithm))

            exec_hyperparameter_search(config_file_list, config_ray)
            print(f"Finished hyperparameter search for {algorithm} and dataset {dataset}.")
            print("=====================================")

if __name__ == '__main__':
    main()
