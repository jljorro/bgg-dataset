import logging
import yaml

from recbole.quick_start import run_recbole, objective_function
from recbole.config import Config

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# CONSTANTES
SEARCH_SPACE_PATHS = '{}_search.yml'
CONFIG_PATHS = '../configs/{}.yml'

DATASET = 'metadata_disc'

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


# 1. Cargamos ficheros de configuración
config_files = [
    CONFIG_PATHS.format('environment_{}'.format(DATASET)), 
    CONFIG_PATHS.format('data_CARS')]

base_config = Config(config_file_list=config_files)
base_config_dict = dict(base_config.final_config_dict)
base_config_dict['nproc'] = 20
print(base_config_dict)


# 3. Definición del Espacio de Búsqueda
search_space = load_search_space(SEARCH_SPACE_PATHS.format('fm'))

# 4. Función de Entrenamiento
def train_rec_sys(config):
    # Unir configuración base + hiperparámetros sugeridos
    merged_config = base_config_dict.copy()
    merged_config.update(config)

    # Ejecutar objective_function de RecBole
    best_valid_score, config_obj, model, dataset, saved_model_file = objective_function(config_dict=merged_config)

    # Reportar a Ray Tune
    tune.report(objective=best_valid_score)

ray.init()

# 5. Configuración del Scheduler de Tune
scheduler = ASHAScheduler(
    metric='ndcg@10',
    mode='max',
    max_t=500,
    grace_period=10,
    reduction_factor=2
)

# 6. Ejecutar Tune
analysis = tune.run(
    tune.with_parameters(train_rec_sys),
    config=base_config_dict,
    num_samples=50,
    scheduler=scheduler,
    resources_per_trial={'cpu': 1}
)

# 7. (Opcional) Guardar el mejor modelo/config
best_trial = analysis.get_best_trial(metric='objective', mode='max', scope='all')
print("======== Mejor configuración encontrada ========")
for key, value in best_trial.config.items():
    print(f"{key}: {value}")
print("=================================================")