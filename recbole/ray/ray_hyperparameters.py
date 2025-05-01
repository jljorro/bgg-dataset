import yaml
import os
from recbole.quick_start import objective_function

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

CONFIG_PATHS = '../configs/{}.yml'
SEARCH_SPACE_PATHS = '{}_search.yml'
DATASET = 'metadata_disc'

def get_config_file_list():
    """
    Get the list of configuration files for the RecBole model.
    """
    config_files = [
        CONFIG_PATHS.format('environment_{}'.format(DATASET)), 
        CONFIG_PATHS.format('data_CARS'),
        CONFIG_PATHS.format('evaluation')
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

def main():

    print(os.path.join(os.getcwd()))

    config_file_list = get_config_file_list()

    ray.init()
    tune.register_trainable("train_func", objective_function)

    config = load_search_space(SEARCH_SPACE_PATHS.format('fm'))

    scheduler = ASHAScheduler(
        metric="ndcg@10", 
        mode="max", 
        max_t=10, 
        grace_period=1, 
        reduction_factor=2
    )

    local_dir = "./ray_results"
    result = tune.run(
        tune.with_parameters(objective_function, config_file_list=config_file_list),
        config=config,
        num_samples=5,
        log_to_file='./logs',
        scheduler=scheduler,
        local_dir=local_dir,
        resources_per_trial={"gpu": 1},
    )

    best_trial = result.get_best_trial("recall@10", "max", "last")
    print("best params: ", best_trial.config)
    print("best result: ", best_trial.last_result)


if __name__ == '__main__':
    main()
