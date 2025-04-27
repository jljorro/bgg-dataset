from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function

HYPER_PATHS = '../hypers/{}.hyper'
CONFIG_PATHS = '../configs/{}.yml'

DATASET = 'reviews_cont'

hp = HyperTuning(
    objective_function=objective_function, 
    algo='bayes', 
    early_stop=15,
    max_evals=100, 
    params_file=HYPER_PATHS.format('fm'), 
    fixed_config_file_list=[CONFIG_PATHS.format('environment_{}'.format(DATASET)), 
                            CONFIG_PATHS.format('data_CARS')],
)

# run
hp.run()

# export result to the file
hp.export_result(output_file='hyper_FM_metadata_cont.result')

# print best parameters
print('best params: ', hp.best_params)

# print best result
print('best result: ')

print(hp.params2result[hp.params2str(hp.best_params)])