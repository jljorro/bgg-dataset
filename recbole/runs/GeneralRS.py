from recbole.quick_start import run_recbole

DATASET = 'bgg_24'
CONFIGS_PATHS = '../configs/{}.yml'

run_recbole(
    model='Random', 
    dataset=DATASET, 
    config_file_list=[CONFIGS_PATHS.format('environment'), 
                      CONFIGS_PATHS.format('data_GeneralRS'), 
                      CONFIGS_PATHS.format('evaluation')])

run_recbole(
    model='Pop',
    dataset=DATASET,
    config_file_list=[CONFIGS_PATHS.format('environment'), 
                      CONFIGS_PATHS.format('data_GeneralRS'), 
                      CONFIGS_PATHS.format('evaluation')])

run_recbole(
    model='ItemKNN', 
    config_file_list=[CONFIGS_PATHS.format('environment'), 
                      CONFIGS_PATHS.format('data_GeneralRS'),
                      CONFIGS_PATHS.format('itemknn'), 
                      CONFIGS_PATHS.format('evaluation')])