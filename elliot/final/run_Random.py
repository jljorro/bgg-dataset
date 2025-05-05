from elliot.run import run_experiment

for i in range (0, 5):
    run_experiment(f'configs/Random_configuration_f{i}.yml')