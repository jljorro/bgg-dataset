from ray import tune
from recbole.config import Config
from typing import Dict, Any, Optional, Tuple
from recbole.utils import init_logger, init_seed
from recbole.utils.utils import get_model, get_trainer
from recbole.data import create_dataset, data_preparation
from ray.tune.schedulers import ASHAScheduler


def get_scheduler():
    scheduler = ASHAScheduler(metric="auc", mode="max", grace_period=1, reduction_factor=2, brackets=1, max_t=100)
    return scheduler


def train_recbole(config_dict=None, config_file_list=None):

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    model_name = config["model"]
    model = get_model(model_name)(config, train_data.dataset).to(config["device"])

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False)
    test_result = trainer.evaluate(test_data)
    model_file = trainer.saved_model_file

    print(f"Best valid score: {best_valid_score}")
    print(f"Best valid result: {best_valid_result}")
    print(f"Test result: {test_result}")
    return model_name, best_valid_score, best_valid_result, test_result, model_file


def objective_function(config_dict=None, config_file_list=None):
    model_name, best_valid_score, best_valid_result, test_result, _ = train_recbole(config_dict=config_dict, config_file_list=config_file_list)

    return {"model": model_name, "AUC": best_valid_score, "best_valid_result/auc@10": best_valid_result, "test_result": test_result}
