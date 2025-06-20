# Boardgames Dataset RecSys 2025

Repository that contains recommender models evaluations for new dataset from BoardGameGeek.

## Installation

To ensure reproducibility and avoid dependency conflicts, we recommend using three separate Python virtual environments managed with `pyenv`:

- One for running experiments with [Elliot](https://elliot.readthedocs.io/en/latest/) (Python 3.8)
- One for running experiments with [RecBole](https://recbole.io/) (Python 3.10 + CUDA 12.5)
- One for evaluation with [Ranx](https://amenra.github.io/ranx/evaluate/) (Python 3.12)

This setup helps maintain clean and compatible environments for each task.

### Prerequisites

  - pyenv installed on your system
  - CUDA 12.5 installed and properly configured (for RecBole experiments)

### Create virtual environment for Elliot (Python 3.8)

```bash
pyenv install 3.8.18
pyenv virtualenv 3.8.18 elliot-env
pyenv activate elliot-env
```

- Navigate to the elliot experiment directory:

```bash
cd elliot
```

- Install dependencies

```bash
pip install -r requirements.txt
```

### Create virtual environment for RecBole (Python 3.10)

```bash
pyenv install 3.10.13
pyenv virtualenv 3.10.13 recbole-env
pyenv activate recbole-env
```

Ensure CUDA 12.5 is correctly set up and detected.

- Navigate to the recbole experiment directory

```bash
cd recbole
```

- Install dependencies

```bash
pip install -r requirements.txt
```

### Create virtual environment for evaluation with Ranx (Python 3.12)

```bash
pyenv install 3.12.3
pyenv virtualenv 3.12.3 ranx-env
pyenv activate ranx-env
```

- Install dependencies (e.g., Ranx)

```bash
pip install ranx
```

You can switch between the environments using:

```bashP
pyenv activate elliot-env # For elliot experiments
pyenv activate recbole-env # For recbole experiments
pyenv activate ranx-env # For evaluation
```

>[!Important] Script Execution Paths
> Most scripts assume that this repository has been cloned into the home directory (~).
> However, in some cases this assumption may not hold, and paths will need to be adjusted manually in the scripts or configuration files.
> For example, in elliot configurations files, you should modify the dataset path as follows:
>
> ```yaml
>...
># Default path
>dataset_path: ~/bgg-recsys25/data/bgg25_raw_ratings/
>
># Updated path (replace [YOUR_PATH] accordingly)
>dataset_path: [YOUR_PATH]/data/bgg25_raw_ratings/
>...
> ```


## Recommender models tested

- *General Recommender Models*
  - Random
  - Pop
  - ItemKNN
  - UserKNN

- *Context-Aware Recommender Models:*
  - ContextRandom
  - ContextPop
  - FM
  - DeepFM

## Folders

- `data`: data corresponding to the dataset described in the paper, divided in ratings, continuous, and discrete, the last two with the context extracted either from metadata or reviews
- `elliot`: configuration files and code to use with Elliot
- `evaluation`: scripts to evaluate the models
- `prefiltering`: new implemented prefiltering methods
- `recbole`: configuration files and code to use with RecBole

> [!IMPORTANT]
> Before running any experiments, make sure to extract the datasets.
> To do this, execute the provided script:
> ```bash
> bash unzip_datasets.sh
> ```
> This will decompress all necessary dataset files into their corresponding directories.


## Elliot Experiments

The Elliot framework is used for implementing general recommendation algorithms.

### Step 1: Hyperparameter Tuning

First, execute all scripts in the `elliot/hypertune` folder to find the best hyperparameters for each model:

```bash
cd elliot/hypertune
./run_experiments.sh
```

Alternatively, you can run each hyperparameter tuning script individually:

```bash
python run_ItemKNN.py
python run_UserKNN.py
# Continue with other hyperparameter tuning scripts
```

### Step 2: Adapting Configuration Files

After identifying the best hyperparameters, modify the YAML configuration files in the `elliot/final` folder to incorporate these optimal values.

### Step 3: Running Final Models

Execute the final models with the optimized configurations:

```bash
cd elliot/final
python run_ItemKNN.py
python run User_KNN.py
# Continue with other scripts
```

## RecBole Experiments

RecBole experiments require GPU acceleration for optimal performance.

### Step 1: Hyperparameter Search

Run the Ray Tune script to search for the best hyperparameters:

```bash
cd ray
python ray_hiperparametres.py
```

**Note**: GPU usage is required for this step.

### Step 2: Configure Final Models

Based on the results from the hyperparameter search, modify the configuration files in the `final/config/hyperparameters` directory.

### Step 3: Execute Final Experiments

Run the CARS (Context-Aware Recommender Systems) experiments:

```bash
cd final
python execute_cars_experiment_v2.py
```

## Prefiltering

To execute the prefiltering process, run the following script:

```bash
python exec_prefiltering.py
```

This will preprocess the data according to the defined filtering criteria.

## Evaluation

After generating recommendations with each recommender system, you need to evaluate their performance.

### Step 1: Copy Predictions

Copy the predictions generated by each recommender type to their respective evaluation folders.

### Step 2: Run Evaluation Scripts

For each recommender type, run the evaluation script:

```bash
# For general recommenders
python evaluation/general/evaluation_script.py

# For context-aware recommenders
python evaluation/context/evaluation_script.py

# For other recommender types (if available)
python evaluation/cars_models/evaluation_script.py
```

## Examples

Here's an example workflow for a complete experiment:

1. Run hyperparameter tuning for Elliot models
2. Configure the best models in Elliot final configs
3. Run the final Elliot models to generate predictions
4. Run hyperparameter search for RecBole models using GPU
5. Configure the best RecBole models
6. Run the final RecBole CARS experiment
7. Execute prefiltering if needed
8. Run evaluation scripts for each recommender type

## Notes

- Make sure you have sufficient disk space to store the model outputs and prediction results
- GPU acceleration is strongly recommended for RecBole experiments
- Check the console output for any error messages during execution


## License

This dataset is intended for research and educational purposes. Please ensure compliance with data source licenses when using or distributing derivative works.
