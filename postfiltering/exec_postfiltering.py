import os
import time
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from postfiltering.context_similarity import ContextPostfiltering

# PATHS
PREDICTIONS_PATH = "./general_predictions/{}_f{}_result.tsv"
TEST_PATH = "../data/bgg25_{}/bgg25_{}.f{}.test.inter"
TRAIN_PATH = "../data/bgg25_{}/bgg25_{}.f{}.train.inter"
RESULTS_PATH = "./results/{}_{}_f{}_postfiltering.tsv"
RESULTS_GENERAL_PATH = "./results/{}_{}_f{}_general_postfiltering.tsv"

# NAMES
DATASETS = ['discrete_metadata', 'continuous_metadata', 'discrete_reviews', 'continuous_reviews']
ALGORITHMS = ['mf', 'puresvd', 'userknn', 'itemknn']

# INPUTS
FOLDS = 5
N_PROCESSES = min(4, cpu_count())  # Ajustable


def process_fold(args):
    dataset, algorithm, fold = args
    print(f"Procesando: {dataset} | {algorithm} | Fold {fold}")

    # Cargar datos de entrenamiento
    train_file = TRAIN_PATH.format(dataset, dataset, fold)
    train_df = pd.read_csv(train_file, sep='\t', header=None, low_memory=False)

    # Cargar datos de test
    test_file = TEST_PATH.format(dataset, dataset, fold)
    test_df = pd.read_csv(test_file, sep='\t', header=None).iloc[1:]

    # Cargar predicciones
    predictions_file = PREDICTIONS_PATH.format(algorithm, fold)
    predictions_df = pd.read_csv(predictions_file, sep='\t', header=None)
    predictions_by_user = predictions_df.groupby(by=0)

    # Inicializar postfiltrado
    context_postfiltering = ContextPostfiltering(train_df)

    result_context = []
    result_general = []

    for idx, row in enumerate(test_df.itertuples(index=False)):
        start_time = time.time()

        user_id = int(row[0])
        game_id = int(row[1])
        general_prediction = float(row[2])
        context = np.array(row[4:], dtype=np.float32)

        try:
            user_predictions = predictions_by_user.get_group(user_id).iloc[:, 1].values
        except KeyError:
            print(f"[Advertencia] Usuario {user_id} no tiene predicciones.")
            continue

        if len(user_predictions) != 600:
            print(f"[Error] Usuario {user_id} tiene {len(user_predictions)} predicciones.")
            continue

        # Postfiltrado contextual
        items_predicted, similarities = context_postfiltering.calculate_similarities(context, user_predictions)

        new_user_id = f"{user_id}_{game_id}"

        result_context.append([new_user_id] + list(items_predicted) + list(similarities))
        result_general.append([new_user_id] + list(user_predictions) + [general_prediction])

        elapsed = time.time() - start_time
        print(f"[Fold {fold}] {idx + 1}/{len(test_df)} - Tiempo por bloque: {elapsed:.3f} segundos")

    # Guardar resultados
    context_df = pd.DataFrame(result_context)
    context_df.to_csv(RESULTS_PATH.format(dataset, algorithm, fold), sep='\t', index=False, header=False)

    general_df = pd.DataFrame(result_general)
    general_df.to_csv(RESULTS_GENERAL_PATH.format(dataset, algorithm, fold), sep='\t', index=False, header=False)


def main():
    args_list = [(dataset, algorithm, fold)
                 for dataset in DATASETS
                 for algorithm in ALGORITHMS
                 for fold in range(FOLDS)]

    print(f"Ejecutando {len(args_list)} tareas con {N_PROCESSES} procesos...")

    with Pool(N_PROCESSES) as pool:
        for i, _ in enumerate(pool.imap_unordered(process_fold, args_list), 1):
            print(f"[INFO] Proceso {i}/{len(args_list)} completado.")



if __name__ == "__main__":
    main()
