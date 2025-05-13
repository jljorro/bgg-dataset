import pandas as pd
import numpy as np
import logging
import time
from multiprocessing import Pool, cpu_count

from context_similarity import ContextSimilarity  # Ya usa GPU

# PATHS
PREDICTIONS_PATH = "./general_predictions/{}_f{}_result.tsv"
TEST_PATH = "../data/bgg25_{}/bgg25_{}.f{}.test.inter"
TRAIN_PATH = "../data/bgg25_{}/bgg25_{}.f{}.train.inter"
RESULTS_PATH = "./results/{}_{}_f{}_postfiltering.tsv"  # Dataset, Algorithm, Fold

# CONFIGURACIÓN
DATASETS = ['discrete_metadata', 'continuous_metadata', 'discrete_reviews', 'continuous_reviews']
ALGORITHMS = ['mf', 'puresvd', 'userknn', 'itemknn']
FOLDS = 5
N_PROCESSES = 1  # Ajusta según tu CPU

# ---------- Logging ----------
def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("postfiltering.log"),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized.")

# ---------- Worker global ----------
global_context_similarity = None

def init_worker(train_df_, predictions_df_):
    """
    Inicializador del proceso worker: crea una instancia compartida de ContextSimilarity (GPU).
    """
    global global_context_similarity
    from context_similarity import ContextSimilarity
    global_context_similarity = ContextSimilarity(train_df_)
    global_context_similarity.set_predictions(predictions_df_)

def process_chunk(chunk):
    """
    Procesamiento de un fragmento del test set usando ContextSimilarity en GPU.
    """
    global global_context_similarity
    result = []
    for row in chunk:
        user_id = int(row[0])
        game_id = int(row[1])
        context = row[4:].astype(float)  # Asegura que sea float32-compatible
        context_rank = global_context_similarity.calculate_context_rank(user_id, game_id, context)
        result.append(context_rank)
    return result

# ---------- Paralelización ----------
def parallel_context_ranking(test_npy, train_df, predictions_df, n_processes):
    chunk_size = int(np.ceil(len(test_npy) / n_processes))
    chunks = [test_npy[i:i + chunk_size] for i in range(0, len(test_npy), chunk_size)]

    with Pool(processes=n_processes, initializer=init_worker, initargs=(train_df, predictions_df)) as pool:
        results = pool.map(process_chunk, chunks)

    flattened = [df for sublist in results for df in sublist]
    context_rank_df = pd.concat(flattened, ignore_index=True)
    return context_rank_df

# ---------- Main ----------
def main():
    init_logging()

    for dataset in DATASETS:
        for algorithm in ALGORITHMS:
            for fold in range(FOLDS):
                logging.info(f"Processing {dataset} with {algorithm} for fold {fold}")

                # Cargar datos
                train_df = pd.read_csv(TRAIN_PATH.format(dataset, dataset, fold), sep="\t")
                test_df = pd.read_csv(TEST_PATH.format(dataset, dataset, fold), sep="\t")
                predictions_df = pd.read_csv(
                    PREDICTIONS_PATH.format(algorithm, fold), sep="\t",
                    names=['user_id:token', 'game_id:token', 'prediction:float']
                )

                test_npy = test_df.to_numpy()

                # Medir tiempo
                start_time = time.time()
                context_rank_df = parallel_context_ranking(test_npy, train_df, predictions_df, N_PROCESSES)
                end_time = time.time()
                logging.info(f"Execution time with {N_PROCESSES} processes: {end_time - start_time:.2f} seconds")

                # Guardar resultados
                context_rank_df.to_csv(RESULTS_PATH.format(dataset, algorithm, fold), sep="\t", index=False)
                logging.info(f"Post-filtered predictions saved for {dataset}, {algorithm}, fold {fold}")

if __name__ == "__main__":
    main()
