import pandas as pd
import numpy as np
import logging
import time

from context_similarity import ContextSimilarity

from multiprocessing import Pool, cpu_count


# PATHS
PREDICTIONS_PATH = "./general_predictions/{}_f{}_result.tsv"
TEST_PATH = "../data/bgg25_{}/bgg25_{}.f{}.test.inter"
TRAIN_PATH = "../data/bgg25_{}/bgg25_{}.f{}.train.inter"
RESULTS_PATH = "./results/{}_{}_f{}_postfiltering.tsv" #Dataset, Algorithm, Fold

# NAMES
DATASETS = ['discrete_metadata', 'continuous_metadata', 'discrete_reviews', 'continuous_reviews']
ALGORITHMS = ['mf', 'puresvd', 'userknn', 'itemknn']

# DATASETS = ['discrete_metadata']
# ALGORITHMS = ['mf']

# INPUTS
FOLDS = 5
N_PROCESSES = 10  # Number of processes to use for parallel processing

def init_logging():
    """
    Initialize logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("postfiltering.log"),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized.")

def process_chunk(chunk, train_df, predictions_df):
    """
    Función ejecutada en cada proceso del pool.
    """
    from context_similarity import ContextSimilarity  # Import dentro del proceso
    context_similarity = ContextSimilarity(train_df)
    context_similarity.set_predictions(predictions_df)

    result = []
    for row in chunk:
        user_id = row[0]
        game_id = row[1]
        context = row[4:]
        context_rank = context_similarity.calculate_context_rank(user_id, game_id, context)
        result.append(context_rank)
    return result  # Lista de DataFrames

def parallel_context_ranking(test_npy, train_df, predictions_df, n_processes):
    """
    Divide los datos de test y ejecuta procesamiento en paralelo.
    """
    chunk_size = int(np.ceil(len(test_npy) / n_processes))
    chunks = [test_npy[i:i + chunk_size] for i in range(0, len(test_npy), chunk_size)]

    # Empaquetar argumentos para starmap
    args = [(chunk, train_df, predictions_df) for chunk in chunks]

    with Pool(processes=n_processes) as pool:
        results = pool.starmap(process_chunk, args)

    # Aplanar la lista de listas de DataFrames
    flattened = [df for sublist in results for df in sublist]
    context_rank_df = pd.concat(flattened, ignore_index=True)
    return context_rank_df


def main():

    init_logging()

    for dataset in DATASETS:
        for algorithm in ALGORITHMS:
            for fold in range(FOLDS):
                logging.info(f"Processing {dataset} with {algorithm} for fold {fold}")

                # Load training and test data
                train_df = pd.read_csv(TRAIN_PATH.format(dataset, dataset, fold), sep="\t")
                test_df = pd.read_csv(TEST_PATH.format(dataset, dataset, fold), sep="\t")

                # Load predictions
                predictions_df = pd.read_csv(PREDICTIONS_PATH.format(algorithm, fold), sep="\t", names=['user_id:token', 'game_id:token', 'prediction:float'])

                # Initialize ContextSimilarity
                context_similarity = ContextSimilarity(train_df)
                context_similarity.set_predictions(predictions_df)

                # Calculate similarity
                context_rank_df = pd.DataFrame(columns=['user_id:token', 'game_id:token', 'similarity'])

                test_npy = test_df.head(100).to_numpy()
                # Iterate over the test dataframe

                # # mesure time execution
                
                # start_time = time.time()
                # for i in range(len(test_npy)):
                #     # Get the user_id and game_id from the test dataframe
                #     user_id = test_npy[i][0]
                #     game_id = test_npy[i][1]
                #     context = test_npy[i][4:]

                #     # Get the context rank for the current user and game
                #     context_rank = context_similarity.calculate_context_rank(user_id, game_id, context)

                #     # Append the result to the new dataframe
                #     context_rank_df = pd.concat([context_rank_df, context_rank], ignore_index=True)
                # end_time = time.time()
                # print(f"Execution time for the first 100 rows: {end_time - start_time} seconds")

                # start_time = time.time()
                # for index, row in test_df.head(100).iterrows():
                #     user_id = row['user_id:token']
                #     game_id = row['game_id:token']
                #     context = row.iloc[4:].tolist()
                    
                #     # Get the context rank for the current user and game
                #     context_rank = context_similarity.calculate_context_rank(user_id, game_id, context)
                    
                #     # Append the result to the new dataframe
                #     context_rank_df = pd.concat([context_rank_df, context_rank], ignore_index=True)
                # end_time = time.time()
                # print(f"Execution time for the first 100 rows: {end_time - start_time} seconds")

                # # Save the context rank to a file
                # context_rank_df.to_csv(RESULTS_PATH.format(dataset, algorithm, fold), sep="\t", index=False)

                 # Medir tiempo
                start_time = time.time()
                context_rank_df = parallel_context_ranking(test_npy, train_df, predictions_df, N_PROCESSES)
                end_time = time.time()
                logging.info(f"Execution time with {N_PROCESSES} processes: {end_time - start_time:.2f} seconds")

                # Guardar resultados
                context_rank_df.to_csv(RESULTS_PATH.format(dataset, algorithm, fold), sep="\t", index=False)
                logging.info(f"Post-filtered predictions saved for {dataset}, {algorithm}, fold {fold}")

                logging.info(f"Post-filtered predictions for {dataset} with {algorithm} for fold {fold} finished.")

if __name__ == "__main__":
    main()

