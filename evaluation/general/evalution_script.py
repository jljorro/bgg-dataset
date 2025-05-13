import pandas as pd
from datetime import datetime

import logging

from ranx import Qrels, Run, evaluate

THRESHOLD = 7
METRICS = ["hits@5", "hits@10",
           "hit_rate@5", "hit_rate@10",
           "precision@5", "precision@10",
           "recall@5", "recall@10",
           "f1@5", "f1@10",
           "mrr@5", "mrr@10",
           "map@5", "map@10",
           "ndcg@5", "ndcg@10",
           "ndcg_burges@5", "ndcg_burges@10"]

ALGORITHMS = ['random', 'mostpop', 'mf', 'puresvd', 'userknn', 'itemknn']
PREDICTIONS_PATH = "./predictions/{}_f{}_result.tsv"
TEST_PATH = "/home/jljorro/Desarrollo/bgg-recsys25/data/bgg25_raw_ratings/bgg25_raw_ratings.f{}.test.inter"

def get_qrel(fold):

    # Load test as qrels
    test_path = TEST_PATH.format(fold)
    test_df = pd.read_csv(test_path, sep="\t", header=None, names=['user', 'item', 'rating', 'timestamp'])

    # Eliminamos la primea fila
    test_df = test_df.drop(index=0)

    # Eliminamos columna timestamp
    test_df = test_df.drop(columns=['timestamp'])
    test_df['rating'] = test_df['rating'].astype(float)
    test_df['rating'] = test_df['rating'].apply(lambda x: 1 if x >= THRESHOLD else 0)

    # Convertimos las columnas user e item a string
    test_df['user'] = test_df['user'].astype(str)
    test_df['item'] = test_df['item'].astype(str)

     # Pasamos test_df a qrels
    qrel = Qrels.from_df(
            df = test_df,
            q_id_col = 'user',
            doc_id_col = 'item',
            score_col = 'rating')
    
    return qrel

def get_run(fold, algorithm):
    # Load predictions as run
    prediction_path = PREDICTIONS_PATH.format(algorithm, fold)
    predictions_df = pd.read_csv(prediction_path, sep="\t", header=None, names=['user', 'item', 'prediction'])

    # Convertimos las columnas user e item a string y prediction a float
    predictions_df['user'] = predictions_df['user'].astype(str)
    predictions_df['item'] = predictions_df['item'].astype(str)
    predictions_df['prediction'] = predictions_df['prediction'].astype(float)

    # Pasamos predictions_df a run
    run = Run.from_df(
                df = predictions_df,
                q_id_col = 'user',
                doc_id_col = 'item',
                score_col = 'prediction')
    
    return run


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Creamos un dataframe vacio para almacenar los resultados
    results_df = pd.DataFrame(columns=['algorithm', 'fold', 'metric', 'value'])

    for fold in range(0, 5):
        logger.info(f"Evaluating fold {fold}")
        qrel = get_qrel(fold)

        for algorithm in ALGORITHMS:
            logger.info(f"Evaluating algorithm {algorithm}")
            run = get_run(fold, algorithm)

            # Evaluamos
            results = evaluate(
                qrels=qrel,
                run=run,
                metrics=METRICS
            )

            # Incluimos los resultados en el dataframe
            for metric in results:
                logger.info(f"Algorithm: {algorithm}, Fold: {fold}, Metric: {metric}, Value: {results[metric]}")
                # Añadimos los resultados al dataframe sin append
                results_df = pd.concat([results_df, pd.DataFrame({
                    'algorithm': [algorithm],
                    'fold': [fold],
                    'metric': [metric],
                    'value': [results[metric]]
                })], ignore_index=True)

    # Guardamos los resultados en un archivo
    results_df.to_csv("results/general_algorithms_results.csv", index=False)


if __name__ == "__main__":
    # Start the evaluation
    start_time = datetime.now()
    print(f"Evaluation started at {start_time}")
    
    main()

    # End the evaluation
    end_time = datetime.now()
    print(f"Evaluation ended at {end_time}")
    print(f"Total time taken: {end_time - start_time}")
    print("Evaluation completed successfully.")