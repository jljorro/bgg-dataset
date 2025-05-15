import pandas as pd
from datetime import datetime

import logging

from ranx import Qrels, Run, evaluate, fuse

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

ALGORITHMS = ['contextpop']
DATASETS = ['continuous_metadata']

PREDICTIONS_CONTEXT_PATH = "./predictions/{}_{}_f{}_prefiltering.tsv" # dataset, algorithm, fold
TEST_PATH = "./tests/{}_test_simplified_f{}.tsv" # dataset, dataset, fold

def get_qrel(dataset, fold):

    # Load test as qrels
    test_path = TEST_PATH.format(dataset, fold)
    test_df = pd.read_csv(test_path, sep="\t")

    # Cambiamos el valor de la culumna 1 user_id:token, por un valor concat de user_id:token + _ + game_id:token
    test_df['user_id:token'] = test_df['user_id:token'].astype(str) + "_" + test_df['context_id'].astype(str)

    # Eliminamos las columnas desde la posición 3 hasta la última
    # test_df = test_df.drop(columns=test_df.columns[3:])

    # Eliminamos la primea fila
    # test_df = test_df.drop(index=0)

    # Eliminamos columna timestamp
    test_df['rating:float'] = test_df['rating:float'].astype(float)
    test_df['rating:float'] = test_df['rating:float'].apply(lambda x: 1 if x >= THRESHOLD else 0)

    # Convertimos las columnas user e item a string
    # test_df['user_id:token'] = test_df['user_id:token'].astype(str)
    test_df['game_id:token'] = test_df['game_id:token'].astype(str)

     # Pasamos test_df a qrels
    qrel = Qrels.from_df(
            df = test_df,
            q_id_col = 'user_id:token',
            doc_id_col = 'game_id:token',
            score_col = 'rating:float')
    
    return qrel

def get_run(prediction_path):
    # Load predictions as run
    predictions_df = pd.read_csv(prediction_path, sep="\t", header=None, names=['user', 'item', 'prediction'])

    # Eliminamos la primea fila
    predictions_df = predictions_df.drop(index=0)

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

    for dataset in DATASETS:
        logger.info(f"Evaluating dataset {dataset}")

        for fold in range(0, 5):
            logger.info(f"Evaluating fold {fold}")
            qrel = get_qrel(dataset, fold)

            for algorithm in ALGORITHMS:
                logger.info(f"Evaluating algorithm {algorithm}")

                context_run_path = PREDICTIONS_CONTEXT_PATH.format(dataset, algorithm, fold)

                context_run = get_run(context_run_path)

                
                # Evaluamos
                results = evaluate(
                    qrels=qrel,
                    run=context_run,
                    metrics=METRICS,
                    make_comparable=True
                )

                # Incluimos los resultados en el dataframe
                for metric in results:
                    logger.info(f"Dataset: {dataset}, Algorithm: {algorithm}, Fold: {fold}, Metric: {metric}, Value: {results[metric]}")
                    # Añadimos los resultados al dataframe sin append
                    results_df = pd.concat([results_df, pd.DataFrame({
                        'dataset': [dataset],
                        'algorithm': [algorithm],
                        'fold': [fold],
                        'metric': [metric],
                        'value': [results[metric]]
                    })], ignore_index=True)

    # Guardamos los resultados en un archivo
    results_df.to_csv("results/prefiltering_results.csv", index=False)


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