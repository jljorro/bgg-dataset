import os
import pandas as pd
from context_similarity_gpu import ContextPostfiltering  # Asegúrate de tener este módulo disponible

# === PATHS ===
PREDICTIONS_PATH = "./general_predictions/{}_f{}_result.tsv"
TEST_PATH = "../data/bgg25_{}/bgg25_{}.f{}.test.inter"
TRAIN_PATH = "../data/bgg25_{}/bgg25_{}.f{}.train.inter"
RESULTS_PATH = "./results/{}_{}_f{}_postfiltering.tsv"  # dataset, algorithm, fold

# === CONFIGURACIÓN ===
DATASETS = ['discrete_metadata', 'continuous_metadata', 'discrete_reviews', 'continuous_reviews']
ALGORITHMS = ['mf', 'puresvd', 'userknn', 'itemknn']
FOLDS = 5
BLOCK_SIZE = 600  # Tamaño de bloque para procesar predicciones en batches

def run_postfiltering():
    for dataset in DATASETS:
        for algorithm in ALGORITHMS:
            for fold in range(FOLDS):
                print(f"\nProcesando: Dataset={dataset}, Algorithm={algorithm}, Fold={fold}")

                # === 1. Cargar datos ===
                train_path = TRAIN_PATH.format(dataset, dataset, fold)
                test_path = TEST_PATH.format(dataset, dataset, fold)
                pred_path = PREDICTIONS_PATH.format(algorithm, fold)
                result_path = RESULTS_PATH.format(dataset, algorithm, fold)

                try:
                    train_df = pd.read_csv(train_path, sep='\t')
                    test_df = pd.read_csv(test_path, sep='\t')
                    pred_df = pd.read_csv(pred_path, sep='\t')
                except FileNotFoundError as e:
                    print(f"[ERROR] Archivo no encontrado: {e}")
                    continue

                # === 2. Inicializar y ajustar modelo de postfiltrado ===
                postfilter = ContextPostfiltering(train_df, test_df, pred_df)
                postfilter.fit()

                # === 3. Procesar predicciones por bloques ===
                rows = []
                for start in range(0, len(pred_df), BLOCK_SIZE):
                    end = start + BLOCK_SIZE
                    block = pred_df.iloc[start:end]

                    # Agrupar por item_id de test dentro del bloque
                    for test_item_id, subblock in block.groupby("item_id"):
                        user_ids = subblock['user_id'].tolist()
                        candidate_ids = subblock['item_id'].tolist()

                        try:
                            sim_matrix = postfilter.get_similarity(test_item_id, [candidate_ids])
                        except KeyError:
                            continue  # Ignorar test_item_id fuera de índice

                        for user_id, (pred_item_id, sim) in zip(user_ids, sim_matrix):
                            rows.append({
                                'user_id': user_id,
                                'item_id': pred_item_id,
                                'similarity': sim
                            })

                # === 4. Guardar resultados ===
                if rows:
                    result_df = pd.DataFrame(rows)
                    result_df.to_csv(result_path, sep='\t', index=False)
                    print(f"[OK] Resultados guardados en: {result_path}")
                else:
                    print("[AVISO] No se generaron resultados para esta combinación.")

if __name__ == "__main__":
    run_postfiltering()
