import pandas as pd
import numpy as np
import time
import multiprocessing as mp
from functools import partial
import os
import hashlib
import argparse

from context_random import ContextRandom
from context_pop import ContextPop

# PATHS
TEST_PATH = "../data/bgg25_{}/bgg25_{}.f{}.test.inter"
TRAIN_PATH = "../data/bgg25_{}/bgg25_{}.f{}.train.inter"
RESULTS_PATH = "./results/{}_{}_f{}_prefiltering.tsv"

# NAMES
DATASETS = ['continuous_metadata']
AVAILABLE_METHODS = ['contextrandom', 'contextpop']

# INPUTS
FOLDS = 5
# Número de chunks para procesar en paralelo - ajustar según el número de núcleos disponibles
NUM_WORKERS = max(1, mp.cpu_count() - 1)

def generate_context_id(context):
    """Genera un identificador único para un contexto"""
    # Convertir el array a string y generar un hash
    context_str = ','.join(map(str, context))
    return hashlib.md5(context_str.encode()).hexdigest()[:8]  # Usamos los primeros 8 caracteres del hash

def group_by_user_context(test_df):
    """Agrupa los datos de test por usuario y contexto"""
    # Extraer las columnas de contexto
    context_columns = test_df.columns[4:]  # Las columnas de contexto comienzan desde la posición 4
    
    # Crear un identificador para cada contexto único si no existe ya
    if 'context_id' not in test_df.columns:
        test_df['context_id'] = test_df.apply(
            lambda row: generate_context_id(row[context_columns].values), 
            axis=1
        )
    
    # Agrupar por usuario_id y context_id
    grouped = test_df.groupby(['user_id:token', 'context_id'])
    
    # Crear un DataFrame con los datos agrupados
    unique_contexts = []
    for (user_id, context_id), group in grouped:
        # Tomamos el primer registro del grupo para obtener el contexto
        first_row = group.iloc[0]
        context_values = first_row[context_columns].values
        
        # Añadimos el usuario, context_id y los valores del contexto
        unique_contexts.append({
            'user_id': user_id,
            'context_id': context_id,
            'context_values': context_values,
            'game_ids': group['game_id:token'].tolist()  # Guardamos todos los game_ids asociados
        })
    
    return unique_contexts

def process_chunk(chunk_data, context_model, total_unique_contexts):
    result = []
    start_idx, chunk = chunk_data
    
    for i, context_data in enumerate(chunk):
        idx = start_idx + i
        start_time = time.time()
        
        user_id = context_data['user_id']
        context_id = context_data['context_id']
        context = context_data['context_values']
        game_ids = context_data['game_ids']  # Esto ahora lo guardamos pero no lo usamos para las recomendaciones
        
        # Obtener los ítems que cumplen con el contexto
        items, relevance = context_model.recommend(context, 20)
        
        # Generamos una fila por cada recomendación para el formato user_id_context_id, item_id, prediction
        for item_id, pred in zip(items, relevance):
            result.append([f"{user_id}_{context_id}", item_id, pred])
            
        end_time = time.time()
        
        if idx % 100 == 0:  # Reducimos la frecuencia de impresión para mejorar rendimiento
            print(f"Procesado: {idx}/{total_unique_contexts} - Tiempo: {end_time - start_time:.4f} segundos")
    
    return result

def process_fold(fold, dataset, method='contextrandom'):
    print(f"Procesando dataset {dataset}, fold {fold}, método {method}")
    
    # Cargar datos
    start_time = time.time()
    train_df = pd.read_csv(TRAIN_PATH.format(dataset, dataset, fold), sep='\t',
    dtype={
        'user_id:token': np.int32,
        'game_id:token': np.int32,
        'rating:float': np.float32,        
        'timestamp:float': np.float32,
        'playing_time_very_short:float': np.int32,
        'playing_time_short:float': np.int32,
        'playing_time_moderate:float': np.int32,
        'playing_time_long:float': np.int32,
        'playing_time_very_long:float': np.int32,
        'gaming_mood_party:float': np.int32,
        'gaming_mood_easy-going:float': np.int32,
        'gaming_mood_expert:float': np.int32,
        'gaming_mood_intense:float': np.int32,
        'gaming_mood_cooperative:float': np.int32,
        'gaming_mood_competitive:float': np.int32,
        'gaming_mood_thematic:float': np.int32,
        'gaming_mood_story-based:float': np.int32,
        'social_companion_1-player:float': np.int32,
        'social_companion_2-players:float': np.int32,
        'social_companion_large-group:float': np.int32,
        'social_companion_toddlers:float': np.int32,
        'social_companion_preschoolers:float': np.int32,
        'social_companion_children:float': np.int32,
        'social_companion_family:float': np.int32,
        'social_companion_friends:float': np.int32})
    test_df = pd.read_csv(TEST_PATH.format(dataset, dataset, fold), sep='\t', dtype={
        'user_id:token': np.int32,
        'game_id:token': np.int32,
        'rating:float': np.float32,        
        'timestamp:float': np.float32,
        'playing_time_very_short:float': np.int32,
        'playing_time_short:float': np.int32,
        'playing_time_moderate:float': np.int32,
        'playing_time_long:float': np.int32,
        'playing_time_very_long:float': np.int32,
        'gaming_mood_party:float': np.int32,
        'gaming_mood_easy-going:float': np.int32,
        'gaming_mood_expert:float': np.int32,
        'gaming_mood_intense:float': np.int32,
        'gaming_mood_cooperative:float': np.int32,
        'gaming_mood_competitive:float': np.int32,
        'gaming_mood_thematic:float': np.int32,
        'gaming_mood_story-based:float': np.int32,
        'social_companion_1-player:float': np.int32,
        'social_companion_2-players:float': np.int32,
        'social_companion_large-group:float': np.int32,
        'social_companion_toddlers:float': np.int32,
        'social_companion_preschoolers:float': np.int32,
        'social_companion_children:float': np.int32,
        'social_companion_family:float': np.int32,
        'social_companion_friends:float': np.int32})
    print(f"Datos cargados en {time.time() - start_time:.2f} segundos")
    
    # Extraer las columnas de contexto
    context_columns = test_df.columns[4:]  # Las columnas de contexto comienzan desde la posición 4
    
    # Crear una copia del DataFrame de test y añadir identificador de contexto
    start_time = time.time()
    test_df_copy = test_df.copy()
    
    # Crear un identificador para cada contexto único
    test_df_copy['context_id'] = test_df_copy.apply(
        lambda row: generate_context_id(row[context_columns].values), 
        axis=1
    )
    
    # Eliminar las columnas de contexto y dejar solo las importantes y el context_id
    test_df_simplified = test_df_copy[['user_id:token', 'game_id:token', 'rating:float', 'timestamp:float', 'context_id']].copy()
    simplified_path = os.path.join(os.path.dirname(RESULTS_PATH.format(dataset, method.lower(), fold)), 
                                  f"{dataset}_test_simplified_f{fold}.tsv")
    os.makedirs(os.path.dirname(simplified_path), exist_ok=True)
    test_df_simplified.to_csv(simplified_path, sep='\t', index=False)
    print(f"Creado dataset simplificado con context_id en: {simplified_path}")
    
    # Agrupar los datos de test por usuario y contexto
    unique_contexts = group_by_user_context(test_df)
    print(f"Datos agrupados en {time.time() - start_time:.2f} segundos")
    print(f"Total de contextos únicos: {len(unique_contexts)} (antes había {len(test_df)} filas)")
    
    # Inicializar el modelo según el método seleccionado
    start_time = time.time()
    if method.lower() == 'contextpop':
        context_model = ContextPop(train_df)
    else:  # Por defecto, usamos ContextRandom
        context_model = ContextRandom(train_df)
    print(f"Índice construido en {time.time() - start_time:.2f} segundos")

    # Preparar el procesamiento en paralelo
    total_unique_contexts = len(unique_contexts)
    pool = mp.Pool(processes=NUM_WORKERS)
    
    # Dividir los contextos únicos en chunks para procesamiento paralelo
    chunk_size = max(1, len(unique_contexts) // NUM_WORKERS)
    chunks = [(i, unique_contexts[i:i+chunk_size]) 
              for i in range(0, len(unique_contexts), chunk_size)]
    
    # Función parcial con los parámetros fijos
    process_func = partial(process_chunk, context_model=context_model, total_unique_contexts=total_unique_contexts)
    
    # Ejecutar el procesamiento en paralelo
    print(f"Iniciando procesamiento paralelo con {NUM_WORKERS} workers...")
    start_time = time.time()
    results = pool.map(process_func, chunks)
    pool.close()
    pool.join()
    
    # Aplanar los resultados
    result_context = [item for sublist in results for item in sublist]
    print(f"Procesamiento paralelo completado en {time.time() - start_time:.2f} segundos")

    # Guardar resultados
    result_df = pd.DataFrame(result_context, columns=['user_id', 'item_id', 'prediction'])
    result_path = RESULTS_PATH.format(dataset, method.lower(), fold)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)  # Asegurar que el directorio exista
    result_df.to_csv(result_path, sep='\t', index=False)
    print(f"Resultados guardados en: {result_path}")

    # Crear un mapeo de context_id a game_ids para análisis posterior (opcional)
    context_mapping = {data['context_id']: data['game_ids'] for data in unique_contexts}
    mapping_path = os.path.join(os.path.dirname(result_path), f"{dataset}_context_mapping_f{fold}.tsv")
    
    # Guardar el mapeo como un archivo TSV
    with open(mapping_path, 'w') as f:
        f.write("context_id\tgame_ids\n")
        for context_id, game_ids in context_mapping.items():
            f.write(f"{context_id}\t{','.join(map(str, game_ids))}\n")
    
    print(f"Mapeo de contexto guardado en: {mapping_path}")

if __name__ == "__main__":
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Ejecutar prefiltering con diferentes métodos.')
    parser.add_argument('--method', type=str, default='contextrandom', 
                        choices=AVAILABLE_METHODS,
                        help='Método de recomendación a utilizar (contextrandom o contextpop)')
    parser.add_argument('--datasets', type=str, nargs='+', default=DATASETS,
                        help='Datasets a procesar')
    parser.add_argument('--folds', type=int, default=FOLDS,
                        help='Número de folds a procesar')
    
    args = parser.parse_args()
    
    start_global = time.time()
    
    for dataset in args.datasets:
        for fold in range(args.folds):
            fold_start = time.time()
            process_fold(fold, dataset, method=args.method)
            print(f"Fold {fold} completado en {time.time() - fold_start:.2f} segundos")
    
    print(f"Procesamiento total completado en {time.time() - start_global:.2f} segundos")