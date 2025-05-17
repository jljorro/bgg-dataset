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
# Number of chunks to process in parallel - adjust according to the number of available cores
NUM_WORKERS = max(1, mp.cpu_count() - 1)

def generate_context_id(context):
    """Generates a unique identifier for a context"""
    # Convert the array to a string and generate a hash
    context_str = ','.join(map(str, context))
    return hashlib.md5(context_str.encode()).hexdigest()[:8]  # We use the first 8 characters of the hash

def group_by_user_context(test_df):
    """Groups test data by user and context"""
    # Extract columns from context
    context_columns = test_df.columns[4:]  # Context columns start from position 4
    
    # Create an identifier for each unique context if it doesn't already exist
    if 'context_id' not in test_df.columns:
        test_df['context_id'] = test_df.apply(
            lambda row: generate_context_id(row[context_columns].values), 
            axis=1
        )
    
    # Group by user_id and context_id
    grouped = test_df.groupby(['user_id:token', 'context_id'])
    
    # Create a DataFrame with the grouped data
    unique_contexts = []
    for (user_id, context_id), group in grouped:
        # We take the first record in the group to get the context
        first_row = group.iloc[0]
        context_values = first_row[context_columns].values
        
        # Add the user, context_id and context values
        unique_contexts.append({
            'user_id': user_id,
            'context_id': context_id,
            'context_values': context_values,
            'game_ids': group['game_id:token'].tolist()  # We save all the associated game_ids
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
        game_ids = context_data['game_ids']  # We now save this but do not use it for recommendations.
        
        # Get the items that meet the context
        items, relevance = context_model.recommend(context, 20)
        
        # We generate a row for each recommendation for the format user_id_context_id, item_id, prediction
        for item_id, pred in zip(items, relevance):
            result.append([f"{user_id}_{context_id}", item_id, pred])
            
        end_time = time.time()
        
        if idx % 100 == 0:  # We reduced the printing frequency to improve performance
            print(f"Processed: {idx}/{total_unique_contexts} - Time: {end_time - start_time:.4f} seconds")
    
    return result

def process_fold(fold, dataset, method='contextrandom'):
    print(f"Processing dataset {dataset}, fold {fold}, method {method}")
    
    # Load data
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
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    
    # Extraer las columnas de contexto
    context_columns = test_df.columns[4:]  # Context columns start from position 4
    
    # Create a copy of the test DataFrame and add context identifier
    start_time = time.time()
    test_df_copy = test_df.copy()
    
    # Create an identifier for each unique context
    test_df_copy['context_id'] = test_df_copy.apply(
        lambda row: generate_context_id(row[context_columns].values), 
        axis=1
    )
    
    # Remove context columns and leave only the important ones and the context_id
    test_df_simplified = test_df_copy[['user_id:token', 'game_id:token', 'rating:float', 'timestamp:float', 'context_id']].copy()
    simplified_path = os.path.join(os.path.dirname(RESULTS_PATH.format(dataset, method.lower(), fold)), 
                                  f"{dataset}_test_simplified_f{fold}.tsv")
    os.makedirs(os.path.dirname(simplified_path), exist_ok=True)
    test_df_simplified.to_csv(simplified_path, sep='\t', index=False)
    print(f"Created simplified dataset with context_id in: {simplified_path}")
    
    # Group test data by user and context
    unique_contexts = group_by_user_context(test_df)
    print(f"Data grouped in {time.time() - start_time:.2f} seconds")
    print(f"Total unique contexts: {len(unique_contexts)} (number of rows before: {len(test_df)})")
    
    # Initialize the model according to the selected method
    start_time = time.time()
    if method.lower() == 'contextpop':
        context_model = ContextPop(train_df)
    else:  # By default, we use ContextRandom
        context_model = ContextRandom(train_df)
    print(f"Index built in {time.time() - start_time:.2f} seconds")

    # Prepare for parallel processing
    total_unique_contexts = len(unique_contexts)
    pool = mp.Pool(processes=NUM_WORKERS)
    
    # Split single contexts into chunks for parallel processing
    chunk_size = max(1, len(unique_contexts) // NUM_WORKERS)
    chunks = [(i, unique_contexts[i:i+chunk_size]) 
              for i in range(0, len(unique_contexts), chunk_size)]
    
    # Partial function with fixed parameters
    process_func = partial(process_chunk, context_model=context_model, total_unique_contexts=total_unique_contexts)
    
    # Run processing in parallel
    print(f"Starting parallel processing with {NUM_WORKERS} workers...")
    start_time = time.time()
    results = pool.map(process_func, chunks)
    pool.close()
    pool.join()
    
    # Flatten the results
    result_context = [item for sublist in results for item in sublist]
    print(f"Parallel processing completed in {time.time() - start_time:.2f} seconds")

    # Save results
    result_df = pd.DataFrame(result_context, columns=['user_id', 'item_id', 'prediction'])
    result_path = RESULTS_PATH.format(dataset, method.lower(), fold)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)  # Ensure that the directory exists
    result_df.to_csv(result_path, sep='\t', index=False)
    print(f"Results saved in: {result_path}")

    # Create a mapping from context_id to game_ids for further analysis (optional)
    context_mapping = {data['context_id']: data['game_ids'] for data in unique_contexts}
    mapping_path = os.path.join(os.path.dirname(result_path), f"{dataset}_context_mapping_f{fold}.tsv")
    
    # Save the mapping as a TSV file
    with open(mapping_path, 'w') as f:
        f.write("context_id\tgame_ids\n")
        for context_id, game_ids in context_mapping.items():
            f.write(f"{context_id}\t{','.join(map(str, game_ids))}\n")
    
    print(f"Context mapping saved in: {mapping_path}")

if __name__ == "__main__":
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Run prefiltering with different methods.')
    parser.add_argument('--method', type=str, default='contextrandom', 
                        choices=AVAILABLE_METHODS,
                        help='Recommendation method to use (contextrandom or contextpop)')
    parser.add_argument('--datasets', type=str, nargs='+', default=DATASETS,
                        help='Datasets to process')
    parser.add_argument('--folds', type=int, default=FOLDS,
                        help='Number of folds to process')
    
    args = parser.parse_args()
    
    start_global = time.time()
    
    for dataset in args.datasets:
        for fold in range(args.folds):
            fold_start = time.time()
            process_fold(fold, dataset, method=args.method)
            print(f"Fold {fold} finished in {time.time() - fold_start:.2f} seconds")
    
    print(f"Total processing finished in {time.time() - start_global:.2f} seconds")