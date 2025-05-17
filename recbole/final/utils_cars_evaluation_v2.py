import torch
import os
import tqdm
import numpy as np

def process_batch_efficiently(users, test_items, all_item_ids, model, dataset, k=10, batch_size=1024):
    """
    Efficiently process predictions by batching users and evaluating only one set of items at a time for each user.

    Parameters:
    - users: Tensor with user IDs
    - test_items: Original test items for each user
    - all_item_ids: All available item IDs
    - model: Trained recommendation model
    - dataset: Dataset with id2token conversion functions
    - k: Number of top-k recommendations to return
    - batch_size: Batch size of users to process at once

    Returns:
    - List of tuples (user_id, item_id, score, test_item_id) for all predictions
    """
    results = []
    n_users = users.shape[0]
    
    # Process users in batches
    for start_idx in range(0, n_users, batch_size):
        end_idx = min(start_idx + batch_size, n_users)
        user_batch = users[start_idx:end_idx]
        test_item_batch = test_items[start_idx:end_idx]
        
        # Process predictions for this user batch
        user_predictions = process_user_batch(user_batch, test_item_batch, all_item_ids, model, dataset, k)
        results.extend(user_predictions)
        
        # Free memory explicitly
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    return results

def process_user_batch(user_batch, test_item_batch, all_item_ids, model, dataset, k=10, item_batch_size=1024):
    """
    Processes a batch of users by evaluating subsets of items at a time.
    
    Parameters:
    - user_batch: Batch of user IDs
    - test_item_batch: Test items corresponding to users
    - all_item_ids: All available item IDs
    - model: Recommendation model
    - dataset: Dataset with conversion functions
    - k: Number of top-k recommendations
    - item_batch_size: How many items to evaluate at once

    Returns:
    - List of tuples with the top-k predictions for each user
    """
    batch_results = []
    n_users = user_batch.shape[0]
    n_items = len(all_item_ids)
    device = next(model.parameters()).device
    
    # To store the top k for each user - we use CPU to save GPU memory
    all_top_scores = torch.full((n_users, k), float('-inf'))
    all_top_indices = torch.zeros((n_users, k), dtype=torch.long)
    
    # Process items in batches for each user
    for item_start_idx in range(0, n_items, item_batch_size):
        item_end_idx = min(item_start_idx + item_batch_size, n_items)
        item_batch = all_item_ids[item_start_idx:item_end_idx]
        
        # Create the batch for the model
        batch = create_user_item_pairs(user_batch, item_batch, device)
        
        # Obtain predictions
        with torch.no_grad():
            predictions = model.predict(batch)
            
        # Move predictions to the CPU to free up GPU memory
        predictions = predictions.cpu()
        
        # Reshape predictions to shape [n_users, n_items_in_batch]
        predictions = predictions.reshape(n_users, -1)
        
        # Update the top-k for each user
        for user_idx in range(n_users):
            scores = predictions[user_idx]
            
            # Concatenate current scores with previous best scores
            combined_scores = torch.cat([all_top_scores[user_idx], scores])
            combined_indices = torch.cat([
                all_top_indices[user_idx], 
                torch.tensor(item_batch, dtype=torch.long)
            ])
            
            # Get the best k ones
            if len(combined_scores) > k:
                top_k_scores, top_k_idxs = torch.topk(combined_scores, k)
                all_top_scores[user_idx] = top_k_scores
                all_top_indices[user_idx] = combined_indices[top_k_idxs]
            else:
                all_top_scores[user_idx] = combined_scores
                all_top_indices[user_idx] = combined_indices
        
        # Force release memory
        del predictions, batch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Convert the results to the desired format
    for user_idx in range(n_users):
        user_id = dataset.id2token('user_id', user_batch[user_idx].item())
        test_item_id = dataset.id2token('game_id', test_item_batch[user_idx].item())
        
        for i in range(k):
            try:
                item_idx = all_top_indices[user_idx][i].item()
                item_id = dataset.id2token('game_id', item_idx)
                score = all_top_scores[user_idx][i].item()
                batch_results.append((user_id, item_id, score, test_item_id))
            except Exception as e:
                # Defensive handling of invalid indexes
                continue
    
    return batch_results

def create_user_item_pairs(users, items, device):
    """
    Creates user-item pairs for efficient evaluation.

    Parameters:
    - users: Tensor with user IDs [n_users]
    - items: Tensor with item IDs [n_items]
    - device: Device to store the tensors on

    Returns:
    - Dictionary of tensors to be used in model.predict()
    """
    n_users = users.shape[0]
    n_items = len(items)
    
    # Create on CPU first to save memory on GPU
    expanded_users = users.repeat_interleave(n_items)
    expanded_items = torch.tensor(items, dtype=torch.long).repeat(n_users)
    
    # Move to appropriate device
    expanded_users = expanded_users.to(device)
    expanded_items = expanded_items.to(device)
    
    # Create the batch for the model
    batch = {
        'user_id': expanded_users,
        'game_id': expanded_items
    }
    
    return batch

def save_predictions(filepath, predictions):
    """
    Saves predictions to a TSV file.

    Parameters:
    - filepath: Path to the output file
    - predictions: List of tuples (user_id, item_id, score, test_item_id)
    """
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Write header if file does not exist
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write("user_id:token\titem_id:token\tscore:float\ttest_item_id:token\n")
    
    # Write predictions
    with open(filepath, 'a') as f:
        for user_id, item_id, score, test_item_id in predictions:
            f.write(f"{user_id}\t{item_id}\t{score}\t{test_item_id}\n")
            
def get_unique_users_and_test_items(test_data):
    """
    Extracts unique users and their corresponding test items.

    Parameters:
    - test_data: Test data with inter_feat

    Returns:
    - users: Tensor with user IDs
    - test_items: Tensor with corresponding test item IDs
    """
    users = test_data.dataset.inter_feat['user_id']
    test_items = test_data.dataset.inter_feat['game_id']
    
    return users, test_items