import torch
import os
import tqdm
import numpy as np

def process_batch_efficiently(users, test_items, all_item_ids, model, dataset, k=10, batch_size=1024):
    """
    Procesa predicciones de forma eficiente procesando por lotes de usuarios
    y evaluando solo un conjunto de items a la vez para cada usuario.
    
    Parámetros:
    - users: Tensor con IDs de usuarios
    - test_items: Items originales de test para cada usuario
    - all_item_ids: Todos los IDs de items disponibles
    - model: Modelo de recomendación entrenado
    - dataset: Dataset con funciones de conversión id2token
    - k: Número de recomendaciones top-k a devolver
    - batch_size: Tamaño del lote de usuarios a procesar a la vez
    
    Devuelve:
    - Lista de tuplas (user_id, item_id, score, test_item_id) para todas las predicciones
    """
    results = []
    n_users = users.shape[0]
    
    # Procesar usuarios por lotes
    for start_idx in range(0, n_users, batch_size):
        end_idx = min(start_idx + batch_size, n_users)
        user_batch = users[start_idx:end_idx]
        test_item_batch = test_items[start_idx:end_idx]
        
        # Procesar predicciones para este lote de usuarios
        user_predictions = process_user_batch(user_batch, test_item_batch, all_item_ids, model, dataset, k)
        results.extend(user_predictions)
        
        # Liberar memoria explícitamente
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    return results

def process_user_batch(user_batch, test_item_batch, all_item_ids, model, dataset, k=10, item_batch_size=1024):
    """
    Procesa un lote de usuarios evaluando subconjuntos de items a la vez.
    
    Parámetros:
    - user_batch: Lote de IDs de usuarios
    - test_item_batch: Items de test correspondientes a los usuarios
    - all_item_ids: Todos los IDs de items disponibles
    - model: Modelo de recomendación
    - dataset: Dataset con funciones de conversión
    - k: Número de recomendaciones top-k 
    - item_batch_size: Cuántos items evaluar a la vez
    
    Devuelve:
    - Lista de tuplas con las predicciones top-k para cada usuario
    """
    batch_results = []
    n_users = user_batch.shape[0]
    n_items = len(all_item_ids)
    device = next(model.parameters()).device
    
    # Para almacenar los k mejores para cada usuario - usamos CPU para economizar memoria GPU
    all_top_scores = torch.full((n_users, k), float('-inf'))
    all_top_indices = torch.zeros((n_users, k), dtype=torch.long)
    
    # Procesar items por lotes para cada usuario
    for item_start_idx in range(0, n_items, item_batch_size):
        item_end_idx = min(item_start_idx + item_batch_size, n_items)
        item_batch = all_item_ids[item_start_idx:item_end_idx]
        
        # Crear el batch para el modelo
        batch = create_user_item_pairs(user_batch, item_batch, device)
        
        # Obtener predicciones
        with torch.no_grad():
            predictions = model.predict(batch)
            
        # Mover las predicciones a CPU para liberar memoria GPU
        predictions = predictions.cpu()
        
        # Reshape las predicciones para tener forma [n_users, n_items_in_batch]
        predictions = predictions.reshape(n_users, -1)
        
        # Actualizar los top-k para cada usuario
        for user_idx in range(n_users):
            scores = predictions[user_idx]
            
            # Concatenar scores actuales con los mejores anteriores
            combined_scores = torch.cat([all_top_scores[user_idx], scores])
            combined_indices = torch.cat([
                all_top_indices[user_idx], 
                torch.tensor(item_batch, dtype=torch.long)
            ])
            
            # Obtener los k mejores
            if len(combined_scores) > k:
                top_k_scores, top_k_idxs = torch.topk(combined_scores, k)
                all_top_scores[user_idx] = top_k_scores
                all_top_indices[user_idx] = combined_indices[top_k_idxs]
            else:
                all_top_scores[user_idx] = combined_scores
                all_top_indices[user_idx] = combined_indices
        
        # Forzar liberación de memoria
        del predictions, batch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Convertir los resultados al formato deseado
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
                # Manejo defensivo de índices no válidos
                continue
    
    return batch_results

def create_user_item_pairs(users, items, device):
    """
    Crea pares usuario-item para evaluación eficiente.
    
    Parámetros:
    - users: Tensor con IDs de usuarios [n_users]
    - items: Tensor con IDs de items [n_items]
    - device: Dispositivo donde colocar los tensores
    
    Devuelve:
    - Diccionario con tensores para ser usado en model.predict()
    """
    n_users = users.shape[0]
    n_items = len(items)
    
    # Crear en CPU primero para ahorrar memoria en GPU
    expanded_users = users.repeat_interleave(n_items)
    expanded_items = torch.tensor(items, dtype=torch.long).repeat(n_users)
    
    # Mover a dispositivo apropiado
    expanded_users = expanded_users.to(device)
    expanded_items = expanded_items.to(device)
    
    # Crear el batch para el modelo
    batch = {
        'user_id': expanded_users,
        'game_id': expanded_items
    }
    
    return batch

def save_predictions(filepath, predictions):
    """
    Guarda las predicciones en un archivo TSV.
    
    Parámetros:
    - filepath: Ruta del archivo de salida
    - predictions: Lista de tuplas (user_id, item_id, score, test_item_id)
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Escribir encabezado si el archivo no existe
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write("user_id:token\titem_id:token\tscore:float\ttest_item_id:token\n")
    
    # Escribir predicciones
    with open(filepath, 'a') as f:
        for user_id, item_id, score, test_item_id in predictions:
            f.write(f"{user_id}\t{item_id}\t{score}\t{test_item_id}\n")
            
def get_unique_users_and_test_items(test_data):
    """
    Extrae usuarios únicos y sus correspondientes items de test.
    
    Parámetros:
    - test_data: Datos de test con inter_feat
    
    Devuelve:
    - users: Tensor con IDs de usuarios
    - test_items: Tensor con IDs de items de test correspondientes
    """
    users = test_data.dataset.inter_feat['user_id']
    test_items = test_data.dataset.inter_feat['game_id']
    
    return users, test_items