import numpy as np
from scipy.spatial.distance import cdist

class ContextPostfiltering:

    def __init__(self, train_df):
        """
        Inicializa el modelo de postfiltrado contextual.
        
        Parámetros:
        - train_df: pd.DataFrame con los datos de entrenamiento.
                    Se espera que la segunda columna (índice 1) contenga los game_id,
                    y que las columnas de contexto comiencen en la columna de índice 4.
        """
        self.context_dict = {}
        
        # Accede solo a las columnas necesarias: game_id y contexto
        game_id_col_idx = 1
        context_start_idx = 4
        cols = [game_id_col_idx] + list(range(context_start_idx, train_df.shape[1]))
        subset = train_df.iloc[:, cols]

        for row in subset.itertuples(index=False):
            game_id = row[0]
            if game_id not in self.context_dict:
                # Convertimos el vector de contexto a tupla (más eficiente como clave si se necesita)
                self.context_dict[game_id] = tuple(row[1:])

    def get_context(self, game_ids):
        """
        Dado una lista de game_ids, devuelve:
        - una matriz NumPy (n_items_context_found x n_features_context) con los vectores de contexto disponibles
        - un diccionario que mapea cada game_id a su índice de fila en la matriz (o -1 si no está disponible)
        """
        context_matrix = []
        index_mapping = {}

        for idx, game_id in enumerate(game_ids):
            context_vector = self.context_dict.get(game_id)
            if context_vector is not None:
                index_mapping[game_id] = len(context_matrix)
                context_matrix.append(context_vector)
            else:
                index_mapping[game_id] = -1

        # Convertimos a NumPy array si se ha encontrado algún contexto
        context_matrix = np.array(context_matrix) if context_matrix else np.empty((0, 0))

        return context_matrix, index_mapping

    def calculate_similarities(self, context_query, items_predictions):
        """
        Calcula la similitud de coseno entre context_query y los contextos de items_predictions.

        Parámetros:
        - context_query: np.array de forma (n_context_features,)
        - items_predictions: lista de game_ids a evaluar

        Retorna:
        - Tupla (game_ids_sorted, similarities_sorted), ambos np.array ordenados por similitud descendente.
        """
        context_matrix, index_mapping = self.get_context(items_predictions)

        game_ids_array = np.array(items_predictions)
        similarities = np.full(len(game_ids_array), -np.inf, dtype=np.float32)

        if context_matrix.size > 0:
            context_query = context_query.reshape(1, -1)
            cosine_sim = 1 - cdist(context_matrix, context_query, metric='cosine').flatten()

            # Mapeamos game_ids válidos a sus posiciones en game_ids_array
            valid_indices = [i for i, gid in enumerate(game_ids_array) if index_mapping[gid] != -1]
            mapped_indices = [index_mapping[game_ids_array[i]] for i in valid_indices]

            # Asignación directa por slicing
            similarities[valid_indices] = cosine_sim[mapped_indices]

        sorted_indices = np.argsort(-similarities)
        return game_ids_array[sorted_indices], similarities[sorted_indices]