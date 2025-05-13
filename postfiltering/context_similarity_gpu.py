import pandas as pd
import cupy as cp
import numpy as np

class ContextPostfiltering:
    """
    Class to calculate context similarity using GPU (CuPy + cuML).
    """

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, predictions: pd.DataFrame):
        """
        Initialize the ContextPostfiltering class.

        Args:
            train (pd.DataFrame): Training data.
            test (pd.DataFrame): Test data.
            predictions (pd.DataFrame): Predictions data.
        """
        
        # Create dictionary to map items IDs to indices
        self.train_item_to_index = {item: index for index, item in enumerate(train['item_id'].unique())}
        self.test_item_to_index = {item: index for index, item in enumerate(test['item_id'].unique())}

        # Extract vectors from DataFrames
        self.train_vectors = self._extract_vectors_from_df(train, self.train_item_to_index)
        self.test_vectors = self._extract_vectors_from_df(test, self.test_item_to_index)

    def _extract_vectors_from_df(self, df, item_to_index):
        """
        Extrae una matriz (n_items, d) de vectores únicos usando columnas [5:] del DataFrame.
        """
        vector_columns = df.columns[5:]  # columnas numéricas para los vectores
        d = len(vector_columns)
        vectors = np.zeros((len(item_to_index), d), dtype=np.float32)

        for item_id, idx in item_to_index.items():
            # Tomamos el primer vector para ese item_id
            item_vector = df[df['item_id'] == item_id][vector_columns].iloc[0].to_numpy(dtype=np.float32)
            vectors[idx] = item_vector

        return vectors
    
    def fit(self):
        """
        Calcula la matriz de similitud coseno entre los ítems de train (filas) y test (columnas).
        Resultado: matriz (n_train_items, n_test_items)
        """
        # Paso 1: Normalizar los vectores (L2)
        train_norm = cp.linalg.norm(self.train_matrix, axis=1, keepdims=True) + 1e-8
        test_norm = cp.linalg.norm(self.test_matrix, axis=1, keepdims=True) + 1e-8

        train_normalized = self.train_matrix / train_norm
        test_normalized = self.test_matrix / test_norm

        # Paso 2: Producto punto → similitud coseno
        # Resultado: (n_train_items, n_test_items)
        self.similarity_matrix = cp.matmul(train_normalized, test_normalized.T)

    def get_similarity(self, test_item_id: int, pred_item_ids: list[list[int]]) -> np.ndarray:
        """
        Dado un ítem de test y una lista de listas de ítems de predicción (por usuario),
        devuelve una matriz con los ítems a comparar y su similitud con el ítem de test.

        Salida: np.ndarray (n_preds, 2) con columnas [item_id, similitud], ordenado desc.
        """
        if self.similarity_matrix is None:
            raise ValueError("Debes ejecutar fit() antes de usar get_similarity().")

        if test_item_id not in self.test_item_to_index:
            raise KeyError(f"El test_item_id '{test_item_id}' no existe en test.")

        test_idx = self.test_item_to_index[test_item_id]

        # Aplanamos la lista de listas a una lista simple (único set de predicciones)
        flat_pred_ids = [item_id for sublist in pred_item_ids for item_id in sublist]

        similarities = []
        for item_id in flat_pred_ids:
            if item_id in self.train_item_to_index:
                train_idx = self.train_item_to_index[item_id]
                sim = float(self.similarity_matrix[train_idx, test_idx])
                similarities.append((item_id, sim))
            else:
                # Si no está en train, puedes asignar 0 o NaN según preferencia
                similarities.append((item_id, 0.0))

        # Ordenamos por similitud descendente
        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

        return np.array(sorted_similarities, dtype=object)
