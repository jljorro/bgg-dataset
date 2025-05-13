import pandas as pd
import cupy as cp
import numpy as np

from cuml.metrics import pairwise_distances


class ContextSimilarity:
    """
    Class to calculate context similarity using GPU (CuPy + cuML).
    """

    def __init__(self, train: pd.DataFrame):
        """
        Initialize the ContextSimilarity class with training data.
        """
        self.train = train

        # Create a dictionary to store context for each item from training data
        self.context_dict = {}
        for _, row in train.iterrows():
            item_id = int(row['game_id:token'])
            context = row.iloc[4:].astype(float).tolist()  # Convert to float for GPU ops
            self.context_dict[item_id] = context

    def set_predictions(self, predictions: pd.DataFrame):
        """
        Set the predictions for the test data.
        """
        self.predictions = predictions

    def calculate_similarity(self, item_id: int, context: list) -> float:
        """
        Compute cosine similarity between training item and current context using GPU.
        """
        if item_id not in self.context_dict:
            return 0.0

        train_context = cp.array(self.context_dict[item_id], dtype=cp.float32).reshape(1, -1)
        context = cp.array(context, dtype=cp.float32).reshape(1, -1)

        # Normalize to unit vectors
        train_context /= cp.linalg.norm(train_context, axis=1, keepdims=True)
        context /= cp.linalg.norm(context, axis=1, keepdims=True)

        # Compute cosine similarity
        similarity = 1 - pairwise_distances(train_context, context, metric='cosine')[0][0]
        return float(similarity)

    def calculate_context_rank(self, user_id: int, item_id: int, context: list) -> pd.DataFrame:
        """
        Compute similarity between context and all predicted items for a user.
        """
        user_predictions = self.predictions[self.predictions['user_id:token'] == user_id]
        item_list = user_predictions['game_id:token'].tolist()

        train_contexts = []
        for i in item_list:
            c = self.context_dict.get(i, [0.0] * len(context))
            train_contexts.append(c)

        train_matrix = cp.array(train_contexts, dtype=cp.float32)
        context_vector = cp.array(context, dtype=cp.float32).reshape(1, -1)

        # Normalize
        train_matrix /= cp.linalg.norm(train_matrix, axis=1, keepdims=True)
        context_vector /= cp.linalg.norm(context_vector, axis=1, keepdims=True)

        similarity_scores = 1 - pairwise_distances(train_matrix, context_vector, metric='cosine').get().flatten()

        return pd.DataFrame({
            'user_id:token': [f"{user_id}_{item_id}"] * len(item_list),
            'game_id:token': item_list,
            'similarity': similarity_scores
        })
