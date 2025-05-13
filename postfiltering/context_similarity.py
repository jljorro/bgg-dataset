import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


class ContextSimilarity:
    """
    Class to calculate context similarity between two sets of data.
    """

    def __init__(self, train: pd.DataFrame):
        """
        Initialize the ContextSimilarity class with training and testing data.

        Args:
            train (pd.DataFrame): Training data.
            test (pd.DataFrame): Testing data.
        """
        self.train = train

        # Create a dictionary to store context for each item from training data
        self.context_dict = {}
        for _, row in train.iterrows():
            item_id = int(row['game_id:token'])
            context = row.iloc[4:].tolist()  # Exclude the first four columns (user_id, game_id, rating, timestamp)
            self.context_dict[item_id] = context

    def set_predictions(self, predictions: pd.DataFrame):
        """
        Set the predictions for the test data.

        Args:
            predictions (pd.DataFrame): Predictions data.
        """
        self.predictions = predictions

    def calculate_similarity(self, item_id: int, context: list) -> float:
        """
        Calculate the similarity between the context of a given item and the context of the training data.

        Args:
            item_id (int): The ID of the item to compare.
            context (list): The context of the item to compare.

        Returns:
            float: The similarity score.
        """
        if item_id not in self.context_dict:
            return 0.0

        train_context = self.context_dict[item_id]
        # Apply consine similarity form library scikit-learn
        

        # Convert lists to numpy arrays
        train_context = np.array(train_context).reshape(1, -1)
        context = np.array(context).reshape(1, -1)
        similarity = cosine_similarity(train_context, context)[0][0]
        return similarity

    def calculate_context_rank(self, user_id: int, item_id: int, context: list) -> float:
        """
        Calculate the context rank for a given user and item.

        Args:
            user_id (int): The ID of the user.
            item_id (int): The ID of the item.
            context (list): The context of the item.

        Returns:
            float: The context rank score.
        """
        # User predictions
        user_predictions = self.predictions[self.predictions['user_id:token'] == user_id]

        # Get item list from user predictions
        item_list = user_predictions['game_id:token'].tolist()

        # Calculate similarity for each item in the list
        similarity_scores = []
        for item in item_list:
            similarity = self.calculate_similarity(item, context)
            similarity_scores.append(similarity)

        # Create matrix with user_id, item_id and similarity scores
        similarity_matrix = pd.DataFrame({
            'user_id:token': [f"{user_id}_{item_id}"] * len(item_list),
            'game_id:token': item_list,
            'similarity': similarity_scores
        })
        # Sort the matrix by similarity scores
        similarity_matrix = similarity_matrix.sort_values(by='similarity', ascending=False)

        return similarity_matrix
        


        
            
        
