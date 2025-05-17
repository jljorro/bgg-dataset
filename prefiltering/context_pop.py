import pandas as pd
import numpy as np
from collections import defaultdict, Counter

class ContextPop:
    """
    Class that implements a recommender based on the popularity of items within each context.
    For each context, it returns the items sorted by frequency of appearance.
    """

    def __init__(self, train_df: pd.DataFrame):
        self.train_df = train_df
        
        # We created a dictionary to index items by context
        # and count their occurrences (popularity)
        self.context_index = defaultdict(Counter)
        
        print("Building context index and calculating popularity...")
        # Preprocessing: We create an index for each unique context
        # and count the occurrences of each item
        for _, row in train_df.iterrows():
            # We use column names or iloc to avoid FutureWarning
            user_id = int(row.iloc[0]) if isinstance(row, pd.Series) else int(row[0])
            game_id = int(row.iloc[1]) if isinstance(row, pd.Series) else int(row[1])
            
            # We extract the context (columns starting from the 5th)
            if isinstance(row, pd.Series):
                context_tuple = tuple(row.iloc[4:].astype(int))
            else:
                context_tuple = tuple(row[4:].astype(int))
            
            # We increment the counter for this item in this context
            self.context_index[context_tuple][game_id] += 1
        
        # We precalculate the ordered lists of items by popularity for each context
        print("Precalculating lists of items ordered by popularity...")
        self.context_items_sorted = {}
        for context, counter in self.context_index.items():
            # We sort the items by descending frequency (popularity)
            items_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
            # We only save the IDs of already sorted items
            self.context_items_sorted[context] = [item_id for item_id, _ in items_sorted]
            
        print(f"Index built with {len(self.context_index)} unique contexts")

    def get_items_for_context(self, context_tuple):
        """Gets the list of items sorted by popularity for a given context"""
        return self.context_items_sorted.get(context_tuple, [])

    def recommend(self, context_query, k):
        """
        Recommends the k most popular items for a given context

        Args:
        context_query: Context vector
        k: Number of items to recommend

        Returns:
        Tuple of (list of item ids, list of relevance values)
        """
        # If context_query is a NumPy array, we convert it to a tuple
        if isinstance(context_query, np.ndarray):
            context_tuple = tuple(context_query)
        else:
            # If it's already a tuple or list, we use it directly
            context_tuple = tuple(context_query)
        
        # Get items sorted by popularity for this context
        items = self.get_items_for_context(context_tuple)
        
        # If there are no items for this context, we return empty lists
        if len(items) == 0:
            return [], []
        
        # We take only the first k items (or all of them if there are less than k)
        items_selected = items[:k]
        
        # We convert to NumPy array
        items_array = np.array(items_selected)
        
        # We generate decreasing relevance: k for the first item, k-1 for the second, etc.
        relevance = np.arange(len(items_array), 0, -1)
        
        return items_array, relevance.tolist()