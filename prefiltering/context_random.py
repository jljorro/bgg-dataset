import pandas as pd
import numpy as np
from collections import defaultdict
from functools import lru_cache

class ContextRandom:

    def __init__(self, train_df: pd.DataFrame):
        self.train_df = train_df
        
        # We created a dictionary to index items by context
        # This will significantly speed up searches
        self.context_index = defaultdict(set)  # Changed to set to avoid duplicates
        
        # Preprocessing: We create an index for each unique context
        for _, row in train_df.iterrows():
            # We use column names or iloc to avoid FutureWarning
            user_id = int(row.iloc[0]) if isinstance(row, pd.Series) else int(row[0])
            game_id = int(row.iloc[1]) if isinstance(row, pd.Series) else int(row[1])
            
            # We extract the context (columns starting from the 5th)
            if isinstance(row, pd.Series):
                context_tuple = tuple(row.iloc[4:].astype(int))
            else:
                context_tuple = tuple(row[4:].astype(int))
            
            # We save the item_id (game_id) for this context in a set to avoid duplicates
            self.context_index[context_tuple].add(game_id)

    def prepare_data(self):
        # We concatenate the context values ​​into a single string for each row
        self.train_df['context'] = self.train_df.iloc[:, 4:].astype(str).agg(''.join, axis=1)

        # We generate a dictionary where for each context we obtain the unique items
        self.context_dict = self.train_df.groupby('context')['game_id:token'].apply(lambda x: x.unique()).to_dict()
    
    @lru_cache(maxsize=1024)  # Cache to avoid recalculations of the same context
    def get_items_for_context(self, context_tuple):
        # We convert the set to a list so we can use it with numpy random choice
        return list(self.context_index.get(context_tuple, set()))

    def recommend(self, context_query, k):
        # We convert the NumPy array to a tuple to use it as the dictionary key
        context_tuple = tuple(context_query)
        
        # Get the items that match the context (no duplicates anymore)
        items = self.get_items_for_context(context_tuple)
        
        # If there are no items for this context, we return empty lists
        if len(items) == 0:
            return [], []
            
        # Convert to NumPy array for easier manipulation
        items_array = np.array(items)
        
        # Additional duplicate check (shouldn't be necessary but for security)
        items_array = np.unique(items_array)
        
        if len(items_array) > k:
            # We have enough items, we select k randomly
            items_selected = np.random.choice(items_array, size=k, replace=False)
        else:
            # We have fewer than k items, we use all of them but in random order
            # We create a random permutation index
            random_indices = np.random.permutation(len(items_array))
            items_selected = items_array[random_indices]
        
        # We make a vector of size len(items_selected) with values ​​len(items_selected) and reducing them one by one
        relevance = np.full(len(items_selected), range(len(items_selected), 0, -1))
        
        return items_selected, relevance.tolist()