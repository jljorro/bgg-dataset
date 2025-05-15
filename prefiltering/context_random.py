import pandas as pd
import numpy as np
from collections import defaultdict
from functools import lru_cache

class ContextRandom:

    def __init__(self, train_df: pd.DataFrame):
        # Guardamos el DataFrame completo
        self.train_df = train_df
        
        # Creamos un diccionario para indexar los ítems por contexto
        # Esto acelerará significativamente las búsquedas
        self.context_index = defaultdict(set)  # Cambiado a set para evitar duplicados
        
        # Preprocesamiento: creamos un índice para cada contexto único
        for _, row in train_df.iterrows():
            # Usamos nombres de columnas o iloc para evitar FutureWarning
            user_id = int(row.iloc[0]) if isinstance(row, pd.Series) else int(row[0])
            game_id = int(row.iloc[1]) if isinstance(row, pd.Series) else int(row[1])
            
            # Extraemos el contexto (columnas a partir de la 5ta)
            if isinstance(row, pd.Series):
                context_tuple = tuple(row.iloc[4:].astype(int))
            else:
                context_tuple = tuple(row[4:].astype(int))
            
            # Guardamos el item_id (game_id) para este contexto en un set para evitar duplicados
            self.context_index[context_tuple].add(game_id)

    def prepare_data(self):
        # Concatenamos los valores de contexto en una sola cadena para cada fila
        self.train_df['context'] = self.train_df.iloc[:, 4:].astype(str).agg(''.join, axis=1)

        # Generamos un diccionario donde por cada contexto obtenemos los items únicos
        self.context_dict = self.train_df.groupby('context')['game_id:token'].apply(lambda x: x.unique()).to_dict()
    
    @lru_cache(maxsize=1024)  # Caché para evitar recálculos del mismo contexto
    def get_items_for_context(self, context_tuple):
        # Convertimos el set a lista para poder usarlo con numpy random choice
        return list(self.context_index.get(context_tuple, set()))

    def recommend(self, context_query, k):
        # Convertimos el array de NumPy a una tupla para usarlo como clave del diccionario
        context_tuple = tuple(context_query)
        
        # Obtener los ítems que cumplen con el contexto (ya sin duplicados)
        items = self.get_items_for_context(context_tuple)
        
        # Si no hay items para este contexto, devolvemos listas vacías
        if len(items) == 0:
            return [], []
            
        # Convertimos a array de NumPy para facilitar la manipulación
        items_array = np.array(items)
        
        # Verificación adicional contra duplicados (no debería ser necesaria pero por seguridad)
        items_array = np.unique(items_array)
        
        if len(items_array) > k:
            # Tenemos suficientes items, seleccionamos k aleatoriamente
            items_selected = np.random.choice(items_array, size=k, replace=False)
        else:
            # Tenemos menos de k items, usamos todos pero en orden aleatorio
            # Creamos un índice de permutación aleatorio
            random_indices = np.random.permutation(len(items_array))
            items_selected = items_array[random_indices]
        
        # Hacemos un vector de tamaño len(items_selected) con valores len(items_selected) y reduciendo uno a uno
        relevance = np.full(len(items_selected), range(len(items_selected), 0, -1))
        
        return items_selected, relevance.tolist()