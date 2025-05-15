import pandas as pd
import numpy as np
from collections import defaultdict, Counter

class ContextPop:
    """
    Clase que implementa un recomendador basado en la popularidad de los ítems dentro de cada contexto.
    Para cada contexto, devuelve los ítems ordenados por frecuencia de aparición.
    """

    def __init__(self, train_df: pd.DataFrame):
        # Guardamos el DataFrame completo
        self.train_df = train_df
        
        # Creamos un diccionario para indexar los ítems por contexto
        # y contar sus apariciones (popularidad)
        self.context_index = defaultdict(Counter)
        
        print("Construyendo índice de contextos y calculando popularidad...")
        # Preprocesamiento: creamos un índice para cada contexto único
        # y contamos las apariciones de cada ítem
        for _, row in train_df.iterrows():
            # Usamos nombres de columnas o iloc para evitar FutureWarning
            user_id = int(row.iloc[0]) if isinstance(row, pd.Series) else int(row[0])
            game_id = int(row.iloc[1]) if isinstance(row, pd.Series) else int(row[1])
            
            # Extraemos el contexto (columnas a partir de la 5ta)
            if isinstance(row, pd.Series):
                context_tuple = tuple(row.iloc[4:].astype(int))
            else:
                context_tuple = tuple(row[4:].astype(int))
            
            # Incrementamos el contador para este ítem en este contexto
            self.context_index[context_tuple][game_id] += 1
        
        # Precalculamos las listas ordenadas de ítems por popularidad para cada contexto
        print("Precalculando listas de items ordenadas por popularidad...")
        self.context_items_sorted = {}
        for context, counter in self.context_index.items():
            # Ordenamos los ítems por frecuencia (popularidad) descendente
            items_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
            # Guardamos solo los IDs de ítems ya ordenados
            self.context_items_sorted[context] = [item_id for item_id, _ in items_sorted]
            
        print(f"Índice construido con {len(self.context_index)} contextos únicos")

    def get_items_for_context(self, context_tuple):
        """Obtiene la lista de ítems ordenados por popularidad para un contexto dado"""
        return self.context_items_sorted.get(context_tuple, [])

    def recommend(self, context_query, k):
        """
        Recomienda los k ítems más populares para un contexto dado
        
        Args:
            context_query: Vector de contexto
            k: Número de ítems a recomendar
            
        Returns:
            Tupla de (lista de ids de ítems, lista de valores de relevancia)
        """
        # Si context_query es un array de NumPy, lo convertimos a tupla
        if isinstance(context_query, np.ndarray):
            context_tuple = tuple(context_query)
        else:
            # Si ya es una tupla o lista, la usamos directamente
            context_tuple = tuple(context_query)
        
        # Obtener los ítems ordenados por popularidad para este contexto
        items = self.get_items_for_context(context_tuple)
        
        # Si no hay items para este contexto, devolvemos listas vacías
        if len(items) == 0:
            return [], []
        
        # Tomamos solo los primeros k ítems (o todos si hay menos de k)
        items_selected = items[:k]
        
        # Convertimos a array de NumPy
        items_array = np.array(items_selected)
        
        # Generamos relevancia decreciente: k para el primer item, k-1 para el segundo, etc.
        relevance = np.arange(len(items_array), 0, -1)
        
        return items_array, relevance.tolist()