#!/usr/bin/env python3
"""
Módulo para cargar y preprocesar el dataset MovieLens.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import os


def load_movielens_100k(data_path='ml-100k'):
    """
    Carga el dataset MovieLens 100K.

    Parámetros:
    -----------
    data_path : str
        Ruta al directorio con los datos

    Retorna:
    --------
    ratings : DataFrame
        Calificaciones con columnas:
        user_id, item_id, rating, timestamp

    users : DataFrame
        Datos demográficos de usuarios

    items : DataFrame
        Información de películas

    R_sparse : csr_matrix
        Matriz de calificaciones en formato disperso
    """

    # Cargar calificaciones
    ratings = pd.read_csv(
        os.path.join(data_path, 'u.data'),
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )

    # Cargar usuarios
    users = pd.read_csv(
        os.path.join(data_path, 'u.user'),
        sep='|',
        names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
    )

    # Cargar películas
    items = pd.read_csv(
        os.path.join(data_path, 'u.item'),
        sep='|',
        encoding='latin-1',
        names=[
            'item_id',
            'title',
            'release_date',
            'video_date',
            'imdb_url'
        ] + [f'genre_{i}' for i in range(19)]
    )

    # Convertir IDs a índices base 0
    user_ids = ratings['user_id'].unique()
    item_ids = ratings['item_id'].unique()

    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    item_to_idx = {iid: i for i, iid in enumerate(item_ids)}

    ratings['user_idx'] = ratings['user_id'].map(user_to_idx)
    ratings['item_idx'] = ratings['item_id'].map(item_to_idx)

    n_users = len(user_ids)
    n_items = len(item_ids)

    # Crear matriz dispersa
    R_sparse = csr_matrix(
        (
            ratings['rating'].values,
            (
                ratings['user_idx'].values,
                ratings['item_idx'].values
            )
        ),
        shape=(n_users, n_items)
    )

    print("Dataset cargado:")
    print(f"- Usuarios: {n_users}")
    print(f"- Items: {n_items}")
    print(f"- Calificaciones: {len(ratings)}")
    print(f"- Densidad: {len(ratings) / (n_users * n_items):.4%}")

    return (
        ratings,
        users,
        items,
        R_sparse,
        user_to_idx,
        item_to_idx
    )


def train_test_split_ratings(ratings, test_size=0.2, random_state=42):
    """
    Divide las calificaciones en entrenamiento y prueba.
    """

    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state,
        stratify=ratings['user_id']
    )

    print(f"Entrenamiento: {len(train)} calificaciones")
    print(f"Prueba: {len(test)} calificaciones")

    return train, test


def build_matrix_from_df(df, n_users, n_items):
    """
    Construye una matriz dispersa a partir de un DataFrame.
    """

    return csr_matrix(
        (
            df['rating'].values,
            (
                df['user_idx'].values,
                df['item_idx'].values
            )
        ),
        shape=(n_users, n_items)
    )