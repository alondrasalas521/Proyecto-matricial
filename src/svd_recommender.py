```python
#!/usr/bin/env python3
"""
Implementación de sistema de recomendación basado en SVD truncada
con estimación previa de valores faltantes.
"""

import numpy as np
from scipy.sparse import csr_matrix


class SVDRecommender:
    """
    Recomendador basado en SVD truncada con estimación de datos faltantes.

    Los valores faltantes se estiman utilizando estadísticas descriptivas
    (medias) de los datos disponibles antes de aplicar SVD.
    """

    def __init__(self, n_factors=50, estimation_method='global_mean'):
        """
        Parámetros:
        -----------
        n_factors : int
            Número de factores latentes (rango de aproximación)

        estimation_method : str
            Método de estimación de valores faltantes:
            - 'global_mean': media global de todas las calificaciones
            - 'user_mean'  : media por usuario
            - 'item_mean'  : media por item
            - 'zero'       : estimar con cero
        """

        self.n_factors = n_factors
        self.estimation_method = estimation_method

        self.U = None
        self.s = None
        self.Vt = None

        self.global_mean = None
        self.user_means = None
        self.item_means = None

    def fit(self, R_train):
        """
        Entrena el modelo.

        Parámetros:
        -----------
        R_train : csr_matrix
            Matriz de calificaciones de entrenamiento
        """

        n_users, n_items = R_train.shape

        # Calcular medias según método de estimación
        if self.estimation_method == 'global_mean':

            self.global_mean = R_train.data.mean()
            estimated_value = self.global_mean

        elif self.estimation_method == 'user_mean':

            self.user_means = np.array(
                R_train.mean(axis=1)
            ).flatten()

            estimated_value = self.user_means.reshape(-1, 1)

        elif self.estimation_method == 'item_mean':

            self.item_means = np.array(
                R_train.mean(axis=0)
            ).flatten()

            estimated_value = self.item_means.reshape(1, -1)

        else:  # 'zero'
            estimated_value = 0

        # Convertir a matriz densa estimando valores faltantes
        R_dense = R_train.toarray()

        if self.estimation_method == 'global_mean':

            missing_mask = (R_dense == 0)
            R_dense[missing_mask] = self.global_mean

        elif self.estimation_method == 'user_mean':

            for i in range(n_users):

                missing_mask = (R_dense[i, :] == 0)
                R_dense[i, missing_mask] = self.user_means[i]

        elif self.estimation_method == 'item_mean':

            for j in range(n_items):

                missing_mask = (R_dense[:, j] == 0)
                R_dense[missing_mask, j] = self.item_means[j]

        # Aplicar SVD truncada
        U, s, Vt = np.linalg.svd(
            R_dense,
            full_matrices=False
        )

        # Truncar
        self.U = U[:, :self.n_factors]
        self.s = s[:self.n_factors]
        self.Vt = Vt[:self.n_factors, :]

    def predict(self, user_idx, item_idx):
        """
        Predice la calificación para un par (usuario, item).
        """

        if self.U is None:
            raise ValueError("El modelo no ha sido entrenado")

        prediction = (
            self.U[user_idx, :] @ self.Vt[:, item_idx]
        )

        return np.clip(prediction, 1, 5)

    def predict_all(self, user_indices=None, item_indices=None):
        """
        Predice calificaciones para múltiples pares.
        """

        if user_indices is None or item_indices is None:

            return self.U @ self.Vt

        else:

            predictions = np.zeros(len(user_indices))

            for i, (u, it) in enumerate(
                zip(user_indices, item_indices)
            ):

                predictions[i] = self.predict(u, it)

            return predictions
```
