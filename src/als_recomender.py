#!/usr/bin/env python3
"""
Implementación de sistema de recomendación basado en ALS.

ALS trabaja directamente con los datos observados sin necesidad
de estimar valores faltantes.
"""

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm


class ALSRecommender:
    """
    Recomendador basado en Alternating Least Squares (ALS).

    A diferencia del enfoque con estimación de valores faltantes,
    ALS trabaja exclusivamente con las calificaciones observadas,
    evitando introducir sesgos artificiales.
    """

    def __init__(
        self,
        n_factors=50,
        lambda_reg=0.1,
        n_iterations=20,
        verbose=True
    ):
        """
        Parámetros:
        -----------
        n_factors : int
            Número de factores latentes

        lambda_reg : float
            Parámetro de regularización

        n_iterations : int
            Número de iteraciones ALS

        verbose : bool
            Si es True, muestra progreso
        """

        self.n_factors = n_factors
        self.lambda_reg = lambda_reg
        self.n_iterations = n_iterations
        self.verbose = verbose

        self.U = None
        self.V = None

        self.train_errors = []
        self.test_errors = []

    def fit(self, R_train, R_test=None):
        """
        Entrena el modelo usando ALS.

        Parámetros:
        -----------
        R_train : csr_matrix
            Matriz de entrenamiento

        R_test : csr_matrix, opcional
            Matriz de prueba para monitoreo
        """

        n_users, n_items = R_train.shape

        # Inicialización aleatoria
        np.random.seed(42)

        self.U = (
            np.random.randn(n_users, self.n_factors) * 0.1
        )

        self.V = (
            np.random.randn(n_items, self.n_factors) * 0.1
        )

        # Precomputar observaciones por usuario
        user_items = [
            R_train[i].nonzero()[1]
            for i in range(n_users)
        ]

        user_ratings = [
            R_train[i].data
            for i in range(n_users)
        ]

        # Precomputar observaciones por item
        item_users = [
            R_train[:, j].nonzero()[0]
            for j in range(n_items)
        ]

        item_ratings = [
            R_train[:, j].data
            for j in range(n_items)
        ]

        # Iteraciones ALS
        iterator = range(self.n_iterations)

        if self.verbose:
            iterator = tqdm(iterator, desc="Entrenando ALS")

        for iteration in iterator:

            # Actualizar factores de usuario
            self._update_users(
                user_items,
                user_ratings
            )

            # Actualizar factores de item
            self._update_items(
                item_users,
                item_ratings
            )

            # Evaluar entrenamiento
            train_rmse = self.compute_rmse(R_train)
            self.train_errors.append(train_rmse)

            if R_test is not None:
                test_rmse = self.compute_rmse(R_test)
                self.test_errors.append(test_rmse)

                if self.verbose:
                    print(
                        f"Iteración {iteration + 1}/"
                        f"{self.n_iterations} | "
                        f"Train RMSE: {train_rmse:.4f} | "
                        f"Test RMSE: {test_rmse:.4f}"
                    )

            else:
                if self.verbose:
                    print(
                        f"Iteración {iteration + 1}/"
                        f"{self.n_iterations} | "
                        f"Train RMSE: {train_rmse:.4f}"
                    )

        return self

    def _update_users(self, user_items, user_ratings):
        """
        Actualiza factores de usuario resolviendo
        problemas de mínimos cuadrados regularizados.
        """

        for i in range(len(user_items)):

            items = user_items[i]
            ratings = user_ratings[i]

            if len(items) == 0:
                continue

            V_i = self.V[items, :]

            # Sistema normal regularizado
            A = (
                V_i.T @ V_i
                + self.lambda_reg * np.eye(self.n_factors)
            )

            b = V_i.T @ ratings

            self.U[i, :] = np.linalg.solve(A, b)

    def _update_items(self, item_users, item_ratings):
        """
        Actualiza factores de item resolviendo
        problemas de mínimos cuadrados regularizados.
        """

        for j in range(len(item_users)):

            users = item_users[j]
            ratings = item_ratings[j]

            if len(users) == 0:
                continue

            U_j = self.U[users, :]

            A = (
                U_j.T @ U_j
                + self.lambda_reg * np.eye(self.n_factors)
            )

            b = U_j.T @ ratings

            self.V[j, :] = np.linalg.solve(A, b)

    def predict(self, user_idx, item_idx):
        """
        Predice la calificación para un usuario-item.
        """

        prediction = (
            self.U[user_idx, :]
            @ self.V[item_idx, :]
        )

        return np.clip(prediction, 1, 5)

    def predict_all(self):
        """
        Reconstruye la matriz completa de predicciones.
        """

        return np.clip(
            self.U @ self.V.T,
            1,
            5
        )

    def compute_rmse(self, R):
        """
        Calcula RMSE sobre las entradas conocidas.
        """

        rows, cols = R.nonzero()

        predictions = np.array([
            self.predict(r, c)
            for r, c in zip(rows, cols)
        ])

        true_values = R.data

        rmse = np.sqrt(
            np.mean((predictions - true_values) ** 2)
        )

        return rmse

    def get_condition_numbers(self, R_train):
        """
        Analiza el condicionamiento de los subproblemas ALS.

        Útil para estudiar estabilidad numérica.
        """

        n_users, _ = R_train.shape

        cond_numbers = []

        for i in range(n_users):

            items = R_train[i].nonzero()[1]

            if len(items) > self.n_factors:

                V_i = self.V[items, :]

                A = (
                    V_i.T @ V_i
                    + self.lambda_reg
                    * np.eye(self.n_factors)
                )

                s = np.linalg.svd(
                    A,
                    compute_uv=False
                )

                cond_numbers.append(
                    s[0] / s[-1]
                )

        return np.array(cond_numbers)