#!/usr/bin/env python3
"""
Script principal para ejecutar experimentos de recomendación.

Compara SVD con estimación de datos faltantes vs ALS.
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("..")

from src.load_data import (
    load_movielens_100k,
    train_test_split_ratings,
    build_matrix_from_df,
)
from src.svd_recommender import SVDRecommender
from src.als_recommender import ALSRecommender
from src.metrics import rmse, mae


def main():
    parser = argparse.ArgumentParser(
        description="Experimentos de recomendación"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="ml-100k",
        help="Ruta a datos",
    )

    parser.add_argument(
        "--n_factors",
        type=int,
        default=20,
        help="Número de factores",
    )

    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=0.1,
        help="Regularización",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Iteraciones ALS",
    )

    parser.add_argument(
        "--estimation_method",
        type=str,
        default="global_mean",
        choices=[
            "global_mean",
            "user_mean",
            "item_mean",
            "zero",
        ],
        help="Método de estimación de valores faltantes para SVD",
    )

    args = parser.parse_args()

    print("Cargando datos...")

    ratings, users, items, R_sparse, user_map, item_map = (
        load_movielens_100k(args.data)
    )

    train_df, test_df = train_test_split_ratings(
        ratings,
        test_size=0.2,
    )

    n_users, n_items = R_sparse.shape

    R_train = build_matrix_from_df(
        train_df,
        n_users,
        n_items,
    )

    R_test = build_matrix_from_df(
        test_df,
        n_users,
        n_items,
    )

    print(f"\nMatriz de entrenamiento: {R_train.shape}")
    print(f"Calificaciones: {R_train.nnz}")
    print(f"Matriz de prueba: {R_test.shape}")
    print(f"Calificaciones: {R_test.nnz}")

    # ==========================================================
    # SVD con estimación de datos faltantes
    # ==========================================================

    print("\n" + "=" * 60)
    print("SVD con estimación de datos faltantes")
    print(f"Método de estimación: {args.estimation_method}")
    print("=" * 60)

    svd = SVDRecommender(
        n_factors=args.n_factors,
        estimation_method=args.estimation_method,
    )

    start = time.time()
    svd.fit(R_train)
    svd_time = time.time() - start

    test_rows, test_cols = R_test.nonzero()

    svd_preds = svd.predict_all(
        test_rows,
        test_cols,
    )

    svd_rmse = rmse(
        svd_preds,
        R_test.data,
    )

    svd_mae = mae(
        svd_preds,
        R_test.data,
    )

    print(f"Tiempo de entrenamiento: {svd_time:.2f} s")
    print(f"RMSE: {svd_rmse:.4f}")
    print(f"MAE: {svd_mae:.4f}")

    # ==========================================================
    # ALS
    # ==========================================================

    print("\n" + "=" * 60)
    print("ALS (Alternating Least Squares)")
    print("Trabaja directamente con datos observados, sin estimación")
    print("=" * 60)

    als = ALSRecommender(
        n_factors=args.n_factors,
        lambda_reg=args.lambda_reg,
        n_iterations=args.iterations,
        verbose=True,
    )

    start = time.time()
    als.fit(R_train, R_test)
    als_time = time.time() - start

    als_rmse = als.compute_rmse(R_test)

    print(f"\nTiempo total: {als_time:.2f} s")
    print(f"RMSE: {als_rmse:.4f}")

    # ==========================================================
    # Análisis de condicionamiento
    # ==========================================================

    print("\n" + "=" * 60)
    print("Análisis de condicionamiento numérico en ALS")
    print("=" * 60)

    cond_numbers = als.get_condition_numbers(R_train)

    print(
        f"Número de condición promedio: "
        f"{np.mean(cond_numbers):.2e}"
    )
    print(
        f"Número de condición máximo: "
        f"{np.max(cond_numbers):.2e}"
    )
    print(
        f"Número de condición mínimo: "
        f"{np.min(cond_numbers):.2e}"
    )

    # ==========================================================
    # Visualización de resultados
    # ==========================================================

    os.makedirs("figures", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    methods = [
        "SVD\n(estimación)",
        "ALS",
    ]

    # 1. Comparación de errores
    rmse_values = [
        svd_rmse,
        als_rmse,
    ]

    axes[0, 0].bar(
        methods,
        rmse_values,
    )
    axes[0, 0].set_ylabel("RMSE")
    axes[0, 0].set_title("Comparación de error (RMSE)")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Convergencia ALS
    axes[0, 1].plot(
        als.train_errors,
        linewidth=2,
        label="Entrenamiento",
    )

    axes[0, 1].plot(
        als.test_errors,
        linewidth=2,
        label="Prueba",
    )

    axes[0, 1].set_xlabel("Iteración")
    axes[0, 1].set_ylabel("RMSE")
    axes[0, 1].set_title("Convergencia de ALS")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Distribución de números de condición
    axes[1, 0].hist(
        cond_numbers,
        bins=50,
        alpha=0.7,
    )
    axes[1, 0].set_xlabel("Número de condición")
    axes[1, 0].set_ylabel("Frecuencia")
    axes[1, 0].set_title("Condicionamiento de subproblemas en ALS")
    axes[1, 0].set_xscale("log")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Tiempos de ejecución
    times = [
        svd_time,
        als_time,
    ]

    axes[1, 1].bar(
        methods,
        times,
    )
    axes[1, 1].set_ylabel("Tiempo (s)")
    axes[1, 1].set_title("Tiempo de entrenamiento")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "figures/experiment_results.png",
        dpi=150,
    )
    plt.show()

    # Guardar resultados
    results = {
        "svd": {
            "rmse": svd_rmse,
            "mae": svd_mae,
            "time": svd_time,
        },
        "als": {
            "rmse": als_rmse,
            "time": als_time,
            "condition_numbers": cond_numbers.tolist(),
        },
        "als_history": {
            "train": als.train_errors,
            "test": als.test_errors,
        },
    }

    np.save(
        "data/results.npy",
        results,
    )

    print("\nResultados guardados en data/results.npy")


if __name__ == "__main__":
    main()