#!/usr/bin/env python3

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.load_data import (
    load_movielens_100k,
    train_test_split_ratings,
    build_matrix_from_df,
)

from src.svd_recommender import SVDRecommender
from src.metrics import mae


def svd_predict(model, user_idx, item_idx):
    prediction = (model.U[user_idx, :] * model.s) @ model.Vt[:, item_idx]
    return np.clip(prediction, 1, 5)


def evaluate_svd(model, R_test):
    rows, cols = R_test.nonzero()

    predictions = np.array([
        svd_predict(model, u, i)
        for u, i in zip(rows, cols)
    ])

    true_values = R_test.data

    test_rmse = np.sqrt(np.mean((predictions - true_values) ** 2))
    test_mae = mae(predictions, true_values)

    return test_rmse, test_mae


def run_svd_experiments(data_path="data/ml-100k"):
    print("Cargando datos desde:", data_path)

    ratings, users, items, R_sparse, user_map, item_map = load_movielens_100k(data_path)

    train_df, test_df = train_test_split_ratings(
        ratings,
        test_size=0.2,
        random_state=42
    )

    n_users, n_items = R_sparse.shape

    R_train = build_matrix_from_df(train_df, n_users, n_items)
    R_test = build_matrix_from_df(test_df, n_users, n_items)

    k_values = [3, 6, 9, 12, 15]

    estimation_methods = [
        "global_mean",
        "user_mean",
        "item_mean",
        "zero"
    ]

    results = []

    for method in estimation_methods:
        for k in k_values:
            print("=" * 60)
            print(f"SVD | método={method} | k={k}")
            print("=" * 60)

            model = SVDRecommender(
                n_factors=k,
                estimation_method=method
            )

            start = time.time()
            model.fit(R_train)
            elapsed_time = time.time() - start

            test_rmse, test_mae = evaluate_svd(model, R_test)

            results.append({
                "metodo": method,
                "k": k,
                "RMSE": test_rmse,
                "MAE": test_mae,
                "tiempo": elapsed_time
            })

            print(f"RMSE: {test_rmse:.4f}")
            print(f"MAE: {test_mae:.4f}")
            print(f"Tiempo: {elapsed_time:.2f} s")

    results_df = pd.DataFrame(results)

    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    results_df.to_csv("data/svd_estimation_results.csv", index=False)

    plot_svd_results(results_df)

    print("\nArchivos guardados:")
    print("- data/svd_estimation_results.csv")
    print("- figures/svd_rmse_estimacion.png")
    print("- figures/svd_mae_estimacion.png")
    print("- figures/svd_tiempo_estimacion.png")

    print("\nMejor configuración por RMSE:")
    print(results_df.loc[results_df["RMSE"].idxmin()])

    return results_df


def plot_svd_results(results_df):
    methods = results_df["metodo"].unique()

    # RMSE
    plt.figure(figsize=(9, 6))

    for method in methods:
        subset = results_df[results_df["metodo"] == method]
        plt.plot(subset["k"], subset["RMSE"], marker="o", label=method)

    plt.xlabel("Número de factores latentes k")
    plt.ylabel("RMSE")
    plt.title("SVD: RMSE para distintos métodos de estimación")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/svd_rmse_estimacion.png", dpi=150)
    plt.show()

    # MAE
    plt.figure(figsize=(9, 6))

    for method in methods:
        subset = results_df[results_df["metodo"] == method]
        plt.plot(subset["k"], subset["MAE"], marker="o", label=method)

    plt.xlabel("Número de factores latentes k")
    plt.ylabel("MAE")
    plt.title("SVD: MAE para distintos métodos de estimación")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/svd_mae_estimacion.png", dpi=150)
    plt.show()

    # Tiempo
    plt.figure(figsize=(9, 6))

    for method in methods:
        subset = results_df[results_df["metodo"] == method]
        plt.plot(subset["k"], subset["tiempo"], marker="o", label=method)

    plt.xlabel("Número de factores latentes k")
    plt.ylabel("Tiempo de entrenamiento (s)")
    plt.title("SVD: tiempo para distintos métodos de estimación")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/svd_tiempo_estimacion.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default="data/ml-100k",
        help="Ruta a la carpeta del dataset MovieLens 100K"
    )

    args = parser.parse_args()

    run_svd_experiments(data_path=args.data)