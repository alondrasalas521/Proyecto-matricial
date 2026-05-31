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

from src.als_recommender import ALSRecommender
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


def evaluate_als(model, R_test):
    rows, cols = R_test.nonzero()

    predictions = np.array([
        model.predict(u, i)
        for u, i in zip(rows, cols)
    ])

    true_values = R_test.data

    test_rmse = np.sqrt(np.mean((predictions - true_values) ** 2))
    test_mae = mae(predictions, true_values)

    return test_rmse, test_mae


def run_comparison(data_path="data/ml-100k"):
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
    best_lambda = 0.1

    results = []

    for k in k_values:
        print("=" * 60)
        print(f"Experimento con k={k}")
        print("=" * 60)

        print("Entrenando ALS...")

        als_model = ALSRecommender(
            n_factors=k,
            lambda_reg=best_lambda,
            n_iterations=20,
            verbose=False
        )

        start = time.time()
        als_model.fit(R_train, R_test)
        als_time = time.time() - start

        als_rmse, als_mae = evaluate_als(als_model, R_test)

        results.append({
            "metodo": "ALS",
            "k": k,
            "lambda": best_lambda,
            "RMSE": als_rmse,
            "MAE": als_mae,
            "tiempo": als_time
        })

        print(f"ALS | RMSE: {als_rmse:.4f} | MAE: {als_mae:.4f} | Tiempo: {als_time:.2f}s")

        print("Entrenando SVD...")

        svd_model = SVDRecommender(
            n_factors=k,
            estimation_method="global_mean"
        )

        start = time.time()
        svd_model.fit(R_train)
        svd_time = time.time() - start

        svd_rmse, svd_mae = evaluate_svd(svd_model, R_test)

        results.append({
            "metodo": "SVD",
            "k": k,
            "lambda": np.nan,
            "RMSE": svd_rmse,
            "MAE": svd_mae,
            "tiempo": svd_time
        })

        print(f"SVD | RMSE: {svd_rmse:.4f} | MAE: {svd_mae:.4f} | Tiempo: {svd_time:.2f}s")

    results_df = pd.DataFrame(results)

    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    results_df.to_csv("data/comparacion_als_svd.csv", index=False)

    plot_comparison(results_df)

    print("\nArchivos guardados:")
    print("- data/comparacion_als_svd.csv")
    print("- figures/als_vs_svd_rmse.png")
    print("- figures/als_vs_svd_mae.png")
    print("- figures/als_vs_svd_tiempo.png")

    return results_df


def plot_comparison(results_df):
    methods = results_df["metodo"].unique()

    plt.figure(figsize=(8, 5))

    for method in methods:
        subset = results_df[results_df["metodo"] == method]
        plt.plot(subset["k"], subset["RMSE"], marker="o", label=method)

    plt.xlabel("Número de factores latentes k")
    plt.ylabel("RMSE")
    plt.title("Comparación de RMSE entre ALS y SVD")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/als_vs_svd_rmse.png", dpi=150)
    plt.show()

    plt.figure(figsize=(8, 5))

    for method in methods:
        subset = results_df[results_df["metodo"] == method]
        plt.plot(subset["k"], subset["MAE"], marker="o", label=method)

    plt.xlabel("Número de factores latentes k")
    plt.ylabel("MAE")
    plt.title("Comparación de MAE entre ALS y SVD")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/als_vs_svd_mae.png", dpi=150)
    plt.show()

    plt.figure(figsize=(8, 5))

    for method in methods:
        subset = results_df[results_df["metodo"] == method]
        plt.plot(subset["k"], subset["tiempo"], marker="o", label=method)

    plt.xlabel("Número de factores latentes k")
    plt.ylabel("Tiempo de entrenamiento (s)")
    plt.title("Comparación de tiempo entre ALS y SVD")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/als_vs_svd_tiempo.png", dpi=150)
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

    run_comparison(data_path=args.data)