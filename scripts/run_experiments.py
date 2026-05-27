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
from src.metrics import rmse, mae


def run_experiments(data_path="ml-100k"):
    print("Cargando datos...")

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
    lambda_values = [0.001, 0.01, 0.1, 1.0, 10.0]

    results = []

    for k in k_values:
        for lambda_reg in lambda_values:

            print("=" * 60)
            print(f"Experimento: k={k}, lambda={lambda_reg}")
            print("=" * 60)

            model = ALSRecommender(
                n_factors=k,
                lambda_reg=lambda_reg,
                n_iterations=20,
                verbose=False
            )

            start = time.time()
            model.fit(R_train, R_test)
            elapsed_time = time.time() - start

            test_rmse = model.compute_rmse(R_test)

            rows, cols = R_test.nonzero()
            predictions = np.array([
                model.predict(u, i)
                for u, i in zip(rows, cols)
            ])

            test_mae = mae(predictions, R_test.data)

            results.append({
                "k": k,
                "lambda": lambda_reg,
                "RMSE": test_rmse,
                "MAE": test_mae,
                "time": elapsed_time
            })

            print(f"RMSE: {test_rmse:.4f}")
            print(f"MAE: {test_mae:.4f}")
            print(f"Tiempo: {elapsed_time:.2f} s")

    results_df = pd.DataFrame(results)

    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    results_df.to_csv("data/experiment_results.csv", index=False)

    plot_results(results_df)

    print("\nResultados finales:")
    print(results_df)

    print("\nArchivos guardados:")
    print("- data/experiment_results.csv")
    print("- figures/rmse_vs_k_lambda.png")
    print("- figures/mae_vs_k_lambda.png")

    return results_df


def plot_results(results_df):

    plt.figure(figsize=(9, 6))

    for lambda_reg in sorted(results_df["lambda"].unique()):
        subset = results_df[results_df["lambda"] == lambda_reg]

        plt.plot(
            subset["k"],
            subset["RMSE"],
            marker="o",
            label=f"lambda={lambda_reg}"
        )

    plt.xlabel("Número de factores k")
    plt.ylabel("RMSE")
    plt.title("RMSE para distintos valores de k y lambda")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/rmse_vs_k_lambda.png", dpi=150)
    plt.show()

    plt.figure(figsize=(9, 6))

    for lambda_reg in sorted(results_df["lambda"].unique()):
        subset = results_df[results_df["lambda"] == lambda_reg]

        plt.plot(
            subset["k"],
            subset["MAE"],
            marker="o",
            label=f"lambda={lambda_reg}"
        )

    plt.xlabel("Número de factores k")
    plt.ylabel("MAE")
    plt.title("MAE para distintos valores de k y lambda")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/mae_vs_k_lambda.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default="ml-100k",
        help="Ruta a la carpeta del dataset MovieLens 100K"
    )

    args = parser.parse_args()

    run_experiments(data_path=args.data)

