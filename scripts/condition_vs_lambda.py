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


def condition_vs_lambda(data_path="data/ml-100k"):

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

    k = 20
    lambda_values = [0.001, 0.01, 0.1, 1, 10, 100]

    results = []

    for lambda_reg in lambda_values:

        print("=" * 60)
        print(f"Entrenando ALS con lambda={lambda_reg}")
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

        cond_numbers = model.get_condition_numbers(R_train)

        cond_mean = np.mean(cond_numbers)
        cond_max = np.max(cond_numbers)
        cond_min = np.min(cond_numbers)

        results.append({
            "lambda": lambda_reg,
            "condition_mean": cond_mean,
            "condition_max": cond_max,
            "condition_min": cond_min,
            "time": elapsed_time
        })

        print(f"Condición promedio: {cond_mean:.4e}")
        print(f"Condición máxima: {cond_max:.4e}")
        print(f"Condición mínima: {cond_min:.4e}")

    results_df = pd.DataFrame(results)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    results_df.to_csv("data/condition_vs_lambda.csv", index=False)

    plt.figure(figsize=(9, 6))

    plt.plot(
        results_df["lambda"],
        results_df["condition_mean"],
        marker="o",
        label="Promedio"
    )

    plt.plot(
        results_df["lambda"],
        results_df["condition_max"],
        marker="o",
        label="Máximo"
    )

    plt.plot(
        results_df["lambda"],
        results_df["condition_min"],
        marker="o",
        label="Mínimo"
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("lambda")
    plt.ylabel("Número de condición")
    plt.title("Número de condición vs lambda en ALS")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig("figures/condition_vs_lambda.png", dpi=150)
    plt.show()

    print("\nResultados guardados en:")
    print("- data/condition_vs_lambda.csv")
    print("- figures/condition_vs_lambda.png")

    return results_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default="data/ml-100k",
        help="Ruta a la carpeta MovieLens 100K"
    )

    args = parser.parse_args()

    condition_vs_lambda(data_path=args.data)