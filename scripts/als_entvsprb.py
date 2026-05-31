#!/usr/bin/env python3

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.load_data import (
    load_movielens_100k,
    train_test_split_ratings,
    build_matrix_from_df,
)

from src.als_recommender import ALSRecommender


def plot_als_convergence(data_path="data/ml-100k"):

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

    k = 5
    lambda_reg = 10
    iterations = 30

    print(f"Entrenando ALS con k={k}, lambda={lambda_reg}, iteraciones={iterations}")

    model = ALSRecommender(
        n_factors=k,
        lambda_reg=lambda_reg,
        n_iterations=iterations,
        verbose=True
    )

    model.fit(R_train, R_test)

    convergence_df = pd.DataFrame({
        "iteration": range(1, iterations + 1),
        "train_rmse": model.train_errors,
        "test_rmse": model.test_errors
    })

    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    convergence_df.to_csv("data/als_convergence.csv", index=False)

    plt.figure(figsize=(9, 6))

    plt.plot(
        convergence_df["iteration"],
        convergence_df["train_rmse"],
        marker="o",
        linewidth=2,
        label="Entrenamiento"
    )

    plt.plot(
        convergence_df["iteration"],
        convergence_df["test_rmse"],
        marker="o",
        linewidth=2,
        label="Prueba"
    )

    plt.xlabel("Iteración")
    plt.ylabel("RMSE")
    plt.title("Curvas de convergencia de ALS")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig("figures/als_convergence_curves.png", dpi=150)
    plt.show()

    print("\nResultados guardados en:")
    print("- data/als_convergence.csv")
    print("- figures/als_convergence_curves.png")

    return convergence_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default="data/ml-100k",
        help="Ruta a MovieLens 100K"
    )

    args = parser.parse_args()

    plot_als_convergence(data_path=args.data)