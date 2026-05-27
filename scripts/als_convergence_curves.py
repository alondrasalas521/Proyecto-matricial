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


def plot_als_convergence_multiple_k(data_path="data/ml-100k"):

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

    # ==========================================================
    # Parámetros
    # ==========================================================

    k_values = [3, 6, 9, 12, 15]

    lambda_reg = 0.1
    iterations = 30

    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    all_results = []

    # ==========================================================
    # Curvas de entrenamiento
    # ==========================================================

    plt.figure(figsize=(10, 6))

    for k in k_values:

        print("=" * 60)
        print(f"Entrenando ALS con k={k}")
        print("=" * 60)

        model = ALSRecommender(
            n_factors=k,
            lambda_reg=lambda_reg,
            n_iterations=iterations,
            verbose=True
        )

        model.fit(R_train, R_test)

        iterations_axis = range(1, iterations + 1)

        # Guardar resultados
        for i in range(iterations):

            all_results.append({
                "k": k,
                "iteration": i + 1,
                "train_rmse": model.train_errors[i],
                "test_rmse": model.test_errors[i]
            })

        # Curvas de prueba
        plt.plot(
            iterations_axis,
            model.test_errors,
            marker="o",
            linewidth=2,
            label=f"k={k}"
        )

    plt.xlabel("Iteración")
    plt.ylabel("RMSE")
    plt.title("Convergencia ALS para distintos valores de k")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        "figures/als_convergence_multiple_k.png",
        dpi=150
    )

    plt.show()

    # ==========================================================
    # Guardar CSV
    # ==========================================================

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        "data/als_convergence_multiple_k.csv",
        index=False
    )

    print("\nResultados guardados en:")
    print("- data/als_convergence_multiple_k.csv")
    print("- figures/als_convergence_multiple_k.png")

    return results_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default="data/ml-100k",
        help="Ruta a MovieLens 100K"
    )

    args = parser.parse_args()

    plot_als_convergence_multiple_k(data_path=args.data)