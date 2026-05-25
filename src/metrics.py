#!/usr/bin/env python3
"""
Módulo de métricas para evaluación de sistemas de recomendación.
"""

import numpy as np


def rmse(predictions, true_values):
    """Root Mean Square Error."""
    return np.sqrt(np.mean((predictions - true_values) ** 2))


def mae(predictions, true_values):
    """Mean Absolute Error."""
    return np.mean(np.abs(predictions - true_values))


def precision_at_k(recommended, relevant, k):
    """Precision@k."""
    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    if len(recommended_k) == 0:
        return 0.0

    hits = sum(1 for item in recommended_k if item in relevant_set)
    return hits / len(recommended_k)


def recall_at_k(recommended, relevant, k):
    """Recall@k."""
    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    if len(relevant_set) == 0:
        return 0.0

    hits = sum(1 for item in recommended_k if item in relevant_set)
    return hits / len(relevant_set)


def ndcg_at_k(recommended, relevant, k):
    """Normalized Discounted Cumulative Gain @ k."""
    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 2)

    ideal_dcg = sum(
        1.0 / np.log2(i + 2)
        for i in range(min(k, len(relevant)))
    )

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
