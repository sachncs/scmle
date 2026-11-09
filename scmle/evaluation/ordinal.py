"""Metrics for ordered categories, computed on integer level ranks."""

import numpy as np
from sklearn import metrics as sk_metrics


def quadratic_weighted_kappa(truth: np.ndarray, guess: np.ndarray) -> float:
    """Agreement beyond chance, with larger misses punished quadratically.

    Args:
        truth: True level ranks.
        guess: Predicted level ranks.

    Returns:
        The quadratic-weighted Cohen's kappa; 1 is perfect agreement.
    """
    return float(
        sk_metrics.cohen_kappa_score(
            np.asarray(truth).astype(int),
            np.asarray(guess).astype(int),
            weights="quadratic",
        )
    )


def mean_absolute_class_error(truth: np.ndarray, guess: np.ndarray) -> float:
    """Average number of levels the guess is away from the truth.

    Args:
        truth: True level ranks.
        guess: Predicted level ranks.

    Returns:
        The mean absolute difference between the ranks.
    """
    return float(
        np.abs(
            np.asarray(truth, dtype=float) - np.asarray(guess, dtype=float)
        ).mean()
    )
