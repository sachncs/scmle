"""Metrics for ordered categories, computed on integer level ranks."""

import numpy
from sklearn import metrics


def quadratic_weighted_kappa(
    truth: numpy.ndarray, guess: numpy.ndarray
) -> float:
    """Agreement beyond chance, with larger misses punished quadratically.

    Args:
        truth: True level ranks.
        guess: Predicted level ranks.

    Returns:
        The quadratic-weighted Cohen's kappa; 1 is perfect agreement.
    """
    return float(
        metrics.cohen_kappa_score(
            numpy.asarray(truth).astype(int),
            numpy.asarray(guess).astype(int),
            weights="quadratic",
        )
    )


def mean_absolute_class_error(
    truth: numpy.ndarray, guess: numpy.ndarray
) -> float:
    """Average number of levels the guess is away from the truth.

    Args:
        truth: True level ranks.
        guess: Predicted level ranks.

    Returns:
        The mean absolute difference between the ranks.
    """
    return float(
        numpy.abs(
            numpy.asarray(truth, dtype=float)
            - numpy.asarray(guess, dtype=float)
        ).mean()
    )
