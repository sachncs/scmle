"""Reliability of predicted probabilities, for two classes or many."""

from collections.abc import Sequence
from typing import Any

import numpy as np

BINS = 10


def top_label_curve(
    probabilities: np.ndarray,
    labels: Sequence[Any] | np.ndarray,
    classes: Sequence[Any],
    bins: int = BINS,
) -> tuple[float, list[dict[str, float]]]:
    """Computes confidence against accuracy of the most likely class.

    Rows are binned by the probability of the predicted class; in a calibrated
    model the share predicted correctly in each bin matches the bin's mean
    confidence. The expected calibration error is the row-weighted gap.

    Args:
        probabilities: Predicted probabilities, one column per class.
        labels: True labels.
        classes: Class labels in column order.
        bins: Number of equal-width confidence bins.

    Returns:
        The expected calibration error and the curve. The curve has one
        mapping per bin that holds rows, in bin order, with keys `bin`
        (index), `count`, `mean_confidence` and `accuracy`.
    """
    matrix = np.asarray(probabilities, dtype=float)
    guess = np.asarray(classes)[matrix.argmax(axis=1)]
    confidence = matrix.max(axis=1)
    correct = (guess == np.asarray(labels)).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    slot = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    error, curve = 0.0, []
    for index in range(bins):
        members = slot == index
        if not members.any():
            continue
        gap = abs(float(confidence[members].mean() - correct[members].mean()))
        error += float(members.mean()) * gap
        curve.append(
            {
                "bin": index,
                "count": int(members.sum()),
                "mean_confidence": float(confidence[members].mean()),
                "accuracy": float(correct[members].mean()),
            }
        )
    return error, curve
