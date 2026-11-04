"""Density-ratio weights for weighted conformal prediction.

A probabilistic classifier separates calibration covariates from unlabeled
target covariates; its odds, corrected for the sample sizes, estimate the
likelihood ratio `p_target(x) / p_calibration(x)`. Weights are clipped, and
the effective sample size `(sum w)^2 / sum w^2` is reported. Below
`MIN_EFFECTIVE_SIZE` the weights are refused: a handful of rows would carry
the interval. With an estimated ratio the coverage guarantee is approximate,
and it is reported that way (A4 of the F12 feature file).
"""

import dataclasses
from typing import Any

import numpy as np
from sklearn import linear_model

MIN_EFFECTIVE_SIZE = 50.0
DEFAULT_CLIP = 20.0


class WeightRefusedError(ValueError):
    """The effective sample size is too small for weighted conformal."""


@dataclasses.dataclass(frozen=True)
class ShiftWeights:
    """Estimated likelihood ratios.

    Attributes:
        calibration: Weights of the calibration rows.
        target: Weights of the target rows.
        effective_size: Kish effective size of the calibration weights.
        clip: Upper bound applied to the odds.
        discriminator: The fitted classifier (target = 1).
        scale: Calibration rows over target rows, the odds correction.
    """

    calibration: np.ndarray
    target: np.ndarray
    effective_size: float
    clip: float
    discriminator: Any = None
    scale: float = 1.0

    def ratio(self, features: np.ndarray) -> np.ndarray:
        """Clipped likelihood ratio at new rows (for test-point weights)."""
        probability = self.discriminator.predict_proba(
            np.asarray(features, dtype=float)
        )[:, 1]
        odds = probability / np.maximum(1.0 - probability, 1e-12)
        return np.minimum(odds * self.scale, self.clip)


def effective_sample_size(weights: np.ndarray) -> float:
    """`(sum w)^2 / sum w^2`."""
    values = np.asarray(weights, dtype=float)
    squares = float((values**2).sum())
    return float(values.sum() ** 2 / squares) if squares > 0 else 0.0


def estimate_weights(
    calibration: np.ndarray,
    target: np.ndarray,
    clip: float = DEFAULT_CLIP,
    minimum: float = MIN_EFFECTIVE_SIZE,
) -> ShiftWeights:
    """Fits the discriminator and returns clipped, size-corrected odds.

    Raises:
        WeightRefusedError: When the effective sample size of the
            calibration weights is below `minimum`.
    """
    first = np.asarray(calibration, dtype=float)
    second = np.asarray(target, dtype=float)
    matrix = np.vstack([first, second])
    labels = np.r_[np.zeros(len(first)), np.ones(len(second))]
    model = linear_model.LogisticRegression(max_iter=1000).fit(matrix, labels)
    probability = model.predict_proba(matrix)[:, 1]
    odds = probability / np.maximum(1.0 - probability, 1e-12)
    ratio = np.minimum(odds * len(first) / len(second), clip)
    result = ShiftWeights(
        ratio[: len(first)],
        ratio[len(first) :],
        effective_sample_size(ratio[: len(first)]),
        clip,
        model,
        len(first) / len(second),
    )
    if result.effective_size < minimum:
        raise WeightRefusedError(
            f"effective sample size {result.effective_size:.1f} is below "
            f"{minimum:.0f}; weighted conformal is refused"
        )
    return result
