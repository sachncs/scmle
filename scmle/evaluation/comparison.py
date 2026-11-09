"""Paired comparison of two candidates scored on the same folds.

Comparing mean scores ignores how noisy the difference is. The correlated
Bayesian t-test (Corani and Benavoli) turns the per-fold score differences
into the probability that one candidate is really better, accounting for the
correlation between folds that share training data. A region of practical
equivalence (ROPE) separates "better" from "no meaningful difference".
"""

from collections.abc import Sequence
import dataclasses
import math

import numpy as np
from scipy import stats

MIN_FOLDS = 3


@dataclasses.dataclass(frozen=True)
class Comparison:
    """Posterior belief about which of two candidates is better.

    Attributes:
        mean_difference: Mean of (first - second) over the folds.
        probability_first_better: Posterior probability that the true
            difference exceeds the ROPE in favour of the first candidate.
        probability_equivalent: Posterior probability that the true
            difference lies inside the ROPE.
        probability_second_better: Posterior probability that the true
            difference exceeds the ROPE in favour of the second candidate.
        folds: Number of paired fold scores used.
    """

    mean_difference: float
    probability_first_better: float
    probability_equivalent: float
    probability_second_better: float
    folds: int

    def decisive(self, threshold: float) -> bool:
        """Whether one side is better with at least `threshold` probability.

        Args:
            threshold: Probability a side must reach.

        Returns:
            True if the larger of the two "better" probabilities is at least
            `threshold`.
        """
        return (
            max(self.probability_first_better, self.probability_second_better)
            >= threshold
        )


def compare_paired(
    first: Sequence[float],
    second: Sequence[float],
    rope: float = 0.0,
    correlation: float | None = None,
) -> Comparison:
    """Compares two candidates scored on identical folds.

    Scores must be oriented so that larger is better.

    Args:
        first: Fold scores of the first candidate.
        second: Fold scores of the second candidate, same folds and order.
        rope: Half-width of the region of practical equivalence.
        correlation: Correlation between fold differences. Defaults to
            `1 / folds`, the share of rows in each validation fold, which is
            the usual value for k-fold cross-validation.

    Returns:
        The comparison: the mean difference, the posterior probabilities
        that the first is better, that they are equivalent and that the
        second is better, and the number of folds. When every fold
        difference is equal, the probabilities are 0 or 1 according to
        whether the mean difference lies inside the ROPE or on which side
        it lies outside.

    Raises:
        ValueError: If the score lists differ in length or are too short.
    """
    if len(first) != len(second):
        raise ValueError("both candidates need scores on the same folds")
    if len(first) < MIN_FOLDS:
        raise ValueError(f"need at least {MIN_FOLDS} folds to compare")
    differences = np.asarray(first, dtype=float) - np.asarray(
        second, dtype=float
    )
    folds = len(differences)
    mean = float(differences.mean())
    variance = float(differences.var(ddof=1))
    rho = correlation if correlation is not None else 1.0 / folds
    scale = math.sqrt((1.0 / folds + rho / (1.0 - rho)) * variance)
    if scale == 0.0:
        inside = abs(mean) <= rope
        return Comparison(
            mean,
            0.0 if inside or mean < 0 else 1.0,
            1.0 if inside else 0.0,
            0.0 if inside or mean > 0 else 1.0,
            folds,
        )
    posterior = stats.t(folds - 1, loc=mean, scale=scale)
    left = float(posterior.cdf(-rope))
    right = float(posterior.sf(rope))
    return Comparison(mean, right, 1.0 - left - right, left, folds)
