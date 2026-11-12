"""Permutation test of covariate balance across treatment arms.

The check that the arms of a supposedly randomized experiment look alike on
the covariates. The statistic is the sum over numeric covariates of the
squared standardized difference in means between the arms. Its null
distribution is built by permuting the arm labels, which keeps the arm sizes,
so the test is exact for assignment that is randomized (given the sizes, every
labelling is equally likely) and its false-alarm rate is the level by
construction; `tests/unit/test_balance.py` checks that by simulation, and also
how often assignment that follows a covariate is flagged. It replaces the
cross-validated AUC cutoff of `uplift.check_propensity` as the judge of
"looks randomized", an uncertified rule, though the AUC stays in that
report as a description.

What it does not do: it sees only numeric covariates, only differences in
means (not interactions), and a pass says no imbalance was found, not that
assignment was randomized. Balance on observed covariates says nothing about
unobserved ones.
"""

import itertools
import operator
from typing import Any

import numpy
from numpy.typing import ArrayLike
import pandas

from scmle.core import randomness

LEVEL = 0.05
PERMUTATIONS = 499
COMPONENT = "evaluation.balance"
NOTE = (
    "numeric covariates, differences in means only; a pass is the absence of "
    "a found imbalance, not evidence that assignment was random"
)


def balance_test(
    features: pandas.DataFrame,
    treatment: ArrayLike,
    seed: int = 0,
    permutations: int = PERMUTATIONS,
    level: float = LEVEL,
) -> dict[str, Any]:
    """Permutation test that the covariates are balanced across the arms.

    Args:
        features: Covariates, one row per unit.
        treatment: 0 or 1 per unit.
        seed: Seed of the permutations.
        permutations: Number of label permutations.
        level: Size of the test; `balanced` is `p_value > level`.

    Returns:
        A mapping with `statistic`, the permutation p-value `p_value` =
        `(1 + #extreme) / (B + 1)`, `balanced`, `level`, `permutations`,
        `covariates` (the number of numeric covariates with non-zero spread
        that entered the statistic) and `note` on what was and was not
        looked at. `statistic`, `p_value` and `balanced` are None when no
        such covariate remains or one arm is empty. Non-finite covariate
        values are replaced by the column median.

    Raises:
        ValueError: On unequal lengths, no permutations, or a level outside
            (0, 1).
    """
    arm = numpy.asarray(treatment).astype(int)
    if len(features) != len(arm):
        raise ValueError("features and treatment must have the same length")
    if permutations < 1:
        raise ValueError("permutations must be positive")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    numeric = features.select_dtypes("number")
    matrix = numeric.to_numpy(dtype=float)
    if matrix.size:
        medians = numpy.nanmedian(
            numpy.where(numpy.isfinite(matrix), matrix, numpy.nan), 0
        )
        medians = numpy.where(numpy.isfinite(medians), medians, 0.0)
        matrix = numpy.where(numpy.isfinite(matrix), matrix, medians)
    spread = matrix.std(axis=0) if matrix.size else numpy.zeros(0)
    keep = spread > 0.0
    base = {
        "level": level,
        "permutations": permutations,
        "covariates": int(keep.sum()),
        "note": NOTE,
    }
    treated = int(arm.sum())
    if not keep.any() or treated in (0, len(arm)):
        return {
            **base,
            "statistic": None,
            "p_value": None,
            "balanced": None,
        }
    standard = (matrix[:, keep] - matrix[:, keep].mean(0)) / spread[keep]
    rng = randomness.component_rng(seed, COMPONENT, 0)

    def statistic(labels: numpy.ndarray) -> numpy.ndarray:
        ones = labels.sum(axis=-1, keepdims=True)
        zeros = labels.shape[-1] - ones
        up = labels @ standard / ones
        down = (1.0 - labels) @ standard / zeros
        return numpy.asarray(((up - down) ** 2).sum(axis=-1))

    observed = float(statistic(arm.astype(float)[None, :])[0])
    draws = numpy.stack(
        list(
            map(
                operator.call,
                itertools.repeat(lambda: rng.permutation(arm), permutations),
            )
        )
    )
    null = statistic(draws.astype(float))
    p_value = float((1 + numpy.sum(null >= observed)) / (permutations + 1))
    return {
        **base,
        "statistic": observed,
        "p_value": p_value,
        "balanced": bool(p_value > level),
    }
