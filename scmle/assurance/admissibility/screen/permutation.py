"""Shared permutations of the outcome and the early stop of each hypothesis."""

from collections.abc import Sequence
from fractions import Fraction
import math

import numpy

from scmle.assurance.admissibility.screen.fitting import permute
from scmle.assurance.admissibility.screen.hypotheses import Pending
from scmle.assurance.admissibility.screen.settings import BATCH
from scmle.assurance.admissibility.screen.settings import TOLERANCE
from scmle.assurance.ledger import multiplicity
from scmle.runtime import randomness


def resamples(family: int, alpha: Fraction) -> int:
    """Finds the smallest `B` with `1 / (B + 1) < alpha / (family * H_family)`.

    Args:
        family: Number of hypotheses adjusted together.
        alpha: Level of the Benjamini-Yekutieli step.

    Returns:
        The number of permutations `B`; 0 when `family` is not positive.
    """
    if family <= 0:
        return 0
    need = family * multiplicity.harmonics(family) / alpha
    # floor, not ceil - 1: when `need` is a whole number (7 hypotheses at
    # 0.05 need 363) ceil - 1 puts 1 / (B + 1) exactly on the threshold, and
    # the float p-value the step reads rounds to just above it. One more
    # permutation makes the inequality strict by 1 / (need * (need + 1)),
    # far above any float rounding.
    return max(0, math.floor(need))


def draw(
    pending: Sequence[Pending],
    labels: numpy.ndarray,
    binned: numpy.ndarray,
    strata: numpy.ndarray,
    total: int,
    alpha: Fraction,
    seed: int,
) -> None:
    """Draws `total` shared permutations and updates every hypothesis in place.

    A hypothesis stops once its final p-value is certain to exceed the
    largest threshold of the step-up rule. The `exceed`, `drawn` and
    `active` fields of each hypothesis in `pending` with a statistic are
    updated; one with an analytic p-value is left unchanged.

    Args:
        pending: The hypotheses of the family.
        labels: Class codes (or values) of the outcome, per row.
        binned: Binned outcome codes, per row.
        strata: Permutation stratum per row.
        total: Number of permutations to draw.
        alpha: Level of the Benjamini-Yekutieli step.
        seed: Run seed of the permutation streams.
    """
    family = len(pending)
    ceiling = alpha / multiplicity.harmonics(family)
    live = [h for h in pending if h.statistic is not None]
    done = 0
    batch = 0
    while done < total and any(h.active for h in live):
        count = min(BATCH, total - done)
        rng = randomness.component_rng(seed, "leakage.screen.batch", batch)
        index = permute(strata, count, rng)
        permuted = labels[index]
        permuted_bins = binned[index]
        for h in live:
            if not h.active or h.statistic is None:
                continue
            values = h.statistic(permuted_bins if h.binned else permuted)
            h.exceed += int((values >= h.observed - TOLERANCE).sum())
            h.drawn += count
            if Fraction(h.exceed + 1, total + 1) > ceiling:
                h.active = False
        done += count
        batch += 1
