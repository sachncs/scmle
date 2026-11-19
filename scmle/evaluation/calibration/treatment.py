"""Calibration of uplift models: effect sizes and the value of the ranking.

An uplift model's effect for one row is never observed, so it is judged on
groups. Rows sorted by predicted effect are cut into groups with enough
rows in both arms, and each group's treated-minus-control difference is
compared with its mean predicted effect by the grouped chi-square test of
`verdict`. Beside it, the Qini coefficient and the area under the uplift
curve get bootstrap intervals; the claim holds only when the Qini interval
lies above zero, the value of a random ranking.
"""

from typing import Any

import numpy

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.evaluation.calibration import verdict
import scmle.evaluation.uplift

ARM = 30
"""Least rows per arm in an uplift group: the usual sample size at which a
mean is close enough to normal for a two-sample z-statistic."""


def area(outcome: Any, treatment: Any, score: Any) -> float:
    """Area under the uplift curve, averaged over the rows walked.

    Walking the rows from the highest predicted uplift, the curve at `k`
    rows is the treated-minus-control mean outcome among them times `k / n`
    (zero until both arms appear).

    Args:
        outcome: Observed outcome of each row.
        treatment: 1 where the row was treated.
        score: Predicted uplift of each row.

    Returns:
        The mean height of the curve.
    """
    order = numpy.argsort(-numpy.asarray(score, dtype=float), kind="stable")
    y = numpy.asarray(outcome, dtype=float)[order]
    t = numpy.asarray(treatment, dtype=float)[order]
    treated, control = numpy.cumsum(t), numpy.cumsum(1 - t)
    both = (treated > 0) & (control > 0)
    gap = numpy.zeros(len(y))
    gap[both] = (
        numpy.cumsum(y * t)[both] / treated[both]
        - numpy.cumsum(y * (1 - t))[both] / control[both]
    )
    walked = numpy.arange(1, len(y) + 1) / len(y)
    return float(numpy.mean(gap * walked))


def spread(values: numpy.ndarray) -> float:
    """Variance of one arm's mean outcome.

    A 0/1 outcome uses the share shrunk by half a success and half a
    failure (Agresti and Coull), so an arm with no positives still has a
    variance; any other outcome uses the sample variance.

    Args:
        values: Outcomes of the arm's rows.

    Returns:
        The variance of their mean.
    """
    if numpy.isin(values, (0.0, 1.0)).all():
        share = (values.sum() + 0.5) / (len(values) + 1)
        return float(share * (1 - share) / len(values))
    return float(values.var(ddof=1) / len(values))


def effects(
    outcome: numpy.ndarray, arm: numpy.ndarray, score: numpy.ndarray
) -> verdict.Check:
    """Tests predicted effects against observed differences in groups.

    Args:
        outcome: Observed outcome of each row.
        arm: True where the row was treated.
        score: Predicted effect of each row.

    Returns:
        The test, groups needing `ARM` rows in each arm.
    """

    def enough(members: numpy.ndarray) -> bool:
        treated = int(arm[members].sum())
        return min(treated, len(members) - treated) >= ARM

    found = []
    for members in verdict.partition(
        numpy.argsort(score, kind="stable"), enough
    ):
        treated = outcome[members][arm[members]]
        control = outcome[members][~arm[members]]
        found.append(
            verdict.Group(
                count=len(members),
                predicted=float(score[members].mean()),
                observed=float(treated.mean() - control.mean()),
                variance=spread(treated) + spread(control),
            )
        )
    return verdict.Check("effect", tuple(found))


def bootstrap(
    outcome: numpy.ndarray,
    arm: numpy.ndarray,
    score: numpy.ndarray,
    draws: int,
    seed: int,
) -> tuple[verdict.Interval, verdict.Interval]:
    """Percentile intervals of the Qini coefficient and the uplift area.

    Rows are resampled within each arm, keeping the arms' sizes, and kept
    in their original order, so rows with tied scores are walked in the
    same order as in the full-sample estimate (walking one arm's ties first
    would bias every resample).

    Args:
        outcome: Observed outcome of each row.
        arm: True where the row was treated.
        score: Predicted effect of each row.
        draws: Resamples.
        seed: Seed of the resampling.

    Returns:
        The Qini interval (null zero, a random ranking's value) and the area
        interval (descriptive).
    """
    rng = numpy.random.default_rng(seed)
    treated, control = numpy.flatnonzero(arm), numpy.flatnonzero(~arm)
    flag = arm.astype(int)
    values = numpy.empty((draws, 2))
    for draw in range(draws):
        rows = numpy.sort(
            numpy.concatenate(
                [
                    rng.choice(treated, len(treated)),
                    rng.choice(control, len(control)),
                ]
            )
        )
        values[draw] = (
            scmle.evaluation.uplift.qini_coefficient(
                outcome[rows], flag[rows], score[rows]
            ),
            area(outcome[rows], flag[rows], score[rows]),
        )
    tail = (1 - verdict.CONFIDENCE) / 2
    lower, upper = numpy.quantile(values, (tail, 1 - tail), axis=0)
    qini = scmle.evaluation.uplift.qini_coefficient(outcome, flag, score)
    return (
        verdict.Interval("qini", qini, float(lower[0]), float(upper[0]), 0.0),
        verdict.Interval(
            "auuc", area(outcome, flag, score), float(lower[1]), float(upper[1])
        ),
    )


def uplift(
    outcome: Any,
    treatment: Any,
    score: Any,
    draws: int = verdict.DRAWS,
    seed: int = 0,
    level: float = verdict.LEVEL,
) -> verdict.Claim:
    """Tests an uplift model's ranking and its effect sizes.

    Args:
        outcome: Observed outcome of each row.
        treatment: 1 where the row was treated, 0 for control.
        score: Predicted effect of each row.
        draws: Bootstrap resamples per interval.
        seed: Seed of the bootstrap.
        level: Family-wise size of the tests.

    Returns:
        The uplift claim: the effect check and the Qini and area intervals.

    Raises:
        InputError: If an arm is empty or the inputs differ in length.
    """
    values = verdict.vector(outcome, "outcomes")
    flag = verdict.vector(treatment, "treatment", len(values))
    effect = verdict.vector(score, "scores", len(values))
    if not numpy.isin(flag, (0.0, 1.0)).all():
        raise errors.InputError("treatment must be 0 or 1")
    arm = flag == 1
    if arm.all() or not arm.any():
        raise errors.InputError("uplift needs rows in both arms")
    if draws < 1:
        raise errors.InputError("draws must be positive")
    return verdict.Claim(
        tasks.Task.UPLIFT,
        verdict.STATEMENTS[tasks.Task.UPLIFT],
        (effects(values, arm, effect),),
        bootstrap(values, arm, effect, draws, seed),
        level,
    )
