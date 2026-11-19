"""The grouped chi-square test every task shares, and the claim it backs.

Rows are sorted by prediction, cut into `GROUPS` groups of equal size, and a
group too small for the normal approximation is merged into its neighbour.
With `O_g` the observed and `E_g` the predicted value of group `g` and `V_g`
the variance of `O_g` when the predictions are right,
`sum (O_g - E_g)^2 / V_g` is chi-square with one degree of freedom per group
(Hosmer and Lemeshow's statistic for predictions not fitted on these rows, so
no degree is lost to estimation).
"""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
import math
from typing import Any

import numpy
from scipy import stats

from scmle.contracts import errors
from scmle.contracts import tasks

LEVEL = 0.05
"""Family-wise size of every claim: the conventional 5% two-sided level,
split evenly across a claim's tests (Bonferroni), so the chance that any test
of a calibrated model rejects stays at most 5% whatever their dependence,
as far as each chi-square test holds its nominal size. The tests are large-
sample approximations, so the size is measured: over 1,000 seeded calibrated
simulations per task (the seeded size test of the calibration unit
tests) the rejection rate was 5.1% ordinal, 4.8% multilabel, 4.9%
survival, 2.3% ranking and 6.1% uplift (5.35% over 8,000), all within the
binomial Monte Carlo margin of 5%."""

GROUPS = 10
"""Risk groups per test: deciles, Hosmer and Lemeshow's choice, which keeps
about 10% of the rows per group so each group's rate has a usable variance."""

EXPECTED = 5.0
"""Least expected count in each cell of a group: Cochran's rule for the
chi-square approximation of a count (Cochran 1954); smaller groups merge."""

DRAWS = 2000
"""Bootstrap resamples per interval: the Monte Carlo error of a 2.5% tail is
`sqrt(0.025 * 0.975 / 2000)`, about 0.35 percentage points of coverage."""

CONFIDENCE = 0.95
"""Coverage of every interval, matching `LEVEL` for one two-sided test."""

STATEMENTS: Mapping[tasks.Task, str] = {
    tasks.Task.ORDINAL: (
        "For every level k, the predicted probability that a row's level is "
        "k or below matches the share of held-out rows at or below k. It "
        "does not say the most likely level is the true one."
    ),
    tasks.Task.MULTILABEL: (
        "For every label, the predicted probability matches the share of "
        "held-out rows carrying the label. It says nothing about the joint "
        "probability of a set of labels."
    ),
    tasks.Task.SURVIVAL: (
        "At each horizon, the predicted probability of being event-free "
        "matches the Kaplan-Meier estimate within groups of similar "
        "predicted risk, under censoring independent of the event given "
        "the group. It says nothing beyond the last horizon."
    ),
    tasks.Task.RANKING: (
        "Within a query, higher scores mark more relevant rows: the mean "
        "NDCG beats a random order of the same rows (gain interval above "
        "zero). An increasing logistic map of the score, learnt on other "
        "queries, gives a probability of relevance that matches the "
        "observed share. Scores are not comparable across queries before "
        "that map."
    ),
    tasks.Task.UPLIFT: (
        "Treating rows in the order of predicted uplift gains more "
        "incremental outcome than treating them at random (Qini interval "
        "above zero), and the predicted effect matches the treated-minus-"
        "control difference within groups. An individual row's effect is "
        "never observed, so no claim is made about one row; the estimates "
        "assume treatment was assigned as in a randomized experiment."
    ),
}
"""The claim each structured task supports, as the reader is told it."""


@dataclasses.dataclass(frozen=True)
class Group:
    """One group of rows with similar predictions.

    Attributes:
        count: Rows in the group.
        predicted: Mean prediction of the rows.
        observed: Observed rate (or effect) of the rows.
        variance: Variance of `observed` if the predictions are right.
    """

    count: int
    predicted: float
    observed: float
    variance: float


@dataclasses.dataclass(frozen=True)
class Check:
    """A grouped chi-square calibration test of one prediction.

    Attributes:
        subject: What was tested, for example `level <= 1`.
        groups: The groups the statistic is summed over.
        tested: Whether any group could be formed.
        statistic: `sum (observed - predicted)^2 / variance`; NaN untested.
        freedom: Degrees of freedom, one per group.
        pvalue: Upper chi-square tail of the statistic; NaN untested.
        error: Row-weighted mean absolute gap (the expected calibration
            error); NaN untested.
    """

    subject: str
    groups: tuple[Group, ...]
    tested: bool = dataclasses.field(init=False)
    statistic: float = dataclasses.field(init=False)
    freedom: int = dataclasses.field(init=False)
    pvalue: float = dataclasses.field(init=False)
    error: float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Derives the statistic from the groups."""
        rows = sum(group.count for group in self.groups)
        terms = [deviation(group) for group in self.groups]
        statistic = float(sum(terms)) if terms else math.nan
        freedom = len(self.groups)
        found = {
            "tested": bool(self.groups),
            "statistic": statistic,
            "freedom": freedom,
            "pvalue": float(stats.chi2.sf(statistic, freedom))
            if self.groups
            else math.nan,
            "error": sum(
                group.count * abs(group.observed - group.predicted)
                for group in self.groups
            )
            / rows
            if rows
            else math.nan,
        }
        for name, value in found.items():
            object.__setattr__(self, name, value)


@dataclasses.dataclass(frozen=True)
class Interval:
    """A bootstrap percentile interval of a score.

    The ranking claim fills one for the NDCG gain over a random order, the
    uplift claim for the Qini coefficient and the area under the uplift
    curve.

    Attributes:
        name: The score, for example `qini`.
        estimate: The score on all rows.
        lower: Lower end at `CONFIDENCE`.
        upper: Upper end at `CONFIDENCE`.
        null: The value a useless model scores, which the interval must lie
            above for the claim to hold; None when the score only describes.
    """

    name: str
    estimate: float
    lower: float
    upper: float
    null: float | None = None


@dataclasses.dataclass(frozen=True)
class Claim:
    """What a task's held-out predictions support, with the tests behind it.

    Attributes:
        task: The task.
        statement: The claim, in words.
        checks: The calibration tests.
        intervals: The intervals of the task's scores.
        level: Family-wise size of the tests.
        threshold: Size of each test: `level` split over the checks.
        supported: Whether every check ran and none rejects at
            `threshold`, and every interval with a null lies above it.
    """

    task: tasks.Task
    statement: str
    checks: tuple[Check, ...]
    intervals: tuple[Interval, ...] = ()
    level: float = LEVEL
    threshold: float = dataclasses.field(init=False)
    supported: bool = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Derives the per-test size and the verdict."""
        threshold = self.level / max(len(self.checks), 1)
        calibrated = all(
            check.tested and check.pvalue >= threshold for check in self.checks
        )
        useful = all(
            interval.null is None or interval.lower > interval.null
            for interval in self.intervals
        )
        evidence = bool(self.checks or self.intervals)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(
            self, "supported", calibrated and useful and evidence
        )


def deviation(group: Group) -> float:
    """One group's term of the chi-square statistic.

    Args:
        group: The group.

    Returns:
        `(observed - predicted)^2 / variance`; zero when both agree exactly
        and the variance is zero, infinite when only the variance is zero.
    """
    gap = (group.observed - group.predicted) ** 2
    if group.variance > 0:
        return gap / group.variance
    return 0.0 if gap == 0 else math.inf


def partition(
    order: numpy.ndarray,
    enough: Callable[[numpy.ndarray], bool],
    groups: int = GROUPS,
) -> list[numpy.ndarray]:
    """Cuts sorted rows into groups, merging any group that is too small.

    Args:
        order: Row positions sorted by prediction.
        enough: Whether a set of row positions is large enough to test.
        groups: Number of equal-size groups cut before merging.

    Returns:
        The groups in prediction order; a short group joins the next one and
        a short tail joins the last group. Empty when all rows together are
        not enough.
    """
    found: list[numpy.ndarray] = []
    pending = numpy.empty(0, dtype=int)
    for chunk in numpy.array_split(order, groups):
        pending = numpy.concatenate([pending, chunk])
        if len(pending) and enough(pending):
            found.append(pending)
            pending = numpy.empty(0, dtype=int)
    if len(pending) and found:
        found[-1] = numpy.concatenate([found[-1], pending])
    return found


def vector(values: Any, name: str, rows: int | None = None) -> numpy.ndarray:
    """A finite one-dimensional float array.

    Args:
        values: The values.
        name: Their name, for the error message.
        rows: The length they must have, if fixed.

    Returns:
        The values as floats.

    Raises:
        InputError: If they are not one-dimensional, not finite or of the
            wrong length.
    """
    array = numpy.asarray(values, dtype=float)
    if array.ndim != 1:
        raise errors.InputError(f"{name} must be one-dimensional")
    if not numpy.isfinite(array).all():
        raise errors.InputError(f"{name} must be finite")
    if rows is not None and len(array) != rows:
        raise errors.InputError(f"{name} has {len(array)} rows, not {rows}")
    return array


def probable(values: numpy.ndarray, name: str) -> numpy.ndarray:
    """Checks that values are probabilities.

    Args:
        values: The values.
        name: Their name, for the error message.

    Returns:
        The values, unchanged.

    Raises:
        InputError: If a value is not finite or lies outside `[0, 1]`.
    """
    if not numpy.isfinite(values).all() or (
        (values < 0).any() or (values > 1).any()
    ):
        raise errors.InputError(f"{name} must lie between 0 and 1")
    return values


def bernoulli(
    subject: str,
    probability: Any,
    outcome: Any,
    expected: float = EXPECTED,
    groups: int = GROUPS,
) -> Check:
    """Tests predicted probabilities of a 0/1 outcome.

    Each group's variance is `sum p_i (1 - p_i) / n^2`, the exact variance
    of the observed share when every row is a Bernoulli trial with its own
    predicted probability.

    Args:
        subject: What the probability is of.
        probability: Predicted probability of each row.
        outcome: 1 where the outcome happened, else 0.
        expected: Least expected count of positives and of negatives in a
            group.
        groups: Groups cut before merging.

    Returns:
        The test.

    Raises:
        InputError: If the inputs differ in length or a probability is
            outside `[0, 1]`.
    """
    chance = probable(vector(probability, "probabilities"), "probabilities")
    happened = vector(outcome, "outcomes", len(chance))

    def enough(members: numpy.ndarray) -> bool:
        share = chance[members]
        return bool(min(share.sum(), (1 - share).sum()) >= expected)

    order = numpy.argsort(chance, kind="stable")
    found = tuple(
        Group(
            count=len(members),
            predicted=float(chance[members].mean()),
            observed=float(happened[members].mean()),
            variance=float(
                (chance[members] * (1 - chance[members])).sum()
                / len(members) ** 2
            ),
        )
        for members in partition(order, enough, groups)
    )
    return Check(subject, found)
