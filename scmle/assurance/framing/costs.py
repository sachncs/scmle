"""Decision-cost model and net benefit for a binary action.

Binary action `a` (act or not), outcome `y` in {0, 1}, cost `c_ay` (a
negative cost is a benefit). With `c_FP = c_10 - c_00` and
`c_FN = c_01 - c_11`, both required positive, the expected cost per row of a
rule is

    Cost = c_00 (1 - P) + c_01 P + c_FP * FP - c_FN * TP
         = const - c_FN * NB(t),     NB(t) = TP - [t / (1 - t)] * FP,

with `TP` and `FP` rates over all rows, `P` the prevalence and
`t = c_FP / (c_FP + c_FN)`. Minimising expected cost is therefore the same
problem as maximising the net benefit of Vickers and Elkin (2006), up to the
positive scale `c_FN`. The identity is checked numerically in the tests. The
definition was read in the author manuscript of Vickers and Elkin (Med Decis
Making 26(6), PMC2577036) on 2026-11-05: net benefit is
`TP/n - FP/n * pt/(1 - pt)`, and their eq. 1 gives the weight as the odds
of the threshold, equal to the ratio of the harm of a false positive to the
harm of a false negative, so
`pt = c_FP / (c_FP + c_FN)` as used here. The R packages named as references
were not available, so the reference used by the tests is an independent loop
implementation of the same definition.
"""

from collections.abc import Mapping
import dataclasses
import enum
import hashlib
from typing import Any
from typing import ClassVar
from typing import Self

import numpy
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.contracts import records
from scmle.contracts import signing

ENTRY_NAMES = ("c_00", "c_01", "c_10", "c_11")


class Basis(enum.StrEnum):
    """Where a number in the cost model came from."""

    DATA = "data"
    HUMAN = "human"
    TEMPLATE = "template"


class DominatedError(ValueError):
    """One action dominates, so there is no decision to make."""


@dataclasses.dataclass(frozen=True, slots=True)
class Tariff:
    """Costs of the four (action, outcome) cells and a plausible range.

    Attributes:
        c_00: Cost of not acting when the outcome is 0.
        c_01: Cost of not acting when the outcome is 1.
        c_10: Cost of acting when the outcome is 0.
        c_11: Cost of acting when the outcome is 1.
        low: Lowest plausible threshold, in (0, 1).
        high: Highest plausible threshold, in (0, 1).
        sources: Source of each cost, in the order c_00, c_01, c_10, c_11.
        basis: Source of the range `[low, high]`.
        migrations: The stored layout's version steps, oldest first; version
            1 spelled `low`, `high` and `basis` as `t_lo`, `t_hi` and
            `range_source`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={"t_lo": "low", "t_hi": "high", "range_source": "basis"}
        ),
    )

    c_00: float
    c_01: float
    c_10: float
    c_11: float
    low: float
    high: float
    sources: tuple[Basis, ...]
    basis: Basis

    def __post_init__(self) -> None:
        """Validates the model, refusing one that is not a decision.

        Raises:
            DominatedError: If `c_FP` or `c_FN` is not positive.
            ValueError: If a cost or a range bound is not a finite number,
                the range does not satisfy `0 < low <= high < 1`, the
                implied threshold lies outside the range, or `sources` or
                `basis` have the wrong type or length.
        """
        for name in ENTRY_NAMES:
            qualification.finite(getattr(self, name), name)
        if self.c_fp <= 0 or self.c_fn <= 0:
            raise DominatedError(
                "one action dominates (c_FP and c_FN must both be positive): "
                f"c_FP={self.c_fp}, c_FN={self.c_fn}"
            )
        low = qualification.finite(self.low, "low")
        high = qualification.finite(self.high, "high")
        if not 0.0 < low <= high < 1.0:
            raise errors.InputError(
                "the range must satisfy 0 < low <= high < 1"
            )
        if not low <= self.threshold <= high:
            raise errors.InputError(
                f"the implied threshold {self.threshold:.4f} lies outside "
                f"the range [{low}, {high}]"
            )
        qualification.tupled(self.sources, "sources")
        if len(self.sources) != len(ENTRY_NAMES) or not all(
            isinstance(s, Basis) for s in self.sources
        ):
            raise errors.InputError("sources must hold four Basis values")
        if not isinstance(self.basis, Basis):
            raise errors.InputError("basis must be a Basis")

    @property
    def c_fp(self) -> float:
        """Extra cost of acting on a row whose outcome is 0."""
        return self.c_10 - self.c_00

    @property
    def c_fn(self) -> float:
        """Extra cost of not acting on a row whose outcome is 1."""
        return self.c_01 - self.c_11

    @property
    def threshold(self) -> float:
        """The implied decision threshold `t = c_FP / (c_FP + c_FN)`."""
        return self.c_fp / (self.c_fp + self.c_fn)

    @property
    def unreviewed(self) -> tuple[str, ...]:
        """Names of the costs that came from data or a template, not a person.

        The names follow the order of `ENTRY_NAMES`.
        """
        return tuple(
            name
            for name, source in zip(ENTRY_NAMES, self.sources, strict=True)
            if source is not Basis.HUMAN
        )

    @property
    def model_hash(self) -> str:
        """SHA-256 of the canonical JSON of the model's version-1 layout.

        The hash is anchored at version 1 (`t_lo`, `t_hi`, `range_source`),
        so a model registered before its keys were renamed keeps its hash.
        """
        anchored = records.downgrade(self.migrations, self.to_dict())
        return hashlib.sha256(signing.canonicalise(anchored)).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Returns the model as a JSON-ready record.

        Returns:
            A mapping of the four costs, the range bounds, the source of
            each cost as its string value, and the source of the range. It
            carries no version key: the model is stored inside a charter or
            a framing directory, whose version it follows.
        """
        return {
            "c_00": self.c_00,
            "c_01": self.c_01,
            "c_10": self.c_10,
            "c_11": self.c_11,
            "low": self.low,
            "high": self.high,
            "sources": [s.value for s in self.sources],
            "basis": self.basis.value,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the model from a record made by `to_dict`.

        A record that is invalid, or fails the checks in `__post_init__`,
        raises `ValueError` or `DominatedError`.

        Args:
            record: A mapping holding exactly the fields of the model, or its
                version-1 layout (`t_lo`, `t_hi`, `range_source`). A record
                without `schema_version` is read as version 1; every step
                only renames keys, so a current layout passes unchanged.

        Returns:
            The model.
        """
        layout = qualification.require(cls, record)
        sources = tuple(Basis(v) for v in layout["sources"])
        if len(sources) != len(ENTRY_NAMES):
            raise errors.InputError("sources must hold one entry per cost")
        return cls(
            c_00=layout["c_00"],
            c_01=layout["c_01"],
            c_10=layout["c_10"],
            c_11=layout["c_11"],
            low=layout["low"],
            high=layout["high"],
            sources=sources,
            basis=Basis(layout["basis"]),
        )


def odds(t: float) -> float:
    """Returns `t / (1 - t)`, the weight of a false positive in net benefit.

    Args:
        t: Threshold, strictly between 0 and 1.

    Returns:
        The odds of the threshold.

    Raises:
        ValueError: If `t` is not a finite number strictly between 0 and 1.
    """
    threshold = qualification.finite(t, "t")
    if not 0.0 < threshold < 1.0:
        raise errors.InputError("t must be strictly between 0 and 1")
    return threshold / (1.0 - threshold)


def dichotomous(values: ArrayLike, name: str) -> NDArray[numpy.float64]:
    """Checks that `values` is a non-empty 1-d array of zeros and ones.

    Args:
        values: The values to check.
        name: Name of the argument, used in error messages.

    Returns:
        The values as a float array.

    Raises:
        ValueError: If the array is not 1-d, is empty, or holds anything
            other than 0 and 1.
    """
    array = numpy.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise errors.InputError(f"{name} must be a non-empty 1-d array")
    if not numpy.isin(array, (0.0, 1.0)).all():
        raise errors.InputError(f"{name} must hold only 0 and 1")
    return array


def pair(
    y: ArrayLike, a: ArrayLike
) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    """Checks and aligns outcomes and actions.

    Args:
        y: Outcomes, 0 or 1.
        a: Actions, 0 or 1.

    Returns:
        The outcomes and the actions as float arrays of equal length.

    Raises:
        ValueError: If either is not a non-empty 1-d array of zeros and ones,
            or the lengths differ.
    """
    outcomes = dichotomous(y, "y")
    actions = dichotomous(a, "a")
    if outcomes.shape != actions.shape:
        raise errors.InputError("y and a must have the same length")
    return outcomes, actions


def net_benefit(y: ArrayLike, a: ArrayLike, t: float) -> float:
    """Returns `NB(t) = TP - [t / (1 - t)] FP` for a fixed action rule.

    Input that `pair` or `odds` rejects raises `ValueError`.

    Args:
        y: Outcomes, 0 or 1.
        a: Actions, 0 or 1.
        t: Threshold, strictly between 0 and 1.

    Returns:
        Net benefit per row.
    """
    outcomes, actions = pair(y, a)
    tp = float(numpy.mean(actions * outcomes))
    fp = float(numpy.mean(actions * (1.0 - outcomes)))
    return tp - odds(t) * fp


def net_benefit_all(y: ArrayLike, t: float) -> float:
    """Returns the net benefit of acting on every row.

    The value is `P - (1 - P) t / (1 - t)`, with `P` the prevalence. Input
    that `dichotomous` or `odds` rejects raises `ValueError`.

    Args:
        y: Outcomes, 0 or 1.
        t: Threshold, strictly between 0 and 1.

    Returns:
        Net benefit per row.
    """
    outcomes = dichotomous(y, "y")
    prevalence = float(numpy.mean(outcomes))
    return prevalence - (1.0 - prevalence) * odds(t)


def net_benefit_none() -> float:
    """Returns the net benefit of acting on no row.

    Returns:
        Always 0.0.
    """
    return 0.0


def expected_cost(y: ArrayLike, a: ArrayLike, model: Tariff) -> float:
    """Returns the mean cost per row, read straight from the four cells.

    Input that `pair` rejects raises `ValueError`.

    Args:
        y: Outcomes, 0 or 1.
        a: Actions, 0 or 1.
        model: The cost model whose cells price each (action, outcome) pair.

    Returns:
        The mean of the cost of each row.
    """
    outcomes, actions = pair(y, a)
    cost = numpy.where(
        actions == 1.0,
        numpy.where(outcomes == 1.0, model.c_11, model.c_10),
        numpy.where(outcomes == 1.0, model.c_01, model.c_00),
    )
    return float(numpy.mean(cost))


def benefits(
    y: ArrayLike, a: ArrayLike, thresholds: ArrayLike
) -> dict[str, list[float]]:
    """Computes model, treat-all and treat-none net benefit on a grid.

    Input that `net_benefit` rejects raises `ValueError`.

    Args:
        y: Outcomes, 0 or 1.
        a: Actions of the frozen rule, 0 or 1.
        thresholds: Thresholds to evaluate, each strictly between 0 and 1.

    Returns:
        A mapping with `threshold` (the grid, flattened) and, for each
        threshold in the same order, `model`, `all` and `none` net benefit.
    """
    grid = [float(t) for t in numpy.asarray(thresholds, dtype=float).ravel()]
    return {
        "threshold": grid,
        "model": [net_benefit(y, a, t) for t in grid],
        "all": [net_benefit_all(y, t) for t in grid],
        "none": [0.0] * len(grid),
    }


def decision_curve(
    y: ArrayLike, scores: ArrayLike, thresholds: ArrayLike
) -> dict[str, list[float]]:
    """Net benefit when the rule is re-thresholded at every `t`.

    This is the usual decision-curve display (act when `score >= t`). It is
    the form the reference-implementation comparison uses. The baseline
    claim does not use it: there the rule is frozen on `X` first. Input that
    `net_benefit` rejects raises `ValueError`.

    Args:
        y: Outcomes, 0 or 1.
        scores: Risk score of each row; the rule acts when `score >= t`.
        thresholds: Thresholds to evaluate, each strictly between 0 and 1.

    Returns:
        A mapping with `threshold` (the grid, flattened) and, for each
        threshold in the same order, `model`, `all` and `none` net benefit.
    """
    risk = numpy.asarray(scores, dtype=float)
    grid = [float(t) for t in numpy.asarray(thresholds, dtype=float).ravel()]
    return {
        "threshold": grid,
        "model": [net_benefit(y, (risk >= t).astype(float), t) for t in grid],
        "all": [net_benefit_all(y, t) for t in grid],
        "none": [0.0] * len(grid),
    }
