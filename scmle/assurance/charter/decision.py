"""The decision block of the charter (format 5).

Everything here is optional: a charter without a `Layer` runs as it
always did and makes no decision claim. The costs of acting are not repeated
here; they stay in the charter's `costs`, which a decision layer
requires. What this block adds is how the cost model is used (baselines,
calibration level, off-policy gates), which fairness constraints are
declared, and the rollout design with its error levels.

Costs, margins and propensities are inputs, never findings: the instrument
tiers a decision claim `attested` at best when they come from a person.
"""

from collections.abc import Mapping
import dataclasses
from fractions import Fraction
from typing import Any
from typing import ClassVar
from typing import Self

import scmle.assurance.ledger.budget
from scmle.contracts import errors
from scmle.contracts import evidence
from scmle.contracts import qualification
from scmle.contracts import records

BASELINES = ("treat_none", "treat_all", "incumbent")
CRITERIA = (
    "demographic_parity",
    "equal_opportunity",
    "equalized_odds",
    "predictive_parity",
)
MECHANISMS = ("group_thresholds", "reweighing")
DEFAULT_CALIBRATION_LEVEL = 0.01
DEFAULT_ESS_FLOOR = 0.2
DEFAULT_MIN_GROUP = 50
HIGHER = "higher"
LOWER = "lower"
DIRECTIONS = (HIGHER, LOWER)


def check(value: Any, name: str) -> float:
    """Checks that a number is finite and strictly greater than zero.

    Args:
        value: The candidate number.
        name: Name used in the error message.

    Returns:
        `value` as a float.

    Raises:
        ValueError: If `value` is not a finite int or float (a bool is
            rejected) or is not greater than 0.
    """
    number = qualification.finite(value, name)
    if number <= 0:
        raise errors.InputError(f"{name} must be greater than 0")
    return number


def confine(value: Any, name: str) -> float:
    """Checks that a number is finite and strictly between 0 and 1.

    Args:
        value: The candidate number.
        name: Name used in the error message.

    Returns:
        `value` as a float.

    Raises:
        ValueError: If `value` is not a finite int or float (a bool is
            rejected) or is outside the open interval (0, 1).
    """
    number = qualification.finite(value, name)
    if not 0 < number < 1:
        raise errors.InputError(f"{name} must be strictly between 0 and 1")
    return number


@dataclasses.dataclass(frozen=True, slots=True)
class Gates:
    """Declared gates of off-policy evaluation, fixed before the outcome.

    Attributes:
        clip: Largest importance weight; larger ones are clipped. None
            leaves the weights as they are.
        propensity: Smallest behaviour probability on which the target
            policy may put mass; below it the value is not identified.
        ess: Floor of the Kish effective sample size (ESS) over `n`.
        penalty: Per-row penalty for each treated unit in policy fitting.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={"min_propensity": "propensity", "ess_floor": "ess"}
        ),
    )

    clip: float | None = None
    propensity: float = 0.01
    ess: float = DEFAULT_ESS_FLOOR
    penalty: float = 0.0

    def __post_init__(self) -> None:
        """Validates the rules.

        Raises:
            ValueError: If `clip` is set and is not a finite number of at
                least 1, `propensity` or `ess` is not strictly in (0, 1), or
                `penalty` is not finite or is negative.
        """
        if self.clip is not None and check(self.clip, "clip") < 1:
            raise errors.InputError("clip must be at least 1")
        confine(self.propensity, "propensity")
        confine(self.ess, "ess")
        if qualification.finite(self.penalty, "penalty") < 0:
            raise errors.InputError("penalty must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Serializes the record.

        Returns:
            A JSON-ready dict of the record's fields.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the rules from a dict.

        Keys other than the fields, and missing fields, are rejected with
        the `ValueError` of `qualification.require`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt rules.
        """
        return cls(**qualification.require(cls, record))


@dataclasses.dataclass(frozen=True, slots=True)
class Policy:
    """How the chartered cost model is used to decide.

    Attributes:
        unit: Unit of the costs (for example "EUR per case").
        baselines: Rules the model must beat over the cost-ratio range.
        cost: Cost per treated unit for causal policies, or None.
        calibration: Level of the calibration check on development data.
        gates: Gates for off-policy evaluation.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            renames={
                "treatment_cost": "cost",
                "calibration_level": "calibration",
                "off_policy": "gates",
            },
            parts={"gates": Gates.migrations[0]},
        ),
    )

    unit: str
    baselines: tuple[str, ...] = BASELINES
    cost: float | None = None
    calibration: float = DEFAULT_CALIBRATION_LEVEL
    gates: Gates = dataclasses.field(default_factory=Gates)

    def __post_init__(self) -> None:
        """Validates the model.

        Raises:
            ValueError: If `unit` is blank, `baselines` is not a tuple, is
                empty, has a repeat or names a rule outside `BASELINES`,
                `cost` is set but not finite, or `calibration` is not
                strictly in (0, 1).
        """
        evidence.require_text(self.unit, "unit")
        qualification.tupled(self.baselines, "baselines")
        if not self.baselines or len(set(self.baselines)) != len(
            self.baselines
        ):
            raise errors.InputError("baselines must be non-empty and distinct")
        for name in self.baselines:
            if name not in BASELINES:
                raise errors.InputError(f"unknown baseline {name!r}")
        if self.cost is not None:
            qualification.finite(self.cost, "cost")
        confine(self.calibration, "calibration")

    def to_dict(self) -> dict[str, Any]:
        """Serializes the model.

        Returns:
            A JSON-ready dict of the fields, with `baselines` as a list and
            `gates` as a nested dict.
        """
        return {
            "unit": self.unit,
            "baselines": list(self.baselines),
            "cost": self.cost,
            "calibration": self.calibration,
            "gates": self.gates.to_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the model from a dict.

        Keys other than the fields, and missing fields, are rejected with
        the `ValueError` of `qualification.require`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt model.
        """
        record = qualification.require(cls, record)
        return cls(
            unit=record["unit"],
            baselines=tuple(record["baselines"]),
            cost=record["cost"],
            calibration=record["calibration"],
            gates=Gates.from_dict(record["gates"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Constraint:
    """A fairness constraint on the frozen rule; explicit, never a default.

    Attributes:
        attribute: Column that defines the groups.
        criterion: One of `CRITERIA`.
        gap: Largest allowed gap between groups, in (0, 1).
        mechanism: One of `MECHANISMS`.
        attestation: Who attests that the mechanism is permitted (group
            thresholds are a legal and policy question the instrument does
            not decide); required for `group_thresholds`.
        minimum: Rows below which a group is `not_assessable`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(renames={"max_gap": "gap", "min_group": "minimum"}),
    )

    attribute: str
    criterion: str
    gap: float
    mechanism: str
    attestation: str | None = None
    minimum: int = DEFAULT_MIN_GROUP

    def __post_init__(self) -> None:
        """Validates the constraint.

        Raises:
            ValueError: If `attribute` is blank, `criterion` is not in
                `CRITERIA`, `gap` is not strictly in (0, 1), `mechanism` is
                not in `MECHANISMS`, `group_thresholds` is chosen without a
                non-blank `attestation`, or `minimum` is not an integer of
                at least 2.
        """
        evidence.require_text(self.attribute, "attribute")
        if self.criterion not in CRITERIA:
            raise errors.InputError(f"unknown criterion {self.criterion!r}")
        confine(self.gap, "gap")
        if self.mechanism not in MECHANISMS:
            raise errors.InputError(f"unknown mechanism {self.mechanism!r}")
        if self.mechanism == "group_thresholds" and not (
            self.attestation and self.attestation.strip()
        ):
            raise errors.InputError(
                "group thresholds need an attestation that they are permitted"
            )
        qualification.count(self.minimum, "minimum", minimum=2)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the record.

        Returns:
            A JSON-ready dict of the record's fields.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the constraint from a dict.

        Keys other than the fields, and missing fields, are rejected with
        the `ValueError` of `qualification.require`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt constraint.
        """
        return cls(**qualification.require(cls, record))


@dataclasses.dataclass(frozen=True, slots=True)
class Guardrail:
    """A metric that must not get worse by more than a margin.

    Attributes:
        name: The metric.
        margin: Largest tolerated harm (challenger minus champion in the
            harmful direction), in the metric's units; the null is that the
            harm is at least the margin.
        low: Smallest value the metric can take.
        high: Largest value it can take.
        direction: The harmful direction, `higher` (a larger value is
            worse) or `lower`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={"higher_is_worse": "direction"},
            change=lambda layout: {
                **layout,
                **(
                    {"direction": HIGHER if layout["direction"] else LOWER}
                    if isinstance(layout.get("direction"), bool)
                    else {}
                ),
            },
            revert=lambda layout: {
                **layout,
                **(
                    {"direction": layout["direction"] == HIGHER}
                    if isinstance(layout.get("direction"), str)
                    else {}
                ),
            },
        ),
    )

    name: str
    margin: float
    low: float = 0.0
    high: float = 1.0
    direction: str = HIGHER

    def __post_init__(self) -> None:
        """Validates the guardrail.

        Raises:
            ValueError: If `name` is blank, `margin` is not finite and
                positive, `low` is not finite and below a finite `high`, or
                `direction` is not in `DIRECTIONS`.
        """
        evidence.require_text(self.name, "name")
        check(self.margin, "margin")
        if self.direction not in DIRECTIONS:
            raise errors.InputError(f"direction must be one of {DIRECTIONS}")
        if not qualification.finite(self.low, "low") < qualification.finite(
            self.high, "high"
        ):
            raise errors.InputError("low must be below high")

    def to_dict(self) -> dict[str, Any]:
        """Serializes the record.

        Returns:
            A JSON-ready dict of the record's fields.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the guardrail from a dict.

        Keys other than the fields, and missing fields, are rejected with
        the `ValueError` of `qualification.require`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt guardrail.
        """
        return cls(**qualification.require(cls, record))


@dataclasses.dataclass(frozen=True, slots=True)
class Experiment:
    """The pre-registered design of an A/B test and its decisions.

    Attributes:
        unit: Randomisation unit.
        primary: Primary metric.
        low: Smallest value of the primary metric.
        high: Largest value of the primary metric.
        assignment: Probability of the challenger arm, in (0, 1).
        delta: Smallest improvement worth acting on, in metric units.
        tolerance: Rollback tolerance `tau`, in metric units.
        limit: Most units the experiment may run.
        promotion: Total false-promotion level (alpha), as an exact
            fraction.
        rollback: Rollback level (alpha) per promoted champion.
        guardrails: Metrics that must not get worse.
        salt: Salt of the hashed assignment.
        srm: Level of the sample-ratio-mismatch (SRM) check.
        horizon: Number of experiments over which the false-promotion level
            is divided equally, or None for the open-ended sequence of
            decreasing shares.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            renames={
                "assignment_prob": "assignment",
                "delta_min": "delta",
                "n_max": "limit",
                "alpha_promote": "promotion",
                "alpha_rollback": "rollback",
                "srm_level": "srm",
            },
            parts={"guardrails": Guardrail.migrations[0]},
        ),
    )

    unit: str
    primary: str
    assignment: float
    delta: float
    tolerance: float
    limit: int
    promotion: Fraction
    rollback: Fraction
    low: float = 0.0
    high: float = 1.0
    guardrails: tuple[Guardrail, ...] = ()
    salt: str = "scmle"
    srm: float = 0.001
    horizon: int | None = None

    def __post_init__(self) -> None:
        """Validates the rollout.

        Raises:
            ValueError: If `unit`, `primary` or `salt` is blank;
                `assignment` or `srm` is not strictly in (0, 1); `delta` or
                `tolerance` is not finite and positive; `limit` is not an
                integer of at least 2; `promotion` or `rollback` is not
                strictly between 0 and 1; `low` is not finite and below a
                finite `high`; `delta` is not smaller than `high - low`;
                `guardrails` is not a tuple; or guardrail names repeat or
                include `primary`.
        """
        evidence.require_text(self.unit, "unit")
        evidence.require_text(self.primary, "primary")
        evidence.require_text(self.salt, "salt")
        confine(self.assignment, "assignment")
        check(self.delta, "delta")
        check(self.tolerance, "tolerance")
        qualification.count(self.limit, "limit", minimum=2)
        for name in ("promotion", "rollback"):
            if not 0 < getattr(self, name) < 1:
                raise errors.InputError(
                    f"{name} must be strictly between 0 and 1"
                )
        if not qualification.finite(self.low, "low") < qualification.finite(
            self.high, "high"
        ):
            raise errors.InputError("low must be below high")
        if self.delta >= self.high - self.low:
            raise errors.InputError(
                "delta must be smaller than the metric range"
            )
        confine(self.srm, "srm")
        qualification.tupled(self.guardrails, "guardrails")
        names = [g.name for g in self.guardrails]
        if len(set(names)) != len(names) or self.primary in names:
            raise errors.InputError(
                "guardrail names must be distinct from each other"
            )

    def budget(self) -> scmle.assurance.ledger.budget.Decision:
        """Builds a decision budget from the rollout's levels.

        Returns:
            A new `budget.Decision` with this rollout's `promotion`,
            `rollback` and `horizon`.
        """
        return scmle.assurance.ledger.budget.Decision(
            promotion=self.promotion,
            rollback=self.rollback,
            horizon=self.horizon,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the rollout.

        Returns:
            A JSON-ready dict of the fields. The two alpha levels are
            written as ``"p/q"`` strings and `guardrails` as a list of
            dicts.
        """
        return {
            "unit": self.unit,
            "primary": self.primary,
            "assignment": self.assignment,
            "delta": self.delta,
            "tolerance": self.tolerance,
            "limit": self.limit,
            "promotion": str(self.promotion),
            "rollback": str(self.rollback),
            "low": self.low,
            "high": self.high,
            "guardrails": [g.to_dict() for g in self.guardrails],
            "salt": self.salt,
            "srm": self.srm,
            "horizon": self.horizon,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the rollout from a dict.

        Keys other than the fields, and missing fields, are rejected with
        the `ValueError` of `qualification.require`; an alpha level
        that is not a valid fraction string raises `ValueError` from
        `budget.parse`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt rollout.
        """
        record = qualification.require(cls, record)
        return cls(
            unit=record["unit"],
            primary=record["primary"],
            assignment=record["assignment"],
            delta=record["delta"],
            tolerance=record["tolerance"],
            limit=record["limit"],
            promotion=scmle.assurance.ledger.budget.parse(
                record["promotion"], "promotion"
            ),
            rollback=scmle.assurance.ledger.budget.parse(
                record["rollback"], "rollback"
            ),
            low=record["low"],
            high=record["high"],
            guardrails=tuple(
                Guardrail.from_dict(g) for g in record["guardrails"]
            ),
            salt=record["salt"],
            srm=record["srm"],
            horizon=record["horizon"],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Layer:
    """The optional decision block of an `Charter`.

    Attributes:
        model: How the chartered costs are used.
        fairness: Declared fairness constraints (possibly none).
        rollout: The rollout design, or None when only offline evidence is
            wanted.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            renames={},
            parts={
                "model": Policy.migrations[0],
                "fairness": Constraint.migrations[0],
                "rollout": Experiment.migrations[0],
            },
        ),
    )

    model: Policy
    fairness: tuple[Constraint, ...] = ()
    rollout: Experiment | None = None

    def __post_init__(self) -> None:
        """Validates the block.

        A `fairness` that is not a tuple raises `ValueError` (from
        `qualification.tupled`).
        """
        qualification.tupled(self.fairness, "fairness")

    def to_dict(self) -> dict[str, Any]:
        """Serializes the block.

        Returns:
            A JSON-ready dict with `model` and `fairness` (a list of
            dicts), plus `rollout` only when it is set.
        """
        record: dict[str, Any] = {
            "model": self.model.to_dict(),
            "fairness": [f.to_dict() for f in self.fairness],
        }
        if self.rollout is not None:
            record["rollout"] = self.rollout.to_dict()
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the block from a dict.

        `rollout` may be absent and then defaults to None. Other missing
        fields, and keys that are not fields, are rejected with the
        `ValueError` of `qualification.require`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt block.
        """
        record = qualification.require(
            cls,
            {
                "rollout": None,
                **records.load(cls.__name__, cls.migrations, record),
            },
        )
        plan = record["rollout"]
        return cls(
            model=Policy.from_dict(record["model"]),
            fairness=tuple(Constraint.from_dict(f) for f in record["fairness"]),
            rollout=None if plan is None else Experiment.from_dict(plan),
        )
