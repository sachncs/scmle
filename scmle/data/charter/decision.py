"""The decision block of the charter (feature F11).

Everything here is optional: a charter without a `DecisionLayer` runs as it
always did and makes no decision claim. The costs of acting are not repeated
here; they stay in the charter's `cost_model` (F8), which a decision layer
requires. What this block adds is how the cost model is used (baselines,
calibration level, off-policy gates), which fairness constraints are declared,
and the rollout plan with its error levels.

Costs, margins and propensities are inputs, never findings: the instrument
tiers a decision claim `attested` at best when they come from a person.
"""

from collections.abc import Mapping
import dataclasses
from fractions import Fraction
from typing import Any
from typing import Self

from scmle.contracts import evidence
from scmle.contracts import qualification
from scmle.data.ledger import budget as budget_module

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


def positive_number(value: Any, name: str) -> float:
    """A finite number strictly greater than zero."""
    number = qualification.finite(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return number


def open_unit(value: Any, name: str) -> float:
    """A finite number strictly between 0 and 1."""
    number = qualification.finite(value, name)
    if not 0 < number < 1:
        raise ValueError(f"{name} must be strictly between 0 and 1")
    return number


@dataclasses.dataclass(frozen=True, slots=True)
class OffPolicyRules:
    """Declared gates of off-policy evaluation, fixed before the outcome.

    Attributes:
        clip: Largest importance weight; larger ones are clipped. None
            leaves the weights as they are.
        min_propensity: Smallest behaviour probability on which the target
            policy may put mass; below it the value is not identified.
        ess_floor: Smallest Kish effective sample size over `n`.
        penalty: Per-row penalty for each treated unit in policy fitting.
    """

    clip: float | None = None
    min_propensity: float = 0.01
    ess_floor: float = DEFAULT_ESS_FLOOR
    penalty: float = 0.0

    def __post_init__(self) -> None:
        """Validates the rules."""
        if self.clip is not None and positive_number(self.clip, "clip") < 1:
            raise ValueError("clip must be at least 1")
        open_unit(self.min_propensity, "min_propensity")
        open_unit(self.ess_floor, "ess_floor")
        if qualification.finite(self.penalty, "penalty") < 0:
            raise ValueError("penalty must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the rules, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(**record)


@dataclasses.dataclass(frozen=True, slots=True)
class DecisionModel:
    """How the chartered cost model is used to decide.

    Attributes:
        unit: Unit of the costs (for example "EUR per case").
        baselines: Rules the model must beat over the cost-ratio range.
        treatment_cost: Cost per treated unit for causal policies, or None.
        calibration_level: Level of the calibration check on development
            data.
        off_policy: Gates for off-policy evaluation.
    """

    unit: str
    baselines: tuple[str, ...] = BASELINES
    treatment_cost: float | None = None
    calibration_level: float = DEFAULT_CALIBRATION_LEVEL
    off_policy: OffPolicyRules = dataclasses.field(
        default_factory=OffPolicyRules
    )

    def __post_init__(self) -> None:
        """Validates the model."""
        evidence.require_text(self.unit, "unit")
        qualification.require_tuple(self.baselines, "baselines")
        if not self.baselines or len(set(self.baselines)) != len(
            self.baselines
        ):
            raise ValueError("baselines must be non-empty and distinct")
        for name in self.baselines:
            if name not in BASELINES:
                raise ValueError(f"unknown baseline {name!r}")
        if self.treatment_cost is not None:
            qualification.finite(self.treatment_cost, "treatment_cost")
        open_unit(self.calibration_level, "calibration_level")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "unit": self.unit,
            "baselines": list(self.baselines),
            "treatment_cost": self.treatment_cost,
            "calibration_level": self.calibration_level,
            "off_policy": self.off_policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the model, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(
            unit=record["unit"],
            baselines=tuple(record["baselines"]),
            treatment_cost=record["treatment_cost"],
            calibration_level=record["calibration_level"],
            off_policy=OffPolicyRules.from_dict(record["off_policy"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class FairnessConstraint:
    """A fairness constraint on the frozen rule; explicit, never a default.

    Attributes:
        attribute: Column that defines the groups.
        criterion: One of `CRITERIA`.
        max_gap: Largest allowed gap between groups, in (0, 1).
        mechanism: One of `MECHANISMS`.
        attestation: Who attests that the mechanism is permitted (group
            thresholds are a legal and policy question the instrument does
            not decide); required for `group_thresholds`.
        min_group: Rows below which a group is `not_assessable`.
    """

    attribute: str
    criterion: str
    max_gap: float
    mechanism: str
    attestation: str | None = None
    min_group: int = DEFAULT_MIN_GROUP

    def __post_init__(self) -> None:
        """Validates the constraint."""
        evidence.require_text(self.attribute, "attribute")
        if self.criterion not in CRITERIA:
            raise ValueError(f"unknown criterion {self.criterion!r}")
        open_unit(self.max_gap, "max_gap")
        if self.mechanism not in MECHANISMS:
            raise ValueError(f"unknown mechanism {self.mechanism!r}")
        if self.mechanism == "group_thresholds" and not (
            self.attestation and self.attestation.strip()
        ):
            raise ValueError(
                "group thresholds need an attestation that they are permitted"
            )
        qualification.count(self.min_group, "min_group", minimum=2)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the constraint, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(**record)


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
        higher_is_worse: Whether a larger value is the harmful direction.
    """

    name: str
    margin: float
    low: float = 0.0
    high: float = 1.0
    higher_is_worse: bool = True

    def __post_init__(self) -> None:
        """Validates the guardrail."""
        evidence.require_text(self.name, "name")
        positive_number(self.margin, "margin")
        if not qualification.finite(self.low, "low") < qualification.finite(
            self.high, "high"
        ):
            raise ValueError("low must be below high")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the guardrail, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(**record)


@dataclasses.dataclass(frozen=True, slots=True)
class RolloutPlan:
    """The pre-registered plan of an A/B test and its decisions.

    Attributes:
        unit: Randomisation unit.
        primary: Primary metric.
        low: Smallest value of the primary metric.
        high: Largest value of the primary metric.
        assignment_prob: Probability of the challenger arm, in (0, 1).
        delta_min: Smallest improvement worth acting on, in metric units.
        tolerance: Rollback tolerance `tau`, in metric units.
        n_max: Most units the experiment may run.
        alpha_promote: Total false-promotion level, as an exact fraction.
        alpha_rollback: Rollback level per promoted champion.
        guardrails: Metrics that must not get worse.
        salt: Salt of the hashed assignment.
        srm_level: Level of the sample-ratio check.
        horizon: Planned number of experiments for the level discount, or
            None for the open-ended sequence.
    """

    unit: str
    primary: str
    assignment_prob: float
    delta_min: float
    tolerance: float
    n_max: int
    alpha_promote: Fraction
    alpha_rollback: Fraction
    low: float = 0.0
    high: float = 1.0
    guardrails: tuple[Guardrail, ...] = ()
    salt: str = "scmle"
    srm_level: float = 0.001
    horizon: int | None = None

    def __post_init__(self) -> None:
        """Validates the plan."""
        evidence.require_text(self.unit, "unit")
        evidence.require_text(self.primary, "primary")
        evidence.require_text(self.salt, "salt")
        open_unit(self.assignment_prob, "assignment_prob")
        positive_number(self.delta_min, "delta_min")
        positive_number(self.tolerance, "tolerance")
        qualification.count(self.n_max, "n_max", minimum=2)
        for name in ("alpha_promote", "alpha_rollback"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be strictly between 0 and 1")
        if not qualification.finite(self.low, "low") < qualification.finite(
            self.high, "high"
        ):
            raise ValueError("low must be below high")
        if self.delta_min >= self.high - self.low:
            raise ValueError("delta_min must be smaller than the metric range")
        open_unit(self.srm_level, "srm_level")
        qualification.require_tuple(self.guardrails, "guardrails")
        names = [g.name for g in self.guardrails]
        if len(set(names)) != len(names) or self.primary in names:
            raise ValueError("guardrail names must be distinct from each other")

    def budget(self) -> budget_module.DecisionBudget:
        """A fresh decision budget with the plan's levels."""
        return budget_module.DecisionBudget(
            alpha_promote=self.alpha_promote,
            alpha_rollback=self.alpha_rollback,
            horizon=self.horizon,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record; fractions are written as ``"p/q"`` strings."""
        return {
            "unit": self.unit,
            "primary": self.primary,
            "assignment_prob": self.assignment_prob,
            "delta_min": self.delta_min,
            "tolerance": self.tolerance,
            "n_max": self.n_max,
            "alpha_promote": str(self.alpha_promote),
            "alpha_rollback": str(self.alpha_rollback),
            "low": self.low,
            "high": self.high,
            "guardrails": [g.to_dict() for g in self.guardrails],
            "salt": self.salt,
            "srm_level": self.srm_level,
            "horizon": self.horizon,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the plan, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(
            unit=record["unit"],
            primary=record["primary"],
            assignment_prob=record["assignment_prob"],
            delta_min=record["delta_min"],
            tolerance=record["tolerance"],
            n_max=record["n_max"],
            alpha_promote=budget_module.parse_fraction(
                record["alpha_promote"], "alpha_promote"
            ),
            alpha_rollback=budget_module.parse_fraction(
                record["alpha_rollback"], "alpha_rollback"
            ),
            low=record["low"],
            high=record["high"],
            guardrails=tuple(
                Guardrail.from_dict(g) for g in record["guardrails"]
            ),
            salt=record["salt"],
            srm_level=record["srm_level"],
            horizon=record["horizon"],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class DecisionLayer:
    """The optional decision block of an `AnalysisCharter`.

    Attributes:
        model: How the chartered costs are used.
        fairness: Declared fairness constraints (possibly none).
        rollout: The rollout plan, or None when only offline evidence is
            wanted.
    """

    model: DecisionModel
    fairness: tuple[FairnessConstraint, ...] = ()
    rollout: RolloutPlan | None = None

    def __post_init__(self) -> None:
        """Validates the block."""
        qualification.require_tuple(self.fairness, "fairness")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record; absent parts are omitted."""
        record: dict[str, Any] = {
            "model": self.model.to_dict(),
            "fairness": [f.to_dict() for f in self.fairness],
        }
        if self.rollout is not None:
            record["rollout"] = self.rollout.to_dict()
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the block, rejecting unknown or missing fields."""
        record = {"rollout": None, **record}
        qualification.require_fields(cls, record)
        plan = record["rollout"]
        return cls(
            model=DecisionModel.from_dict(record["model"]),
            fairness=tuple(
                FairnessConstraint.from_dict(f) for f in record["fairness"]
            ),
            rollout=None if plan is None else RolloutPlan.from_dict(plan),
        )
