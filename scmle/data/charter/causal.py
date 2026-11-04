"""The causal block of a charter estimand (charter format 4).

A `kind="causal"` estimand may carry a `CausalSpec`: the treatment, the
outcome, the estimand, the design columns, the assumptions the analyst
attests, the priority order of strategies, the pre-registered negative
controls and the sensitivity benchmark. The identification step reads this
block and nothing else to choose a strategy (F10), so the choice is a function
of the charter and never of the outcome values.

Versioning. An estimand without the block serialises exactly as before (the
key is omitted), so every charter signed under format 1 to 3 keeps its hash
and its signature and loads unchanged.
"""

from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import Self

from scmle.contracts import evidence
from scmle.contracts import qualification

CAUSAL_ESTIMANDS = ("ATE", "ATT", "LATE", "ATT_DID", "ATT_SC", "LATE_RD")
UNCONFOUNDEDNESS_PREFIX = "unconfoundedness_given:"
FIXED_ASSUMPTIONS = (
    "positivity",
    "no_interference",
    "randomized_assignment",
    "parallel_trends",
    "no_anticipation",
    "instrument_relevance",
    "exclusion",
    "independence",
    "monotonicity",
    "continuity_at_cutoff",
    "no_manipulation",
    "synthetic_hull",
)
NEGATIVE_CONTROL_KINDS = ("outcome", "exposure")
BENCHMARKS = ("strongest_observed", "fixed")


def covariates_of(assumptions: Iterable[str]) -> tuple[str, ...] | None:
    """The covariates named by `unconfoundedness_given:...`, if declared.

    Returns:
        The names in declared order, or None when the assumption is absent.
    """
    for item in assumptions:
        if item.startswith(UNCONFOUNDEDNESS_PREFIX):
            names = item[len(UNCONFOUNDEDNESS_PREFIX) :].split(",")
            found = tuple(n.strip() for n in names if n.strip())
            return () if found == ("none",) else found
    return None


def check_assumption(item: str) -> None:
    """Raises `ValueError` for a word outside the closed vocabulary."""
    if item in FIXED_ASSUMPTIONS:
        return
    if item.startswith(UNCONFOUNDEDNESS_PREFIX):
        if (
            covariates_of([item]) is None
            or not item[len(UNCONFOUNDEDNESS_PREFIX) :].strip()
        ):
            raise ValueError(
                "unconfoundedness_given needs covariates, or 'none' for an "
                "attestation that no adjustment is needed"
            )
        return
    raise ValueError(
        f"unknown assumption {item!r}; allowed: {FIXED_ASSUMPTIONS} and "
        f"'{UNCONFOUNDEDNESS_PREFIX}<columns>'"
    )


@dataclasses.dataclass(frozen=True, slots=True)
class NegativeControl:
    """A variable the treatment cannot affect that shares the confounding.

    Attributes:
        kind: `outcome` (an outcome the treatment cannot move) or `exposure`
            (a treatment that cannot move the outcome).
        variable: The column.
    """

    kind: str
    variable: str

    def __post_init__(self) -> None:
        """Validates the record."""
        if self.kind not in NEGATIVE_CONTROL_KINDS:
            raise ValueError(f"kind must be one of {NEGATIVE_CONTROL_KINDS}")
        evidence.require_text(self.variable, "variable")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the control, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(**record)


@dataclasses.dataclass(frozen=True, slots=True)
class SensitivitySpec:
    """How strong an unmeasured confounder the claim must survive.

    Attributes:
        benchmark: `strongest_observed` lets the confounder be `strength`
            times as strong as the strongest observed adjustment covariate;
            `fixed` gives the partial R-squared of the confounder with the
            treatment and with the outcome directly (`strength`).
        strength: The multiplier, or the partial R-squared in (0, 1).
        m_max: For difference in differences, the largest multiple `M` of the
            biggest pre-period change that the claim must survive.
    """

    benchmark: str = "strongest_observed"
    strength: float = 1.0
    m_max: float = 1.0

    def __post_init__(self) -> None:
        """Validates the record."""
        if self.benchmark not in BENCHMARKS:
            raise ValueError(f"benchmark must be one of {BENCHMARKS}")
        value = qualification.finite(self.strength, "strength")
        if value <= 0:
            raise ValueError("strength must be positive")
        if self.benchmark == "fixed" and value >= 1:
            raise ValueError("a fixed strength is a partial R2 below 1")
        if qualification.finite(self.m_max, "m_max") < 0:
            raise ValueError("m_max must not be negative")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the record, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(**record)


@dataclasses.dataclass(frozen=True, slots=True)
class CausalSpec:
    """The causal question of a charter.

    Attributes:
        treatment: Treatment column (binary for back-door, DiD, SC).
        outcome: Outcome column.
        estimand: One of `CAUSAL_ESTIMANDS`.
        unit: Unit identifier column (panel designs).
        time: Period column (panel designs).
        adoption: Column holding the first treated period of the unit, empty
            for a never-treated unit (DiD).
        instrument: Instrument column (one, binary or numeric).
        running_variable: Running variable (RD).
        cutoff: Cutoff of the running variable (RD).
        dag: Path of a hashed DAG hypothesis (informational for the
            selector; the back-door check needs the graph object).
        assumptions: Attested assumptions, a closed vocabulary.
        strategies: Strategy names in priority order, fixed here.
        negative_controls: Pre-registered falsification controls.
        strata: Replication strata columns.
        sensitivity: Unmeasured-confounding benchmark, if declared.
        trim: Propensity trimming bound `eps` in [0, 0.5): units with an
            estimated propensity outside `[eps, 1 - eps]` are dropped and the
            report names the changed estimand. Zero means no trimming.
        min_share: Smallest share of units that may remain after trimming.
    """

    treatment: str
    outcome: str
    estimand: str
    unit: str | None = None
    time: str | None = None
    adoption: str | None = None
    instrument: str | None = None
    running_variable: str | None = None
    cutoff: float | None = None
    dag: str | None = None
    assumptions: tuple[str, ...] = ()
    strategies: tuple[str, ...] = ()
    negative_controls: tuple[NegativeControl, ...] = ()
    strata: tuple[str, ...] = ()
    sensitivity: SensitivitySpec | None = None
    trim: float = 0.0
    min_share: float = 0.5

    def __post_init__(self) -> None:
        """Validates the block."""
        evidence.require_text(self.treatment, "treatment")
        evidence.require_text(self.outcome, "outcome")
        if self.estimand not in CAUSAL_ESTIMANDS:
            raise ValueError(f"estimand must be one of {CAUSAL_ESTIMANDS}")
        for name in (
            "assumptions",
            "strategies",
            "negative_controls",
            "strata",
        ):
            qualification.require_tuple(getattr(self, name), name)
        for item in self.assumptions:
            check_assumption(item)
        if len(set(self.assumptions)) != len(self.assumptions):
            raise ValueError("duplicate assumption")
        if len(set(self.strategies)) != len(self.strategies):
            raise ValueError("duplicate strategy")
        if self.cutoff is not None:
            qualification.finite(self.cutoff, "cutoff")
        trim = qualification.finite(self.trim, "trim")
        if not 0.0 <= trim < 0.5:
            raise ValueError("trim must be within [0, 0.5)")
        share = qualification.finite(self.min_share, "min_share")
        if not 0.0 < share <= 1.0:
            raise ValueError("min_share must be within (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        record = dataclasses.asdict(self)
        record["negative_controls"] = [
            c.to_dict() for c in self.negative_controls
        ]
        record["sensitivity"] = (
            None if self.sensitivity is None else self.sensitivity.to_dict()
        )
        for name in ("assumptions", "strategies", "strata"):
            record[name] = list(record[name])
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the block, rejecting unknown or missing fields."""
        defaults = {
            f.name: f.default
            for f in dataclasses.fields(cls)
            if f.default is not dataclasses.MISSING
        }
        record = {**defaults, **record}
        qualification.require_fields(cls, record)
        sensitivity = record["sensitivity"]
        return cls(
            treatment=record["treatment"],
            outcome=record["outcome"],
            estimand=record["estimand"],
            unit=record["unit"],
            time=record["time"],
            adoption=record["adoption"],
            instrument=record["instrument"],
            running_variable=record["running_variable"],
            cutoff=record["cutoff"],
            dag=record["dag"],
            assumptions=tuple(record["assumptions"]),
            strategies=tuple(record["strategies"]),
            negative_controls=tuple(
                NegativeControl.from_dict(c)
                for c in record["negative_controls"]
            ),
            strata=tuple(record["strata"]),
            sensitivity=(
                None
                if sensitivity is None
                else SensitivitySpec.from_dict(sensitivity)
            ),
            trim=record["trim"],
            min_share=record["min_share"],
        )
