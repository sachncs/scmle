"""Contracts of the analysis charter (feature F1).

The charter fixes, before any data is profiled, what is being asked, how a
claim will be judged and what error the run may spend. Its estimand follows
the attributes of ICH E9(R1): population, target variable, intercurrent
events and their handling strategy, and a summary measure; a prediction task
adds the prediction moment. The class is named `AnalysisEstimand` because
`EstimandSpec` already exists for parameter estimation.

The charter hash is the SHA-256 of its canonical JSON, so it is identical
across runs and platforms and any one-byte edit changes it.
"""

from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import enum
import hashlib
import json
from pathlib import Path
from typing import Any
from typing import Self

from scmle.contracts import errors
from scmle.contracts import evidence
from scmle.contracts import qualification
from scmle.contracts import signing
from scmle.data.ledger import budget as budget_module

FORMAT_VERSION = 1
ESTIMAND_KINDS = ("predictive", "causal")
MAX_CONFIRMATION_FRACTION = 0.5


class DecisionAction(enum.StrEnum):
    """Closed set of actions a check may trigger."""

    DROP_COLUMN = "drop_column"
    QUARANTINE_COLUMN = "quarantine_column"
    REWEIGHT = "reweight"
    SWITCH_SPLIT = "switch_split"
    BLOCK_RUN = "block_run"
    DOWNGRADE_TIER = "downgrade_tier"


def texts(values: Iterable[Any], name: str) -> tuple[str, ...]:
    """Returns a tuple of non-empty strings."""
    return tuple(evidence.require_text(str(v), name) for v in values)


@dataclasses.dataclass(frozen=True, slots=True)
class IntercurrentEvent:
    """An event after the start that affects the target, and its strategy.

    Attributes:
        event: What happens (for example "account closed").
        strategy: How it is handled (censor, exclude, treat as outcome, ...).
    """

    event: str
    strategy: str

    def __post_init__(self) -> None:
        """Validates the record."""
        evidence.require_text(self.event, "event")
        evidence.require_text(self.strategy, "strategy")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the event, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(**record)


@dataclasses.dataclass(frozen=True, slots=True)
class AnalysisEstimand:
    """What the analysis estimates, in ICH E9(R1) terms.

    Attributes:
        kind: `predictive` or `causal`; selects the admissibility rule (F4).
        population: Who or what the claim is about.
        target: The variable, with its measurement window.
        prediction_moment: When a prediction is issued (the as-of rule).
        intercurrent_events: Events and the strategy for each (at least one;
            declare "none" with strategy "not applicable" if there are none).
        summary_measure: How the target is summarised (for example "ROC AUC").
    """

    kind: str
    population: str
    target: str
    prediction_moment: str
    intercurrent_events: tuple[IntercurrentEvent, ...]
    summary_measure: str

    def __post_init__(self) -> None:
        """Validates the estimand."""
        if self.kind not in ESTIMAND_KINDS:
            raise ValueError(f"kind must be one of {ESTIMAND_KINDS}")
        evidence.require_text(self.population, "population")
        evidence.require_text(self.target, "target")
        evidence.require_text(self.prediction_moment, "prediction_moment")
        evidence.require_text(self.summary_measure, "summary_measure")
        qualification.require_tuple(
            self.intercurrent_events, "intercurrent_events"
        )
        if not self.intercurrent_events:
            raise ValueError(
                "intercurrent_events must not be empty: state a strategy "
                "for each event, or declare 'none' as not applicable"
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "kind": self.kind,
            "population": self.population,
            "target": self.target,
            "prediction_moment": self.prediction_moment,
            "intercurrent_events": [
                e.to_dict() for e in self.intercurrent_events
            ],
            "summary_measure": self.summary_measure,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the estimand, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(
            kind=record["kind"],
            population=record["population"],
            target=record["target"],
            prediction_moment=record["prediction_moment"],
            intercurrent_events=tuple(
                IntercurrentEvent.from_dict(e)
                for e in record["intercurrent_events"]
            ),
            summary_measure=record["summary_measure"],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class DecisionRow:
    """Binds one check to the action its outcome triggers.

    Attributes:
        check_id: The check.
        trigger: A test outcome at the budgeted level (not a raw threshold).
        action: What happens when the trigger fires.
        certificate_id: The detector certificate that licenses gating
            (required for every gating check; see the charter lint).
    """

    check_id: str
    trigger: str
    action: DecisionAction
    certificate_id: str = ""

    def __post_init__(self) -> None:
        """Validates the row."""
        evidence.require_text(self.check_id, "check_id")
        evidence.require_text(self.trigger, "trigger")
        if not isinstance(self.action, DecisionAction):
            raise ValueError("action must be a DecisionAction")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "check_id": self.check_id,
            "trigger": self.trigger,
            "action": self.action.value,
            "certificate_id": self.certificate_id,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the row, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(
            check_id=record["check_id"],
            trigger=record["trigger"],
            action=DecisionAction(record["action"]),
            certificate_id=record["certificate_id"],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class SetSizes:
    """How the development rows are divided (F2).

    Attributes:
        confirmation_fraction: Share of development rows carved out as the
            confirmation set `C`, in (0, 0.5].
        embargo_rows: Rows dropped around temporal confirmation blocks so
            serial dependence cannot cross the boundary.
    """

    confirmation_fraction: float
    embargo_rows: int = 0

    def __post_init__(self) -> None:
        """Validates the sizes."""
        fraction = qualification.finite(
            self.confirmation_fraction, "confirmation_fraction"
        )
        if not 0 < fraction <= MAX_CONFIRMATION_FRACTION:
            raise ValueError(
                "confirmation_fraction must be in (0, "
                f"{MAX_CONFIRMATION_FRACTION}] so exploration keeps a majority"
            )
        qualification.count(self.embargo_rows, "embargo_rows")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the sizes, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(**record)


@dataclasses.dataclass(frozen=True, slots=True)
class Choice:
    """A defensible analysis choice declared before exploration (F6).

    Attributes:
        name: The choice (for example "imputation").
        options: The defensible options.
        default: The option used in the primary analysis.
    """

    name: str
    options: tuple[str, ...]
    default: str

    def __post_init__(self) -> None:
        """Validates the choice."""
        evidence.require_text(self.name, "name")
        qualification.require_tuple(self.options, "options")
        if not self.options or len(set(self.options)) != len(self.options):
            raise ValueError("options must be non-empty and distinct")
        if self.default not in self.options:
            raise ValueError("default must be one of the options")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "name": self.name,
            "options": list(self.options),
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the choice, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(
            name=record["name"],
            options=tuple(record["options"]),
            default=record["default"],
        )


def positive(value: Any, name: str) -> float:
    """A finite number strictly greater than zero."""
    number = qualification.finite(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return number


@dataclasses.dataclass(frozen=True, slots=True)
class AnalysisCharter:
    """The signed plan of an analysis, fixed before any exploration.

    Attributes:
        question: The question being answered.
        estimand: The estimand.
        unit: Unit of analysis (for example "customer-month").
        deployment_population: Reference sample URI or stated marginals.
        gating_checks: Checks allowed to gate a run; each needs a decision
            row and a certificate id.
        decision_map: Check to action bindings.
        sets: How development rows are divided.
        min_effect: Smallest standardised effect worth acting on.
        k_max: Most claims the run may carry into confirmation.
        power: Target power at `min_effect`, in (0, 1).
        error_budget: Alpha, exploration wealth, permutations, tolerances.
        multiverse: Defensible choices declared in advance.
        seed: Seed of every random draw the charter governs.
    """

    question: str
    estimand: AnalysisEstimand
    unit: str
    deployment_population: str
    gating_checks: tuple[str, ...]
    decision_map: tuple[DecisionRow, ...]
    sets: SetSizes
    min_effect: float
    k_max: int
    power: float
    error_budget: budget_module.ErrorBudget
    multiverse: tuple[Choice, ...]
    seed: int

    def __post_init__(self) -> None:
        """Validates the charter, including the decision-map lint."""
        evidence.require_text(self.question, "question")
        evidence.require_text(self.unit, "unit")
        evidence.require_text(
            self.deployment_population, "deployment_population"
        )
        for name in ("gating_checks", "decision_map", "multiverse"):
            qualification.require_tuple(getattr(self, name), name)
        positive(self.min_effect, "min_effect")
        qualification.count(self.k_max, "k_max", minimum=1)
        power = qualification.finite(self.power, "power")
        if not 0 < power < 1:
            raise ValueError("power must be strictly between 0 and 1")
        qualification.count(self.seed, "seed")
        names = [c.name for c in self.multiverse]
        if len(set(names)) != len(names):
            raise ValueError("duplicate name in multiverse")
        self.check_decision_map()

    def check_decision_map(self) -> None:
        """Every gating check has exactly one row and a certificate id."""
        rows: dict[str, DecisionRow] = {}
        for row in self.decision_map:
            if row.check_id in rows:
                raise ValueError(f"duplicate decision row for {row.check_id}")
            rows[row.check_id] = row
        for check_id in self.gating_checks:
            bound = rows.get(check_id)
            if bound is None:
                raise ValueError(f"gating check {check_id} has no decision row")
            if not bound.certificate_id.strip():
                raise ValueError(
                    f"gating check {check_id} needs a certificate_id"
                )

    def lint(self, known_checks: Iterable[str]) -> None:
        """Rejects rows or gates that name a check that does not exist.

        Args:
            known_checks: Identifiers of the registered checks.

        Raises:
            ValueError: If any row or gating check is unknown.
        """
        known = set(known_checks)
        for check_id in (
            *(r.check_id for r in self.decision_map),
            *self.gating_checks,
        ):
            if check_id not in known:
                raise ValueError(f"unknown check {check_id} in the charter")

    @property
    def charter_hash(self) -> str:
        """SHA-256 of the canonical JSON of the charter (64 hex digits)."""
        return hashlib.sha256(
            signing.canonical_payload(self.to_dict())
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record; fractions are written as ``"p/q"`` strings."""
        return {
            "question": self.question,
            "estimand": self.estimand.to_dict(),
            "unit": self.unit,
            "deployment_population": self.deployment_population,
            "gating_checks": list(self.gating_checks),
            "decision_map": [r.to_dict() for r in self.decision_map],
            "sets": self.sets.to_dict(),
            "min_effect": self.min_effect,
            "k_max": self.k_max,
            "power": self.power,
            "error_budget": self.error_budget.to_dict(),
            "multiverse": [c.to_dict() for c in self.multiverse],
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the charter, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(
            question=record["question"],
            estimand=AnalysisEstimand.from_dict(record["estimand"]),
            unit=record["unit"],
            deployment_population=record["deployment_population"],
            gating_checks=tuple(record["gating_checks"]),
            decision_map=tuple(
                DecisionRow.from_dict(r) for r in record["decision_map"]
            ),
            sets=SetSizes.from_dict(record["sets"]),
            min_effect=record["min_effect"],
            k_max=record["k_max"],
            power=record["power"],
            error_budget=budget_module.ErrorBudget.from_dict(
                record["error_budget"]
            ),
            multiverse=tuple(Choice.from_dict(c) for c in record["multiverse"]),
            seed=record["seed"],
        )


def template() -> dict[str, Any]:
    """A skeleton charter whose required text fields are blank.

    It deliberately fails validation until the analyst fills it in.
    """
    return {
        "question": "",
        "estimand": {
            "kind": "predictive",
            "population": "",
            "target": "",
            "prediction_moment": "",
            "intercurrent_events": [],
            "summary_measure": "",
        },
        "unit": "",
        "deployment_population": "",
        "gating_checks": [],
        "decision_map": [],
        "sets": {"confirmation_fraction": 0.25, "embargo_rows": 0},
        "min_effect": 0.2,
        "k_max": 10,
        "power": 0.8,
        "error_budget": budget_module.ErrorBudget.create(0.05, 0.1).to_dict(),
        "multiverse": [],
        "seed": 0,
    }


def charter_from_record(record: Mapping[str, Any]) -> AnalysisCharter:
    """Builds a charter, reporting problems as a `ConfigError`."""
    try:
        return AnalysisCharter.from_dict(record)
    except (ValueError, KeyError, TypeError) as error:
        raise errors.ConfigError(
            f"invalid analysis charter: {error}"
        ) from error


def load_json(path: str | Path) -> dict[str, Any]:
    """Reads a JSON object, reporting problems as a `ConfigError`."""
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise errors.ConfigError(
            f"cannot read charter {path}: {error}"
        ) from error
    if not isinstance(record, dict):
        raise errors.ConfigError(f"charter {path} is not a JSON object")
    return record


def read_charter(path: str | Path) -> AnalysisCharter:
    """Reads a charter file, bare or in its signed wrapper (unverified)."""
    record = load_json(path)
    inner = record.get("charter", record)
    return charter_from_record(inner)
