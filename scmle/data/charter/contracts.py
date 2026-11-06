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
from scmle.data.charter import causal as causal_module
from scmle.data.charter import decision as decision_module
from scmle.data.charter import ingest as ingest_module
from scmle.data.charter import rules as rules_module
from scmle.data.framing import costs
from scmle.data.ledger import budget as budget_module

FORMAT_VERSION = 9
OLDEST_READABLE_FORMAT = 1
ESTIMAND_KINDS = ("predictive", "causal")
MAX_CONFIRMATION_FRACTION = 0.5


NEW_ENTITY = "new_entity"
KNOWN_ENTITY_FUTURE = "known_entity_future"
UNITS_OF_PREDICTION = (NEW_ENTITY, KNOWN_ENTITY_FUTURE)


class DecisionAction(enum.StrEnum):
    """Closed set of actions a check may trigger."""

    DROP_COLUMN = "drop_column"
    QUARANTINE_COLUMN = "quarantine_column"
    REWEIGHT = "reweight"
    SWITCH_SPLIT = "switch_split"
    BLOCK_RUN = "block_run"
    DOWNGRADE_TIER = "downgrade_tier"
    REFUSE_CLAIM = "refuse_claim"


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
        causal: The causal block (format 4); None for a predictive estimand
            and for every charter written before format 4, which serialise
            and hash as they did.
    """

    kind: str
    population: str
    target: str
    prediction_moment: str
    intercurrent_events: tuple[IntercurrentEvent, ...]
    summary_measure: str
    causal: causal_module.CausalSpec | None = None

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
        if self.causal is not None and self.kind != "causal":
            raise ValueError("a causal block needs kind='causal'")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        record: dict[str, Any] = {
            "kind": self.kind,
            "population": self.population,
            "target": self.target,
            "prediction_moment": self.prediction_moment,
            "intercurrent_events": [
                e.to_dict() for e in self.intercurrent_events
            ],
            "summary_measure": self.summary_measure,
        }
        if self.causal is not None:
            record["causal"] = self.causal.to_dict()
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the estimand, rejecting unknown or missing fields."""
        record = {"causal": None, **record}
        qualification.require_fields(cls, record)
        block = record["causal"]
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
            causal=(
                None
                if block is None
                else causal_module.CausalSpec.from_dict(block)
            ),
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
        cost_model: Costs of acting and not acting (F8), fixed before `C`
            is opened; None for a charter that does not claim a decision
            benefit. Format 1 charters have none and load unchanged.
        rules: Pre-registered ledger rule, confirmation rule, failure
            policy and stability thresholds (format 3); None for a charter
            that does not fix them, which serialises, hashes and loads
            exactly as a format 1 or 2 charter did.
        decision: The decision layer (format 5): how the cost model is used,
            fairness constraints and the rollout plan; None for a charter
            that makes no decision claim. It needs a `cost_model`.
        unit_of_prediction: What a deployed prediction is for (format 6):
            `new_entity` (a row of an entity not seen in training) or
            `known_entity_future` (a later row of a known entity); None when
            the charter does not say, which serialises, hashes and loads as
            a format 1-5 charter did. `new_entity` needs a group column in
            the data and makes interval calibration use one score per entity.
        ingest: The ingestion block (format 7): source budget, `B_feat`,
            windows, lag, reconciliation delay and key tolerances; None for
            a charter that reads a file, which serialises, hashes and loads
            as a format 1-6 charter did.
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
    cost_model: costs.DecisionCostModel | None = None
    rules: rules_module.RunRules | None = None
    decision: decision_module.DecisionLayer | None = None
    unit_of_prediction: str | None = None
    ingest: ingest_module.IngestSpec | None = None

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
        if self.decision is not None and self.cost_model is None:
            raise ValueError("a decision layer needs a cost_model")
        if (
            self.unit_of_prediction is not None
            and self.unit_of_prediction not in UNITS_OF_PREDICTION
        ):
            raise ValueError(
                f"unit_of_prediction must be one of {UNITS_OF_PREDICTION}"
            )

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
        record = {
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
        if self.cost_model is not None:
            record["cost_model"] = self.cost_model.to_dict()
        if self.rules is not None:
            record["rules"] = self.rules.to_dict()
        if self.decision is not None:
            record["decision"] = self.decision.to_dict()
        if self.unit_of_prediction is not None:
            record["unit_of_prediction"] = self.unit_of_prediction
        if self.ingest is not None:
            record["ingest"] = self.ingest.to_dict()
        return record

    @property
    def format_version(self) -> int:
        """Oldest format that carries this charter.

        2; 3 with rules; 4 with a causal block; 5 with a decision layer; 6
        with a unit of prediction; 7 with an ingest block; 8 when the ingest
        block carries an entity or reconcile sub-block; 9 when the causal
        block carries subgroups or attestors.
        """
        found = 2 if self.rules is None else 3
        causal_block = self.estimand.causal
        if causal_block is not None:
            extended = (
                causal_block.subgroups
                or causal_block.attestors
                or causal_block.attestations
            )
            found = max(found, 9 if extended else 4)
        if self.decision is not None:
            found = max(found, 5)
        if self.unit_of_prediction is not None:
            found = max(found, 6)
        if self.ingest is not None:
            sub = self.ingest.entity or self.ingest.reconcile
            found = max(found, 7 if sub is None else 8)
        return found

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the charter, rejecting unknown or missing fields."""
        record = {
            "cost_model": None,
            "rules": None,
            "decision": None,
            "unit_of_prediction": None,
            "ingest": None,
            **record,
        }
        qualification.require_fields(cls, record)
        cost = record["cost_model"]
        chosen = record["rules"]
        layer = record["decision"]
        block = record["ingest"]
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
            cost_model=(
                None
                if cost is None
                else costs.DecisionCostModel.from_dict(cost)
            ),
            rules=(
                None
                if chosen is None
                else rules_module.RunRules.from_dict(chosen)
            ),
            decision=(
                None
                if layer is None
                else decision_module.DecisionLayer.from_dict(layer)
            ),
            unit_of_prediction=record["unit_of_prediction"],
            ingest=(
                None
                if block is None
                else ingest_module.IngestSpec.from_dict(block)
            ),
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
