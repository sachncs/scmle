"""The parts of a charter: estimand, decision rows, set sizes, choices."""

from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import enum
from typing import Any
from typing import Self

import scmle.assurance.charter.causal
from scmle.contracts import errors
from scmle.contracts import evidence
from scmle.contracts import qualification

FORMAT_VERSION = 11
OLDEST_READABLE_FORMAT = 1
ESTIMAND_KINDS = ("predictive", "causal")
MAX_CONFIRMATION_FRACTION = 0.5


NEW_ENTITY = "new_entity"
KNOWN_ENTITY_FUTURE = "known_entity_future"
UNITS_OF_PREDICTION = (NEW_ENTITY, KNOWN_ENTITY_FUTURE)


class Edict(enum.StrEnum):
    """Closed set of actions a check may trigger."""

    DROP_COLUMN = "drop_column"
    QUARANTINE_COLUMN = "quarantine_column"
    REWEIGHT = "reweight"
    SWITCH_SPLIT = "switch_split"
    BLOCK_RUN = "block_run"
    DOWNGRADE_TIER = "downgrade_tier"
    REFUSE_CLAIM = "refuse_claim"


def texts(values: Iterable[Any], name: str) -> tuple[str, ...]:
    """Converts values to text and requires each to be non-blank.

    A blank value raises `ValueError` (from `evidence.require_text`).

    Args:
        values: The items; each is converted with `str`.
        name: Field name used in the error message.

    Returns:
        The stripped strings, in the order of `values`.
    """
    return tuple(evidence.require_text(str(v), name) for v in values)


@dataclasses.dataclass(frozen=True, slots=True)
class Event:
    """An event after the start that affects the target, and its strategy.

    Attributes:
        event: What happens (for example "account closed").
        strategy: How it is handled (censor, exclude, treat as outcome, ...).
    """

    event: str
    strategy: str

    def __post_init__(self) -> None:
        """Validates the record.

        A blank `event` or `strategy` raises `ValueError` (from
        `evidence.require_text`).
        """
        evidence.require_text(self.event, "event")
        evidence.require_text(self.strategy, "strategy")

    def to_dict(self) -> dict[str, Any]:
        """Serializes the record.

        Returns:
            A JSON-ready dict of the record's fields.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the event from a dict.

        Keys other than the fields, and missing fields, are rejected with
        the `ValueError` of `qualification.require`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt event.
        """
        qualification.require(cls, record)
        return cls(**record)


@dataclasses.dataclass(frozen=True, slots=True)
class AnalysisEstimand:
    """What the analysis estimates, in terms of the ICH E9 addendum.

    Attributes:
        kind: `predictive` or `causal`; selects the admissibility rule.
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
    intercurrent_events: tuple[Event, ...]
    summary_measure: str
    causal: scmle.assurance.charter.causal.Question | None = None

    def __post_init__(self) -> None:
        """Validates the estimand.

        Raises:
            ValueError: If `kind` is not in `ESTIMAND_KINDS`, `population`,
                `target`, `prediction_moment` or `summary_measure` is
                blank, `intercurrent_events` is not a tuple or is empty, or
                a causal block is given and `kind` is not `causal`.
        """
        if self.kind not in ESTIMAND_KINDS:
            raise errors.InputError(f"kind must be one of {ESTIMAND_KINDS}")
        evidence.require_text(self.population, "population")
        evidence.require_text(self.target, "target")
        evidence.require_text(self.prediction_moment, "prediction_moment")
        evidence.require_text(self.summary_measure, "summary_measure")
        qualification.tupled(self.intercurrent_events, "intercurrent_events")
        if not self.intercurrent_events:
            raise errors.InputError(
                "intercurrent_events must not be empty: state a strategy "
                "for each event, or declare 'none' as not applicable"
            )
        if self.causal is not None and self.kind != "causal":
            raise errors.InputError("a causal block needs kind='causal'")

    def to_dict(self) -> dict[str, Any]:
        """Serializes the estimand.

        Returns:
            A JSON-ready dict of the fields, with `intercurrent_events` as a
            list of dicts. `causal` is included, as a dict, only when set.
        """
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
        """Rebuilds the estimand from a dict.

        `causal` may be absent and then defaults to None. Other missing
        fields, and keys that are not fields, are rejected with the
        `ValueError` of `qualification.require`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt estimand.
        """
        record = {"causal": None, **record}
        qualification.require(cls, record)
        block = record["causal"]
        return cls(
            kind=record["kind"],
            population=record["population"],
            target=record["target"],
            prediction_moment=record["prediction_moment"],
            intercurrent_events=tuple(
                Event.from_dict(e) for e in record["intercurrent_events"]
            ),
            summary_measure=record["summary_measure"],
            causal=(
                None
                if block is None
                else scmle.assurance.charter.causal.Question.from_dict(block)
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Trigger:
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
    action: Edict
    certificate_id: str = ""

    def __post_init__(self) -> None:
        """Validates the row.

        Raises:
            ValueError: If `check_id` or `trigger` is blank, or `action` is
                not an `Edict`.
        """
        evidence.require_text(self.check_id, "check_id")
        evidence.require_text(self.trigger, "trigger")
        if not isinstance(self.action, Edict):
            raise errors.InputError("action must be an Edict")

    def to_dict(self) -> dict[str, Any]:
        """Serializes the row.

        Returns:
            A JSON-ready dict of the fields, with `action` as its string
            value.
        """
        return {
            "check_id": self.check_id,
            "trigger": self.trigger,
            "action": self.action.value,
            "certificate_id": self.certificate_id,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the row from a dict.

        Keys other than the fields, and missing fields, are rejected with
        the `ValueError` of `qualification.require`; an `action`
        that is not an `Edict` value also raises `ValueError`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt row.
        """
        qualification.require(cls, record)
        return cls(
            check_id=record["check_id"],
            trigger=record["trigger"],
            action=Edict(record["action"]),
            certificate_id=record["certificate_id"],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class SetSizes:
    """How the development rows are divided.

    Attributes:
        confirmation_fraction: Share of development rows carved out as the
            confirmation set `C`, in (0, 0.5].
        embargo_rows: Rows dropped around temporal confirmation blocks so
            serial dependence cannot cross the boundary.
    """

    confirmation_fraction: float
    embargo_rows: int = 0

    def __post_init__(self) -> None:
        """Validates the sizes.

        Raises:
            ValueError: If `confirmation_fraction` is not finite or is
                outside (0, `MAX_CONFIRMATION_FRACTION`], or `embargo_rows`
                is not a non-negative integer.
        """
        fraction = qualification.finite(
            self.confirmation_fraction, "confirmation_fraction"
        )
        if not 0 < fraction <= MAX_CONFIRMATION_FRACTION:
            raise errors.InputError(
                "confirmation_fraction must be in (0, "
                f"{MAX_CONFIRMATION_FRACTION}] so exploration keeps a majority"
            )
        qualification.count(self.embargo_rows, "embargo_rows")

    def to_dict(self) -> dict[str, Any]:
        """Serializes the record.

        Returns:
            A JSON-ready dict of the record's fields.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the sizes from a dict.

        Keys other than the fields, and missing fields, are rejected with
        the `ValueError` of `qualification.require`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt sizes.
        """
        qualification.require(cls, record)
        return cls(**record)


@dataclasses.dataclass(frozen=True, slots=True)
class Choice:
    """A defensible analysis choice declared before exploration.

    Attributes:
        name: The choice (for example "imputation").
        options: The defensible options.
        default: The option used in the primary analysis.
    """

    name: str
    options: tuple[str, ...]
    default: str

    def __post_init__(self) -> None:
        """Validates the choice.

        Raises:
            ValueError: If `name` is blank, `options` is not a tuple, is
                empty or has a repeat, or `default` is not one of the
                options.
        """
        evidence.require_text(self.name, "name")
        qualification.tupled(self.options, "options")
        if not self.options or len(set(self.options)) != len(self.options):
            raise errors.InputError("options must be non-empty and distinct")
        if self.default not in self.options:
            raise errors.InputError("default must be one of the options")

    def to_dict(self) -> dict[str, Any]:
        """Serializes the choice.

        Returns:
            A JSON-ready dict of the fields, with `options` as a list.
        """
        return {
            "name": self.name,
            "options": list(self.options),
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the choice from a dict.

        Keys other than the fields, and missing fields, are rejected with
        the `ValueError` of `qualification.require`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt choice.
        """
        qualification.require(cls, record)
        return cls(
            name=record["name"],
            options=tuple(record["options"]),
            default=record["default"],
        )


def positive(value: Any, name: str) -> float:
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
