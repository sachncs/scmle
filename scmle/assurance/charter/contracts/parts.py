"""The parts of a charter: estimand, decision rows, set sizes, choices."""

from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import enum
from typing import Any
from typing import ClassVar
from typing import Self

import scmle.assurance.charter.causal
from scmle.contracts import errors
from scmle.contracts import evidence
from scmle.contracts import qualification
from scmle.contracts import records

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

    migrations: ClassVar[tuple[records.Migration, ...]] = ()

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
        return cls(**qualification.require(cls, record))


@dataclasses.dataclass(frozen=True, slots=True)
class Estimand:
    """What the analysis estimates, in terms of the ICH E9 addendum.

    Attributes:
        kind: `predictive` or `causal`; selects the admissibility rule.
        population: Who or what the claim is about.
        target: The variable, with its measurement window.
        moment: When a prediction is issued (the as-of rule).
        intercurrent: Intercurrent events and the strategy for each (at least
            one; declare "none" with strategy "not applicable" if there are
            none).
        summary: How the target is summarised (for example "ROC AUC").
        causal: The causal block (format 4); None for a predictive estimand
            and for every charter written before format 4, which serialise
            and hash as they did.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            renames={
                "prediction_moment": "moment",
                "intercurrent_events": "intercurrent",
                "summary_measure": "summary",
            },
            parts={
                "causal": scmle.assurance.charter.causal.Question.migrations[0]
            },
        ),
    )

    kind: str
    population: str
    target: str
    moment: str
    intercurrent: tuple[Event, ...]
    summary: str
    causal: scmle.assurance.charter.causal.Question | None = None

    def __post_init__(self) -> None:
        """Validates the estimand.

        Raises:
            ValueError: If `kind` is not in `ESTIMAND_KINDS`, `population`,
                `target`, `moment` or `summary` is blank, `intercurrent` is
                not a tuple or is empty, or a causal block is given and
                `kind` is not `causal`.
        """
        if self.kind not in ESTIMAND_KINDS:
            raise errors.InputError(f"kind must be one of {ESTIMAND_KINDS}")
        evidence.require_text(self.population, "population")
        evidence.require_text(self.target, "target")
        evidence.require_text(self.moment, "moment")
        evidence.require_text(self.summary, "summary")
        qualification.tupled(self.intercurrent, "intercurrent")
        if not self.intercurrent:
            raise errors.InputError(
                "intercurrent must not be empty: state a strategy "
                "for each event, or declare 'none' as not applicable"
            )
        if self.causal is not None and self.kind != "causal":
            raise errors.InputError("a causal block needs kind='causal'")

    def to_dict(self) -> dict[str, Any]:
        """Serializes the estimand.

        Returns:
            A JSON-ready dict of the fields, with `intercurrent` as a list of
            dicts. `causal` is included, as a dict, only when set.
        """
        record: dict[str, Any] = {
            "kind": self.kind,
            "population": self.population,
            "target": self.target,
            "moment": self.moment,
            "intercurrent": [e.to_dict() for e in self.intercurrent],
            "summary": self.summary,
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
        record = qualification.require(
            cls,
            {
                "causal": None,
                **records.load(cls.__name__, cls.migrations, record),
            },
        )
        block = record["causal"]
        return cls(
            kind=record["kind"],
            population=record["population"],
            target=record["target"],
            moment=record["moment"],
            intercurrent=tuple(
                Event.from_dict(e) for e in record["intercurrent"]
            ),
            summary=record["summary"],
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
        check: The check.
        trigger: A test outcome at the budgeted level (not a raw threshold).
        action: What happens when the trigger fires.
        certificate: The detector certificate that licenses gating
            (required for every gating check; see the charter lint).
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={"check_id": "check", "certificate_id": "certificate"}
        ),
    )

    check: str
    trigger: str
    action: Edict
    certificate: str = ""

    def __post_init__(self) -> None:
        """Validates the row.

        Raises:
            ValueError: If `check` or `trigger` is blank, or `action` is
                not an `Edict`.
        """
        evidence.require_text(self.check, "check")
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
            "check": self.check,
            "trigger": self.trigger,
            "action": self.action.value,
            "certificate": self.certificate,
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
        record = qualification.require(cls, record)
        return cls(
            check=record["check"],
            trigger=record["trigger"],
            action=Edict(record["action"]),
            certificate=record["certificate"],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Sizes:
    """How the development rows are divided.

    Attributes:
        confirmation: Share of development rows carved out as the
            confirmation set `C`, in (0, 0.5].
        embargo: Rows dropped around temporal confirmation blocks so serial
            dependence cannot cross the boundary.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "confirmation_fraction": "confirmation",
                "embargo_rows": "embargo",
            }
        ),
    )

    confirmation: float
    embargo: int = 0

    def __post_init__(self) -> None:
        """Validates the sizes.

        Raises:
            ValueError: If `confirmation` is not finite or is outside (0,
                `MAX_CONFIRMATION_FRACTION`], or `embargo` is not a
                non-negative integer.
        """
        fraction = qualification.finite(self.confirmation, "confirmation")
        if not 0 < fraction <= MAX_CONFIRMATION_FRACTION:
            raise errors.InputError(
                "confirmation must be in (0, "
                f"{MAX_CONFIRMATION_FRACTION}] so exploration keeps a majority"
            )
        qualification.count(self.embargo, "embargo")

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
        return cls(**qualification.require(cls, record))


@dataclasses.dataclass(frozen=True, slots=True)
class Choice:
    """A defensible analysis choice declared before exploration.

    Attributes:
        name: The choice (for example "imputation").
        options: The defensible options.
        default: The option used in the primary analysis.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = ()

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
        record = qualification.require(cls, record)
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
