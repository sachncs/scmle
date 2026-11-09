"""Field-level provenance and evidence tiers.

A `FieldProvenance` record says how one field's values come to exist: where
they originate, who writes them, when, whether they are overwritten, and the
as-of rule that says how long before the prediction moment each value is
known. The tier of the as-of rule is:

* `confirmed` only when the as-of rule was derived from a change log and the
  reconstruction from that log agrees with the stored values;
* `attested` when a named person supplied the as-of rule and named evidence a
  third party could check, and the attestation has not expired;
* `unverified` otherwise, including every field whose as-of rule is unknown.

An attestation here is a record with a name and checkable evidence (who
attested, in what role, on what date and for what scope). Its cryptographic
signature is not verified by this module.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import datetime
import enum
import hashlib
import json
from typing import Any
from typing import Self

from scmle.contracts import evidence
from scmle.contracts import qualification

FORMAT_VERSION = 1


class Origin(enum.StrEnum):
    """How a value first comes to exist."""

    INSTRUMENT = "instrument"
    SYSTEM_DEFAULT = "system_default"
    DERIVED = "derived"
    MANUAL = "manual"


class WrittenAt(enum.StrEnum):
    """Clock that stamps the write."""

    EVENT_TIME = "event_time"
    BATCH_TIME = "batch_time"
    UNKNOWN = "unknown"


class Overwritten(enum.StrEnum):
    """Whether a stored value can change after it was first written."""

    NEVER = "never"
    ON_UPDATE = "on_update"
    UNKNOWN = "unknown"


class AsOfKind(enum.StrEnum):
    """How the as-of rule is expressed."""

    TIMESTAMP = "timestamp"
    LAG = "lag"
    UNKNOWN = "unknown"


class AsOfSource(enum.StrEnum):
    """Where the as-of rule came from."""

    LOG = "log"
    ATTESTED = "attested"
    NONE = "none"


class Availability(enum.StrEnum):
    """Whether the field exists at serving time."""

    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


def number(value: Any, name: str) -> float:
    """Check that a value is a finite number.

    Args:
        value: The value to check.
        name: Field name used in the error message.

    Returns:
        `value` as a float. A value that is not a finite number is
        rejected with a `ValueError` from `qualification.finite`.
    """
    return qualification.finite(value, name)


@dataclasses.dataclass(frozen=True, slots=True)
class AsOf:
    """How long before the prediction moment a value is known.

    Attributes:
        kind: `timestamp` (a write time compared with the moment), `lag` (a
            constant delay) or `unknown`.
        lead_seconds: Prediction moment minus the time the value is known,
            in seconds. Positive means known before the moment, negative
            after it. None exactly when `kind` is `unknown`.
    """

    kind: AsOfKind
    lead_seconds: float | None

    def __post_init__(self) -> None:
        """Validates the rule.

        Raises:
            ValueError: If `kind` is not an `AsOfKind`, if an unknown rule
                has a lead, or if a known rule has no finite lead.
        """
        if not isinstance(self.kind, AsOfKind):
            raise ValueError("kind must be an AsOfKind")
        if self.kind is AsOfKind.UNKNOWN:
            if self.lead_seconds is not None:
                raise ValueError("an unknown as-of rule has no lead")
        else:
            if self.lead_seconds is None:
                raise ValueError("a known as-of rule needs lead_seconds")
            number(self.lead_seconds, "lead_seconds")

    @classmethod
    def unknown(cls) -> Self:
        """Builds the rule of a field nobody has stated or derived.

        Returns:
            A rule of kind `unknown` with no lead.
        """
        return cls(AsOfKind.UNKNOWN, None)

    @property
    def is_known(self) -> bool:
        """Whether a lead is stated."""
        return self.kind is not AsOfKind.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        """Converts the rule to a JSON-ready record.

        Returns:
            A dict with `kind` (its string value) and `lead_seconds`.
        """
        return {"kind": self.kind.value, "lead_seconds": self.lead_seconds}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a rule from a record, rejecting unknown or missing fields.

        Args:
            record: A mapping with exactly `kind` and `lead_seconds`.

        Returns:
            The rule. Unexpected or missing fields, an unknown `kind` and
            an invalid rule are rejected with a `ValueError`.
        """
        qualification.require_fields(cls, record)
        return cls(AsOfKind(record["kind"]), record["lead_seconds"])


def parse_date(value: str, name: str) -> datetime.date:
    """Parses an ISO calendar date.

    Args:
        value: Text in the form YYYY-MM-DD.
        name: Field name used in the error message.

    Returns:
        The date.

    Raises:
        ValueError: If `value` is not an ISO date.
    """
    try:
        return datetime.date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD)") from error


@dataclasses.dataclass(frozen=True, slots=True)
class Attestation:
    """A named person's statement about a field's origin and as-of rule.

    Attributes:
        name_or_id: Who made the statement.
        role: Their role (for example "data owner").
        statement: Origin, as-of and meaning, in words.
        evidence: A log table, DDL, ticket or document a third party could
            check.
        date: ISO date of the statement.
        scope: The dataset version (hash) the statement covers.
        expires: ISO date after which the statement no longer counts.
    """

    name_or_id: str
    role: str
    statement: str
    evidence: str
    date: str
    scope: str
    expires: str

    def __post_init__(self) -> None:
        """Validates the record.

        Raises:
            ValueError: If a text field is empty, a date is not an ISO date,
                or `expires` precedes `date`.
        """
        for name in (
            "name_or_id",
            "role",
            "statement",
            "evidence",
            "scope",
        ):
            evidence.require_text(getattr(self, name), name)
        issued = parse_date(self.date, "date")
        if parse_date(self.expires, "expires") < issued:
            raise ValueError("expires must not precede date")

    def valid_on(self, day: datetime.date) -> bool:
        """Tests whether the statement counts on a day.

        Args:
            day: The day to test.

        Returns:
            True if `day` lies between `date` and `expires`, both included.
        """
        issued = parse_date(self.date, "date")
        return issued <= day <= parse_date(self.expires, "expires")

    def to_dict(self) -> dict[str, Any]:
        """Converts the record to a JSON-ready dict.

        Returns:
            A dict of the record's fields and their values.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the record, rejecting unknown or missing fields.

        Args:
            record: A mapping holding exactly the record's fields.

        Returns:
            The record. Unexpected or missing fields and invalid values are
            rejected with a `ValueError`.
        """
        qualification.require_fields(cls, record)
        return cls(**record)


@dataclasses.dataclass(frozen=True, slots=True)
class LogCheck:
    """Result of reconstructing a field from a change log.

    Attributes:
        rows: Rows compared.
        mismatched: Rows whose reconstruction differs from the stored value.
        tolerance: Largest mismatch share the field may have.
    """

    rows: int
    mismatched: int
    tolerance: float

    def __post_init__(self) -> None:
        """Validates the record.

        Raises:
            ValueError: If `rows` is not a positive integer, `mismatched` is
                not a non-negative integer or exceeds `rows`, or `tolerance`
                is outside [0, 1].
        """
        qualification.count(self.rows, "rows", minimum=1)
        qualification.count(self.mismatched, "mismatched")
        if self.mismatched > self.rows:
            raise ValueError("mismatched must not exceed rows")
        qualification.probability(self.tolerance, "tolerance")

    @property
    def rate(self) -> float:
        """Share of rows that disagree."""
        return self.mismatched / self.rows

    @property
    def passed(self) -> bool:
        """Whether the disagreement share is within the tolerance."""
        return self.rate <= self.tolerance

    def to_dict(self) -> dict[str, Any]:
        """Converts the record to a JSON-ready dict.

        Returns:
            A dict of the record's fields and their values.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the record, rejecting unknown or missing fields.

        Args:
            record: A mapping holding exactly the record's fields.

        Returns:
            The record. Unexpected or missing fields and invalid values are
            rejected with a `ValueError`.
        """
        qualification.require_fields(cls, record)
        return cls(**record)


@dataclasses.dataclass(frozen=True, slots=True)
class FieldProvenance:
    """How one field's values come to exist.

    Attributes:
        name: Column name.
        origin: How values first come to exist.
        writer: System or person that writes the value.
        written_at: Clock that stamps the write.
        overwritten: Whether stored values can change later.
        definition_history: Dated notes on changes to the definition.
        as_of: The as-of rule.
        as_of_source: `log`, `attested` or `none`; `none` requires an
            unknown rule and a known rule requires a source.
        available_at_serving: Whether the field exists when predicting.
        attestation: Required when `as_of_source` is `attested`.
        log_check: Required when `as_of_source` is `log`.
    """

    name: str
    origin: Origin
    writer: str
    written_at: WrittenAt
    overwritten: Overwritten
    definition_history: tuple[str, ...]
    as_of: AsOf
    as_of_source: AsOfSource
    available_at_serving: Availability
    attestation: Attestation | None = None
    log_check: LogCheck | None = None

    def __post_init__(self) -> None:
        """Validates the record.

        Raises:
            ValueError: If `name` or `writer` is empty, a definition note is
                empty, `definition_history` is not a tuple, an enum field has
                the wrong type, or the as-of rule, its source, the
                attestation and the log check are inconsistent.
        """
        evidence.require_text(self.name, "name")
        evidence.require_text(self.writer, "writer")
        qualification.require_tuple(
            self.definition_history, "definition_history"
        )
        for note in self.definition_history:
            evidence.require_text(note, "definition_history")
        for value, kind in (
            (self.origin, Origin),
            (self.written_at, WrittenAt),
            (self.overwritten, Overwritten),
            (self.as_of_source, AsOfSource),
            (self.available_at_serving, Availability),
            (self.as_of, AsOf),
        ):
            if not isinstance(value, kind):
                raise ValueError(f"expected a {kind.__name__}")
        known = self.as_of.is_known
        if known and self.as_of_source is AsOfSource.NONE:
            raise ValueError("a known as-of rule needs a source")
        if not known and self.as_of_source is not AsOfSource.NONE:
            raise ValueError("an unknown as-of rule cannot have a source")
        if (
            self.as_of_source is AsOfSource.ATTESTED
            and self.attestation is None
        ):
            raise ValueError("an attested as-of rule needs an attestation")
        if self.as_of_source is AsOfSource.LOG and self.log_check is None:
            raise ValueError("a log-derived as-of rule needs a log_check")

    def tier(self, today: datetime.date | None = None) -> qualification.Tier:
        """Evidence tier of the field's as-of rule.

        Args:
            today: Day on which an attestation must still be valid; None
                skips the expiry test.

        Returns:
            `confirmed` for a log-derived rule whose log check passed;
            `attested` for an unexpired attestation, or for a log-derived
            rule that also carries one; `unverified` otherwise.
        """
        attested = self.attestation is not None and (
            today is None or self.attestation.valid_on(today)
        )
        if self.as_of_source is AsOfSource.LOG:
            if self.log_check is not None and self.log_check.passed:
                return qualification.Tier.CONFIRMED
            return (
                qualification.Tier.ATTESTED
                if attested
                else qualification.Tier.UNVERIFIED
            )
        if self.as_of_source is AsOfSource.ATTESTED and attested:
            return qualification.Tier.ATTESTED
        return qualification.Tier.UNVERIFIED

    def to_dict(self) -> dict[str, Any]:
        """Converts the record to a JSON-ready dict.

        Returns:
            A dict of the fields; enums as strings, nested records as dicts
            (None when absent) and the history as a list.
        """
        return {
            "name": self.name,
            "origin": self.origin.value,
            "writer": self.writer,
            "written_at": self.written_at.value,
            "overwritten": self.overwritten.value,
            "definition_history": list(self.definition_history),
            "as_of": self.as_of.to_dict(),
            "as_of_source": self.as_of_source.value,
            "available_at_serving": self.available_at_serving.value,
            "attestation": None
            if self.attestation is None
            else self.attestation.to_dict(),
            "log_check": None
            if self.log_check is None
            else self.log_check.to_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild a record, rejecting unknown or missing fields.

        Args:
            record: A mapping in the form `to_dict` returns.

        Returns:
            The record. Unexpected or missing fields, an unknown enum
            value and an inconsistent record are rejected with a
            `ValueError`.
        """
        qualification.require_fields(cls, record)
        attestation = record["attestation"]
        log_check = record["log_check"]
        return cls(
            name=record["name"],
            origin=Origin(record["origin"]),
            writer=record["writer"],
            written_at=WrittenAt(record["written_at"]),
            overwritten=Overwritten(record["overwritten"]),
            definition_history=tuple(record["definition_history"]),
            as_of=AsOf.from_dict(record["as_of"]),
            as_of_source=AsOfSource(record["as_of_source"]),
            available_at_serving=Availability(record["available_at_serving"]),
            attestation=None
            if attestation is None
            else Attestation.from_dict(attestation),
            log_check=None
            if log_check is None
            else LogCheck.from_dict(log_check),
        )


def undeclared(name: str) -> FieldProvenance:
    """The record of a field nobody described: everything unknown.

    Args:
        name: Column name.

    Returns:
        A record whose tier is `unverified`.
    """
    return FieldProvenance(
        name=name,
        origin=Origin.DERIVED,
        writer="undeclared",
        written_at=WrittenAt.UNKNOWN,
        overwritten=Overwritten.UNKNOWN,
        definition_history=(),
        as_of=AsOf.unknown(),
        as_of_source=AsOfSource.NONE,
        available_at_serving=Availability.UNKNOWN,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ProvenanceTable:
    """The data dictionary written as `field_provenance.json`.

    Attributes:
        fields: One record per field, in a fixed order, names unique.
    """

    fields: tuple[FieldProvenance, ...]

    def __post_init__(self) -> None:
        """Validates the table.

        Raises:
            ValueError: If `fields` is not a tuple or names repeat.
        """
        qualification.require_tuple(self.fields, "fields")
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ValueError("field names must be unique")

    @classmethod
    def covering(
        cls,
        columns: Sequence[str],
        declared: Sequence[FieldProvenance] = (),
    ) -> Self:
        """A table with a record for every column.

        Args:
            columns: Every column of the data.
            declared: Records someone supplied; each must name a column.

        Returns:
            Declared records, plus `undeclared` records for the rest, in
            column order.

        Raises:
            ValueError: If a declared record names no column or repeats.
        """
        by_name = {f.name: f for f in declared}
        if len(by_name) != len(declared):
            raise ValueError("declared field names must be unique")
        unknown = sorted(set(by_name) - set(columns))
        if unknown:
            raise ValueError(f"declared fields not in the data: {unknown}")
        return cls(tuple(by_name.get(c) or undeclared(c) for c in columns))

    def get(self, name: str) -> FieldProvenance:
        """Looks up the record of a field.

        Args:
            name: Column name.

        Returns:
            The record for `name`.

        Raises:
            KeyError: If the table has no such field.
        """
        for item in self.fields:
            if item.name == name:
                return item
        raise KeyError(name)

    def tiers(
        self, today: datetime.date | None = None
    ) -> dict[str, qualification.Tier]:
        """Computes the evidence tier of every field.

        Args:
            today: Day on which attestations must still be valid; None
                skips the expiry test.

        Returns:
            A mapping from field name to tier, in field order.
        """
        return {f.name: f.tier(today) for f in self.fields}

    def to_dict(self) -> dict[str, Any]:
        """Converts the table to a JSON-ready document.

        Returns:
            A dict with `format_version` and the list of field records.
        """
        return {
            "format_version": FORMAT_VERSION,
            "fields": [f.to_dict() for f in self.fields],
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuild a table, rejecting unknown versions and fields.

        Args:
            document: A mapping in the form `to_dict` returns.

        Returns:
            The table.

        Raises:
            ValueError: If `format_version` is not supported, the document
                has keys other than `format_version` and `fields`, or a
                field record is invalid.
        """
        if document.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"unsupported format_version: {document.get('format_version')}"
            )
        extra = sorted(set(document) - {"format_version", "fields"})
        if extra:
            raise ValueError(f"unexpected field(s): {extra}")
        return cls(
            tuple(FieldProvenance.from_dict(f) for f in document["fields"])
        )

    def digest(self) -> str:
        """Hashes the table.

        Returns:
            The hex SHA-256 of the table's JSON with sorted keys and no
            whitespace.
        """
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
