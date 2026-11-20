"""Field-level provenance and evidence tiers.

A `Pedigree` record says how one field's values come to exist: where
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

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import datetime
import enum
import hashlib
import json
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.contracts import errors
from scmle.contracts import evidence
from scmle.contracts import qualification
from scmle.contracts import records

LEGACY = "format_version"


class Origin(enum.StrEnum):
    """How a value first comes to exist."""

    INSTRUMENT = "instrument"
    SYSTEM_DEFAULT = "system_default"
    DERIVED = "derived"
    MANUAL = "manual"


class Stamp(enum.StrEnum):
    """Clock that stamps the write."""

    EVENT_TIME = "event_time"
    BATCH_TIME = "batch_time"
    UNKNOWN = "unknown"


class Overwritten(enum.StrEnum):
    """Whether a stored value can change after it was first written."""

    NEVER = "never"
    ON_UPDATE = "on_update"
    UNKNOWN = "unknown"


class Cutoff(enum.StrEnum):
    """How the as-of rule is expressed."""

    TIMESTAMP = "timestamp"
    LAG = "lag"
    UNKNOWN = "unknown"


class Provenance(enum.StrEnum):
    """Where the as-of rule came from."""

    LOG = "log"
    ATTESTED = "attested"
    NONE = "none"


class Availability(enum.StrEnum):
    """Whether the field exists at serving time."""

    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


@dataclasses.dataclass(frozen=True, slots=True)
class Lead:
    """How long before the prediction moment a value is known.

    Attributes:
        kind: `timestamp` (a write time compared with the moment), `lag` (a
            constant delay) or `unknown`.
        seconds: Prediction moment minus the time the value is known,
            in seconds. Positive means known before the moment, negative
            after it. None exactly when `kind` is `unknown`.
        migrations: The layout's version steps, oldest first; version 1
            spelled `seconds` as `lead_seconds`. A rule is stored inside a
            `Pedigree`, whose version it follows.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(renames={"lead_seconds": "seconds"}),
    )

    kind: Cutoff
    seconds: float | None

    def __post_init__(self) -> None:
        """Validates the rule.

        Raises:
            ValueError: If `kind` is not an `Cutoff`, if an unknown rule
                has a lead, or if a known rule has no finite lead.
        """
        if not isinstance(self.kind, Cutoff):
            raise errors.InputError("kind must be an Cutoff")
        if self.kind is Cutoff.UNKNOWN:
            if self.seconds is not None:
                raise errors.InputError("an unknown as-of rule has no lead")
        else:
            if self.seconds is None:
                raise errors.InputError("a known as-of rule needs seconds")
            qualification.finite(self.seconds, "seconds")

    @classmethod
    def unknown(cls) -> Self:
        """Builds the rule of a field nobody has stated or derived.

        Returns:
            A rule of kind `unknown` with no lead.
        """
        return cls(Cutoff.UNKNOWN, None)

    @property
    def given(self) -> bool:
        """Whether a lead is stated."""
        return self.kind is not Cutoff.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        """Converts the rule to a JSON-ready record.

        Returns:
            A dict with `kind` (its string value) and `seconds`.
        """
        return {"kind": self.kind.value, "seconds": self.seconds}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a rule from a record, rejecting unknown or missing fields.

        Args:
            record: A mapping with exactly `kind` and `seconds` (version 1:
                `lead_seconds`).

        Returns:
            The rule. Unexpected or missing fields, an unknown `kind` and
            an invalid rule are rejected with a `ValueError`.
        """
        layout = qualification.require(cls, record)
        return cls(Cutoff(layout["kind"]), layout["seconds"])


def parse(value: str, name: str) -> datetime.date:
    """Parses an ISO calendar date.

    Args:
        value: ISO 8601 date text, normally YYYY-MM-DD (any form that
            `datetime.date.fromisoformat` accepts).
        name: Field name used in the error message.

    Returns:
        The date.

    Raises:
        ValueError: If `value` is not an ISO date.
    """
    try:
        return datetime.date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise errors.InputError(
            f"{name} must be an ISO date (YYYY-MM-DD)"
        ) from error


@dataclasses.dataclass(frozen=True, slots=True)
class Attestation:
    """A named person's statement about a field's origin and as-of rule.

    Attributes:
        attestor: Who made the statement.
        role: Their role (for example "data owner").
        statement: Origin, as-of and meaning, in words.
        evidence: A log table, DDL, ticket or document a third party could
            check.
        date: ISO date of the statement.
        scope: The dataset version (hash) the statement covers.
        expires: ISO date after which the statement no longer counts.
        migrations: The layout's version steps, oldest first; version 1
            spelled `attestor` as `name_or_id`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(renames={"name_or_id": "attestor"}),
    )

    attestor: str
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
            "attestor",
            "role",
            "statement",
            "evidence",
            "scope",
        ):
            if not isinstance(getattr(self, name), str):
                raise errors.InputError(f"{name} must be text")
            evidence.require(getattr(self, name), name)
        issued = parse(self.date, "date")
        if parse(self.expires, "expires") < issued:
            raise errors.InputError("expires must not precede date")

    def effective(self, day: datetime.date) -> bool:
        """Tests whether the statement counts on a day.

        Args:
            day: The day to test.

        Returns:
            True if `day` lies between `date` and `expires`, both included.
        """
        issued = parse(self.date, "date")
        return issued <= day <= parse(self.expires, "expires")

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
            record: A mapping holding exactly the record's fields (version
                1: `name_or_id` for `attestor`).

        Returns:
            The record. Unexpected or missing fields, empty text and invalid
            dates are rejected with a `ValueError`.
        """
        layout = qualification.require(cls, record)
        return cls(**layout)


@dataclasses.dataclass(frozen=True, slots=True)
class Reconstruction:
    """Result of reconstructing a field from a change log.

    Attributes:
        rows: Rows compared.
        mismatched: Rows whose reconstruction differs from the stored value.
        tolerance: Largest mismatch share the field may have.
        migrations: The layout's version steps; none so far.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = ()

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
            raise errors.InputError("mismatched must not exceed rows")
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
        return cls(**qualification.require(cls, record))


def promote(layout: records.Layout) -> records.Layout:
    """Moves the parts of a renamed version-1 `Pedigree` to version 2.

    Args:
        layout: The record after its own keys were renamed.

    Returns:
        The record with its rule and attestation in the version-2 layout.
    """
    return shift(layout, records.upgrade)


def demote(layout: records.Layout) -> records.Layout:
    """Moves the parts of a version-2 `Pedigree` back to version 1.

    Args:
        layout: The record before its own keys are renamed back.

    Returns:
        The record with its rule and attestation in the version-1 layout.
    """
    return shift(layout, records.downgrade)


def shift(
    layout: records.Layout,
    move: Callable[
        [tuple[records.Migration, ...], records.Layout, int], records.Layout
    ],
) -> records.Layout:
    """Moves the rule and the attestation of a `Pedigree` layout.

    Args:
        layout: The record, its own keys at version 2.
        move: `records.upgrade` or `records.downgrade`.

    Returns:
        A copy with `lead` and `attestation` moved; a part that is not an
        object is left for the reader to reject.
    """
    moved = dict(layout)
    for key, kind in (("lead", Lead), ("attestation", Attestation)):
        if isinstance(moved.get(key), Mapping):
            moved[key] = move(kind.migrations, dict(moved[key]), records.ORIGIN)
    return moved


@dataclasses.dataclass(frozen=True, slots=True)
class Pedigree:
    """How one field's values come to exist.

    Attributes:
        name: Column name.
        origin: How values first come to exist.
        writer: System or person that writes the value.
        stamp: Clock that stamps the write.
        overwritten: Whether stored values can change later.
        definitions: Dated notes on changes to the definition.
        lead: The as-of rule.
        provenance: `log`, `attested` or `none`; `none` requires an
            unknown rule and a known rule requires a source.
        availability: Whether the field exists when predicting.
        attestation: Required when `provenance` is `attested`.
        reconstruction: Required when `provenance` is `log`.
        migrations: The layout's version steps, oldest first; version 1
            spelled `stamp`, `definitions`, `lead`, `provenance`,
            `availability` and `reconstruction` as `written_at`,
            `definition_history`, `as_of`, `as_of_source`,
            `available_at_serving` and `log_check`, and its rule and
            attestation in their version-1 layouts. A record is stored inside
            a `Dictionary`, whose version it follows.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "written_at": "stamp",
                "definition_history": "definitions",
                "as_of": "lead",
                "as_of_source": "provenance",
                "available_at_serving": "availability",
                "log_check": "reconstruction",
            },
            change=promote,
            revert=demote,
        ),
    )

    name: str
    origin: Origin
    writer: str
    stamp: Stamp
    overwritten: Overwritten
    definitions: tuple[str, ...]
    lead: Lead
    provenance: Provenance
    availability: Availability
    attestation: Attestation | None = None
    reconstruction: Reconstruction | None = None

    def __post_init__(self) -> None:
        """Validates the record.

        Raises:
            ValueError: If `name` or `writer` is empty, a definition note is
                empty, `definitions` is not a tuple, an enum field has
                the wrong type, or the as-of rule, its source, the
                attestation and the reconstruction are inconsistent.
        """
        evidence.require(self.name, "name")
        evidence.require(self.writer, "writer")
        qualification.tupled(self.definitions, "definitions")
        for note in self.definitions:
            evidence.require(note, "definitions")
        for value, kind in (
            (self.origin, Origin),
            (self.stamp, Stamp),
            (self.overwritten, Overwritten),
            (self.provenance, Provenance),
            (self.availability, Availability),
            (self.lead, Lead),
        ):
            if not isinstance(value, kind):
                raise errors.InputError(f"expected a {kind.__name__}")
        known = self.lead.given
        if known and self.provenance is Provenance.NONE:
            raise errors.InputError("a known as-of rule needs a source")
        if not known and self.provenance is not Provenance.NONE:
            raise errors.InputError(
                "an unknown as-of rule cannot have a source"
            )
        if self.provenance is Provenance.ATTESTED and self.attestation is None:
            raise errors.InputError(
                "an attested as-of rule needs an attestation"
            )
        if self.provenance is Provenance.LOG and self.reconstruction is None:
            raise errors.InputError(
                "a log-derived as-of rule needs a reconstruction"
            )

    def tier(self, today: datetime.date | None = None) -> qualification.Tier:
        """Returns the evidence tier of the field's as-of rule.

        Args:
            today: Day on which an attestation must still be valid; None
                skips the expiry test.

        Returns:
            `confirmed` for a log-derived rule whose log check passed;
            `attested` when the rule's source is `attested`, or is `log`
            with a failed check, and an unexpired attestation is present;
            `unverified` otherwise.
        """
        attested = self.attestation is not None and (
            today is None or self.attestation.effective(today)
        )
        if self.provenance is Provenance.LOG:
            if self.reconstruction is not None and self.reconstruction.passed:
                return qualification.Tier.CONFIRMED
            return (
                qualification.Tier.ATTESTED
                if attested
                else qualification.Tier.UNVERIFIED
            )
        if self.provenance is Provenance.ATTESTED and attested:
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
            "stamp": self.stamp.value,
            "overwritten": self.overwritten.value,
            "definitions": list(self.definitions),
            "lead": self.lead.to_dict(),
            "provenance": self.provenance.value,
            "availability": self.availability.value,
            "attestation": None
            if self.attestation is None
            else self.attestation.to_dict(),
            "reconstruction": None
            if self.reconstruction is None
            else self.reconstruction.to_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a record, rejecting unknown or missing fields.

        Args:
            record: A mapping in the form `to_dict` returns, or its version-1
                layout.

        Returns:
            The record. Unexpected or missing fields, an unknown enum
            value and an inconsistent record are rejected with a
            `ValueError`.
        """
        layout = qualification.require(cls, record)
        attestation = layout["attestation"]
        reconstruction = layout["reconstruction"]
        return cls(
            name=layout["name"],
            origin=Origin(layout["origin"]),
            writer=layout["writer"],
            stamp=Stamp(layout["stamp"]),
            overwritten=Overwritten(layout["overwritten"]),
            definitions=tuple(layout["definitions"]),
            lead=Lead.from_dict(layout["lead"]),
            provenance=Provenance(layout["provenance"]),
            availability=Availability(layout["availability"]),
            attestation=None
            if attestation is None
            else Attestation.from_dict(attestation),
            reconstruction=None
            if reconstruction is None
            else Reconstruction.from_dict(reconstruction),
        )


def undeclared(name: str) -> Pedigree:
    """Returns the record of a field nobody described: everything unknown.

    Args:
        name: Column name.

    Returns:
        A record whose tier is `unverified`.
    """
    return Pedigree(
        name=name,
        origin=Origin.DERIVED,
        writer="undeclared",
        stamp=Stamp.UNKNOWN,
        overwritten=Overwritten.UNKNOWN,
        definitions=(),
        lead=Lead.unknown(),
        provenance=Provenance.NONE,
        availability=Availability.UNKNOWN,
    )


def forward(document: records.Layout) -> records.Layout:
    """Moves every field record of a version-1 `Dictionary` to version 2.

    Args:
        document: The table without its version key.

    Returns:
        The table with each field record in the version-2 layout.
    """
    return travel(document, records.upgrade)


def backward(document: records.Layout) -> records.Layout:
    """Moves every field record of a version-2 `Dictionary` to version 1.

    Args:
        document: The table without its version key.

    Returns:
        The table with each field record in the version-1 layout.
    """
    return travel(document, records.downgrade)


def travel(
    document: records.Layout,
    move: Callable[
        [tuple[records.Migration, ...], records.Layout, int], records.Layout
    ],
) -> records.Layout:
    """Moves every field record of a `Dictionary` document.

    Args:
        document: The table without its version key.
        move: `records.upgrade` or `records.downgrade`.

    Returns:
        A copy with each field record that is an object moved; anything else
        is left for the reader to reject.
    """
    fields = document.get("fields")
    if not isinstance(fields, list):
        return dict(document)
    return {
        **document,
        "fields": [
            move(Pedigree.migrations, dict(f), records.ORIGIN)
            if isinstance(f, Mapping)
            else f
            for f in fields
        ],
    }


@dataclasses.dataclass(frozen=True, slots=True)
class Dictionary:
    """The data dictionary of every field, which ingest writes as JSON.

    Attributes:
        fields: One record per field, in a fixed order, names unique.
        migrations: The document's version steps, oldest first; version 1
            spelled its version key `format_version` and held its field
            records in their version-1 layout.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(change=forward, revert=backward),
    )

    fields: tuple[Pedigree, ...]

    def __post_init__(self) -> None:
        """Validates the table.

        Raises:
            ValueError: If `fields` is not a tuple or names repeat.
        """
        qualification.tupled(self.fields, "fields")
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise errors.InputError("field names must be unique")

    @classmethod
    def covering(
        cls,
        columns: Sequence[str],
        declared: Sequence[Pedigree] = (),
    ) -> Self:
        """Builds a table with a record for every column.

        Args:
            columns: Every column of the data.
            declared: Records someone supplied; each must name a column.

        Returns:
            Declared records, plus `undeclared` records for the rest, in
            column order.

        Raises:
            ValueError: If a declared record names no column or repeats, or
                `columns` repeats a name.
        """
        by_name = {f.name: f for f in declared}
        if len(by_name) != len(declared):
            raise errors.InputError("declared field names must be unique")
        unknown = sorted(set(by_name) - set(columns))
        if unknown:
            raise errors.InputError(
                f"declared fields not in the data: {unknown}"
            )
        return cls(tuple(by_name.get(c) or undeclared(c) for c in columns))

    def get(self, name: str) -> Pedigree:
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
            A dict with `schema_version` and the list of field records.
        """
        return {
            records.KEY: records.ORIGIN + len(self.migrations),
            "fields": [f.to_dict() for f in self.fields],
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuilds a table of any stored version, rejecting unknown fields.

        Args:
            document: A mapping in the form `to_dict` returns, or the
                version-1 form with `format_version` 1.

        Returns:
            The table.

        Raises:
            ValueError: If the document carries no version, a version newer
                than this build, keys other than its version and `fields`,
                or an invalid field record.
        """
        if records.KEY not in document and LEGACY not in document:
            raise errors.InputError("the table carries no schema_version")
        layout = records.load(cls.__name__, cls.migrations, document, LEGACY)
        extra = sorted(set(layout) - {records.KEY, "fields"})
        if extra:
            raise errors.InputError(f"unexpected field(s): {extra}")
        if "fields" not in layout:
            raise errors.InputError("missing field(s): ['fields']")
        return cls(tuple(Pedigree.from_dict(f) for f in layout["fields"]))

    def digest(self) -> str:
        """Hashes the table, anchored at its version-1 form.

        A table stored before a key was renamed therefore keeps its digest.

        Returns:
            The hex SHA-256 of the version-1 document (`format_version` 1 and
            the field records in their version-1 layout) as JSON with sorted
            keys and no whitespace.
        """
        layout = self.to_dict()
        del layout[records.KEY]
        anchored = {
            LEGACY: records.ORIGIN,
            **records.downgrade(self.migrations, layout),
        }
        text = json.dumps(anchored, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
