"""Per-column attestations that a strong predictor is legitimate.

The leakage screen (`screen.py`) cannot tell a strong legitimate predictor
from a leak without provenance: both have a large, significant effect. In a
charter run a certified suspect is therefore quarantined, and the only way to
keep it is evidence a person is willing to put their name to. That evidence is
a `Affidavit`: the column, who states what, the evidence a third
party could check, the dates it is valid, the charter it covers and how long
before the prediction moment the value is known. It is the same record as an
as-of attestation in `provenance.py`, so the admissibility rules treat the
column exactly as they treat any attested field: it is admitted with a flag,
its tier is `attested`, never `confirmed`.

Rules, all enforced here or in `qualify.py`:

* An attestation counts only for the charter named in its `scope` and only
  on days between `date` and `expires`. Anything else is ignored, the column
  stays blocked, and the run says why.
* An attestation never changes what the screen measured. The suspect, its
  p-values and its bound stay in `leakage_suspicions` and in the ledger; the
  override is a separate ledger diagnostic (`leakage.attestation:<column>`)
  with the attester, the dates and the screen's p-value.
* An attestation about a column the screen did not flag changes nothing and
  is reported as unused.

What this is not: a proof that the column is legitimate, and not a
verification of the attester. The signature of the person is not checked; the
file is as trustworthy as the process that produced it, and the tier says
`attested`, not `confirmed`.

File format (JSON, schema version 2; version 1 spelled the version key
`format_version`, `attestor` as `name_or_id` and `seconds` as
`lead_seconds`, and still loads)::

    {
        "schema_version": 2,
        "attestations": [
            {
                "column": "...",
                "attestor": "...",
                "role": "...",
                "statement": "...",
                "evidence": "...",
                "date": "YYYY-MM-DD",
                "expires": "YYYY-MM-DD",
                "scope": "<charter_hash>",
                "seconds": 0,
            }
        ],
    }
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import json
from pathlib import Path
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.assurance.admissibility import provenance
from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.contracts import records

LEGACY = "format_version"
WRITER = "attested legitimate predictor"
OWN_FIELDS = ("column", "seconds")


@dataclasses.dataclass(frozen=True, slots=True)
class Affidavit:
    """A named person's statement that one column is a legitimate predictor.

    Attributes:
        column: The column the statement is about.
        attestation: The statement, its evidence, dates and scope.
        seconds: How long before the prediction moment the value is
            known (>= 0 for a value known no later than the moment).
        migrations: The layout's version steps, oldest first; version 1
            spelled `seconds` as `lead_seconds` and the attestation's
            `attestor` as `name_or_id`. A record is stored in an attestation
            file, whose version it follows.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={"lead_seconds": "seconds", "name_or_id": "attestor"}
        ),
    )

    column: str
    attestation: provenance.Attestation
    seconds: float

    def __post_init__(self) -> None:
        """Validates the record.

        Raises:
            ValueError: If `column` is empty or not a string, or
                `seconds` is not a finite number.
        """
        if not self.column or not isinstance(self.column, str):
            raise errors.InputError("column must be a non-empty string")
        qualification.finite(self.seconds, "seconds")

    def field(self) -> provenance.Pedigree:
        """Builds the provenance record the admissibility rules consume.

        Returns:
            A record for the column with origin `derived`, `WRITER` as its
            writer, a `lag` as-of rule of `seconds` whose source is
            `attested`, serving availability `yes`, no definition history, an
            unknown write time and overwrite status, and this attestation.
        """
        return provenance.Pedigree(
            name=self.column,
            origin=provenance.Origin.DERIVED,
            writer=WRITER,
            stamp=provenance.Stamp.UNKNOWN,
            overwritten=provenance.Overwritten.UNKNOWN,
            definitions=(),
            lead=provenance.Lead(provenance.Cutoff.LAG, self.seconds),
            provenance=provenance.Provenance.ATTESTED,
            availability=provenance.Availability.YES,
            attestation=self.attestation,
        )

    def to_dict(self) -> dict[str, Any]:
        """Converts the record to a JSON-ready dict.

        Returns:
            A dict with `column`, `seconds` and the fields of the
            attestation.
        """
        return {
            "column": self.column,
            "seconds": self.seconds,
            **self.attestation.to_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the record, rejecting unknown or missing fields.

        Args:
            record: A mapping with `column`, `seconds` (a finite number, not
                a bool or text) and the fields of an attestation, or its
                version-1 layout (`lead_seconds`, `name_or_id`).

        Returns:
            The record.

        Raises:
            ValueError: If `column` or `seconds` is missing, `seconds` is not
                a finite number, or the attestation fields are unexpected,
                missing or invalid.
        """
        layout = records.load(cls.__name__, cls.migrations, record)
        del layout[records.KEY]
        body = {k: v for k, v in layout.items() if k not in OWN_FIELDS}
        own = {k: layout[k] for k in OWN_FIELDS if k in layout}
        missing = sorted(set(OWN_FIELDS) - set(own))
        if missing:
            raise errors.InputError(f"missing field(s): {missing}")
        return cls(
            own["column"],
            provenance.Attestation.from_dict(body),
            qualification.finite(own["seconds"], "seconds"),
        )


def forward(document: records.Layout) -> records.Layout:
    """Moves every entry of a version-1 attestation file to version 2.

    Args:
        document: The file without its version key.

    Returns:
        The file with each entry that is an object in the version-2 layout.
    """
    return travel(document, records.upgrade)


def backward(document: records.Layout) -> records.Layout:
    """Moves every entry of a version-2 attestation file to version 1.

    Args:
        document: The file without its version key.

    Returns:
        The file with each entry that is an object in the version-1 layout.
    """
    return travel(document, records.downgrade)


def travel(
    document: records.Layout,
    move: Callable[
        [tuple[records.Migration, ...], records.Layout, int], records.Layout
    ],
) -> records.Layout:
    """Moves every entry of an attestation file.

    Args:
        document: The file without its version key.
        move: `records.upgrade` or `records.downgrade`.

    Returns:
        A copy with each entry that is an object moved; anything else is left
        for the reader to reject.
    """
    entries = document.get("attestations")
    if not isinstance(entries, list):
        return dict(document)
    return {
        **document,
        "attestations": [
            move(Affidavit.migrations, dict(e), records.ORIGIN)
            if isinstance(e, Mapping)
            else e
            for e in entries
        ],
    }


MIGRATIONS = (records.Migration(change=forward, revert=backward),)
VERSION = records.ORIGIN + len(MIGRATIONS)


@dataclasses.dataclass(frozen=True, slots=True)
class Applied:
    """What became of the attestations given to a run.

    Attributes:
        valid: Attestations in force, by column.
        rejected: Column -> why its attestation does not count.
    """

    valid: Mapping[str, Affidavit]
    rejected: Mapping[str, str]


def load(path: str | Path) -> tuple[Affidavit, ...]:
    """Reads an attestation file.

    Args:
        path: JSON file in the format of the module docstring.

    Returns:
        The attestations, in file order.

    Raises:
        ValueError: On an unreadable file, a missing or unknown schema
            version, a malformed record or two attestations for one column.
    """
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise errors.InputError(
            f"cannot read attestations {path}: {error}"
        ) from error
    if not isinstance(document, dict) or not (
        records.KEY in document or LEGACY in document
    ):
        raise errors.InputError(
            f"attestations must be an object with schema_version {VERSION}"
        )
    layout = records.load("attestations", MIGRATIONS, document, LEGACY)
    entries = layout.get("attestations")
    if not isinstance(entries, list):
        raise errors.InputError(
            "attestations must hold a list under 'attestations'"
        )
    found = tuple(Affidavit.from_dict(r) for r in entries)
    names = [a.column for a in found]
    if len(set(names)) != len(names):
        raise errors.InputError("at most one attestation per column")
    return found


def apply(
    attestations: Sequence[Affidavit],
    scope: str,
    today: str,
) -> Applied:
    """Separates the attestations that count from those that do not.

    Args:
        attestations: Everything the run was given.
        scope: The charter hash the run is bound to.
        today: ISO date on which attestations must be valid.

    Returns:
        The valid attestations and a reason for each rejected one, each
        keyed by column. An attestation is rejected when its scope is not
        `scope` or `today` falls outside its dates. A `today` that is not
        an ISO date is rejected with a `ValueError`.
    """
    day = provenance.parse(today, "today")
    valid: dict[str, Affidavit] = {}
    rejected: dict[str, str] = {}
    for item in attestations:
        given = item.attestation
        if given.scope != scope:
            rejected[item.column] = (
                f"scope {given.scope!r} is not this charter ({scope!r})"
            )
        elif not given.effective(day):
            rejected[item.column] = (
                f"valid {given.date} to {given.expires}, not on {today}"
            )
        else:
            valid[item.column] = item
    return Applied(valid, rejected)
