"""Per-column attestations that a strong predictor is legitimate.

The leakage screen (`screen.py`) cannot tell a strong legitimate predictor
from a leak without provenance: both have a large, significant effect. In a
charter run a certified suspect is therefore quarantined, and the only way to
keep it is evidence a person is willing to put their name to. That evidence is
a `LeakageAttestation`: the column, who states what, the evidence a third
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

File format (JSON)::

    {
        "format_version": 1,
        "attestations": [
            {
                "column": "...",
                "name_or_id": "...",
                "role": "...",
                "statement": "...",
                "evidence": "...",
                "date": "YYYY-MM-DD",
                "expires": "YYYY-MM-DD",
                "scope": "<charter_hash>",
                "lead_seconds": 0,
            }
        ],
    }
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import json
from pathlib import Path
from typing import Any
from typing import Self

from scmle.contracts import qualification
from scmle.data.admissibility import provenance

FORMAT_VERSION = 1
WRITER = "attested legitimate predictor"
OWN_FIELDS = ("column", "lead_seconds")


@dataclasses.dataclass(frozen=True, slots=True)
class LeakageAttestation:
    """A named person's statement that one column is a legitimate predictor.

    Attributes:
        column: The column the statement is about.
        attestation: The statement, its evidence, dates and scope.
        lead_seconds: How long before the prediction moment the value is
            known (>= 0 for a value known no later than the moment).
    """

    column: str
    attestation: provenance.Attestation
    lead_seconds: float

    def __post_init__(self) -> None:
        """Validates the record.

        Raises:
            ValueError: If `column` is empty or not a string, or
                `lead_seconds` is not a finite number.
        """
        if not self.column or not isinstance(self.column, str):
            raise ValueError("column must be a non-empty string")
        qualification.finite(self.lead_seconds, "lead_seconds")

    def field(self) -> provenance.FieldProvenance:
        """Builds the provenance record the admissibility rules consume.

        Returns:
            A record for the column with origin `derived`, `WRITER` as its
            writer, a `lag` as-of rule of `lead_seconds` whose source is
            `attested`, serving availability `yes`, no definition history, an
            unknown write time and overwrite status, and this attestation.
        """
        return provenance.FieldProvenance(
            name=self.column,
            origin=provenance.Origin.DERIVED,
            writer=WRITER,
            written_at=provenance.WrittenAt.UNKNOWN,
            overwritten=provenance.Overwritten.UNKNOWN,
            definition_history=(),
            as_of=provenance.AsOf(provenance.AsOfKind.LAG, self.lead_seconds),
            as_of_source=provenance.AsOfSource.ATTESTED,
            available_at_serving=provenance.Availability.YES,
            attestation=self.attestation,
        )

    def to_dict(self) -> dict[str, Any]:
        """Converts the record to a JSON-ready dict.

        Returns:
            A dict with `column`, `lead_seconds` and the fields of the
            attestation.
        """
        return {
            "column": self.column,
            "lead_seconds": self.lead_seconds,
            **self.attestation.to_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the record, rejecting unknown or missing fields.

        Args:
            record: A mapping with `column`, `lead_seconds` and the fields
                of an attestation.

        Returns:
            The record.

        Raises:
            ValueError: If `column` or `lead_seconds` is missing, or the
                attestation fields are unexpected, missing or invalid.
        """
        body = {k: v for k, v in record.items() if k not in OWN_FIELDS}
        own = {k: record[k] for k in OWN_FIELDS if k in record}
        missing = sorted(set(OWN_FIELDS) - set(own))
        if missing:
            raise ValueError(f"missing field(s): {missing}")
        return cls(
            own["column"],
            provenance.Attestation.from_dict(body),
            float(own["lead_seconds"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Applied:
    """What became of the attestations given to a run.

    Attributes:
        valid: Attestations in force, by column.
        rejected: Column -> why its attestation does not count.
    """

    valid: Mapping[str, LeakageAttestation]
    rejected: Mapping[str, str]


def load(path: str | Path) -> tuple[LeakageAttestation, ...]:
    """Reads an attestation file.

    Args:
        path: JSON file in the format of the module docstring.

    Returns:
        The attestations, in file order.

    Raises:
        ValueError: On an unreadable file, an unknown format version, a
            malformed record or two attestations for one column.
    """
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read attestations {path}: {error}") from error
    if not isinstance(document, dict) or (
        document.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError(
            f"attestations must be an object with format_version "
            f"{FORMAT_VERSION}"
        )
    records = document.get("attestations")
    if not isinstance(records, list):
        raise ValueError("attestations must hold a list under 'attestations'")
    found = tuple(LeakageAttestation.from_dict(r) for r in records)
    names = [a.column for a in found]
    if len(set(names)) != len(names):
        raise ValueError("at most one attestation per column")
    return found


def apply(
    attestations: Sequence[LeakageAttestation],
    scope: str,
    today: str,
) -> Applied:
    """Separate the attestations that count from those that do not.

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
    day = provenance.parse_date(today, "today")
    valid: dict[str, LeakageAttestation] = {}
    rejected: dict[str, str] = {}
    for item in attestations:
        given = item.attestation
        if given.scope != scope:
            rejected[item.column] = (
                f"scope {given.scope!r} is not this charter ({scope!r})"
            )
        elif not given.valid_on(day):
            rejected[item.column] = (
                f"valid {given.date} to {given.expires}, not on {today}"
            )
        else:
            valid[item.column] = item
    return Applied(valid, rejected)
