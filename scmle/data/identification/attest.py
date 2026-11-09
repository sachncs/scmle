"""Signed attestations of the identification assumptions.

A causal claim rests on assumptions no data can verify. The charter names who
may vouch for which assumption (`causal.Attestor`) and where the signed
statements are (`CausalSpec.attestations`). An attestation counts for an
assumption only when all of these hold; otherwise it is rejected with a
reason and the assumption has no named attestor, which keeps the claim at
`unverified` (`tier.decide` is unchanged):

* the attestor is named in the charter and may attest that assumption;
* the scope is the hash of this charter;
* the run date is between `date` and `expires`;
* the signature, an HMAC-SHA256 over the other fields under the signing key,
  verifies. With no key nothing can verify and nothing counts.

Limits. The signature is symmetric: it shows that someone holding the project
signing key issued the statement unchanged, not which person did, and not
that the assumption is true. The tier stays `attested` at most from this
source; `confirmed` still needs every other condition.

File format (JSON)::

    {
        "format_version": 1,
        "attestations": [
            {
                "assumption": "...",
                "attestor": "...",
                "statement": "...",
                "evidence": "...",
                "date": "YYYY-MM-DD",
                "expires": "YYYY-MM-DD",
                "scope": "<charter_hash>",
                "signature": "<hex>",
            }
        ],
    }
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import hmac
import json
from pathlib import Path
from typing import Any
from typing import Self

from scmle.contracts import evidence
from scmle.contracts import qualification
from scmle.contracts import signing
from scmle.data.admissibility import provenance
from scmle.data.charter import causal

FORMAT_VERSION = 1


@dataclasses.dataclass(frozen=True, slots=True)
class Attestation:
    """One named person's statement about one assumption.

    Attributes:
        assumption: A word of the closed assumption vocabulary.
        attestor: The name the charter lists.
        statement: What they vouch for, in words.
        evidence: A document a third party could check.
        date: ISO date of the statement.
        expires: ISO date after which it no longer counts.
        scope: Hash of the charter it covers.
        signature: HMAC-SHA256 hex over the other fields; empty if unsigned.
    """

    assumption: str
    attestor: str
    statement: str
    evidence: str
    date: str
    expires: str
    scope: str
    signature: str = ""

    def __post_init__(self) -> None:
        """Validates the record.

        Raises:
            ValueError: If `expires` precedes `date`. A blank text field or a
                date that cannot be parsed is also refused with a
                `ValueError`.
        """
        for name in (
            "assumption",
            "attestor",
            "statement",
            "evidence",
            "scope",
        ):
            evidence.require_text(getattr(self, name), name)
        issued = provenance.parse_date(self.date, "date")
        if provenance.parse_date(self.expires, "expires") < issued:
            raise ValueError("expires must not precede date")

    def payload(self) -> dict[str, Any]:
        """Gives the signed content.

        Returns:
            A mapping of every field but the signature.
        """
        record = dataclasses.asdict(self)
        del record["signature"]
        return record

    def signed(self, key: str | bytes) -> Self:
        """Signs the attestation.

        Args:
            key: The signing key.

        Returns:
            A copy carrying the signature computed under `key`.
        """
        return dataclasses.replace(
            self, signature=signing.sign_payload(self.payload(), key)
        )

    def verify(self, key: str | bytes | None) -> bool:
        """Checks the signature against the content.

        Args:
            key: The signing key, or None to use the key from the environment.

        Returns:
            True when a key is available, the attestation is signed and the
            signature matches the content; False otherwise.
        """
        resolved = signing.resolve_key(key)
        if resolved is None or not self.signature:
            return False
        expected = signing.sign_payload(self.payload(), resolved)
        return hmac.compare_digest(expected, self.signature)

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record.

        Returns:
            A mapping of every field.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the record, rejecting unknown or missing fields.

        Args:
            record: A mapping as written by `to_dict`.

        Returns:
            The attestation described by the record.
        """
        qualification.require_fields(cls, record)
        return cls(**record)


@dataclasses.dataclass(frozen=True, slots=True)
class Applied:
    """What became of the attestations given to a run.

    Attributes:
        accepted: Assumption to the attestor who verifiably signed it.
        records: The accepted records, with their signatures.
        rejected: Assumption to why no attestation counts for it.
        evaluated_on: The ISO date the validity windows were checked on.
    """

    accepted: Mapping[str, str]
    records: tuple[Attestation, ...]
    rejected: Mapping[str, str]
    evaluated_on: str

    def to_dict(self) -> dict[str, Any]:
        """Builds the `attestations` section of `causal_claims.json`.

        Returns:
            A mapping of the evaluation date, the accepted records and the
            rejection reason per assumption.
        """
        return {
            "evaluated_on": self.evaluated_on,
            "accepted": [r.to_dict() for r in self.records],
            "rejected": dict(self.rejected),
        }


def load(path: str | Path) -> tuple[Attestation, ...]:
    """Reads an attestation file.

    Args:
        path: The attestation JSON file.

    Returns:
        The attestations, in file order.

    Raises:
        ValueError: On an unreadable file, an unknown format version, a
            malformed record or two attestations for one assumption.
    """
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read attestations {path}: {error}") from error
    if (
        not isinstance(document, dict)
        or document.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError(
            f"attestations must be an object with format_version "
            f"{FORMAT_VERSION}"
        )
    records = document.get("attestations")
    if not isinstance(records, list):
        raise ValueError("attestations must hold a list under 'attestations'")
    found = tuple(Attestation.from_dict(r) for r in records)
    names = [a.assumption for a in found]
    if len(set(names)) != len(names):
        raise ValueError("at most one attestation per assumption")
    return found


def apply(
    attestors: Sequence[causal.Attestor],
    given: Sequence[Attestation],
    needed: Sequence[str],
    scope: str,
    today: str,
    key: str | bytes | None,
) -> Applied:
    """Decides which attestations count for the assumptions a claim needs.

    Args:
        attestors: The charter's attestors.
        given: Everything the attestation file holds.
        needed: The assumptions the chosen strategy relies on.
        scope: The charter hash the run is bound to.
        today: ISO date on which attestations must be valid.
        key: Signing key; None accepts nothing.

    Returns:
        The accepted attestors by assumption, the accepted records and a
        reason for every needed assumption that has none.
    """
    day = provenance.parse_date(today, "today")
    allowed = {a.name: set(a.assumptions) for a in attestors}
    by_assumption = {a.assumption: a for a in given}
    accepted: dict[str, str] = {}
    records: list[Attestation] = []
    rejected: dict[str, str] = {}
    for assumption in needed:
        item = by_assumption.get(assumption)
        if item is None:
            rejected[assumption] = "no signed attestation in the file"
        elif item.attestor not in allowed:
            rejected[assumption] = (
                f"attestor {item.attestor!r} is not named in the charter"
            )
        elif assumption not in allowed[item.attestor]:
            rejected[assumption] = (
                f"attestor {item.attestor!r} is not permitted to attest this "
                "assumption"
            )
        elif item.scope != scope:
            rejected[assumption] = (
                f"scope {item.scope!r} is not this charter ({scope!r})"
            )
        elif not (
            provenance.parse_date(item.date, "date")
            <= day
            <= provenance.parse_date(item.expires, "expires")
        ):
            rejected[assumption] = (
                f"valid {item.date} to {item.expires}, not on {today}"
            )
        elif signing.resolve_key(key) is None:
            rejected[assumption] = "no signing key to verify the signature"
        elif not item.verify(key):
            rejected[assumption] = "the signature does not verify"
        else:
            accepted[assumption] = item.attestor
            records.append(item)
    return Applied(accepted, tuple(records), rejected, today)
