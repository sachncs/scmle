"""Release evidence records and deterministic evidence indexes."""

from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import datetime
import enum
import hmac
import json
from typing import Any
from typing import Self

from scmle.contracts import signing


class Outcome(enum.StrEnum):
    """Observed result of one release criterion."""

    PASS = "pass"  # nosec B105 - evidence status, not a credential
    FAIL = "fail"
    WAIVED = "waived"


class Disposition(enum.StrEnum):
    """Release disposition of an evidence record."""

    REQUIRED = "required"
    ACCEPTED = "accepted"
    BLOCKED = "blocked"


def require_text(value: str, field_name: str) -> str:
    """Returns a non-empty, trimmed string for a contract field.

    Args:
        value: The text to check.
        field_name: The field name, for the error message.

    Returns:
        `value` with surrounding whitespace removed.

    Raises:
        ValueError: If the trimmed text is empty.
    """
    result = value.strip()
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class Record:
    """One independently reviewable release claim.

    Attributes:
        criterion_id: Stable identifier for the acceptance criterion.
        claim: Exact claim supported by this record.
        scope: Modality, task, platform, and release scope of the claim.
        raw_inputs: Immutable source identities used to produce the result.
        command_or_test: Reproducible command or test selector.
        artifact: Evidence artifact metadata, including a digest when present.
        result: Observed pass, fail, or explicitly waived result.
        reviewer: Person or automation that reviewed the evidence.
        observed_at: Time the result was observed, in UTC ISO-8601 form.
        limitations: Known uncertainty or excluded scope.
        decision: Whether the criterion is accepted, required, or blocked.
    """

    criterion_id: str
    claim: str
    scope: Mapping[str, str]
    raw_inputs: Mapping[str, str]
    command_or_test: str
    artifact: Mapping[str, str]
    result: Outcome
    reviewer: str
    observed_at: str
    limitations: tuple[str, ...]
    decision: Disposition

    def __post_init__(self) -> None:
        """Validates the release-proof boundary.

        Raises:
            ValueError: If a text field is blank, the scope, raw inputs,
                artifact or limitations are empty, a limitation is blank, the
                result or decision is not a valid value, the observation time
                is not an ISO-8601 timestamp, or it has no timezone.
        """
        for field_name, value in (
            ("criterion_id", self.criterion_id),
            ("claim", self.claim),
            ("command_or_test", self.command_or_test),
            ("reviewer", self.reviewer),
            ("observed_at", self.observed_at),
        ):
            require_text(value, field_name)
        if not self.scope:
            raise ValueError("scope must not be empty")
        if not self.raw_inputs:
            raise ValueError("raw_inputs must not be empty")
        if not self.artifact:
            raise ValueError("artifact must not be empty")
        if not self.limitations:
            raise ValueError("limitations must state known uncertainty")
        if not all(item.strip() for item in self.limitations):
            raise ValueError("limitations must not contain empty values")
        if not isinstance(self.result, Outcome):
            object.__setattr__(self, "result", Outcome(self.result))
        if not isinstance(self.decision, Disposition):
            object.__setattr__(self, "decision", Disposition(self.decision))
        timestamp = datetime.datetime.fromisoformat(
            self.observed_at.replace("Z", "+00:00")
        )
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("observed_at must include timezone information")

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible release evidence record."""
        return {
            "criterion_id": self.criterion_id,
            "claim": self.claim,
            "scope": dict(sorted(self.scope.items())),
            "raw_inputs": dict(sorted(self.raw_inputs.items())),
            "command_or_test": self.command_or_test,
            "artifact": dict(sorted(self.artifact.items())),
            "result": self.result.value,
            "reviewer": self.reviewer,
            "observed_at": self.observed_at,
            "limitations": list(self.limitations),
            "decision": self.decision.value,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a release evidence record from serialized data.

        Args:
            record: A mapping as produced by `to_dict`.

        Returns:
            The evidence record.
        """
        return cls(
            criterion_id=str(record["criterion_id"]),
            claim=str(record["claim"]),
            scope=dict(record["scope"]),
            raw_inputs=dict(record["raw_inputs"]),
            command_or_test=str(record["command_or_test"]),
            artifact=dict(record["artifact"]),
            result=Outcome(record["result"]),
            reviewer=str(record["reviewer"]),
            observed_at=str(record["observed_at"]),
            limitations=tuple(record["limitations"]),
            decision=Disposition(record["decision"]),
        )


class Index:
    """Deterministic collection of unique release evidence records."""

    def __init__(
        self,
        records: Iterable[Record] = (),
        signature: str | None = None,
    ) -> None:
        """Creates an index and registers the supplied records.

        A non-blank signature marks the index as signed, and a signed index
        refuses new records, so records can be supplied together with a
        signature only when the iterable is empty.

        Args:
            records: The evidence records to register.
            signature: The index signature, if it is already signed.
        """
        self.records: dict[str, Record] = {}
        self.signature = signature
        if signature is not None:
            require_text(signature, "signature")
        for record in records:
            self.register(record)

    def register(self, record: Record) -> None:
        """Registers one criterion without silently replacing evidence.

        Args:
            record: The evidence record.

        Raises:
            ValueError: If the index is signed or the criterion is already
                registered.
        """
        if self.signature is not None:
            raise ValueError("a signed evidence index cannot be changed")
        if record.criterion_id in self.records:
            raise ValueError(
                f"duplicate evidence criterion: {record.criterion_id}"
            )
        self.records[record.criterion_id] = record

    def find(self, criterion_id: str) -> Record | None:
        """Returns a criterion record, or ``None`` when it is absent.

        Args:
            criterion_id: The criterion to look up.

        Returns:
            The record, or None.
        """
        return self.records.get(criterion_id)

    def pending(self) -> tuple[Record, ...]:
        """Returns records that cannot yet support an accepted release."""
        return tuple(
            record
            for record in self.ordered()
            if record.result is not Outcome.PASS
            or record.decision is not Disposition.ACCEPTED
        )

    def ordered(self) -> tuple[Record, ...]:
        """Returns records in deterministic criterion order."""
        return tuple(self.records[key] for key in sorted(self.records))

    def ready(self) -> bool:
        """Whether every registered criterion is an accepted pass.

        Returns:
            True if there is at least one record and none is pending.
        """
        return bool(self.records) and not self.pending()

    def unsigned_dict(self) -> dict[str, Any]:
        """Returns the versioned content covered by the index signature."""
        return {
            "schema_version": 1,
            "records": [record.to_dict() for record in self.ordered()],
        }

    def sign(self, signing_key: str | bytes) -> str:
        """Signs the complete index and returns its HMAC value.

        Args:
            signing_key: The key to sign with.

        Returns:
            The signature, which is also stored on the index.
        """
        self.signature = signing.sign_payload(self.unsigned_dict(), signing_key)
        return self.signature

    def verify(self, signing_key: str | bytes) -> bool:
        """Verifies the index signature against its current records.

        Args:
            signing_key: The key to verify with.

        Returns:
            True if the index is signed and the signature matches its records.
        """
        if self.signature is None:
            return False
        expected = signing.sign_payload(self.unsigned_dict(), signing_key)
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> dict[str, Any]:
        """Returns a deterministic, JSON-compatible evidence index."""
        record = self.unsigned_dict()
        if self.signature is not None:
            record["signature"] = {
                "algorithm": signing.SIGNATURE_ALGORITHM,
                "value": self.signature,
            }
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds and validates a versioned evidence index.

        Args:
            record: A mapping with schema version 1, a list of records and an
                optional signature object.

        Returns:
            The index, signed if the record carried a signature.

        Raises:
            ValueError: If the schema version is unsupported, the records are
                not a list, or the signature is malformed or uses an
                unsupported algorithm.
        """
        if record.get("schema_version") != 1:
            raise ValueError("unsupported evidence index schema version")
        raw_records = record.get("records")
        if not isinstance(raw_records, list):
            raise ValueError("evidence index records must be a list")
        signature_record = record.get("signature")
        signature: str | None = None
        if signature_record is not None:
            if not isinstance(signature_record, Mapping):
                raise ValueError("evidence index signature must be an object")
            if signature_record.get("algorithm") != signing.SIGNATURE_ALGORITHM:
                raise ValueError(
                    "unsupported evidence index signature algorithm"
                )
            signature = require_text(
                str(signature_record.get("value", "")),
                "signature",
            )
        index = cls(Record.from_dict(item) for item in raw_records)
        index.signature = signature
        return index

    def canonical_json(self) -> str:
        """Returns stable JSON suitable for hashing and signing."""
        return json.dumps(
            self.unsigned_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
