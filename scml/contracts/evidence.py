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

from scml.contracts import signing


class EvidenceResult(enum.StrEnum):
    """Observed result of one release criterion."""

    PASS = "pass"  # nosec B105 - evidence status, not a credential
    FAIL = "fail"
    WAIVED = "waived"


class EvidenceDecision(enum.StrEnum):
    """Release disposition of an evidence record."""

    REQUIRED = "required"
    ACCEPTED = "accepted"
    BLOCKED = "blocked"


def require_text(value: str, field_name: str) -> str:
    """Returns a non-empty, trimmed string for a contract field."""
    result = value.strip()
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class EvidenceRecord:
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
    result: EvidenceResult
    reviewer: str
    observed_at: str
    limitations: tuple[str, ...]
    decision: EvidenceDecision

    def __post_init__(self) -> None:
        """Validate the release-proof boundary."""
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
        if not isinstance(self.result, EvidenceResult):
            object.__setattr__(self, "result", EvidenceResult(self.result))
        if not isinstance(self.decision, EvidenceDecision):
            object.__setattr__(
                self, "decision", EvidenceDecision(self.decision)
            )
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
        """Rebuilds a release evidence record from serialized data."""
        return cls(
            criterion_id=str(record["criterion_id"]),
            claim=str(record["claim"]),
            scope=dict(record["scope"]),
            raw_inputs=dict(record["raw_inputs"]),
            command_or_test=str(record["command_or_test"]),
            artifact=dict(record["artifact"]),
            result=EvidenceResult(record["result"]),
            reviewer=str(record["reviewer"]),
            observed_at=str(record["observed_at"]),
            limitations=tuple(record["limitations"]),
            decision=EvidenceDecision(record["decision"]),
        )


class EvidenceIndex:
    """Deterministic collection of unique release evidence records."""

    def __init__(
        self,
        records: Iterable[EvidenceRecord] = (),
        signature: str | None = None,
    ) -> None:
        """Creates an index and registers the supplied records."""
        self.records: dict[str, EvidenceRecord] = {}
        self.signature = signature
        if signature is not None:
            require_text(signature, "signature")
        for record in records:
            self.register(record)

    def register(self, record: EvidenceRecord) -> None:
        """Registers one criterion without silently replacing evidence."""
        if self.signature is not None:
            raise ValueError("a signed evidence index cannot be changed")
        if record.criterion_id in self.records:
            raise ValueError(
                f"duplicate evidence criterion: {record.criterion_id}"
            )
        self.records[record.criterion_id] = record

    def find(self, criterion_id: str) -> EvidenceRecord | None:
        """Returns a criterion record, or ``None`` when it is absent."""
        return self.records.get(criterion_id)

    def pending(self) -> tuple[EvidenceRecord, ...]:
        """Returns records that cannot yet support an accepted release."""
        return tuple(
            record
            for record in self.ordered()
            if record.result is not EvidenceResult.PASS
            or record.decision is not EvidenceDecision.ACCEPTED
        )

    def ordered(self) -> tuple[EvidenceRecord, ...]:
        """Returns records in deterministic criterion order."""
        return tuple(self.records[key] for key in sorted(self.records))

    def ready(self) -> bool:
        """Whether every registered criterion is an accepted pass."""
        return bool(self.records) and not self.pending()

    def unsigned_dict(self) -> dict[str, Any]:
        """Return the versioned content covered by the index signature."""
        return {
            "schema_version": 1,
            "records": [record.to_dict() for record in self.ordered()],
        }

    def sign(self, signing_key: str | bytes) -> str:
        """Sign the complete index and return its HMAC value."""
        self.signature = signing.sign_payload(self.unsigned_dict(), signing_key)
        return self.signature

    def verify(self, signing_key: str | bytes) -> bool:
        """Verify the index signature against its current records."""
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
        """Rebuild and validate a versioned evidence index."""
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
        index = cls(EvidenceRecord.from_dict(item) for item in raw_records)
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
