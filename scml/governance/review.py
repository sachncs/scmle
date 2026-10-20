"""Independent release review evidence and fail-closed verification."""

from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import datetime
import enum
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from typing import Self

from scml import durability
from scml.governance import attestation


class ReviewDecision(enum.StrEnum):
    """Disposition recorded by an independent release reviewer."""

    APPROVED = "approved"
    BLOCKED = "blocked"


class ReviewClaimKind(enum.StrEnum):
    """Critical result categories required for independent proof."""

    METRIC = "metric"
    FAIRNESS = "fairness"
    EXPLANATION = "explanation"
    RELIABILITY = "reliability"
    SECURITY = "security"


def require_text(value: str, field_name: str) -> str:
    """Return a non-empty release-review string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value.strip()


@dataclasses.dataclass(frozen=True, slots=True)
class ReviewClaim:
    """One independently recomputed claim in a release review."""

    criterion_id: str
    kind: ReviewClaimKind
    passed: bool
    observed: str
    reference: str
    evidence_uri: str
    notes: str

    def __post_init__(self) -> None:
        """Validate the claim's identity and independent evidence link."""
        for field_name in (
            "criterion_id",
            "observed",
            "reference",
            "evidence_uri",
            "notes",
        ):
            require_text(getattr(self, field_name), field_name)
        if not isinstance(self.kind, ReviewClaimKind):
            object.__setattr__(self, "kind", ReviewClaimKind(self.kind))
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be bool")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-native claim."""
        return {
            "criterion_id": self.criterion_id,
            "kind": self.kind.value,
            "passed": self.passed,
            "observed": self.observed,
            "reference": self.reference,
            "evidence_uri": self.evidence_uri,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild a claim from a persisted record."""
        return cls(
            criterion_id=str(record["criterion_id"]),
            kind=ReviewClaimKind(record["kind"]),
            passed=record["passed"],
            observed=str(record["observed"]),
            reference=str(record["reference"]),
            evidence_uri=str(record["evidence_uri"]),
            notes=str(record["notes"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class IndependentReleaseReview:
    """Signed proof that an independent reviewer checked a release."""

    release_id: str
    source_commit: str
    clean_clone_digest: str
    reviewer_id: str
    implementer_ids: tuple[str, ...]
    environment_digest: str
    commands: tuple[str, ...]
    claims: tuple[ReviewClaim, ...]
    artifact_digests: Mapping[str, str]
    observed_at: str
    decision: ReviewDecision
    signature: str | None = None

    def __post_init__(self) -> None:
        """Reject reviews that cannot establish independence or replay."""
        for field_name in (
            "release_id",
            "source_commit",
            "clean_clone_digest",
            "reviewer_id",
            "environment_digest",
            "observed_at",
        ):
            require_text(getattr(self, field_name), field_name)
        if not self.implementer_ids:
            raise ValueError("implementer_ids must not be empty")
        if any(
            not isinstance(item, str) or not item.strip()
            for item in self.implementer_ids
        ):
            raise ValueError("implementer_ids must contain non-empty values")
        if self.reviewer_id in self.implementer_ids:
            raise ValueError("reviewer must be independent of implementers")
        if not self.commands or any(not item.strip() for item in self.commands):
            raise ValueError("commands must contain reproducible commands")
        if not self.claims:
            raise ValueError("claims must not be empty")
        if not all(isinstance(item, ReviewClaim) for item in self.claims):
            raise TypeError("claims must contain ReviewClaim values")
        claim_ids = [claim.criterion_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claims must not contain duplicate criteria")
        if not self.artifact_digests:
            raise ValueError("artifact_digests must not be empty")
        if any(
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(digest, str)
            or not digest.strip()
            for name, digest in self.artifact_digests.items()
        ):
            raise ValueError("artifact_digests must contain named digests")
        if not isinstance(self.decision, ReviewDecision):
            object.__setattr__(self, "decision", ReviewDecision(self.decision))
        if self.signature is not None:
            require_text(self.signature, "signature")
        timestamp = datetime.datetime.fromisoformat(
            self.observed_at.replace("Z", "+00:00")
        )
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("observed_at must include timezone information")

    def unsigned_dict(self) -> dict[str, Any]:
        """Return the content covered by the review signature."""
        return {
            "schema_version": 1,
            "release_id": self.release_id,
            "source_commit": self.source_commit,
            "clean_clone_digest": self.clean_clone_digest,
            "reviewer_id": self.reviewer_id,
            "implementer_ids": list(self.implementer_ids),
            "environment_digest": self.environment_digest,
            "commands": list(self.commands),
            "claims": [claim.to_dict() for claim in self.claims],
            "artifact_digests": dict(sorted(self.artifact_digests.items())),
            "observed_at": self.observed_at,
            "decision": self.decision.value,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a persisted review record."""
        record = self.unsigned_dict()
        if self.signature is not None:
            record["signature"] = self.signature
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild a review from a persisted JSON-native record."""
        claims = tuple(
            ReviewClaim.from_dict(item) for item in record.get("claims", ())
        )
        return cls(
            release_id=str(record["release_id"]),
            source_commit=str(record["source_commit"]),
            clean_clone_digest=str(record["clean_clone_digest"]),
            reviewer_id=str(record["reviewer_id"]),
            implementer_ids=tuple(record["implementer_ids"]),
            environment_digest=str(record["environment_digest"]),
            commands=tuple(record["commands"]),
            claims=claims,
            artifact_digests=dict(record["artifact_digests"]),
            observed_at=str(record["observed_at"]),
            decision=ReviewDecision(record["decision"]),
            signature=str(record["signature"])
            if record.get("signature") is not None
            else None,
        )

    def canonical_json(self) -> str:
        """Return canonical unsigned review content."""
        return json.dumps(
            self.unsigned_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def sign(self, signing_key: str | bytes) -> Self:
        """Return this review signed over its immutable content."""
        return dataclasses.replace(
            self,
            signature=attestation.sign_payload(
                self.unsigned_dict(), signing_key
            ),
        )

    def verify_signature(self, signing_key: str | bytes) -> bool:
        """Verify the review signature without trusting its decision."""
        if self.signature is None:
            return False
        expected = attestation.sign_payload(self.unsigned_dict(), signing_key)
        return self.signature == expected

    def write(self, path: str | Path) -> None:
        """Persist the review atomically and flush its directory entry."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            durability.sync_directory(target.parent)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Load and validate a persisted review record."""
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(record, Mapping):
            raise ValueError("release review must be a JSON object")
        return cls.from_dict(record)

    def accepted(
        self,
        source_commit: str,
        required_kinds: Iterable[ReviewClaimKind],
        signing_key: str | bytes,
    ) -> bool:
        """Return whether this signed review satisfies release proof gates."""
        if self.source_commit != require_text(source_commit, "source_commit"):
            return False
        if self.decision is not ReviewDecision.APPROVED:
            return False
        if not self.verify_signature(signing_key):
            return False
        if not all(claim.passed for claim in self.claims):
            return False
        available = {claim.kind for claim in self.claims}
        required = {
            item if isinstance(item, ReviewClaimKind) else ReviewClaimKind(item)
            for item in required_kinds
        }
        return required.issubset(available)


__all__ = [
    "IndependentReleaseReview",
    "ReviewClaim",
    "ReviewClaimKind",
    "ReviewDecision",
    "require_text",
]
