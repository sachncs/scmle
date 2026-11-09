"""Independent release review evidence and fail-closed verification."""

from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import datetime
import enum
import hmac
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from typing import Self

from scmle import durability
from scmle.contracts import signing


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


def require_text(value: object, field_name: str) -> str:
    """Returns a non-empty release-review string.

    Args:
        value: The candidate text.
        field_name: The field name, for the error message.

    Returns:
        `value` with surrounding whitespace removed.

    Raises:
        ValueError: If `value` is not a string or is blank.
    """
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
        """Checks the claim's identity and independent evidence link.

        A `kind` given as text is converted to a `ReviewClaimKind`.

        Raises:
            TypeError: If `passed` is not a bool.
            ValueError: If a text field is blank or not a string, or the kind
                is not a valid `ReviewClaimKind` value.
        """
        for field_name in (
            "criterion_id",
            "observed",
            "reference",
            "evidence_uri",
            "notes",
        ):
            require_text(getattr(self, field_name), field_name)
        kind: Any = self.kind  # may be text from a persisted record
        if not isinstance(kind, ReviewClaimKind):
            object.__setattr__(self, "kind", ReviewClaimKind(kind))
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be bool")

    def to_dict(self) -> dict[str, Any]:
        """Returns a deterministic JSON-native claim."""
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
        """Rebuilds a claim from a persisted record.

        Text fields are converted with ``str``; `passed` is taken as is and
        must be a bool. A missing key raises `KeyError`, and the validation
        of `__post_init__` applies.

        Args:
            record: A mapping with the claim's fields.

        Returns:
            The claim.
        """
        return cls(
            criterion_id=str(record["criterion_id"]),
            kind=ReviewClaimKind(record["kind"]),
            passed=record["passed"],
            observed=str(record["observed"]),
            reference=str(record["reference"]),
            evidence_uri=str(record["evidence_uri"]),
            notes=str(record["notes"]),
        )


def require_independence(
    reviewer_id: str, implementer_ids: tuple[object, ...]
) -> None:
    """Checks the implementers are named and do not include the reviewer.

    Args:
        reviewer_id: Identity of the reviewer.
        implementer_ids: Identities of the implementers.

    Raises:
        ValueError: If there are no implementers, one is not non-empty text, or
            the reviewer is one of them.
    """
    if not implementer_ids:
        raise ValueError("implementer_ids must not be empty")
    if any(
        not isinstance(item, str) or not item.strip()
        for item in implementer_ids
    ):
        raise ValueError("implementer_ids must contain non-empty values")
    if reviewer_id in implementer_ids:
        raise ValueError("reviewer must be independent of implementers")


def require_claims(claims: tuple[ReviewClaim, ...]) -> None:
    """Checks the claims are present, typed and about distinct criteria.

    Args:
        claims: The claims of a review.

    Raises:
        TypeError: If a claim is not a `ReviewClaim`.
        ValueError: If there are no claims or a criterion repeats.
    """
    if not claims:
        raise ValueError("claims must not be empty")
    if not all(isinstance(item, ReviewClaim) for item in claims):
        raise TypeError("claims must contain ReviewClaim values")
    claim_ids = [claim.criterion_id for claim in claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("claims must not contain duplicate criteria")


def require_artifact_digests(artifact_digests: Mapping[Any, Any]) -> None:
    """Checks the artifact digests are present and named.

    Args:
        artifact_digests: Digest of each reviewed artifact, by name.

    Raises:
        ValueError: If there are none or a name or digest is not non-empty
            text.
    """
    if not artifact_digests:
        raise ValueError("artifact_digests must not be empty")
    if any(
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(digest, str)
        or not digest.strip()
        for name, digest in artifact_digests.items()
    ):
        raise ValueError("artifact_digests must contain named digests")


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
        """Rejects reviews that cannot establish independence or replay.

        A `decision` given as text is converted to a `ReviewDecision`.

        Raises:
            TypeError: If a claim is not a `ReviewClaim`.
            ValueError: If an identity or digest field is missing, there are no
                implementers, the reviewer is one of them, commands, claims or
                artifact digests are empty or malformed, a criterion repeats,
                the decision is not a valid `ReviewDecision` value, the
                signature is blank, or the observation time is not an ISO 8601
                time or has no timezone.
        """
        for field_name in (
            "release_id",
            "source_commit",
            "clean_clone_digest",
            "reviewer_id",
            "environment_digest",
            "observed_at",
        ):
            require_text(getattr(self, field_name), field_name)
        require_independence(self.reviewer_id, self.implementer_ids)
        if not self.commands or any(not item.strip() for item in self.commands):
            raise ValueError("commands must contain reproducible commands")
        require_claims(self.claims)
        require_artifact_digests(self.artifact_digests)
        decision: Any = self.decision  # may be text from a persisted record
        if not isinstance(decision, ReviewDecision):
            object.__setattr__(self, "decision", ReviewDecision(decision))
        if self.signature is not None:
            require_text(self.signature, "signature")
        timestamp = datetime.datetime.fromisoformat(
            self.observed_at.replace("Z", "+00:00")  # noqa: FURB162 -- keeps the exact set of accepted inputs
        )
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("observed_at must include timezone information")

    def unsigned_dict(self) -> dict[str, Any]:
        """Returns the content covered by the review signature."""
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
        """Returns a persisted review record."""
        record = self.unsigned_dict()
        if self.signature is not None:
            record["signature"] = self.signature
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a review from a persisted JSON-native record.

        A missing key (other than `claims` and `signature`) raises
        `KeyError`, and the validation of `__post_init__` applies.

        Args:
            record: A mapping with the review's fields.

        Returns:
            The review.
        """
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
        """Returns canonical unsigned review content."""
        return json.dumps(
            self.unsigned_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def sign(self, signing_key: str | bytes) -> Self:
        """Returns this review signed over its immutable content.

        Args:
            signing_key: The key to sign with.

        Returns:
            A copy of the review carrying the signature.
        """
        return dataclasses.replace(
            self,
            signature=signing.sign_payload(self.unsigned_dict(), signing_key),
        )

    def verify_signature(self, signing_key: str | bytes) -> bool:
        """Checks the review signature without trusting its decision.

        An empty key is refused by the signing helper with a `ValueError`
        rather than returning False.

        Args:
            signing_key: The key to verify with.

        Returns:
            True if the review is signed and the signature matches its content.
        """
        if self.signature is None:
            return False
        expected = signing.sign_payload(self.unsigned_dict(), signing_key)
        return hmac.compare_digest(self.signature, expected)

    def write(self, path: str | Path) -> None:
        """Persists the review atomically and flushes its directory entry.

        The temporary file is removed and the error re-raised if writing
        fails.

        Args:
            path: The file to write; parent directories are created.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(
                    json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
            durability.sync_directory(target.parent)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Loads and validates a persisted review record.

        A missing file raises `OSError`, invalid JSON raises a `ValueError`
        subclass, and the checks of `from_dict` apply.

        Args:
            path: The file written by `write`.

        Returns:
            The review.

        Raises:
            ValueError: If the file does not hold a JSON object.
        """
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(record, Mapping):
            raise ValueError("release review must be a JSON object")
        return cls.from_dict(record)

    def accepted(
        self,
        source_commit: str,
        required_kinds: Iterable[ReviewClaimKind | str],
        signing_key: str | bytes,
    ) -> bool:
        """Returns whether this signed review satisfies release proof gates.

        A blank `source_commit`, a `required_kinds` entry that is not a claim
        kind, or an empty key raises `ValueError` (the key only once the
        commit and decision checks have passed).

        Args:
            source_commit: The commit the release is built from.
            required_kinds: The claim kinds the release requires.
            signing_key: The key to verify the signature with.

        Returns:
            True if the review is of that commit, approved, correctly signed,
            every claim passed and every required kind is covered.
        """
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
