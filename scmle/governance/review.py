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
from typing import ClassVar
from typing import Self

from scmle import durability
from scmle.contracts import errors
from scmle.contracts import records
from scmle.contracts import signing

CLAIM = records.Migration(
    renames={"criterion_id": "criterion", "evidence_uri": "uri"}
)
"""Version 1 to 2 of a recomputed claim (versioned by its review)."""


class Verdict(enum.StrEnum):
    """Disposition recorded by an independent release reviewer."""

    APPROVED = "approved"
    BLOCKED = "blocked"


class Kind(enum.StrEnum):
    """Critical result categories required for independent proof."""

    METRIC = "metric"
    FAIRNESS = "fairness"
    EXPLANATION = "explanation"
    RELIABILITY = "reliability"
    SECURITY = "security"


def require(value: object, field_name: str) -> str:
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
        raise errors.InputError(f"{field_name} must not be empty")
    return value.strip()


@dataclasses.dataclass(frozen=True, slots=True)
class Recomputation:
    """One independently recomputed claim in a release review.

    Attributes:
        criterion: The criterion recomputed.
        kind: The claim's category.
        passed: Whether the recomputation passed.
        observed: The value recomputed.
        reference: The value it was compared with.
        uri: Where the independent evidence is.
        notes: The reviewer's notes.
    """

    criterion: str
    kind: Kind
    passed: bool
    observed: str
    reference: str
    uri: str
    notes: str

    def __post_init__(self) -> None:
        """Checks the claim's identity and independent evidence link.

        A `kind` given as text is converted to a `Kind`.

        Raises:
            TypeError: If `passed` is not a bool.
            ValueError: If a text field is blank or not a string, or the kind
                is not a valid `Kind` value.
        """
        for field_name in (
            "criterion",
            "observed",
            "reference",
            "uri",
            "notes",
        ):
            require(getattr(self, field_name), field_name)
        kind: Any = self.kind  # may be text from a persisted record
        if not isinstance(kind, Kind):
            object.__setattr__(self, "kind", Kind(kind))
        if not isinstance(self.passed, bool):
            raise errors.KindError("passed must be bool")

    def to_dict(self) -> dict[str, Any]:
        """Returns a deterministic JSON-native claim."""
        return {
            "criterion": self.criterion,
            "kind": self.kind.value,
            "passed": self.passed,
            "observed": self.observed,
            "reference": self.reference,
            "uri": self.uri,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a claim from a persisted record.

        Text fields are converted with ``str``; `passed` is taken as is and
        must be a bool. A missing key raises `KeyError`, and the validation
        of `__post_init__` applies.

        Args:
            record: A mapping with the claim's fields at the current version
                (`Signoff.from_dict` migrates older claims).

        Returns:
            The claim.
        """
        return cls(
            criterion=str(record["criterion"]),
            kind=Kind(record["kind"]),
            passed=record["passed"],
            observed=str(record["observed"]),
            reference=str(record["reference"]),
            uri=str(record["uri"]),
            notes=str(record["notes"]),
        )


def separation(reviewer: str, implementers: tuple[object, ...]) -> None:
    """Checks the implementers are named and do not include the reviewer.

    Args:
        reviewer: Identity of the reviewer.
        implementers: Identities of the implementers.

    Raises:
        ValueError: If there are no implementers, one is not non-empty text, or
            the reviewer is one of them.
    """
    if not implementers:
        raise errors.InputError("implementers must not be empty")
    if any(
        not isinstance(item, str) or not item.strip() for item in implementers
    ):
        raise errors.InputError("implementers must contain non-empty values")
    if reviewer in implementers:
        raise errors.InputError("reviewer must be independent of implementers")


def vet(claims: tuple[Recomputation, ...]) -> None:
    """Checks the claims are present, typed and about distinct criteria.

    Args:
        claims: The claims of a review.

    Raises:
        TypeError: If a claim is not a `Recomputation`.
        ValueError: If there are no claims or a criterion repeats.
    """
    if not claims:
        raise errors.InputError("claims must not be empty")
    if not all(isinstance(item, Recomputation) for item in claims):
        raise errors.KindError("claims must contain Recomputation values")
    claim_ids = [claim.criterion for claim in claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise errors.InputError("claims must not contain duplicate criteria")


def enforce(artifact_digests: Mapping[Any, Any]) -> None:
    """Checks the artifact digests are present and named.

    Args:
        artifact_digests: Digest of each reviewed artifact, by name.

    Raises:
        ValueError: If there are none or a name or digest is not non-empty
            text.
    """
    if not artifact_digests:
        raise errors.InputError("digests.artifacts must not be empty")
    if any(
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(digest, str)
        or not digest.strip()
        for name, digest in artifact_digests.items()
    ):
        raise errors.InputError("digests.artifacts must contain named digests")


@dataclasses.dataclass(frozen=True, slots=True)
class Signoff:
    """Signed proof that an independent reviewer checked a release.

    A review keeps the version it was stored in, and its signature covers
    the content in that layout, so a version-1 review still verifies.

    Attributes:
        release: The release reviewed.
        commit: The commit the release is built from.
        digests: `clone` (the clean clone), `environment` (the review
            environment) and `artifacts` (each reviewed artifact's digest,
            by name).
        reviewer: Identity of the reviewer.
        implementers: Identities of the implementers.
        commands: The commands the reviewer ran.
        claims: The claims recomputed.
        observed: When the review was made.
        decision: The reviewer's verdict.
        signature: The HMAC over `preimage`, or None if unsigned.
        schema_version: The layout the review was stored (and signed) in.
        migrations: The review's version steps, oldest first.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            {
                "release_id": "release",
                "source_commit": "commit",
                "reviewer_id": "reviewer",
                "implementer_ids": "implementers",
                "observed_at": "observed",
            },
            parts={"claims": CLAIM},
            groups={
                "digests": {
                    "clean_clone_digest": "clone",
                    "environment_digest": "environment",
                    "artifact_digests": "artifacts",
                }
            },
        ),
    )

    release: str
    commit: str
    digests: Mapping[str, Any]
    reviewer: str
    implementers: tuple[str, ...]
    commands: tuple[str, ...]
    claims: tuple[Recomputation, ...]
    observed: str
    decision: Verdict
    signature: str | None = None
    schema_version: int = records.ORIGIN + len(migrations)

    def __post_init__(self) -> None:
        """Rejects reviews that cannot establish independence or replay.

        A `decision` given as text is converted to a `Verdict`.

        Raises:
            TypeError: If a claim is not a `Recomputation`.
            ValueError: If an identity or digest field is missing, there are no
                implementers, the reviewer is one of them, commands, claims or
                artifact digests are empty or malformed, a criterion repeats,
                the decision is not a valid `Verdict` value, the
                signature is blank, or the observation time is not an ISO 8601
                time or has no timezone.
        """
        for field_name in ("release", "commit", "reviewer", "observed"):
            require(getattr(self, field_name), field_name)
        for name in ("clone", "environment"):
            require(self.digests.get(name), f"digests.{name}")
        separation(self.reviewer, self.implementers)
        if not self.commands or any(not item.strip() for item in self.commands):
            raise errors.InputError(
                "commands must contain reproducible commands"
            )
        vet(self.claims)
        enforce(self.digests.get("artifacts") or {})
        decision: Any = self.decision  # may be text from a persisted record
        if not isinstance(decision, Verdict):
            object.__setattr__(self, "decision", Verdict(decision))
        if self.signature is not None:
            require(self.signature, "signature")
        timestamp = datetime.datetime.fromisoformat(
            self.observed.replace("Z", "+00:00")  # noqa: FURB162 -- keeps the exact set of accepted inputs
        )
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise errors.InputError(
                "observed must include timezone information"
            )

    def preimage(self) -> dict[str, Any]:
        """Returns the content covered by the review signature.

        Returns:
            The content in the layout the review is stored in.
        """
        record: records.Layout = {
            "release": self.release,
            "commit": self.commit,
            "digests": {
                "clone": self.digests["clone"],
                "environment": self.digests["environment"],
                "artifacts": dict(sorted(self.digests["artifacts"].items())),
            },
            "reviewer": self.reviewer,
            "implementers": list(self.implementers),
            "commands": list(self.commands),
            "claims": [claim.to_dict() for claim in self.claims],
            "observed": self.observed,
            "decision": self.decision.value,
        }
        record = records.downgrade(self.migrations, record, self.schema_version)
        return {records.KEY: self.schema_version, **record}

    def to_dict(self) -> dict[str, Any]:
        """Returns a persisted review record."""
        record = self.preimage()
        if self.signature is not None:
            record["signature"] = self.signature
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a review from a persisted JSON-native record.

        A version-1 review is migrated and remembers its version. A missing
        key (other than `claims` and `signature`) raises `KeyError`, and the
        validation of `__post_init__` applies.

        Args:
            record: A mapping with the review's fields.

        Returns:
            The review.

        Raises:
            ValueError: If the version is unknown.
        """
        values = {"claims": [], **record}
        stored = records.detect("review", values)
        if stored > records.ORIGIN + len(cls.migrations):
            raise errors.InputError(f"unsupported review version {stored}")
        values = records.upgrade(cls.migrations, values, stored)
        digests = values["digests"]
        return cls(
            release=str(values["release"]),
            commit=str(values["commit"]),
            digests={
                "clone": str(digests["clone"]),
                "environment": str(digests["environment"]),
                "artifacts": dict(digests["artifacts"]),
            },
            reviewer=str(values["reviewer"]),
            implementers=tuple(values["implementers"]),
            commands=tuple(values["commands"]),
            claims=tuple(
                Recomputation.from_dict(item)
                for item in values.get("claims", ())
            ),
            observed=str(values["observed"]),
            decision=Verdict(values["decision"]),
            signature=str(values["signature"])
            if values.get("signature") is not None
            else None,
            schema_version=stored,
        )

    def canonize(self) -> str:
        """Returns canonical unsigned review content."""
        return json.dumps(
            self.preimage(),
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
            signature=signing.mac(self.preimage(), signing_key),
        )

    def verify(self, signing_key: str | bytes) -> bool:
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
        expected = signing.mac(self.preimage(), signing_key)
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
            durability.flush(target.parent)
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
            raise errors.InputError("release review must be a JSON object")
        return cls.from_dict(record)

    def accepted(
        self,
        source_commit: str,
        required_kinds: Iterable[Kind | str],
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
        if self.commit != require(source_commit, "source_commit"):
            return False
        if self.decision is not Verdict.APPROVED:
            return False
        if not self.verify(signing_key):
            return False
        if not all(claim.passed for claim in self.claims):
            return False
        available = {claim.kind for claim in self.claims}
        required = {
            item if isinstance(item, Kind) else Kind(item)
            for item in required_kinds
        }
        return required.issubset(available)


__all__ = [
    "Kind",
    "Recomputation",
    "Signoff",
    "Verdict",
    "require",
]
