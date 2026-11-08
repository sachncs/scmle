"""File-backed signing-key lifecycle metadata and verification policy.

The registry deliberately stores metadata, not key material. Secret custody
belongs to the deployment secret manager; this module makes identity,
expiration, retirement, revocation, and rotation decisions explicit at the
artifact boundary.
"""

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
import threading
from typing import Any
from typing import Self

from scmle import durability
from scmle.governance import attestation


class KeyState(enum.StrEnum):
    """Lifecycle state of a signing key identifier."""

    ACTIVE = "active"
    RETIRED = "retired"
    REVOKED = "revoked"


def require_text(value: str, field_name: str) -> str:
    """Return a non-empty string suitable for a key record."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value.strip()


def parse_timestamp(value: str, field_name: str) -> datetime.datetime:
    """Parse an offset-aware ISO-8601 timestamp."""
    timestamp = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information")
    return timestamp


@dataclasses.dataclass(frozen=True, slots=True)
class SigningKeyRecord:
    """Public lifecycle metadata for one externally-held signing secret."""

    key_id: str
    owner: str
    created_at: str
    expires_at: str | None = None
    state: KeyState = KeyState.ACTIVE
    state_changed_at: str | None = None
    reason: str = ""
    replacement_key_id: str | None = None

    def __post_init__(self) -> None:
        """Validate lifecycle transitions and time boundaries."""
        require_text(self.key_id, "key_id")
        require_text(self.owner, "owner")
        created = parse_timestamp(self.created_at, "created_at")
        if self.expires_at is not None:
            expires = parse_timestamp(self.expires_at, "expires_at")
            if expires <= created:
                raise ValueError("expires_at must be after created_at")
        if not isinstance(self.state, KeyState):
            object.__setattr__(self, "state", KeyState(self.state))
        if self.state is not KeyState.ACTIVE:
            if self.state_changed_at is None:
                raise ValueError("non-active keys require state_changed_at")
            parse_timestamp(self.state_changed_at, "state_changed_at")
            require_text(self.reason, "reason")
        if self.state_changed_at is not None:
            changed = parse_timestamp(self.state_changed_at, "state_changed_at")
            if changed < created:
                raise ValueError("state_changed_at must not precede created_at")
        if self.replacement_key_id is not None:
            require_text(self.replacement_key_id, "replacement_key_id")
            if self.replacement_key_id == self.key_id:
                raise ValueError("replacement_key_id must differ from key_id")

    def can_sign(self, observed_at: str) -> bool:
        """Return whether this key may create a signature now."""
        observed = parse_timestamp(observed_at, "observed_at")
        created = parse_timestamp(self.created_at, "created_at")
        expires = (
            parse_timestamp(self.expires_at, "expires_at")
            if self.expires_at is not None
            else None
        )
        return (
            self.state is KeyState.ACTIVE
            and observed >= created
            and (expires is None or observed < expires)
        )

    def can_verify(
        self,
        signed_at: str,
        observed_at: str,
        historical: bool = False,
    ) -> bool:
        """Return whether a signature may be accepted under this lifecycle.

        Revocation is immediate and cannot be bypassed by historical mode.
        Retired or expired keys can verify an artifact signed before the
        transition only when the caller explicitly requests historical proof.
        """
        signed = parse_timestamp(signed_at, "signed_at")
        observed = parse_timestamp(observed_at, "observed_at")
        created = parse_timestamp(self.created_at, "created_at")
        if self.state is KeyState.REVOKED or signed < created:
            return False
        if self.expires_at is not None:
            expires = parse_timestamp(self.expires_at, "expires_at")
            if signed >= expires:
                return False
            if observed >= expires and not historical:
                return False
        if self.state is KeyState.ACTIVE:
            return True
        if not historical or self.state_changed_at is None:
            return False
        changed = parse_timestamp(self.state_changed_at, "state_changed_at")
        return signed < changed

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic public key metadata."""
        return {
            "key_id": self.key_id,
            "owner": self.owner,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "state": self.state.value,
            "state_changed_at": self.state_changed_at,
            "reason": self.reason,
            "replacement_key_id": self.replacement_key_id,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild key metadata from a JSON object."""
        return cls(
            key_id=str(record["key_id"]),
            owner=str(record["owner"]),
            created_at=str(record["created_at"]),
            expires_at=(
                str(record["expires_at"])
                if record.get("expires_at") is not None
                else None
            ),
            state=KeyState(record.get("state", KeyState.ACTIVE)),
            state_changed_at=(
                str(record["state_changed_at"])
                if record.get("state_changed_at") is not None
                else None
            ),
            reason=str(record.get("reason", "")),
            replacement_key_id=(
                str(record["replacement_key_id"])
                if record.get("replacement_key_id") is not None
                else None
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class SignatureEnvelope:
    """A signature plus the metadata needed to evaluate its lifecycle."""

    key_id: str
    signed_at: str
    value: str
    algorithm: str = attestation.SIGNATURE_ALGORITHM

    def __post_init__(self) -> None:
        """Validate signature identity and algorithm."""
        require_text(self.key_id, "key_id")
        parse_timestamp(self.signed_at, "signed_at")
        require_text(self.value, "value")
        if self.algorithm != attestation.SIGNATURE_ALGORITHM:
            raise ValueError("unsupported signature algorithm")

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible signature envelope."""
        return {
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "signed_at": self.signed_at,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild a signature envelope from JSON."""
        return cls(
            key_id=str(record["key_id"]),
            signed_at=str(record["signed_at"]),
            value=str(record["value"]),
            algorithm=str(record.get("algorithm", "")),
        )


class SigningKeyRegistry:
    """Metadata registry that enforces signing and verification policy."""

    def __init__(self, records: Iterable[SigningKeyRecord] = ()) -> None:
        """Create a registry from unique key records."""
        self.lock = threading.RLock()
        self.records: dict[str, SigningKeyRecord] = {}
        for record in records:
            self.register(record)

    def register(self, record: SigningKeyRecord) -> None:
        """Add a key record without replacing an existing identity."""
        if not isinstance(record, SigningKeyRecord):
            raise TypeError("record must be SigningKeyRecord")
        with self.lock:
            if record.key_id in self.records:
                raise ValueError(f"key already registered: {record.key_id}")
            self.records[record.key_id] = record

    def get(self, key_id: str) -> SigningKeyRecord:
        """Return metadata for a key or fail closed."""
        try:
            return self.records[key_id]
        except KeyError as error:
            raise ValueError(f"unknown signing key: {key_id}") from error

    def retire(
        self,
        key_id: str,
        changed_at: str,
        reason: str,
        replacement_key_id: str | None = None,
    ) -> SigningKeyRecord:
        """Stop new signatures while retaining historical verification."""
        return self.transition(
            key_id,
            KeyState.RETIRED,
            changed_at,
            reason,
            replacement_key_id,
        )

    def revoke(
        self, key_id: str, changed_at: str, reason: str
    ) -> SigningKeyRecord:
        """Immediately invalidate signatures made by a compromised key."""
        return self.transition(key_id, KeyState.REVOKED, changed_at, reason)

    def transition(
        self,
        key_id: str,
        state: KeyState,
        changed_at: str,
        reason: str,
        replacement_key_id: str | None = None,
    ) -> SigningKeyRecord:
        """Apply one explicit lifecycle transition."""
        with self.lock:  # Read, validate and replace as one step.
            current = self.get(key_id)
            if current.state is not KeyState.ACTIVE:
                raise ValueError(
                    f"key is already {current.state.value}: {key_id}"
                )
            if state is KeyState.ACTIVE:
                raise ValueError("active is not a terminal transition")
            updated = dataclasses.replace(
                current,
                state=state,
                state_changed_at=changed_at,
                reason=reason,
                replacement_key_id=replacement_key_id,
            )
            if replacement_key_id is not None:
                replacement = self.get(replacement_key_id)
                if replacement.state is not KeyState.ACTIVE:
                    raise ValueError("replacement key must be active")
            self.records[key_id] = updated
            return updated

    def sign(
        self,
        payload: Mapping[str, Any],
        key_id: str,
        secret: str | bytes,
        signed_at: str,
    ) -> SignatureEnvelope:
        """Create a signature only with an active, non-expired key."""
        record = self.get(key_id)
        if not record.can_sign(signed_at):
            raise ValueError(f"signing key cannot sign: {key_id}")
        return SignatureEnvelope(
            key_id=key_id,
            signed_at=signed_at,
            value=attestation.sign_payload(payload, secret),
        )

    def verify(
        self,
        payload: Mapping[str, Any],
        envelope: SignatureEnvelope,
        secrets: Mapping[str, str | bytes],
        observed_at: str,
        historical: bool = False,
    ) -> bool:
        """Verify a signature against lifecycle state and external custody."""
        if not isinstance(envelope, SignatureEnvelope):
            raise TypeError("envelope must be SignatureEnvelope")
        record = self.records.get(envelope.key_id)
        secret = secrets.get(envelope.key_id)
        if record is None or secret is None:
            return False
        if not record.can_verify(envelope.signed_at, observed_at, historical):
            return False
        expected = attestation.sign_payload(payload, secret)
        return hmac.compare_digest(envelope.value, expected)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic metadata without secret material."""
        return {
            "schema_version": 1,
            "keys": [
                record.to_dict()
                for key, record in sorted(dict(self.records).items())
            ],
        }

    def to_json(self) -> str:
        """Return canonical registry JSON."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild and validate a registry from JSON-native metadata."""
        if record.get("schema_version") != 1:
            raise ValueError("unsupported signing-key registry schema version")
        keys = record.get("keys")
        if not isinstance(keys, list):
            raise ValueError("signing-key registry keys must be a list")
        return cls(SigningKeyRecord.from_dict(item) for item in keys)

    @classmethod
    def from_json(cls, value: str) -> Self:
        """Parse and validate registry JSON."""
        record = json.loads(value)
        if not isinstance(record, Mapping):
            raise ValueError("signing-key registry must be an object")
        return cls.from_dict(record)

    def write(self, path: str | Path) -> None:
        """Atomically persist metadata and flush the containing directory."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(self.to_dict(), indent=2) + "\n")
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
        """Read and validate a persisted registry."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "KeyState",
    "SignatureEnvelope",
    "SigningKeyRecord",
    "SigningKeyRegistry",
    "parse_timestamp",
    "require_text",
]
