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
from scmle.contracts import signing


class KeyState(enum.StrEnum):
    """Lifecycle state of a signing key identifier."""

    ACTIVE = "active"
    RETIRED = "retired"
    REVOKED = "revoked"


def require_text(value: object, field_name: str) -> str:
    """Returns a non-empty string suitable for a key record.

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


def parse_timestamp(value: str, field_name: str) -> datetime.datetime:
    """Parses an offset-aware ISO-8601 timestamp.

    Args:
        value: The timestamp text; a ``Z`` means UTC.
        field_name: The field name, for the error message.

    Returns:
        The timestamp as a timezone-aware datetime.

    Raises:
        ValueError: If the timestamp is malformed or has no timezone.
    """
    timestamp = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))  # noqa: FURB162 -- keeps the exact set of accepted inputs
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
        """Validates lifecycle fields and time boundaries.

        A `state` given as text is converted to `KeyState`.

        Raises:
            ValueError: If the key id, owner or a timestamp is invalid, the key
                expires at or before its creation, the state is not a
                `KeyState` value, a non-active key has no change time or
                reason, the change precedes creation, or the replacement key
                is blank or the key itself.
        """
        require_text(self.key_id, "key_id")
        require_text(self.owner, "owner")
        created = parse_timestamp(self.created_at, "created_at")
        if self.expires_at is not None:
            expires = parse_timestamp(self.expires_at, "expires_at")
            if expires <= created:
                raise ValueError("expires_at must be after created_at")
        state: Any = self.state  # may be text from a persisted record
        if not isinstance(state, KeyState):
            object.__setattr__(self, "state", KeyState(state))
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
        """Returns whether this key may create a signature at a given time.

        Args:
            observed_at: The time of signing, as an offset-aware timestamp.

        Returns:
            True if the key is active, already created and not yet expired at
            that time.
        """
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
        """Returns whether a signature may be accepted under this lifecycle.

        Revocation is immediate and cannot be bypassed by historical mode.
        Retired or expired keys can verify an artifact signed before the
        transition only when the caller explicitly requests historical proof.
        A signature dated before the key was created or at or after its
        expiry is never accepted.

        Args:
            signed_at: When the signature was made.
            observed_at: When it is being checked.
            historical: Whether to accept signatures made before a retirement
                or expiry.

        Returns:
            True if the signature is acceptable under the key's lifecycle.
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
        """Returns deterministic public key metadata."""
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
        """Rebuilds key metadata from a JSON object.

        A missing `key_id`, `owner` or `created_at` raises `KeyError`; other
        invalid fields raise the `ValueError` of construction.

        Args:
            record: A mapping with the key's fields; the state defaults to
                active and unknown entries are ignored.

        Returns:
            The key record.
        """
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
    algorithm: str = signing.SIGNATURE_ALGORITHM

    def __post_init__(self) -> None:
        """Validates signature identity and algorithm.

        Raises:
            ValueError: If the key id or value is blank, the timestamp is
                invalid, or the algorithm is unsupported.
        """
        require_text(self.key_id, "key_id")
        parse_timestamp(self.signed_at, "signed_at")
        require_text(self.value, "value")
        if self.algorithm != signing.SIGNATURE_ALGORITHM:
            raise ValueError("unsupported signature algorithm")

    def to_dict(self) -> dict[str, str]:
        """Returns a JSON-compatible signature envelope."""
        return {
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "signed_at": self.signed_at,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a signature envelope from JSON.

        A missing algorithm is read as empty and refused as unsupported.

        Args:
            record: A mapping with the key id, signing time, value and
                algorithm.

        Returns:
            The signature envelope.
        """
        return cls(
            key_id=str(record["key_id"]),
            signed_at=str(record["signed_at"]),
            value=str(record["value"]),
            algorithm=str(record.get("algorithm", "")),
        )


class SigningKeyRegistry:
    """Metadata registry that enforces signing and verification policy."""

    def __init__(self, records: Iterable[SigningKeyRecord] = ()) -> None:
        """Creates a registry from unique key records.

        Args:
            records: The key records to register.
        """
        self.lock = threading.RLock()
        self.records: dict[str, SigningKeyRecord] = {}
        for record in records:
            self.register(record)

    def register(self, record: SigningKeyRecord) -> None:
        """Adds a key record without replacing an existing identity.

        Args:
            record: The key record.

        Raises:
            TypeError: If `record` is not a `SigningKeyRecord`.
            ValueError: If the key id is already registered.
        """
        if not isinstance(record, SigningKeyRecord):
            raise TypeError("record must be SigningKeyRecord")
        with self.lock:
            if record.key_id in self.records:
                raise ValueError(f"key already registered: {record.key_id}")
            self.records[record.key_id] = record

    def get(self, key_id: str) -> SigningKeyRecord:
        """Returns the record of a key, refusing an unknown one.

        Args:
            key_id: Identifier of the key.

        Returns:
            The key's record.

        Raises:
            ValueError: If the key is not registered.
        """
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
        """Stops new signatures while retaining historical verification.

        The refusals are those of `transition`.

        Args:
            key_id: Identifier of the key.
            changed_at: When the key is retired.
            reason: Why it is retired.
            replacement_key_id: The active key that takes over, if any.

        Returns:
            The updated record.
        """
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
        """Immediately invalidates signatures made by a compromised key.

        An active or a retired key can be revoked (a retired key still
        verifies historical signatures, so a later compromise must be able to
        end that); the refusals are those of `transition`.

        Args:
            key_id: Identifier of the key.
            changed_at: When the key is revoked.
            reason: Why it is revoked.

        Returns:
            The updated record.
        """
        return self.transition(key_id, KeyState.REVOKED, changed_at, reason)

    def transition(
        self,
        key_id: str,
        state: KeyState,
        changed_at: str,
        reason: str,
        replacement_key_id: str | None = None,
    ) -> SigningKeyRecord:
        """Applies one explicit lifecycle transition.

        Args:
            key_id: Identifier of the key.
            state: The state to move to; it cannot be active.
            changed_at: When the state changes.
            reason: Why it changes.
            replacement_key_id: The active key that takes over, if any.

        Returns:
            The updated record.

        Raises:
            ValueError: If the key or the replacement key is unknown, the key
                is not active (a retired key may only be revoked), the target
                state is active, the replacement key is not active, or the new
                record is invalid (for example a malformed time, a blank reason
                or the key as its own replacement). Nothing changes in that
                case.
        """
        with self.lock:  # Read, validate and replace as one step.
            current = self.get(key_id)
            if current.state is not KeyState.ACTIVE and not (
                current.state is KeyState.RETIRED and state is KeyState.REVOKED
            ):
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
                replacement_key_id=(
                    replacement_key_id
                    if replacement_key_id is not None
                    else current.replacement_key_id
                ),
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
        """Creates a signature only with an active, non-expired key.

        Args:
            payload: The content to sign.
            key_id: Identifier of the key.
            secret: The key's secret.
            signed_at: The time of signing.

        Returns:
            The signature envelope.

        Raises:
            ValueError: If the key is not registered or may not sign at that
                time. An empty secret and a malformed time also fail with
                `ValueError`.
        """
        record = self.get(key_id)
        if not record.can_sign(signed_at):
            raise ValueError(f"signing key cannot sign: {key_id}")
        return SignatureEnvelope(
            key_id=key_id,
            signed_at=signed_at,
            value=signing.sign_payload(payload, secret),
        )

    def verify(
        self,
        payload: Mapping[str, Any],
        envelope: SignatureEnvelope,
        secrets: Mapping[str, str | bytes],
        observed_at: str,
        historical: bool = False,
    ) -> bool:
        """Verifies a signature against lifecycle state and external custody.

        Args:
            payload: The signed content.
            envelope: The signature to check.
            secrets: The secrets the verifier holds, by key id.
            observed_at: When the check is made.
            historical: Whether to accept signatures made before a retirement
                or expiry.

        Returns:
            True if the key is known, its secret is held, its lifecycle accepts
            the signature and the signature matches the payload.

        Raises:
            TypeError: If `envelope` is not a `SignatureEnvelope`.
        """
        if not isinstance(envelope, SignatureEnvelope):
            raise TypeError("envelope must be SignatureEnvelope")
        record = self.records.get(envelope.key_id)
        secret = secrets.get(envelope.key_id)
        if record is None or secret is None:
            return False
        if not record.can_verify(envelope.signed_at, observed_at, historical):
            return False
        expected = signing.sign_payload(payload, secret)
        return hmac.compare_digest(envelope.value, expected)

    def to_dict(self) -> dict[str, Any]:
        """Returns deterministic metadata without secret material."""
        return {
            "schema_version": 1,
            "keys": [
                record.to_dict()
                for key, record in sorted(dict(self.records).items())
            ],
        }

    def to_json(self) -> str:
        """Returns canonical registry JSON."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds and validates a registry from JSON-native metadata.

        Args:
            record: A mapping with `schema_version` 1 and a list of `keys`.

        Returns:
            The registry.

        Raises:
            ValueError: If the schema version is unsupported, the keys are not
                a list, a key record is invalid (a missing required field is a
                `KeyError`) or a key id appears twice.
        """
        if record.get("schema_version") != 1:
            raise ValueError("unsupported signing-key registry schema version")
        keys = record.get("keys")
        if not isinstance(keys, list):
            raise ValueError("signing-key registry keys must be a list")
        return cls(SigningKeyRecord.from_dict(item) for item in keys)

    @classmethod
    def from_json(cls, value: str) -> Self:
        """Parses and validates registry JSON.

        Args:
            value: The JSON text.

        Returns:
            The registry.

        Raises:
            ValueError: If the JSON is invalid or not an object, or
                `from_dict` refuses its content.
        """
        record = json.loads(value)
        if not isinstance(record, Mapping):
            raise ValueError("signing-key registry must be an object")
        return cls.from_dict(record)

    def write(self, path: str | Path) -> None:
        """Atomically persists metadata and flushes the containing directory.

        The registry is written as indented JSON to a temporary file in the
        same directory, flushed and renamed over `path`; on failure the
        temporary file is removed and the error propagates.

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
                stream.write(json.dumps(self.to_dict(), indent=2) + "\n")
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
        """Reads and validates a persisted registry.

        A missing file raises `OSError`; invalid content raises the
        `ValueError` of `from_json`.

        Args:
            path: The file written by `write`.

        Returns:
            The registry.
        """
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "KeyState",
    "SignatureEnvelope",
    "SigningKeyRecord",
    "SigningKeyRegistry",
    "parse_timestamp",
    "require_text",
]
