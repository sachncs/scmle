"""Signed records a person hands the loop: replies and envelopes."""

from collections.abc import Mapping
import dataclasses
import hmac
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.contracts import errors
from scmle.contracts import records
from scmle.contracts import signing
from scmle.governance.deployment_ledger import kinds

REPLY = (
    records.nest(
        {"escalation_id": "escalation", "name_or_id": "attestor"},
        groups={"digests": {"question_hash": "question"}},
    ),
)
"""A reply's version steps, oldest first."""
GRANT = (
    records.Migration(
        renames={"envelope_id": "identifier", "name_or_id": "attestor"}
    ),
)
"""An envelope's version steps, oldest first."""
VERSION = records.ORIGIN + len(REPLY)  # replies and envelopes move together


def store(
    layout: records.Layout, migrations: tuple[records.Migration, ...]
) -> records.Layout:
    """Returns a record's signed content in the layout it is stored in.

    Args:
        layout: The record's fields at the current version, with its
            `schema_version` (the version it was stored in).
        migrations: The kind's version steps.

    Returns:
        The content at the stored version; a version-1 record carries no
        version key, as it was written.
    """
    stored = layout[records.KEY]
    content = {k: v for k, v in layout.items() if k != records.KEY}
    content = records.downgrade(migrations, content, stored)
    return (
        content
        if stored == records.ORIGIN
        else {records.KEY: stored, **content}
    )


def load(
    record: Mapping[str, Any], migrations: tuple[records.Migration, ...]
) -> records.Layout:
    """Moves a stored record to the current version, keeping its version.

    Args:
        record: The stored record; one without `schema_version` is version 1.
        migrations: The kind's version steps.

    Returns:
        The fields at the current version, with `schema_version` the stored
        version.

    Raises:
        MisuseError: If the version is unknown.
    """
    values = dict(record)
    try:
        stored = records.detect("ledger record", values)
    except errors.InputError as error:
        raise kinds.MisuseError(str(error)) from error
    if stored > records.ORIGIN + len(migrations):
        raise kinds.MisuseError(f"unsupported record version {stored}")
    values = records.upgrade(migrations, values, stored)
    values[records.KEY] = stored
    return values


@dataclasses.dataclass(frozen=True, slots=True)
class Attestation:
    """A person's signed reply to one escalation.

    The fields are a domain owner's sign-off: who, in what role, on
    what date, for what scope. The signature covers every
    other field, so a reply cannot be moved to another question, scope or
    date. HMAC is symmetric: whoever can verify can also sign, so the verifier
    must keep the key where the system cannot read it; the ledger
    refuses the fingerprints of keys the system declared as its own.

    A reply keeps the version it was stored in, and its signature covers
    the content in that layout, so a version-1 reply still verifies.

    Attributes:
        escalation: The escalation answered.
        digests: `question`, the hash of that escalation's question.
        decision: What the person decides.
        statement: What the person asserts.
        evidence: Ticket, log table or document id.
        role: The person's role.
        attestor: The person.
        date: Date of the reply.
        scope: Model version hash and stream id the reply applies to.
        expires: Last day the reply is valid.
        key_id: Fingerprint of the signing key.
        signature: HMAC-SHA256 of the other fields.
        schema_version: The layout the reply was stored (and signed) in.
        migrations: The reply's version steps, oldest first.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = REPLY

    escalation: str
    digests: Mapping[str, str]
    decision: kinds.Reply
    statement: str
    evidence: str
    role: str
    attestor: str
    date: str
    scope: str
    expires: str
    key_id: str
    signature: str = ""
    schema_version: int = VERSION

    def __post_init__(self) -> None:
        """Validates the record's shape.

        Raises:
            MisuseError: If a required text field is blank, the
                decision is not a `Reply`, a date is not
                ``YYYY-MM-DD``, or the expiry is before the date.
        """
        for name in (
            "escalation",
            "statement",
            "evidence",
            "role",
            "attestor",
            "scope",
            "key_id",
        ):
            if not str(getattr(self, name)).strip():
                raise kinds.MisuseError(f"attestation {name} is required")
        if not str(self.digests.get("question", "")).strip():
            raise kinds.MisuseError("attestation digests.question is required")
        if not isinstance(self.decision, kinds.Reply):
            raise kinds.MisuseError("decision must be a Reply")
        issued = kinds.parse_date(self.date, "date")
        if kinds.parse_date(self.expires, "expires") < issued:
            raise kinds.MisuseError("expires is before the date")

    def preimage(self) -> dict[str, Any]:
        """The signed content.

        Returns:
            The record's fields in its stored layout, with the decision as
            text and without the signature.
        """
        record = dataclasses.asdict(self)
        record["decision"] = self.decision.value
        record["digests"] = dict(self.digests)
        del record["signature"]
        return store(record, REPLY)

    def to_dict(self) -> dict[str, Any]:
        """The full record.

        Returns:
            The signed content plus the signature.
        """
        return {**self.preimage(), "signature": self.signature}

    @classmethod
    def create(cls, key: str | bytes, **fields: Any) -> Self:
        """Builds and signs a reply with `key` (the person's key).

        Args:
            key: Signing key; its fingerprint becomes `key_id`.
            **fields: The other fields, all of them required except
                `signature`; `decision` may be given as its text value.

        Returns:
            The signed reply. A `decision` that is not a `Reply`
            value fails with `ValueError`, a missing one with `KeyError`; the
            fields are otherwise validated as in construction.
        """
        fields["decision"] = kinds.Reply(fields["decision"])
        unsigned = cls(key_id=kinds.thumbprint(key), **fields)
        signature = signing.mac(unsigned.preimage(), key)
        return dataclasses.replace(unsigned, signature=signature)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a reply from `to_dict`.

        Args:
            record: A mapping as produced by `to_dict`; one without a
                `schema_version` is version 1 and is migrated.

        Returns:
            The reply.

        Raises:
            MisuseError: If the version is unknown, the fields differ from
                the reply's fields or fail the checks of construction. A
                `decision` that is not a `Reply` value fails with
                `ValueError`.
        """
        values = load(record, REPLY)
        names = {f.name for f in dataclasses.fields(cls)}
        if set(values) != names:
            raise kinds.MisuseError("unexpected or missing reply fields")
        values["decision"] = kinds.Reply(values["decision"])
        values["digests"] = dict(values["digests"])
        return cls(**values)

    def verify(self, key: str | bytes) -> bool:
        """Whether the signature matches under `key`.

        Args:
            key: The verification key.

        Returns:
            True if the reply is signed, was made with a key of this
            fingerprint and the signature matches its content.
        """
        if not self.signature or self.key_id != kinds.thumbprint(key):
            return False
        expected = signing.mac(self.preimage(), key)
        return hmac.compare_digest(self.signature, expected)


@dataclasses.dataclass(frozen=True, slots=True)
class Envelope:
    """Authority a person signs in advance for actions off the never list.

    The never list (`scmle.governance.never_list`) stays as it is: an
    envelope cannot name any of its actions, only the actions in
    `ENVELOPE_ACTIONS`. It bounds what the unattended controller may do by
    action, scope, count and time, and the signature covers every other
    field. HMAC is symmetric, so the verifier must keep the key where the
    system cannot read it; the ledger refuses the fingerprints of
    keys the system declared as its own.

    An envelope keeps the version it was stored in, and its signature
    covers the content in that layout.

    Attributes:
        identifier: Identifier, unique in the ledger.
        limits: For each permitted action, how many times it may be used.
        statement: What the person authorises, in words.
        role: The person's role.
        attestor: The person.
        date: First day the envelope is valid.
        scope: Model version hash and stream id it applies to.
        expires: Last day the envelope is valid.
        key_id: Fingerprint of the signing key.
        signature: HMAC-SHA256 of the other fields.
        schema_version: The layout the envelope was stored (and signed) in.
        migrations: The envelope's version steps, oldest first.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = GRANT

    identifier: str
    limits: Mapping[str, int]
    statement: str
    role: str
    attestor: str
    date: str
    scope: str
    expires: str
    key_id: str
    signature: str = ""
    schema_version: int = VERSION

    def __post_init__(self) -> None:
        """Validates the record's shape.

        Raises:
            MisuseError: If a required text field is blank, no action
                is permitted, an action cannot be put in an envelope, a limit is
                not a positive integer, a date is not ``YYYY-MM-DD``, or the
                expiry is before the date.
        """
        for name in (
            "identifier",
            "statement",
            "role",
            "attestor",
            "scope",
            "key_id",
        ):
            if not str(getattr(self, name)).strip():
                raise kinds.MisuseError(f"envelope {name} is required")
        if not self.limits:
            raise kinds.MisuseError("an envelope permits at least one action")
        for action, raw_limit in self.limits.items():
            limit: object = raw_limit  # may be any value from a record
            if action not in kinds.ENVELOPE_ACTIONS:
                raise kinds.MisuseError(
                    f"{action!r} cannot be put in an envelope; only "
                    f"{list(kinds.ENVELOPE_ACTIONS)} can"
                )
            if (
                isinstance(limit, bool)
                or not isinstance(limit, int)
                or limit < 1
            ):
                raise kinds.MisuseError(
                    f"the limit for {action!r} must be a positive integer"
                )
        issued = kinds.parse_date(self.date, "date")
        if kinds.parse_date(self.expires, "expires") < issued:
            raise kinds.MisuseError("expires is before the date")

    def preimage(self) -> dict[str, Any]:
        """The signed content.

        Returns:
            The record's fields in its stored layout, without the signature.
        """
        record = dataclasses.asdict(self)
        record["limits"] = dict(self.limits)
        del record["signature"]
        return store(record, GRANT)

    def to_dict(self) -> dict[str, Any]:
        """The full record.

        Returns:
            The signed content plus the signature.
        """
        return {**self.preimage(), "signature": self.signature}

    @classmethod
    def create(cls, key: str | bytes, **fields: Any) -> Self:
        """Builds and signs an envelope with `key` (the person's key).

        Args:
            key: Signing key; its fingerprint becomes `key_id`.
            **fields: The other envelope fields, all of them required except
                `signature`; they are validated as in construction.

        Returns:
            The signed envelope.
        """
        unsigned = cls(key_id=kinds.thumbprint(key), **fields)
        signature = signing.mac(unsigned.preimage(), key)
        return dataclasses.replace(unsigned, signature=signature)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds an envelope from `to_dict`.

        Args:
            record: A mapping as produced by `to_dict`; one without a
                `schema_version` is version 1 and is migrated.

        Returns:
            The envelope.

        Raises:
            MisuseError: If the version is unknown, or the fields differ from
                the envelope's fields or fail the checks of construction.
        """
        values = load(record, GRANT)
        names = {f.name for f in dataclasses.fields(cls)}
        if set(values) != names:
            raise kinds.MisuseError("unexpected or missing envelope fields")
        return cls(**values)

    def verify(self, key: str | bytes) -> bool:
        """Whether the signature matches under `key`.

        Args:
            key: The verification key.

        Returns:
            True if the envelope is signed, was made with a key of this
            fingerprint and the signature matches its content.
        """
        if not self.signature or self.key_id != kinds.thumbprint(key):
            return False
        expected = signing.mac(self.preimage(), key)
        return hmac.compare_digest(self.signature, expected)
