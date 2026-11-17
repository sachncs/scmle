"""Signed records a person hands the loop: replies and envelopes."""

from collections.abc import Mapping
import dataclasses
import hmac
from typing import Any
from typing import Self

from scmle.contracts import signing
from scmle.governance.deployment_ledger import kinds


@dataclasses.dataclass(frozen=True, slots=True)
class Attestation:
    """A person's signed reply to one escalation.

    The fields are a domain owner's sign-off: who, in what role, on
    what date, for what scope. The signature covers every
    other field, so a reply cannot be moved to another question, scope or
    date. HMAC is symmetric: whoever can verify can also sign, so the verifier
    must keep the key where the system cannot read it; the ledger
    refuses the fingerprints of keys the system declared as its own.

    Attributes:
        escalation_id: The escalation answered.
        question_hash: Hash of that escalation's question.
        decision: What the person decides.
        statement: What the person asserts.
        evidence: Ticket, log table or document id.
        role: The person's role.
        name_or_id: The person.
        date: Date of the reply.
        scope: Model version hash and stream id the reply applies to.
        expires: Last day the reply is valid.
        key_id: Fingerprint of the signing key.
        signature: HMAC-SHA256 of the other fields.
    """

    escalation_id: str
    question_hash: str
    decision: kinds.Reply
    statement: str
    evidence: str
    role: str
    name_or_id: str
    date: str
    scope: str
    expires: str
    key_id: str
    signature: str = ""

    def __post_init__(self) -> None:
        """Validates the record's shape.

        Raises:
            MisuseError: If a required text field is blank, the
                decision is not a `Reply`, a date is not
                ``YYYY-MM-DD``, or the expiry is before the date.
        """
        for name in (
            "escalation_id",
            "question_hash",
            "statement",
            "evidence",
            "role",
            "name_or_id",
            "scope",
            "key_id",
        ):
            if not str(getattr(self, name)).strip():
                raise kinds.MisuseError(f"attestation {name} is required")
        if not isinstance(self.decision, kinds.Reply):
            raise kinds.MisuseError("decision must be a Reply")
        issued = kinds.parse_date(self.date, "date")
        if kinds.parse_date(self.expires, "expires") < issued:
            raise kinds.MisuseError("expires is before the date")

    def preimage(self) -> dict[str, Any]:
        """The signed content.

        Returns:
            The record's fields, with the decision as text and without the
            signature.
        """
        record = dataclasses.asdict(self)
        record["decision"] = self.decision.value
        del record["signature"]
        return record

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
            record: A mapping as produced by `to_dict`.

        Returns:
            The reply.

        Raises:
            MisuseError: If the fields differ from the reply's fields
                or fail the checks of construction. A `decision` that is not
                a `Reply` value fails with `ValueError`.
        """
        names = {f.name for f in dataclasses.fields(cls)}
        if set(record) != names:
            raise kinds.MisuseError("unexpected or missing reply fields")
        values = dict(record)
        values["decision"] = kinds.Reply(values["decision"])
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

    Attributes:
        envelope_id: Identifier, unique in the ledger.
        limits: For each permitted action, how many times it may be used.
        statement: What the person authorises, in words.
        role: The person's role.
        name_or_id: The person.
        date: First day the envelope is valid.
        scope: Model version hash and stream id it applies to.
        expires: Last day the envelope is valid.
        key_id: Fingerprint of the signing key.
        signature: HMAC-SHA256 of the other fields.
    """

    envelope_id: str
    limits: Mapping[str, int]
    statement: str
    role: str
    name_or_id: str
    date: str
    scope: str
    expires: str
    key_id: str
    signature: str = ""

    def __post_init__(self) -> None:
        """Validates the record's shape.

        Raises:
            MisuseError: If a required text field is blank, no action
                is permitted, an action cannot be put in an envelope, a limit is
                not a positive integer, a date is not ``YYYY-MM-DD``, or the
                expiry is before the date.
        """
        for name in (
            "envelope_id",
            "statement",
            "role",
            "name_or_id",
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
            The record's fields without the signature.
        """
        record = dataclasses.asdict(self)
        record["limits"] = dict(self.limits)
        del record["signature"]
        return record

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
            record: A mapping as produced by `to_dict`.

        Returns:
            The envelope.

        Raises:
            MisuseError: If the fields differ from the envelope's
                fields or fail the checks of construction.
        """
        names = {f.name for f in dataclasses.fields(cls)}
        if set(record) != names:
            raise kinds.MisuseError("unexpected or missing envelope fields")
        return cls(**dict(record))

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
