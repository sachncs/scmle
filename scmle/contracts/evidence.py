"""Release evidence records and deterministic evidence indexes."""

from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import datetime
import enum
import hmac
import json
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.contracts import errors
from scmle.contracts import records
from scmle.contracts import signing


class Outcome(enum.StrEnum):
    """Observed result of one release criterion."""

    PASS = "pass"  # noqa: S105 -- evidence status, not a credential # nosec B105
    FAIL = "fail"
    WAIVED = "waived"


class Disposition(enum.StrEnum):
    """Release disposition of an evidence record."""

    REQUIRED = "required"
    ACCEPTED = "accepted"
    BLOCKED = "blocked"


def require(value: str, field_name: str) -> str:
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
        raise errors.InputError(f"{field_name} must not be empty")
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class Record:
    """One independently reviewable release claim.

    Attributes:
        criterion: Stable identifier for the acceptance criterion.
        claim: Exact claim supported by this record.
        scope: Modality, task, platform, and release scope of the claim.
        inputs: Immutable source identities used to produce the result.
        command: Reproducible command or test selector.
        artifact: Evidence artifact metadata, including a digest when present.
        result: Observed pass, fail, or explicitly waived result.
        reviewer: Person or automation that reviewed the evidence.
        observed: Time the result was observed, in UTC ISO-8601 form.
        limitations: Known uncertainty or excluded scope.
        decision: Whether the criterion is accepted, required, or blocked.
        migrations: Version steps of the stored record, which is versioned
            by the index that holds it.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "criterion_id": "criterion",
                "raw_inputs": "inputs",
                "command_or_test": "command",
                "observed_at": "observed",
            }
        ),
    )

    criterion: str
    claim: str
    scope: Mapping[str, str]
    inputs: Mapping[str, str]
    command: str
    artifact: Mapping[str, str]
    result: Outcome
    reviewer: str
    observed: str
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
            ("criterion", self.criterion),
            ("claim", self.claim),
            ("command", self.command),
            ("reviewer", self.reviewer),
            ("observed", self.observed),
        ):
            require(value, field_name)
        if not self.scope:
            raise errors.InputError("scope must not be empty")
        if not self.inputs:
            raise errors.InputError("inputs must not be empty")
        if not self.artifact:
            raise errors.InputError("artifact must not be empty")
        if not self.limitations:
            raise errors.InputError("limitations must state known uncertainty")
        if not all(item.strip() for item in self.limitations):
            raise errors.InputError("limitations must not contain empty values")
        result: Any = self.result  # may be a raw value at runtime
        if not isinstance(result, Outcome):
            object.__setattr__(self, "result", Outcome(result))
        decision: Any = self.decision  # may be a raw value at runtime
        if not isinstance(decision, Disposition):
            object.__setattr__(self, "decision", Disposition(decision))
        timestamp = datetime.datetime.fromisoformat(self.observed)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise errors.InputError(
                "observed must include timezone information"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible release evidence record."""
        return {
            "criterion": self.criterion,
            "claim": self.claim,
            "scope": dict(sorted(self.scope.items())),
            "inputs": dict(sorted(self.inputs.items())),
            "command": self.command,
            "artifact": dict(sorted(self.artifact.items())),
            "result": self.result.value,
            "reviewer": self.reviewer,
            "observed": self.observed,
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
            criterion=str(record["criterion"]),
            claim=str(record["claim"]),
            scope=dict(record["scope"]),
            inputs=dict(record["inputs"]),
            command=str(record["command"]),
            artifact=dict(record["artifact"]),
            result=Outcome(record["result"]),
            reviewer=str(record["reviewer"]),
            observed=str(record["observed"]),
            limitations=tuple(record["limitations"]),
            decision=Disposition(record["decision"]),
        )


VERSION = records.ORIGIN + len(Record.migrations)


class Index:
    """Deterministic collection of unique release evidence records.

    Attributes:
        records: The evidence records by criterion.
        signature: The index signature, or None while unsigned.
        version: The layout the signature covers: the current one for a new
            index, the stored one for a loaded index, so an index signed in
            an older layout still verifies after its records migrate.
    """

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
        self.version = VERSION
        if signature is not None:
            require(signature, "signature")
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
            raise errors.InputError("a signed evidence index cannot be changed")
        if record.criterion in self.records:
            raise errors.InputError(
                f"duplicate evidence criterion: {record.criterion}"
            )
        self.records[record.criterion] = record

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

    def preimage(self) -> dict[str, Any]:
        """Returns the versioned content covered by the index signature.

        The records are laid out at the index's `version`, so the signature
        of a stored index is checked against the bytes it was made over.
        """
        return {
            "schema_version": self.version,
            "records": [
                records.downgrade(
                    Record.migrations, record.to_dict(), self.version
                )
                for record in self.ordered()
            ],
        }

    def sign(self, signing_key: str | bytes) -> str:
        """Signs the complete index and returns its HMAC value.

        Args:
            signing_key: The key to sign with.

        Returns:
            The signature, which is also stored on the index.
        """
        self.signature = signing.mac(self.preimage(), signing_key)
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
        expected = signing.mac(self.preimage(), signing_key)
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> dict[str, Any]:
        """Returns a deterministic, JSON-compatible evidence index."""
        record = self.preimage()
        if self.signature is not None:
            record["signature"] = {
                "algorithm": signing.SIGNATURE_ALGORITHM,
                "value": self.signature,
            }
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds and validates a versioned evidence index.

        An index stored at an older schema version keeps that version, so
        its signature verifies against the layout it was made over, while
        its records are migrated to the current layout.

        Args:
            record: A mapping with a supported schema version, a list of
                records and an optional signature object.

        Returns:
            The index, signed if the record carried a signature.

        Raises:
            ValueError: If the schema version is unsupported, the records are
                not a list, or the signature is malformed or uses an
                unsupported algorithm.
        """
        version = record.get("schema_version")
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or not records.ORIGIN <= version <= VERSION
        ):
            raise errors.InputError("unsupported evidence index schema version")
        raw_records = record.get("records")
        if not isinstance(raw_records, list):
            raise errors.InputError("evidence index records must be a list")
        signature_record = record.get("signature")
        signature: str | None = None
        if signature_record is not None:
            if not isinstance(signature_record, Mapping):
                raise errors.InputError(
                    "evidence index signature must be an object"
                )
            if signature_record.get("algorithm") != signing.SIGNATURE_ALGORITHM:
                raise errors.InputError(
                    "unsupported evidence index signature algorithm"
                )
            signature = require(
                str(signature_record.get("value", "")),
                "signature",
            )
        index = cls(
            Record.from_dict(
                records.upgrade(Record.migrations, dict(item), version)
            )
            for item in raw_records
        )
        index.signature = signature
        index.version = version
        return index

    def canonize(self) -> str:
        """Returns stable JSON suitable for hashing and signing."""
        return json.dumps(
            self.preimage(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
