"""Row kinds, the hash of a record and the hypothesis records."""

from collections.abc import Mapping
import dataclasses
from fractions import Fraction
import hashlib
from typing import Any

from scmle.assurance.ledger import bivariate
from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.contracts import signing
import scmle.contracts.evidence

GENESIS = "0" * 64
LEDGER_VERSION = 1
FILE_NAME = "hypothesis_ledger.jsonl"
REJECT = "reject"
RETAIN = "retain"
NOT_TESTED = "not_tested"
OPEN = "open"
REGISTER = "register"
RESULT = "result"
CONTROL = "control"
DIAGNOSTIC = "diagnostic"
CLAIM = "claim"
SURVIVORS = "survivors"
CONFIRMATION = "confirmation"


class LedgerError(errors.ArtifactError):
    """The ledger was misused or its contents are inconsistent."""


def digest(body: Mapping[str, Any]) -> str:
    """Hashes the canonical JSON of a record.

    Args:
        body: The JSON-ready record.

    Returns:
        The SHA-256 hex digest of the canonical JSON form of `body`.
    """
    return hashlib.sha256(signing.canonicalise(body)).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class Test:
    """What is tested, fixed before the data is touched.

    Attributes:
        name: Human-readable name, unique within the ledger.
        family: Family the hypothesis belongs to (for example `bivariate`).
        statistic: Registered statistic (see `bivariate.STATISTICS`).
        null: The null hypothesis, in words.
        direction: `two-sided`, `greater` or `less`.
        set_id: Set the test reads; exploration tests read `X`.
        column: Feature column the statistic reads, if any.
        code_hash: Hash of the code path that computes the statistic.
    """

    name: str
    family: str
    statistic: str
    null: str
    direction: str
    set_id: qualification.Subset
    column: str | None
    code_hash: str

    def __post_init__(self) -> None:
        """Validates the specification.

        Raises:
            ValueError: If `direction` is not a known alternative or
                `set_id` is not a `Subset`. A blank text field is refused with
                a `ValueError` by `evidence.require_text`.
        """
        for field, value in (
            ("name", self.name),
            ("family", self.family),
            ("statistic", self.statistic),
            ("null", self.null),
            ("code_hash", self.code_hash),
        ):
            scmle.contracts.evidence.require_text(value, field)
        if self.direction not in bivariate.DIRECTIONS:
            raise errors.InputError(
                f"direction must be one of {bivariate.DIRECTIONS}"
            )
        if not isinstance(self.set_id, qualification.Subset):
            raise errors.InputError("set_id must be a Subset")

    @classmethod
    def create(
        cls,
        *,
        name: str,
        family: str,
        statistic: str,
        null: str,
        column: str | None = None,
        direction: str = bivariate.TWO_SIDED,
        set_id: qualification.Subset = qualification.Subset.EXPLORATION,
    ) -> "Test":
        """Builds a specification, filling the code hash from the registry.

        Args:
            name: Human-readable name, unique within the ledger.
            family: Family the hypothesis belongs to.
            statistic: Registered statistic; an unknown name is refused by
                `bivariate.code_hash` with a `ValueError`.
            null: The null hypothesis, in words.
            column: Feature column the statistic reads, if any.
            direction: `two-sided`, `greater` or `less`.
            set_id: Set the test reads (exploration by default).

        Returns:
            The specification, with the hash of the statistic's code.
        """
        return cls(
            name=name,
            family=family,
            statistic=statistic,
            null=null,
            direction=direction,
            set_id=set_id,
            column=column,
            code_hash=bivariate.code_hash(statistic),
        )

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record.

        Returns:
            A mapping of every field; the set is stored under ``set`` as its
            value.
        """
        return {
            "name": self.name,
            "family": self.family,
            "statistic": self.statistic,
            "null": self.null,
            "direction": self.direction,
            "set": self.set_id.value,
            "column": self.column,
            "code_hash": self.code_hash,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Test":
        """Rebuilds a specification, rejecting unknown or missing fields.

        Args:
            record: A mapping as written by `to_dict`.

        Returns:
            The specification described by the record.

        Raises:
            ValueError: If the record has unexpected or missing fields. An
                unknown set value or an invalid field is also refused with a
                `ValueError`.
        """
        names = {
            "name",
            "family",
            "statistic",
            "null",
            "direction",
            "set",
            "column",
            "code_hash",
        }
        if set(record) != names:
            raise errors.InputError("unexpected or missing hypothesis fields")
        return cls(
            name=record["name"],
            family=record["family"],
            statistic=record["statistic"],
            null=record["null"],
            direction=record["direction"],
            set_id=qualification.Subset(record["set"]),
            column=record["column"],
            code_hash=record["code_hash"],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Hypothesis:
    """A registered hypothesis.

    Attributes:
        hypothesis_id: Identifier assigned at registration (`h000001`, ...).
        spec: The specification.
    """

    hypothesis_id: str
    spec: Test


@dataclasses.dataclass(frozen=True, slots=True)
class Outcome:
    """The recorded outcome of one hypothesis.

    Attributes:
        hypothesis_id: The hypothesis.
        check: The `Check` (None when the hypothesis was not tested).
        e_value: E-value used by e-LOND, or None.
        level: Test level the rule assigned.
        decision: `reject`, `retain` or `not_tested`.
        wealth_before: Wealth before the test.
        wealth_after: Wealth after the test.
        reason: Why a hypothesis was not tested, or an empty string.
    """

    hypothesis_id: str
    check: qualification.Check | None
    e_value: float | None
    level: Fraction
    decision: str
    wealth_before: Fraction
    wealth_after: Fraction
    reason: str = ""

    @property
    def rejected(self) -> bool:
        """Whether the hypothesis was rejected."""
        return self.decision == REJECT


@dataclasses.dataclass(frozen=True, slots=True)
class Selection:
    """The frozen selection that goes to the confirmation set.

    Attributes:
        hypothesis_ids: Rejected hypotheses, in registration order.
        sha256: Hash of their ids and specifications.
    """

    hypothesis_ids: tuple[str, ...]
    sha256: str


def fingerprint(ids: tuple[str, ...], specs: Mapping[str, Test]) -> str:
    """Hashes the survivor ids together with their specifications.

    Args:
        ids: Survivor hypothesis ids, in registration order.
        specs: Specification of every hypothesis, by id.

    Returns:
        The digest that binds the ids to their specifications.
    """
    return digest(
        {"ids": list(ids), "specs": [specs[i].to_dict() for i in ids]}
    )
