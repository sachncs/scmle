"""Row kinds, the hash of a record and the hypothesis records."""

from collections.abc import Mapping
import dataclasses
from fractions import Fraction
import hashlib
from typing import Any
from typing import ClassVar

from scmle.assurance.ledger import bivariate
from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.contracts import records
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
class Digests:
    """Hashes a hypothesis specification binds.

    Attributes:
        code: Hash of the code path that computes the statistic.
    """

    code: str


@dataclasses.dataclass(frozen=True, slots=True)
class Test:
    """What is tested, fixed before the data is touched.

    The specification is stored at schema version 2; a version 1 record
    (`set`, `code_hash`) loads through `migrations`. The survivor
    fingerprint is computed on the version 1 layout (`anchor`), so a frozen
    selection keeps its hash.

    Attributes:
        name: Human-readable name, unique within the ledger.
        family: Family the hypothesis belongs to (for example `bivariate`).
        statistic: Registered statistic (see `bivariate.STATISTICS`).
        null: The null hypothesis, in words.
        direction: `two-sided`, `greater` or `less`.
        subset: Set the test reads; exploration tests read `X`.
        column: Feature column the statistic reads, if any.
        digests: Hashes the specification binds (the statistic's code).
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            renames={"set": "subset"},
            groups={"digests": {"code_hash": "code"}},
        ),
    )

    name: str
    family: str
    statistic: str
    null: str
    direction: str
    subset: qualification.Subset
    column: str | None
    digests: Digests

    def __post_init__(self) -> None:
        """Validates the specification.

        Raises:
            ValueError: If `direction` is not a known alternative,
                `subset` is not a `Subset` or `digests` is not a `Digests`.
                A blank text field is refused with a `ValueError` by
                `evidence.require_text`.
        """
        if not isinstance(self.digests, Digests):
            raise errors.InputError("digests must be a Digests")
        for field, value in (
            ("name", self.name),
            ("family", self.family),
            ("statistic", self.statistic),
            ("null", self.null),
            ("digests.code", self.digests.code),
        ):
            scmle.contracts.evidence.require(value, field)
        if self.direction not in bivariate.DIRECTIONS:
            raise errors.InputError(
                f"direction must be one of {bivariate.DIRECTIONS}"
            )
        if not isinstance(self.subset, qualification.Subset):
            raise errors.InputError("subset must be a Subset")

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
        subset: qualification.Subset = qualification.Subset.EXPLORATION,
    ) -> "Test":
        """Builds a specification, filling the code hash from the registry.

        Args:
            name: Human-readable name, unique within the ledger.
            family: Family the hypothesis belongs to.
            statistic: Registered statistic; an unknown name is refused by
                `bivariate.fingerprint` with a `ValueError`.
            null: The null hypothesis, in words.
            column: Feature column the statistic reads, if any.
            direction: `two-sided`, `greater` or `less`.
            subset: Set the test reads (exploration by default).

        Returns:
            The specification, with the hash of the statistic's code.
        """
        return cls(
            name=name,
            family=family,
            statistic=statistic,
            null=null,
            direction=direction,
            subset=subset,
            column=column,
            digests=Digests(code=bivariate.fingerprint(statistic)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record at the current schema version.

        Returns:
            A mapping of every field and the `schema_version`; the set is
            stored as its value and the digests as a nested mapping.
        """
        return {
            records.KEY: records.ORIGIN + len(self.migrations),
            "name": self.name,
            "family": self.family,
            "statistic": self.statistic,
            "null": self.null,
            "direction": self.direction,
            "subset": self.subset.value,
            "column": self.column,
            "digests": {"code": self.digests.code},
        }

    def anchor(self) -> records.Layout:
        """Builds the record in its version 1 layout.

        Returns:
            The specification as version 1 stored it (`set`, `code_hash`),
            without `schema_version`.
        """
        current = self.to_dict()
        del current[records.KEY]
        return records.downgrade(self.migrations, current)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Test":
        """Rebuilds a specification, rejecting unknown or missing fields.

        Args:
            record: A mapping as written by `to_dict`, at any schema
                version.

        Returns:
            The specification described by the record.

        Raises:
            ValueError: If the record has unexpected or missing fields or a
                newer schema version. An unknown set value or an invalid
                field is also refused with a `ValueError`.
        """
        record = records.load(cls.__name__, cls.migrations, record)
        del record[records.KEY]
        names = {
            "name",
            "family",
            "statistic",
            "null",
            "direction",
            "subset",
            "column",
            "digests",
        }
        digests = record.get("digests")
        if (
            set(record) != names
            or not isinstance(digests, Mapping)
            or set(digests) != {"code"}
        ):
            raise errors.InputError("unexpected or missing hypothesis fields")
        return cls(
            name=record["name"],
            family=record["family"],
            statistic=record["statistic"],
            null=record["null"],
            direction=record["direction"],
            subset=qualification.Subset(record["subset"]),
            column=record["column"],
            digests=Digests(code=digests["code"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Hypothesis:
    """A registered hypothesis.

    Attributes:
        identifier: Identifier assigned at registration (`h000001`, ...).
        spec: The specification.
    """

    identifier: str
    spec: Test


@dataclasses.dataclass(frozen=True, slots=True)
class Outcome:
    """The recorded outcome of one hypothesis.

    Attributes:
        hypothesis: The hypothesis.
        check: The `Check` (None when the hypothesis was not tested).
        e_value: E-value used by e-LOND, or None.
        level: Test level the rule assigned.
        decision: `reject`, `retain` or `not_tested`.
        wealth_before: Wealth before the test.
        wealth_after: Wealth after the test.
        reason: Why a hypothesis was not tested, or an empty string.
    """

    hypothesis: str
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
        hypotheses: Rejected hypotheses, in registration order.
        sha256: Hash of their ids and specifications.
    """

    hypotheses: tuple[str, ...]
    sha256: str


def fingerprint(ids: tuple[str, ...], specs: Mapping[str, Test]) -> str:
    """Hashes the survivor ids together with their specifications.

    Args:
        ids: Survivor hypothesis ids, in registration order.
        specs: Specification of every hypothesis, by id.

    Returns:
        The digest that binds the ids to their specifications, computed on
        the version 1 layout of each specification (`Test.anchor`).
    """
    return digest({"ids": list(ids), "specs": [specs[i].anchor() for i in ids]})
