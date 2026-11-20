"""Records for statistical data qualification.

A `Check` is one test or estimator with its null, statistic, evidence and
decision. A `Claim` is a finding tested on the confirmation set. A
`Certificate` records the measured operating characteristics of a detector on
planted defects. Every record carries an evidence `Tier`, and only a record
tied to the confirmation set may be `confirmed`.
"""

from collections.abc import Mapping
import dataclasses
import enum
import math
from typing import Any
from typing import ClassVar
from typing import Self

from scipy import stats

from scmle.contracts import errors
from scmle.contracts import evidence
from scmle.contracts import records

SeedPath = tuple[str | int, ...]
DEFAULT_CONFIDENCE = 0.95
INTERVAL_LENGTH = 2


class Tier(enum.StrEnum):
    """How strongly a result is supported."""

    CONFIRMED = "confirmed"
    ATTESTED = "attested"
    UNVERIFIED = "unverified"


class Decision(enum.StrEnum):
    """Action a check triggers."""

    PASS = "pass"  # noqa: S105 -- check outcome, not a credential # nosec B105
    WARN = "warn"
    QUARANTINE = "quarantine"
    BLOCK = "block"


class Subset(enum.StrEnum):
    """Which data set a check read."""

    EXPLORATION = "X"
    CONFIRMATION = "C"
    FORWARD = "F"
    ALL = "all"


def finite(value: Any, name: str) -> float:
    """Returns `value` as a float, requiring a finite real number.

    Args:
        value: The value to check.
        name: The field name, for the error message.

    Returns:
        The value as a float.

    Raises:
        ValueError: If the value is a bool, not an int or float, or not
            finite.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise errors.InputError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise errors.InputError(f"{name} must be a finite number")
    return number


def probability(value: Any, name: str) -> float:
    """Returns `value` as a float in [0, 1].

    Args:
        value: The value to check.
        name: The field name, for the error message.

    Returns:
        The value as a float.

    Raises:
        ValueError: If the value is not a finite number within [0, 1].
    """
    number = finite(value, name)
    if not 0.0 <= number <= 1.0:
        raise errors.InputError(f"{name} must be within [0, 1]")
    return number


def count(value: Any, name: str, minimum: int = 0) -> int:
    """Returns `value` as an integer of at least `minimum`.

    Args:
        value: The value to check.
        name: The field name, for the error message.
        minimum: The smallest accepted value.

    Returns:
        The value, unchanged.

    Raises:
        ValueError: If the value is not an integer (a bool does not count) or
            is below the minimum.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise errors.InputError(f"{name} must be an integer")
    if value < minimum:
        label = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise errors.InputError(f"{name} must be {label}")
    return value


def require(cls: Any, record: Mapping[str, Any]) -> records.Layout:
    """Migrates `record` and requires it to hold the dataclass fields of `cls`.

    Args:
        cls: The dataclass whose fields are expected; its `migrations`, if
            it has any, lift a record stored at an older schema version.
        record: The mapping to check, at any supported schema version.

    Returns:
        The record (at the current version, without its version key, when
        `cls` has migrations).

    Raises:
        ValueError: If the record has a field `cls` lacks or lacks one it
            has, or its schema version is unsupported.
    """
    steps = getattr(cls, "migrations", None)
    if steps is None:
        record = dict(record)
    else:
        record = records.load(cls.__name__, steps, record)
        del record[records.KEY]
    names = {field.name for field in dataclasses.fields(cls)}
    unexpected = sorted(set(record) - names)
    missing = sorted(names - set(record))
    if unexpected:
        raise errors.InputError(f"unexpected field(s): {unexpected}")
    if missing:
        raise errors.InputError(f"missing field(s): {missing}")
    return record


def tupled(value: Any, name: str) -> None:
    """Requires a tuple so that records compare equal after a round trip.

    Args:
        value: The value to check.
        name: The field name, for the error message.

    Raises:
        ValueError: If the value is not a tuple.
    """
    if not isinstance(value, tuple):
        raise errors.InputError(f"{name} must be a tuple")


@dataclasses.dataclass(frozen=True, slots=True)
class Check:
    """One statistical test or estimator and its decision.

    Attributes:
        identifier: Stable identifier of the check.
        null: The null hypothesis being tested, in words.
        statistic: Test statistic or estimate.
        p_value: Probability under the null, when the check is a test.
        interval: Confidence interval, when the check is an estimator.
        effect: Effect size on a scale stated by the check.
        n: Rows the check used.
        decision: Action the result triggers.
        subset: Data set the check read.
        seeds: (run seed, component name, index) of its random stream.
        tier: Evidence tier. `confirmed` requires the confirmation set.
        migrations: Version steps of the stored check.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "check_id": "identifier",
                "null_hypothesis": "null",
                "set_id": "subset",
                "seed_path": "seeds",
            }
        ),
    )

    identifier: str
    null: str
    statistic: float
    p_value: float | None
    interval: tuple[float, float] | None
    effect: float
    n: int
    decision: Decision
    subset: Subset
    seeds: SeedPath
    tier: Tier

    def __post_init__(self) -> None:
        """Validates the record.

        Raises:
            ValueError: If an identity is blank, the statistic or effect is
                not a finite number, `n` is not a non-negative integer, the
                p-value is not within [0, 1], the interval is not a pair of
                finite numbers in a tuple with low <= high, neither a p-value
                nor an interval is given, the seed path is not a tuple, a
                decision, set or tier has the wrong type, or the confirmed
                tier is not on the confirmation set.
        """
        evidence.require(self.identifier, "identifier")
        evidence.require(self.null, "null")
        finite(self.statistic, "statistic")
        finite(self.effect, "effect")
        count(self.n, "n")
        if self.p_value is not None:
            probability(self.p_value, "p_value")
        if self.interval is not None:
            tupled(self.interval, "interval")
            if len(self.interval) != INTERVAL_LENGTH:
                raise errors.InputError("interval must hold (low, high)")
            low = finite(self.interval[0], "interval")
            high = finite(self.interval[1], "interval")
            if low > high:
                raise errors.InputError("interval must satisfy low <= high")
        if self.p_value is None and self.interval is None:
            raise errors.InputError("a check needs a p_value or interval")
        tupled(self.seeds, "seeds")
        for decision, kind in (
            (self.decision, Decision),
            (self.subset, Subset),
            (self.tier, Tier),
        ):
            if not isinstance(decision, kind):
                raise errors.InputError(f"expected a {kind.__name__}")
        on_confirmation = self.subset is Subset.CONFIRMATION
        if self.tier is Tier.CONFIRMED and not on_confirmation:
            raise errors.InputError(
                "confirmed tier requires the confirmation set 'C'"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-ready record.

        Returns:
            The check's fields as a dict, with enums as their values and
            sequences as lists.
        """
        return {
            "schema_version": records.ORIGIN + len(self.migrations),
            "identifier": self.identifier,
            "null": self.null,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "interval": None if self.interval is None else list(self.interval),
            "effect": self.effect,
            "n": self.n,
            "decision": self.decision.value,
            "subset": self.subset.value,
            "seeds": list(self.seeds),
            "tier": self.tier.value,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a check, rejecting unknown or missing fields.

        Args:
            record: A mapping as produced by `to_dict`, at any supported
                schema version.

        Returns:
            The check.
        """
        record = require(cls, record)
        interval = record["interval"]
        return cls(
            identifier=record["identifier"],
            null=record["null"],
            statistic=record["statistic"],
            p_value=record["p_value"],
            interval=None if interval is None else tuple(interval),
            effect=record["effect"],
            n=record["n"],
            decision=Decision(record["decision"]),
            subset=Subset(record["subset"]),
            seeds=tuple(record["seeds"]),
            tier=Tier(record["tier"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Claim:
    """A finding that survived exploration and was tested on `C`.

    Attributes:
        identifier: Stable identifier of the claim.
        statement: The finding, in words.
        checks: Ids of the checks that support it.
        adjusted_p_value: Multiplicity-adjusted p-value from confirmation.
        effect: Confirmed effect size.
        tier: Evidence tier. `confirmed` needs an adjusted p-value.
        assumptions: Assumptions the guarantee depends on.
        migrations: Version steps of the stored claim.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={"claim_id": "identifier", "check_ids": "checks"}
        ),
    )

    identifier: str
    statement: str
    checks: tuple[str, ...]
    adjusted_p_value: float | None
    effect: float
    tier: Tier
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validates the record.

        Raises:
            ValueError: If the id or statement is blank, `checks` or
                `assumptions` is not a tuple, there are no check ids or one is
                blank, the adjusted p-value is not within [0, 1], the effect
                is not a finite number, the tier has the wrong type, or a
                confirmed claim has no adjusted p-value.
        """
        evidence.require(self.identifier, "identifier")
        evidence.require(self.statement, "statement")
        tupled(self.checks, "checks")
        tupled(self.assumptions, "assumptions")
        if not self.checks:
            raise errors.InputError("checks must not be empty")
        for check in self.checks:
            evidence.require(check, "checks")
        if self.adjusted_p_value is not None:
            probability(self.adjusted_p_value, "adjusted_p_value")
        finite(self.effect, "effect")
        if not isinstance(self.tier, Tier):
            raise errors.InputError("expected a Tier")
        if self.tier is Tier.CONFIRMED and self.adjusted_p_value is None:
            raise errors.InputError("confirmed claims need an adjusted_p_value")

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-ready record.

        Returns:
            The claim's fields as a dict, with sequences as lists and the tier
            as its value.
        """
        return {
            "schema_version": records.ORIGIN + len(self.migrations),
            "identifier": self.identifier,
            "statement": self.statement,
            "checks": list(self.checks),
            "adjusted_p_value": self.adjusted_p_value,
            "effect": self.effect,
            "tier": self.tier.value,
            "assumptions": list(self.assumptions),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a claim, rejecting unknown or missing fields.

        Args:
            record: A mapping as produced by `to_dict`, at any supported
                schema version.

        Returns:
            The claim.
        """
        record = require(cls, record)
        return cls(
            identifier=record["identifier"],
            statement=record["statement"],
            checks=tuple(record["checks"]),
            adjusted_p_value=record["adjusted_p_value"],
            effect=record["effect"],
            tier=Tier(record["tier"]),
            assumptions=tuple(record["assumptions"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Certificate:
    """Measured operating characteristics of a detector.

    The detector ran on `planted` known defects and on `clean` rows
    without defects. Plants are synthetic, so recall is optimistic; the
    Wilson lower bound states how well the count supports it.

    Attributes:
        detector: Id of the detector that was measured.
        defect: Kind of defect planted.
        planted: Number of planted defects.
        detected: Planted defects the detector found.
        clean: Rows in the clean copy.
        alarms: Clean rows the detector flagged (false alarms).
        seeds: (run seed, component name, index) of the injection.
        migrations: Version steps of the stored certificate.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "detector_id": "detector",
                "defect_type": "defect",
                "clean_rows": "clean",
                "false_alarms": "alarms",
                "seed_path": "seeds",
            }
        ),
    )

    detector: str
    defect: str
    planted: int
    detected: int
    clean: int
    alarms: int
    seeds: SeedPath

    def __post_init__(self) -> None:
        """Validates the record.

        Raises:
            ValueError: If an identity is blank, a count is not an integer,
                the planted or clean-row count is below one, the detected or
                false-alarm count is negative, the detected count exceeds the
                planted count, the alarms exceed the clean rows, or the
                seed path is not a tuple.
        """
        evidence.require(self.detector, "detector")
        evidence.require(self.defect, "defect")
        count(self.planted, "planted", minimum=1)
        count(self.detected, "detected")
        if self.detected > self.planted:
            raise errors.InputError("detected must be between 0 and planted")
        count(self.clean, "clean", minimum=1)
        count(self.alarms, "alarms")
        if self.alarms > self.clean:
            raise errors.InputError("alarms must be between 0 and clean")
        tupled(self.seeds, "seeds")

    @property
    def recall(self) -> float:
        """Share of planted defects that were found."""
        return self.detected / self.planted

    @property
    def fallout(self) -> float:
        """Share of clean rows that were flagged."""
        return self.alarms / self.clean

    def bound(self, confidence: float = DEFAULT_CONFIDENCE) -> float:
        """Returns the lower end of the two-sided Wilson interval for recall.

        Args:
            confidence: Level of the two-sided interval, in (0, 1).

        Returns:
            A value in [0, recall].

        Raises:
            ValueError: If `confidence` is outside (0, 1).
        """
        if not 0.0 < confidence < 1.0:
            raise errors.InputError("confidence must be within (0, 1)")
        z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
        n = self.planted
        p = self.recall
        scale = 1.0 + z * z / n
        centre = (p + z * z / (2.0 * n)) / scale
        half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / scale
        return max(0.0, centre - half)

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-ready record.

        Returns:
            The certificate's fields as a dict, with the seed path as a list.
        """
        return {
            "schema_version": records.ORIGIN + len(self.migrations),
            "detector": self.detector,
            "defect": self.defect,
            "planted": self.planted,
            "detected": self.detected,
            "clean": self.clean,
            "alarms": self.alarms,
            "seeds": list(self.seeds),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a certificate, rejecting unknown or missing fields.

        Args:
            record: A mapping as produced by `to_dict`, at any supported
                schema version.

        Returns:
            The certificate.
        """
        record = require(cls, record)
        return cls(
            detector=record["detector"],
            defect=record["defect"],
            planted=record["planted"],
            detected=record["detected"],
            clean=record["clean"],
            alarms=record["alarms"],
            seeds=tuple(record["seeds"]),
        )
