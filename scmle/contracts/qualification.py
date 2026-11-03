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
from typing import Self

from scipy import stats

from scmle.contracts import evidence

SeedPath = tuple[str | int, ...]
DEFAULT_CONFIDENCE = 0.95


class Tier(enum.StrEnum):
    """How strongly a result is supported."""

    CONFIRMED = "confirmed"
    ATTESTED = "attested"
    UNVERIFIED = "unverified"


class Decision(enum.StrEnum):
    """Action a check triggers."""

    PASS = "pass"  # nosec B105 - check outcome, not a credential
    WARN = "warn"
    QUARANTINE = "quarantine"
    BLOCK = "block"


class SetId(enum.StrEnum):
    """Which data set a check read."""

    EXPLORATION = "X"
    CONFIRMATION = "C"
    FORWARD = "F"
    ALL = "all"


def finite(value: Any, name: str) -> float:
    """Returns `value` as a float, requiring a finite real number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def probability(value: Any, name: str) -> float:
    """Returns `value` as a float in [0, 1]."""
    number = finite(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return number


def count(value: Any, name: str, minimum: int = 0) -> int:
    """Returns `value` as an integer of at least `minimum`."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        label = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise ValueError(f"{name} must be {label}")
    return value


def require_fields(cls: type, record: Mapping[str, Any]) -> None:
    """Requires `record` to hold exactly the dataclass fields of `cls`."""
    names = {field.name for field in dataclasses.fields(cls)}
    unexpected = sorted(set(record) - names)
    missing = sorted(names - set(record))
    if unexpected:
        raise ValueError(f"unexpected field(s): {unexpected}")
    if missing:
        raise ValueError(f"missing field(s): {missing}")


def require_tuple(value: Any, name: str) -> None:
    """Requires a tuple so that records compare equal after a round trip."""
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple")


@dataclasses.dataclass(frozen=True, slots=True)
class Check:
    """One statistical test or estimator and its decision.

    Attributes:
        check_id: Stable identifier of the check.
        null_hypothesis: The null being tested, in words.
        statistic: Test statistic or estimate.
        p_value: Probability under the null, when the check is a test.
        interval: Confidence interval, when the check is an estimator.
        effect: Effect size on a scale stated by the check.
        n: Rows the check used.
        decision: Action the result triggers.
        set_id: Data set the check read.
        seed_path: (run seed, component name, index) of its random stream.
        tier: Evidence tier. `confirmed` requires the confirmation set.
    """

    check_id: str
    null_hypothesis: str
    statistic: float
    p_value: float | None
    interval: tuple[float, float] | None
    effect: float
    n: int
    decision: Decision
    set_id: SetId
    seed_path: SeedPath
    tier: Tier

    def __post_init__(self) -> None:
        """Validates the record."""
        evidence.require_text(self.check_id, "check_id")
        evidence.require_text(self.null_hypothesis, "null_hypothesis")
        finite(self.statistic, "statistic")
        finite(self.effect, "effect")
        count(self.n, "n")
        if self.p_value is not None:
            probability(self.p_value, "p_value")
        if self.interval is not None:
            require_tuple(self.interval, "interval")
            if len(self.interval) != 2:
                raise ValueError("interval must hold (low, high)")
            low = finite(self.interval[0], "interval")
            high = finite(self.interval[1], "interval")
            if low > high:
                raise ValueError("interval must satisfy low <= high")
        if self.p_value is None and self.interval is None:
            raise ValueError("a check needs a p_value or interval")
        require_tuple(self.seed_path, "seed_path")
        for decision, kind in (
            (self.decision, Decision),
            (self.set_id, SetId),
            (self.tier, Tier),
        ):
            if not isinstance(decision, kind):
                raise ValueError(f"expected a {kind.__name__}")
        on_confirmation = self.set_id is SetId.CONFIRMATION
        if self.tier is Tier.CONFIRMED and not on_confirmation:
            raise ValueError("confirmed tier requires the confirmation set 'C'")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "check_id": self.check_id,
            "null_hypothesis": self.null_hypothesis,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "interval": None if self.interval is None else list(self.interval),
            "effect": self.effect,
            "n": self.n,
            "decision": self.decision.value,
            "set_id": self.set_id.value,
            "seed_path": list(self.seed_path),
            "tier": self.tier.value,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a check, rejecting unknown or missing fields."""
        require_fields(cls, record)
        interval = record["interval"]
        return cls(
            check_id=record["check_id"],
            null_hypothesis=record["null_hypothesis"],
            statistic=record["statistic"],
            p_value=record["p_value"],
            interval=None if interval is None else tuple(interval),
            effect=record["effect"],
            n=record["n"],
            decision=Decision(record["decision"]),
            set_id=SetId(record["set_id"]),
            seed_path=tuple(record["seed_path"]),
            tier=Tier(record["tier"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Claim:
    """A finding that survived exploration and was tested on `C`.

    Attributes:
        claim_id: Stable identifier of the claim.
        statement: The finding, in words.
        check_ids: Checks that support it.
        adjusted_p_value: Multiplicity-adjusted p-value from confirmation.
        effect: Confirmed effect size.
        tier: Evidence tier. `confirmed` needs an adjusted p-value.
        assumptions: Assumptions the guarantee depends on.
    """

    claim_id: str
    statement: str
    check_ids: tuple[str, ...]
    adjusted_p_value: float | None
    effect: float
    tier: Tier
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validates the record."""
        evidence.require_text(self.claim_id, "claim_id")
        evidence.require_text(self.statement, "statement")
        require_tuple(self.check_ids, "check_ids")
        require_tuple(self.assumptions, "assumptions")
        if not self.check_ids:
            raise ValueError("check_ids must not be empty")
        for check_id in self.check_ids:
            evidence.require_text(check_id, "check_ids")
        if self.adjusted_p_value is not None:
            probability(self.adjusted_p_value, "adjusted_p_value")
        finite(self.effect, "effect")
        if not isinstance(self.tier, Tier):
            raise ValueError("expected a Tier")
        if self.tier is Tier.CONFIRMED and self.adjusted_p_value is None:
            raise ValueError("confirmed claims need an adjusted_p_value")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "check_ids": list(self.check_ids),
            "adjusted_p_value": self.adjusted_p_value,
            "effect": self.effect,
            "tier": self.tier.value,
            "assumptions": list(self.assumptions),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a claim, rejecting unknown or missing fields."""
        require_fields(cls, record)
        return cls(
            claim_id=record["claim_id"],
            statement=record["statement"],
            check_ids=tuple(record["check_ids"]),
            adjusted_p_value=record["adjusted_p_value"],
            effect=record["effect"],
            tier=Tier(record["tier"]),
            assumptions=tuple(record["assumptions"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Certificate:
    """Measured operating characteristics of a detector.

    The detector ran on `planted` known defects and on `clean_rows` rows
    without defects. Plants are synthetic, so recall is optimistic; the
    Wilson lower bound states how well the count supports it.

    Attributes:
        detector_id: Detector that was measured.
        defect_type: Kind of defect planted.
        planted: Number of planted defects.
        detected: Planted defects the detector found.
        clean_rows: Rows in the clean copy.
        false_alarms: Clean rows the detector flagged.
        seed_path: (run seed, component name, index) of the injection.
    """

    detector_id: str
    defect_type: str
    planted: int
    detected: int
    clean_rows: int
    false_alarms: int
    seed_path: SeedPath

    def __post_init__(self) -> None:
        """Validates the record."""
        evidence.require_text(self.detector_id, "detector_id")
        evidence.require_text(self.defect_type, "defect_type")
        count(self.planted, "planted", minimum=1)
        count(self.detected, "detected")
        if self.detected > self.planted:
            raise ValueError("detected must be between 0 and planted")
        count(self.clean_rows, "clean_rows", minimum=1)
        count(self.false_alarms, "false_alarms")
        if self.false_alarms > self.clean_rows:
            raise ValueError("false_alarms must be between 0 and clean_rows")
        require_tuple(self.seed_path, "seed_path")

    @property
    def recall(self) -> float:
        """Share of planted defects that were found."""
        return self.detected / self.planted

    @property
    def false_alarm_rate(self) -> float:
        """Share of clean rows that were flagged."""
        return self.false_alarms / self.clean_rows

    def recall_lower_bound(
        self, confidence: float = DEFAULT_CONFIDENCE
    ) -> float:
        """Lower end of the two-sided Wilson interval for recall.

        Args:
            confidence: Level of the two-sided interval, in (0, 1).

        Returns:
            A value in [0, recall].

        Raises:
            ValueError: If `confidence` is outside (0, 1).
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be within (0, 1)")
        z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
        n = self.planted
        p = self.recall
        scale = 1.0 + z * z / n
        centre = (p + z * z / (2.0 * n)) / scale
        half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / scale
        return max(0.0, centre - half)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "detector_id": self.detector_id,
            "defect_type": self.defect_type,
            "planted": self.planted,
            "detected": self.detected,
            "clean_rows": self.clean_rows,
            "false_alarms": self.false_alarms,
            "seed_path": list(self.seed_path),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a certificate, rejecting unknown or missing fields."""
        require_fields(cls, record)
        return cls(
            detector_id=record["detector_id"],
            defect_type=record["defect_type"],
            planted=record["planted"],
            detected=record["detected"],
            clean_rows=record["clean_rows"],
            false_alarms=record["false_alarms"],
            seed_path=tuple(record["seed_path"]),
        )
