"""The error budget: exploration wealth and confirmation alpha.

The exploration alpha-wealth is a triage budget; the final claim guarantee
comes only from `alpha_confirm`, spent once on the confirmation set.
The two never add, so they are distinct types with no shared arithmetic, and
`spend` can only touch the exploration side.

Amounts are exact `fractions.Fraction` values (a decimal float is read as it
is written, so `0.05` is exactly 1/20). That makes conservation of wealth and
order-independence of spending exact rather than true up to rounding.
"""

from collections.abc import Mapping
import dataclasses
import enum
from fractions import Fraction
import math
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.contracts import errors
from scmle.contracts import evidence
from scmle.contracts import qualification
from scmle.contracts import records

MIN_PERMUTATIONS = 20
EXACT_HARMONIC_LIMIT = 10_000
EULER_GAMMA = 0.5772156649015329

Number = int | float | Fraction


class ExhaustedError(ValueError):
    """Raised when a spend exceeds the remaining exploration wealth."""


class Correction(enum.StrEnum):
    """Rule that adjusts the confirmation level for the survivor count."""

    BONFERRONI = "bonferroni"
    HOLM = "holm"
    BH = "bh"
    BY = "by"


def exact(value: Any, name: str) -> Fraction:
    """Reads a number exactly; a float is taken as its shortest decimal.

    Args:
        value: An int, a float or a `Fraction`; a bool is refused.
        name: Field name used in the error message.

    Returns:
        The value as an exact fraction.

    Raises:
        ValueError: If `value` is a bool, a non-finite float or not a
            number.
    """
    if isinstance(value, bool):
        raise errors.InputError(f"{name} must be a finite number")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float) and math.isfinite(value):
        return Fraction(repr(value))
    raise errors.InputError(f"{name} must be a finite number")


def parse(text: Any, name: str) -> Fraction:
    """Parses a serialized fraction such as ``"1/20"``.

    Args:
        text: The serialized fraction (``"1/20"`` or a decimal such as
            ``"0.05"``); any value is converted with `str`.
        name: Field name used in the error message.

    Returns:
        The parsed fraction.

    Raises:
        ValueError: If the text is not a valid fraction or has a zero
            denominator.
    """
    try:
        return Fraction(str(text))
    except (ValueError, ZeroDivisionError) as error:
        raise errors.InputError(f"{name}: invalid fraction {text!r}") from error


@dataclasses.dataclass(frozen=True, slots=True)
class Confirmation:
    """Level spent once on the confirmation set.

    Attributes:
        alpha_confirm: Target FDR (or FWER) of the final claims, in (0, 1).
    """

    alpha_confirm: Fraction

    def __post_init__(self) -> None:
        """Validates the level.

        Raises:
            ValueError: If `alpha_confirm` is not strictly between 0 and 1.
        """
        if not 0 < self.alpha_confirm < 1:
            raise errors.InputError(
                "alpha_confirm must be strictly between 0 and 1"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class Exploration:
    """Alpha-wealth for the exploration phase.

    Attributes:
        initial: Wealth at the start of exploration, in [0, 1].
        wealth: Wealth still unspent, in [0, initial].
    """

    initial: Fraction
    wealth: Fraction

    def __post_init__(self) -> None:
        """Validates the state.

        Raises:
            ValueError: If `initial` is outside [0, 1] or `wealth` is outside
                [0, initial].
        """
        if not 0 <= self.initial <= 1:
            raise errors.InputError("initial must be within [0, 1]")
        if not 0 <= self.wealth <= self.initial:
            raise errors.InputError("wealth must be between 0 and initial")


@dataclasses.dataclass(frozen=True, slots=True)
class Tolerance:
    """What a detector certificate must show before the detector may gate.

    Attributes:
        detector: The detector.
        alarms: Largest tolerated false-alarm rate.
        recall: Smallest tolerated Wilson lower bound of the recall.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "detector_id": "detector",
                "max_false_alarm_rate": "alarms",
                "min_recall_lower_bound": "recall",
            }
        ),
    )

    detector: str
    alarms: float
    recall: float

    def __post_init__(self) -> None:
        """Validates the tolerances.

        A blank detector id or a rate that is not a finite number in [0, 1]
        is refused with a `ValueError` by the checks it calls.
        """
        evidence.require_text(self.detector, "detector")
        qualification.probability(self.alarms, "alarms")
        qualification.probability(self.recall, "recall")

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record.

        Returns:
            A mapping of the three fields to their values.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the tolerance, rejecting unknown or missing fields.

        Args:
            record: A mapping holding exactly the fields of the class.

        Returns:
            The tolerance described by the record. A record with an unknown
            or missing field, or with values the constructor refuses, raises
            a `ValueError` from `require` or the constructor.
        """
        return cls(**qualification.require(cls, record))


@dataclasses.dataclass(frozen=True, slots=True)
class Alpha:
    """Everything a run may spend, fixed in the charter.

    Attributes:
        confirmation: Level for the single confirmation test.
        exploration: Alpha-wealth for the exploration phase.
        permutations: Permutations `B` for negative controls (at least 20).
        tolerances: Certificate thresholds per detector.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            renames={
                "permutations_b": "permutations",
                "detector_tolerances": "tolerances",
            },
            parts={"tolerances": Tolerance.migrations[0]},
        ),
    )

    confirmation: Confirmation
    exploration: Exploration
    permutations: int
    tolerances: tuple[Tolerance, ...]

    def __post_init__(self) -> None:
        """Validates the budget.

        Raises:
            ValueError: If a detector appears twice among the tolerances.
                A `permutations` below the minimum or not an integer, or
                tolerances that are not a tuple, are refused by the
                `qualification` checks with a `ValueError`.
        """
        qualification.count(
            self.permutations, "permutations", minimum=MIN_PERMUTATIONS
        )
        qualification.tupled(self.tolerances, "tolerances")
        names = [t.detector for t in self.tolerances]
        if len(set(names)) != len(names):
            raise errors.InputError("duplicate detector in tolerances")

    @classmethod
    def create(
        cls,
        alpha_confirm: Number,
        wealth_exploration: Number,
        permutations: int = MIN_PERMUTATIONS,
        tolerances: tuple[Tolerance, ...] = (),
    ) -> Self:
        """Builds a fresh budget from plain numbers.

        Args:
            alpha_confirm: Level of the final claims, in (0, 1).
            wealth_exploration: Initial exploration wealth, in [0, 1].
            permutations: Negative-control permutations.
            tolerances: Certificate thresholds per detector.

        Returns:
            A budget whose exploration wealth equals its initial wealth.

        Raises:
            ValueError: On any out-of-range or non-finite input.
        """
        alpha = exact(alpha_confirm, "alpha_confirm")
        wealth = exact(wealth_exploration, "wealth_exploration")
        if not 0 <= wealth <= 1:
            raise errors.InputError("wealth_exploration must be within [0, 1]")
        return cls(
            confirmation=Confirmation(alpha),
            exploration=Exploration(initial=wealth, wealth=wealth),
            permutations=permutations,
            tolerances=tuple(tolerances),
        )

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record.

        Returns:
            A nested mapping in which fractions are written as ``"p/q"``
            strings.
        """
        return {
            "confirmation": {
                "alpha_confirm": str(self.confirmation.alpha_confirm)
            },
            "exploration": {
                "initial": str(self.exploration.initial),
                "wealth": str(self.exploration.wealth),
            },
            "permutations": self.permutations,
            "tolerances": [t.to_dict() for t in self.tolerances],
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a budget exactly, rejecting unknown or missing fields.

        Args:
            record: A mapping as written by `to_dict`.

        Returns:
            The budget described by the record.

        Raises:
            ValueError: If the record or its confirmation or exploration
                part has unexpected or missing fields.
        """
        record = qualification.require(cls, record)
        confirmation = record["confirmation"]
        exploration = record["exploration"]
        names = {"alpha_confirm"}
        if set(confirmation) != names:
            raise errors.InputError("unexpected or missing confirmation fields")
        if set(exploration) != {"initial", "wealth"}:
            raise errors.InputError("unexpected or missing exploration fields")
        return cls(
            confirmation=Confirmation(
                parse(confirmation["alpha_confirm"], "alpha_confirm")
            ),
            exploration=Exploration(
                initial=parse(exploration["initial"], "initial"),
                wealth=parse(exploration["wealth"], "wealth"),
            ),
            permutations=record["permutations"],
            tolerances=tuple(
                Tolerance.from_dict(item) for item in record["tolerances"]
            ),
        )


def remaining(budget: Alpha) -> float:
    """Reports the exploration wealth still unspent.

    Args:
        budget: The run's budget.

    Returns:
        The unspent exploration wealth as a float.
    """
    return float(budget.exploration.wealth)


def spend(budget: Alpha, amount: Number) -> Alpha:
    """Spends exploration wealth and returns the new budget.

    Args:
        budget: The budget before the spend; it is not modified.
        amount: Wealth to spend, at least 0.

    Returns:
        A budget with the amount deducted. Its confirmation budget is the
        same object.

    Raises:
        ValueError: If `amount` is negative or not a finite number.
        ExhaustedError: If `amount` exceeds the remaining wealth.
    """
    cost = exact(amount, "amount")
    if cost < 0:
        raise errors.InputError("amount must be non-negative")
    left = budget.exploration.wealth
    if cost > left:
        raise ExhaustedError(
            f"amount {float(cost):.6g} exceeds "
            f"remaining wealth {float(left):.6g}"
        )
    return dataclasses.replace(
        budget,
        exploration=Exploration(
            initial=budget.exploration.initial, wealth=left - cost
        ),
    )


def apportion(budget: Alpha, weights: Mapping[str, Number]) -> dict[str, Alpha]:
    """Divides a budget among hypothesis families in proportion to weights.

    The confirmation levels of the parts sum exactly to the original level,
    and so do the initial and remaining wealth.

    Args:
        budget: The budget to divide.
        weights: Positive weight per family name.

    Returns:
        One budget per family, in the order of `weights`.

    Raises:
        ValueError: If `weights` is empty, a name is blank or a weight is not
            a positive finite number.
    """
    if not weights:
        raise errors.InputError("weights must not be empty")
    exact_weights: dict[str, Fraction] = {}
    for name, weight in weights.items():
        evidence.require_text(name, "family name")
        value = exact(weight, f"weight[{name}]")
        if value <= 0:
            raise errors.InputError(f"weight[{name}] must be positive")
        exact_weights[name] = value
    total = sum(exact_weights.values(), Fraction(0))
    parts = {}
    for name, weight in exact_weights.items():
        share = weight / total
        parts[name] = dataclasses.replace(
            budget,
            confirmation=Confirmation(
                budget.confirmation.alpha_confirm * share
            ),
            exploration=Exploration(
                initial=budget.exploration.initial * share,
                wealth=budget.exploration.wealth * share,
            ),
        )
    return parts


def harmonic(n: int) -> float:
    """Computes the n-th harmonic number.

    Args:
        n: Number of terms; the exact sum is used up to 10,000 terms and an
            asymptotic series beyond.

    Returns:
        The harmonic number as a float.
    """
    if n > EXACT_HARMONIC_LIMIT:
        return math.log(n) + EULER_GAMMA + 1 / (2 * n) - 1 / (12 * n * n)
    return math.fsum(1.0 / k for k in range(1, n + 1))


def allot(budget: Alpha, n_survivors: int, rule: Correction) -> float:
    """Returns the smallest per-hypothesis level for `n_survivors`.

    This is the rank-1 threshold of the step rule, so a hypothesis with a
    p-value at or below it is rejected whatever the others are: `alpha / n`
    for Bonferroni, Holm and Benjamini-Hochberg, and `alpha / (n * H_n)` for
    Benjamini-Yekutieli.
    The full rank-dependent thresholds belong to the ledger. With no
    survivors there is nothing to correct and the whole level is returned.

    Args:
        budget: The run's budget.
        n_survivors: Hypotheses that survived exploration.
        rule: Multiplicity rule for the confirmation test.

    Returns:
        A level in (0, alpha_confirm].

    Raises:
        ValueError: If `n_survivors` is not a non-negative integer or `rule`
            is not a `Correction`.
    """
    qualification.count(n_survivors, "n_survivors")
    if not isinstance(rule, Correction):
        raise errors.InputError("rule must be a Correction")
    alpha = budget.confirmation.alpha_confirm
    if n_survivors == 0:
        return float(alpha)
    if rule is Correction.BY:
        return float(alpha) / (n_survivors * harmonic(n_survivors))
    return float(alpha / n_survivors)


def gammas(index: int, horizon: int | None) -> Fraction:
    """Computes `gamma_index` of the discount sequence.

    With no horizon ``gamma_j = 1 / (j (j + 1))``, which sums to less than one
    for every finite count. With an integer ``H`` it is ``1 / H`` up to ``H``
    and zero afterwards. An `index` or `horizon` that is not an integer of at
    least 1 is refused with a `ValueError` by `qualification.count`.

    Args:
        index: Position ``j`` in the sequence, at least 1.
        horizon: ``H``, at least 1, or None for the open-ended sequence.

    Returns:
        The weight ``gamma_j`` as an exact fraction.
    """
    qualification.count(index, "index", minimum=1)
    if horizon is None:
        return Fraction(1, index * (index + 1))
    qualification.count(horizon, "horizon", minimum=1)
    return Fraction(1, horizon) if index <= horizon else Fraction(0)


@dataclasses.dataclass(frozen=True, slots=True)
class Decision:
    """Levels for sequential promote and rollback decisions.

    It sits beside the `Alpha` and shares no arithmetic with it: the
    promotion levels are spent by started experiments, and each promoted
    champion gets the whole rollback level for its own monitoring lifetime.
    Promotion levels follow the discount-only rule (valid by the union bound
    over experiments): experiment ``j`` is assigned
    ``promotion * gamma_j``. A level is spent when the experiment starts,
    whether or not it ends in a promotion. The record is stored at schema
    version 2; a version 1 record loads through `migrations`.

    Attributes:
        promotion: Total level (alpha) for false promotions, in (0, 1).
        rollback: Level (alpha) of each champion's rollback monitor, in
            (0, 1).
        horizon: Number of experiments the discount sequence covers, or None
            for the open-ended sequence ``gamma_j = 1 / (j (j + 1))``.
        assigned: ``(experiment_id, level)`` in assignment order.
        monitored: Champions whose rollback monitor has been started.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={"alpha_promote": "promotion", "alpha_rollback": "rollback"}
        ),
    )

    promotion: Fraction
    rollback: Fraction
    horizon: int | None = None
    assigned: tuple[tuple[str, Fraction], ...] = ()
    monitored: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validates the state, including that no level was overspent.

        Raises:
            ValueError: If a level is outside (0, 1), an experiment id or
                champion appears twice, an assigned level is not positive,
                or the assigned levels exceed `promotion`. An invalid
                `horizon` is refused by the `qualification` check with a
                `ValueError`.
        """
        for name in ("promotion", "rollback"):
            if not 0 < getattr(self, name) < 1:
                raise errors.InputError(
                    f"{name} must be strictly between 0 and 1"
                )
        if self.horizon is not None:
            qualification.count(self.horizon, "horizon", minimum=1)
        ids = [pair[0] for pair in self.assigned]
        if len(set(ids)) != len(ids):
            raise errors.InputError("duplicate experiment id in assigned")
        if any(pair[1] <= 0 for pair in self.assigned):
            raise errors.InputError("assigned levels must be positive")
        if self.spent > self.promotion:
            raise errors.InputError("assigned levels exceed promotion")
        if len(set(self.monitored)) != len(self.monitored):
            raise errors.InputError("duplicate champion in monitored")

    @classmethod
    def create(
        cls,
        promotion: Number,
        rollback: Number,
        horizon: int | None = None,
    ) -> Self:
        """Builds a fresh budget from plain numbers (floats read exactly).

        Args:
            promotion: Total level for false promotions, in (0, 1).
            rollback: Level of each champion's rollback monitor, in (0, 1).
            horizon: Number of experiments the discount sequence covers, or
                None for the open-ended sequence.

        Returns:
            A budget with no level assigned and no champion monitored. A
            level outside (0, 1), a non-finite number or an invalid horizon
            raises a `ValueError` from `exact` or the constructor.
        """
        return cls(
            promotion=exact(promotion, "promotion"),
            rollback=exact(rollback, "rollback"),
            horizon=horizon,
        )

    @property
    def spent(self) -> Fraction:
        """The sum of the assigned promotion levels."""
        return sum((pair[1] for pair in self.assigned), Fraction(0))

    @property
    def remaining(self) -> Fraction:
        """The promotion level not yet assigned."""
        return self.promotion - self.spent

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record.

        Returns:
            A mapping with its `schema_version`, in which fractions are
            written as ``"p/q"`` strings and the assignments as
            ``[experiment_id, level]`` pairs.
        """
        return {
            records.KEY: records.ORIGIN + len(self.migrations),
            "promotion": str(self.promotion),
            "rollback": str(self.rollback),
            "horizon": self.horizon,
            "assigned": [[name, str(level)] for name, level in self.assigned],
            "monitored": list(self.monitored),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the budget exactly, rejecting unknown or missing fields.

        Args:
            record: A mapping as written by `to_dict`, of any schema
                version.

        Returns:
            The budget described by the record. An unknown or missing
            field, an invalid fraction, a newer schema version or an
            overspent state raises a `ValueError` from `require`,
            `parse` or the constructor.
        """
        record = qualification.require(cls, record)
        return cls(
            promotion=parse(record["promotion"], "promotion"),
            rollback=parse(record["rollback"], "rollback"),
            horizon=record["horizon"],
            assigned=tuple(
                (str(name), parse(level, "level"))
                for name, level in record["assigned"]
            ),
            monitored=tuple(str(name) for name in record["monitored"]),
        )


def enroll(
    budget: Decision,
    experiment_id: str,
    level: Number | None = None,
) -> tuple[Decision, Fraction]:
    """Assigns a promotion level to an experiment before it has any traffic.

    Args:
        budget: The budget before the assignment; it is not modified.
        experiment_id: A new, non-blank identifier.
        level: An explicit level; by default ``promotion * gamma_j`` for
            the next index ``j``.

    Returns:
        The new budget and the level assigned.

    Raises:
        ValueError: If the id is blank or already used, or the level is not
            positive.
        ExhaustedError: If the level would take the total past
            `promotion`, or the horizon is used up.
    """
    evidence.require_text(experiment_id, "experiment_id")
    if any(pair[0] == experiment_id for pair in budget.assigned):
        raise errors.InputError(
            f"experiment {experiment_id} already has a level"
        )
    if level is None:
        gamma = gammas(len(budget.assigned) + 1, budget.horizon)
        chosen = budget.promotion * gamma
        if chosen == 0:
            raise ExhaustedError("the experiment horizon is used up")
    else:
        chosen = exact(level, "level")
        if chosen <= 0:
            raise errors.InputError("level must be positive")
    if chosen > budget.remaining:
        raise ExhaustedError(
            f"level {float(chosen):.6g} exceeds the remaining promotion "
            f"level {float(budget.remaining):.6g}"
        )
    return (
        dataclasses.replace(
            budget, assigned=(*budget.assigned, (experiment_id, chosen))
        ),
        chosen,
    )


def watch(budget: Decision, champion_id: str) -> tuple[Decision, Fraction]:
    """Starts the rollback monitor of a promoted champion.

    The rollback level is per champion and is not shared, so starting the
    monitor never reduces the promotion budget. A champion is monitored once.

    Args:
        budget: The budget before the start; it is not modified.
        champion_id: Identifier of the promoted champion.

    Returns:
        The new budget with the champion recorded as monitored, and the
        rollback level that champion's monitor may spend.

    Raises:
        ValueError: If the id is blank or already monitored.
    """
    evidence.require_text(champion_id, "champion_id")
    if champion_id in budget.monitored:
        raise errors.InputError(f"champion {champion_id} is already monitored")
    return (
        dataclasses.replace(budget, monitored=(*budget.monitored, champion_id)),
        budget.rollback,
    )
