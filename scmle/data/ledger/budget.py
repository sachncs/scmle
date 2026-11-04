"""The error budget: exploration wealth and confirmation alpha.

Canon P2: the exploration alpha-wealth is a triage budget; the final claim
guarantee comes only from `alpha_confirm`, spent once on the confirmation set.
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
from typing import Self

from scmle.contracts import evidence
from scmle.contracts import qualification

MIN_PERMUTATIONS = 20
EXACT_HARMONIC_LIMIT = 10_000
EULER_GAMMA = 0.5772156649015329

Number = int | float | Fraction


class BudgetExhaustedError(ValueError):
    """Raised when a spend exceeds the remaining exploration wealth."""


class MultiplicityRule(enum.StrEnum):
    """Rule that adjusts the confirmation level for the survivor count."""

    BONFERRONI = "bonferroni"
    HOLM = "holm"
    BH = "bh"
    BY = "by"


def exact(value: Any, name: str) -> Fraction:
    """Reads a number exactly; a float is taken as its shortest decimal."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float) and math.isfinite(value):
        return Fraction(repr(value))
    raise ValueError(f"{name} must be a finite number")


def parse_fraction(text: Any, name: str) -> Fraction:
    """Parses a serialized fraction such as ``"1/20"``."""
    try:
        return Fraction(str(text))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{name}: invalid fraction {text!r}") from error


@dataclasses.dataclass(frozen=True, slots=True)
class ConfirmationBudget:
    """Level spent once on the confirmation set.

    Attributes:
        alpha_confirm: Target FDR (or FWER) of the final claims, in (0, 1).
    """

    alpha_confirm: Fraction

    def __post_init__(self) -> None:
        """Validates the level."""
        if not 0 < self.alpha_confirm < 1:
            raise ValueError("alpha_confirm must be strictly between 0 and 1")


@dataclasses.dataclass(frozen=True, slots=True)
class ExplorationBudget:
    """Alpha-wealth for the exploration phase.

    Attributes:
        initial: Wealth at the start of exploration, in [0, 1].
        wealth: Wealth still unspent, in [0, initial].
    """

    initial: Fraction
    wealth: Fraction

    def __post_init__(self) -> None:
        """Validates the state."""
        if not 0 <= self.initial <= 1:
            raise ValueError("initial must be within [0, 1]")
        if not 0 <= self.wealth <= self.initial:
            raise ValueError("wealth must be between 0 and initial")


@dataclasses.dataclass(frozen=True, slots=True)
class DetectorTolerance:
    """What a detector certificate must show before the detector may gate.

    Attributes:
        detector_id: The detector.
        max_false_alarm_rate: Largest tolerated false-alarm rate.
        min_recall_lower_bound: Smallest tolerated Wilson lower bound.
    """

    detector_id: str
    max_false_alarm_rate: float
    min_recall_lower_bound: float

    def __post_init__(self) -> None:
        """Validates the tolerances."""
        evidence.require_text(self.detector_id, "detector_id")
        qualification.probability(
            self.max_false_alarm_rate, "max_false_alarm_rate"
        )
        qualification.probability(
            self.min_recall_lower_bound, "min_recall_lower_bound"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the tolerance, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(**record)


@dataclasses.dataclass(frozen=True, slots=True)
class ErrorBudget:
    """Everything a run may spend, fixed in the charter.

    Attributes:
        confirmation: Level for the single confirmation test.
        exploration: Alpha-wealth for the exploration phase.
        permutations_b: Permutations for negative controls (at least 20).
        detector_tolerances: Certificate thresholds per detector.
    """

    confirmation: ConfirmationBudget
    exploration: ExplorationBudget
    permutations_b: int
    detector_tolerances: tuple[DetectorTolerance, ...]

    def __post_init__(self) -> None:
        """Validates the budget."""
        qualification.count(
            self.permutations_b, "permutations_b", minimum=MIN_PERMUTATIONS
        )
        qualification.require_tuple(
            self.detector_tolerances, "detector_tolerances"
        )
        names = [t.detector_id for t in self.detector_tolerances]
        if len(set(names)) != len(names):
            raise ValueError("duplicate detector_id in detector_tolerances")

    @classmethod
    def create(
        cls,
        alpha_confirm: Number,
        wealth_exploration: Number,
        permutations_b: int = MIN_PERMUTATIONS,
        detector_tolerances: tuple[DetectorTolerance, ...] = (),
    ) -> Self:
        """Builds a fresh budget from plain numbers.

        Args:
            alpha_confirm: Level of the final claims, in (0, 1).
            wealth_exploration: Initial exploration wealth, in [0, 1].
            permutations_b: Negative-control permutations.
            detector_tolerances: Certificate thresholds per detector.

        Returns:
            A budget whose exploration wealth equals its initial wealth.

        Raises:
            ValueError: On any out-of-range or non-finite input.
        """
        alpha = exact(alpha_confirm, "alpha_confirm")
        wealth = exact(wealth_exploration, "wealth_exploration")
        if not 0 <= wealth <= 1:
            raise ValueError("wealth_exploration must be within [0, 1]")
        return cls(
            confirmation=ConfirmationBudget(alpha),
            exploration=ExplorationBudget(initial=wealth, wealth=wealth),
            permutations_b=permutations_b,
            detector_tolerances=tuple(detector_tolerances),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record; fractions are written as ``"p/q"`` strings."""
        return {
            "confirmation": {
                "alpha_confirm": str(self.confirmation.alpha_confirm)
            },
            "exploration": {
                "initial": str(self.exploration.initial),
                "wealth": str(self.exploration.wealth),
            },
            "permutations_b": self.permutations_b,
            "detector_tolerances": [
                t.to_dict() for t in self.detector_tolerances
            ],
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a budget exactly, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        confirmation = record["confirmation"]
        exploration = record["exploration"]
        names = {"alpha_confirm"}
        if set(confirmation) != names:
            raise ValueError("unexpected or missing confirmation fields")
        if set(exploration) != {"initial", "wealth"}:
            raise ValueError("unexpected or missing exploration fields")
        return cls(
            confirmation=ConfirmationBudget(
                parse_fraction(confirmation["alpha_confirm"], "alpha_confirm")
            ),
            exploration=ExplorationBudget(
                initial=parse_fraction(exploration["initial"], "initial"),
                wealth=parse_fraction(exploration["wealth"], "wealth"),
            ),
            permutations_b=record["permutations_b"],
            detector_tolerances=tuple(
                DetectorTolerance.from_dict(item)
                for item in record["detector_tolerances"]
            ),
        )


def remaining(budget: ErrorBudget) -> float:
    """Exploration wealth still unspent."""
    return float(budget.exploration.wealth)


def spend(budget: ErrorBudget, amount: Number) -> ErrorBudget:
    """Spends exploration wealth and returns the new budget.

    Args:
        budget: The budget before the spend; it is not modified.
        amount: Wealth to spend, at least 0.

    Returns:
        A budget with the amount deducted. Its confirmation budget is the
        same object.

    Raises:
        ValueError: If `amount` is negative or not a finite number.
        BudgetExhaustedError: If `amount` exceeds the remaining wealth.
    """
    cost = exact(amount, "amount")
    if cost < 0:
        raise ValueError("amount must be non-negative")
    left = budget.exploration.wealth
    if cost > left:
        raise BudgetExhaustedError(
            f"amount {float(cost):.6g} exceeds "
            f"remaining wealth {float(left):.6g}"
        )
    return dataclasses.replace(
        budget,
        exploration=ExplorationBudget(
            initial=budget.exploration.initial, wealth=left - cost
        ),
    )


def split_budget(
    budget: ErrorBudget, weights: Mapping[str, Number]
) -> dict[str, ErrorBudget]:
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
        raise ValueError("weights must not be empty")
    exact_weights: dict[str, Fraction] = {}
    for name, weight in weights.items():
        evidence.require_text(name, "family name")
        value = exact(weight, f"weight[{name}]")
        if value <= 0:
            raise ValueError(f"weight[{name}] must be positive")
        exact_weights[name] = value
    total = sum(exact_weights.values(), Fraction(0))
    parts = {}
    for name, weight in exact_weights.items():
        share = weight / total
        parts[name] = dataclasses.replace(
            budget,
            confirmation=ConfirmationBudget(
                budget.confirmation.alpha_confirm * share
            ),
            exploration=ExplorationBudget(
                initial=budget.exploration.initial * share,
                wealth=budget.exploration.wealth * share,
            ),
        )
    return parts


def harmonic(n: int) -> float:
    """The n-th harmonic number (asymptotic series beyond 10,000)."""
    if n > EXACT_HARMONIC_LIMIT:
        return math.log(n) + EULER_GAMMA + 1 / (2 * n) - 1 / (12 * n * n)
    return math.fsum(1.0 / k for k in range(1, n + 1))


def confirmation_alpha(
    budget: ErrorBudget, n_survivors: int, rule: MultiplicityRule
) -> float:
    """Smallest per-hypothesis level the rule applies to `n_survivors`.

    This is the level at which a lone hypothesis is certain to be rejected
    (the rank-1 threshold of the step rule): `alpha / n` for Bonferroni, Holm
    and Benjamini-Hochberg, and `alpha / (n * H_n)` for Benjamini-Yekutieli.
    The full rank-dependent thresholds belong to the ledger (Phase 3). With no
    survivors there is nothing to correct and the whole level is returned.

    Args:
        budget: The run's budget.
        n_survivors: Hypotheses that survived exploration.
        rule: Multiplicity rule for the confirmation test.

    Returns:
        A level in (0, alpha_confirm].

    Raises:
        ValueError: If `n_survivors` is negative or `rule` is unknown.
    """
    qualification.count(n_survivors, "n_survivors")
    if not isinstance(rule, MultiplicityRule):
        raise ValueError("rule must be a MultiplicityRule")
    alpha = budget.confirmation.alpha_confirm
    if n_survivors == 0:
        return float(alpha)
    if rule is MultiplicityRule.BY:
        return float(alpha) / (n_survivors * harmonic(n_survivors))
    return float(alpha / n_survivors)


def discount_weight(index: int, horizon: int | None) -> Fraction:
    """`gamma_index` of the discount sequence (F11).

    With no horizon ``gamma_j = 1 / (j (j + 1))``, which sums to less than one
    for every finite count. With an integer ``H`` it is ``1 / H`` up to ``H``
    and zero afterwards.
    """
    qualification.count(index, "index", minimum=1)
    if horizon is None:
        return Fraction(1, index * (index + 1))
    qualification.count(horizon, "horizon", minimum=1)
    return Fraction(1, horizon) if index <= horizon else Fraction(0)


@dataclasses.dataclass(frozen=True, slots=True)
class DecisionBudget:
    """Levels for sequential promote and rollback decisions (F11).

    It sits beside the `ErrorBudget` and shares no arithmetic with it: the
    promotion levels are spent by started experiments, and each promoted
    champion gets the whole rollback level for its own monitoring lifetime.
    Promotion levels follow the discount-only rule (valid by the union bound
    over experiments): experiment ``j`` is assigned
    ``alpha_promote * gamma_j``. A level is spent when the experiment starts,
    whether or not it ends in a promotion.

    Attributes:
        alpha_promote: Total level for false promotions, in (0, 1).
        alpha_rollback: Level of each champion's rollback monitor, in (0, 1).
        horizon: Planned number of experiments, or None for the open-ended
            sequence ``gamma_j = 1 / (j (j + 1))``.
        assigned: ``(experiment_id, level)`` in assignment order.
        monitored: Champions whose rollback monitor has been started.
    """

    alpha_promote: Fraction
    alpha_rollback: Fraction
    horizon: int | None = None
    assigned: tuple[tuple[str, Fraction], ...] = ()
    monitored: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validates the state, including that no level was overspent."""
        for name in ("alpha_promote", "alpha_rollback"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be strictly between 0 and 1")
        if self.horizon is not None:
            qualification.count(self.horizon, "horizon", minimum=1)
        ids = [name for name, _ in self.assigned]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate experiment id in assigned")
        if any(level <= 0 for _, level in self.assigned):
            raise ValueError("assigned levels must be positive")
        if self.spent > self.alpha_promote:
            raise ValueError("assigned levels exceed alpha_promote")
        if len(set(self.monitored)) != len(self.monitored):
            raise ValueError("duplicate champion in monitored")

    @classmethod
    def create(
        cls,
        alpha_promote: Number,
        alpha_rollback: Number,
        horizon: int | None = None,
    ) -> Self:
        """Builds a fresh budget from plain numbers (floats read exactly)."""
        return cls(
            alpha_promote=exact(alpha_promote, "alpha_promote"),
            alpha_rollback=exact(alpha_rollback, "alpha_rollback"),
            horizon=horizon,
        )

    @property
    def spent(self) -> Fraction:
        """Sum of the assigned promotion levels."""
        return sum((level for _, level in self.assigned), Fraction(0))

    @property
    def remaining(self) -> Fraction:
        """Promotion level not yet assigned."""
        return self.alpha_promote - self.spent

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record; fractions are written as ``"p/q"`` strings."""
        return {
            "alpha_promote": str(self.alpha_promote),
            "alpha_rollback": str(self.alpha_rollback),
            "horizon": self.horizon,
            "assigned": [[name, str(level)] for name, level in self.assigned],
            "monitored": list(self.monitored),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the budget exactly, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(
            alpha_promote=parse_fraction(
                record["alpha_promote"], "alpha_promote"
            ),
            alpha_rollback=parse_fraction(
                record["alpha_rollback"], "alpha_rollback"
            ),
            horizon=record["horizon"],
            assigned=tuple(
                (str(name), parse_fraction(level, "level"))
                for name, level in record["assigned"]
            ),
            monitored=tuple(str(name) for name in record["monitored"]),
        )


def assign_experiment(
    budget: DecisionBudget,
    experiment_id: str,
    level: Number | None = None,
) -> tuple[DecisionBudget, Fraction]:
    """Assigns a promotion level to an experiment before it has any traffic.

    Args:
        budget: The budget before the assignment; it is not modified.
        experiment_id: A new, non-blank identifier.
        level: An explicit level; by default ``alpha_promote * gamma_j`` for
            the next index ``j``.

    Returns:
        The new budget and the level assigned.

    Raises:
        ValueError: If the id is blank or already used, or the level is not
            positive.
        BudgetExhaustedError: If the level would take the total past
            `alpha_promote`, or the horizon is used up.
    """
    evidence.require_text(experiment_id, "experiment_id")
    if any(name == experiment_id for name, _ in budget.assigned):
        raise ValueError(f"experiment {experiment_id} already has a level")
    if level is None:
        gamma = discount_weight(len(budget.assigned) + 1, budget.horizon)
        chosen = budget.alpha_promote * gamma
        if chosen == 0:
            raise BudgetExhaustedError("the experiment horizon is used up")
    else:
        chosen = exact(level, "level")
        if chosen <= 0:
            raise ValueError("level must be positive")
    if chosen > budget.remaining:
        raise BudgetExhaustedError(
            f"level {float(chosen):.6g} exceeds the remaining promotion "
            f"level {float(budget.remaining):.6g}"
        )
    return (
        dataclasses.replace(
            budget, assigned=(*budget.assigned, (experiment_id, chosen))
        ),
        chosen,
    )


def start_rollback_monitor(
    budget: DecisionBudget, champion_id: str
) -> tuple[DecisionBudget, Fraction]:
    """Starts the rollback monitor of a promoted champion.

    The rollback level is per champion and is not shared, so starting the
    monitor never reduces the promotion budget. A champion is monitored once.

    Raises:
        ValueError: If the id is blank or already monitored.
    """
    evidence.require_text(champion_id, "champion_id")
    if champion_id in budget.monitored:
        raise ValueError(f"champion {champion_id} is already monitored")
    return (
        dataclasses.replace(budget, monitored=(*budget.monitored, champion_id)),
        budget.alpha_rollback,
    )
