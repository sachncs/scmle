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
