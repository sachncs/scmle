"""Rule names, evidence, state and the base class of every online rule."""

import abc
import dataclasses
from fractions import Fraction
import math
from typing import Any
from typing import ClassVar

from scmle.assurance.ledger import budget
from scmle.contracts import errors
from scmle.contracts import qualification

P_FLOOR = 1e-300
DEFAULT_KAPPA = 0.5
DEFAULT_LAMBDA = Fraction(1, 2)
ELOND = "e-lond"
LORD_PLUS_PLUS = "lord++"
SAFFRON = "saffron"
RULE_NAMES = (ELOND, LORD_PLUS_PLUS, SAFFRON)


class InvariantError(RuntimeError):
    """Raised if a rule's own bookkeeping would break (a bug, never data)."""


@dataclasses.dataclass(frozen=True, slots=True)
class Guarantee:
    """What a rule promises and under which assumptions.

    Attributes:
        statement: The guarantee in one sentence.
        assumptions: Conditions the guarantee needs.
        citation: Where the statement was verified.
        verified: Whether the statement was read in the primary source.
        dependence: `arbitrary` or `independent`.
    """

    statement: str
    assumptions: tuple[str, ...]
    citation: str
    verified: bool
    dependence: str


@dataclasses.dataclass(frozen=True, slots=True)
class Discount:
    """Discount sequence `gamma_j`, nonincreasing, positive up to the horizon.

    Attributes:
        horizon: With None, ``gamma_j = 1 / (j (j + 1))`` (sums to one, never
            runs out). With an integer ``H``, ``gamma_j = 1 / H`` for
            ``j <= H`` and zero afterwards, which is the right choice when the
            number of hypotheses is known in advance.
    """

    horizon: int | None = None

    def __post_init__(self) -> None:
        """Validates the horizon.

        A horizon that is not an integer of at least 1 is refused with a
        `ValueError` by `qualification.count`.
        """
        if self.horizon is not None:
            qualification.count(self.horizon, "horizon", minimum=1)

    def weight(self, index: int) -> Fraction:
        """Computes `gamma_index` for ``index >= 1``.

        Args:
            index: Position in the sequence, at least 1.

        Returns:
            The weight ``gamma_index``: ``1 / (index (index + 1))`` without a
            horizon, otherwise ``1 / H`` up to the horizon ``H`` and zero
            after it.

        Raises:
            ValueError: If `index` is below 1.
        """
        if index < 1:
            raise errors.InputError("index must be at least 1")
        if self.horizon is None:
            return Fraction(1, index * (index + 1))
        return (
            Fraction(1, self.horizon) if index <= self.horizon else Fraction(0)
        )

    def cumulative(self, count: int) -> Fraction:
        """Computes ``sum_{j <= count} gamma_j`` in closed form.

        Args:
            count: Number of terms; a negative or non-integer count is
                refused with a `ValueError` by `qualification.count`.

        Returns:
            The partial sum of the first `count` weights.
        """
        qualification.count(count, "count")
        if self.horizon is None:
            return Fraction(count, count + 1)
        return Fraction(min(count, self.horizon), self.horizon)


@dataclasses.dataclass(frozen=True, slots=True)
class Evidence:
    """What one test yields: a p-value, an e-value, or both.

    Attributes:
        p: A super-uniform p-value in [0, 1], or None.
        e: An e-value (nonnegative, ``E[e] <= 1`` under the null), or None.
    """

    p: float | None = None
    e: float | None = None

    def __post_init__(self) -> None:
        """Validates the values.

        Raises:
            ValueError: If neither a p-value nor an e-value is given, or the
                e-value is negative. A p-value outside [0, 1] or a
                non-finite e-value is refused with a `ValueError` by the
                `qualification` checks.
        """
        if self.p is None and self.e is None:
            raise errors.InputError("evidence needs a p-value or an e-value")
        if self.p is not None:
            qualification.probability(self.p, "p")
        if self.e is not None:
            value = qualification.finite(self.e, "e")
            if value < 0.0:
                raise errors.InputError("e must be non-negative")


@dataclasses.dataclass(frozen=True, slots=True)
class State:
    """Everything a rule remembers.

    Attributes:
        tests: Hypotheses tested so far.
        rejections: 1-based times of the rejections.
        candidates: 1-based times of the SAFFRON candidates (rejections are
            candidates too); empty for the other rules.
        spent: Level spent so far, as the rule defines it.
    """

    tests: int = 0
    rejections: tuple[int, ...] = ()
    candidates: tuple[int, ...] = ()
    spent: Fraction = Fraction(0)


@dataclasses.dataclass(frozen=True, slots=True)
class Step:
    """The outcome of one test.

    Attributes:
        level: Test level `alpha_t` the rule assigned.
        rejected: Whether the hypothesis was rejected.
        candidate: Whether it was a SAFFRON candidate (False otherwise).
        e_value: The e-value used by e-LOND (None for the p-value rules).
        wealth_before: Wealth before the test.
        wealth_after: Wealth after the test.
        state: The state after the test.
    """

    level: Fraction
    rejected: bool
    candidate: bool
    e_value: float | None
    wealth_before: Fraction
    wealth_after: Fraction
    state: State


def calibrate(p: float, kappa: float = DEFAULT_KAPPA) -> float:
    """Turns a p-value into an e-value with ``kappa * p ** (kappa - 1)``.

    Args:
        p: A super-uniform p-value in [0, 1]. Values below 1e-300 are raised
            to 1e-300, which only weakens the evidence.
        kappa: Calibrator parameter, strictly between 0 and 1.

    Returns:
        A finite e-value. ``kappa = 0.5`` uses only an exactly rounded square
        root, so the result is identical on every platform.

    Raises:
        ValueError: If `p` is outside [0, 1] or `kappa` outside (0, 1).
    """
    qualification.probability(p, "p")
    level = qualification.finite(kappa, "kappa")
    if not 0.0 < level < 1.0:
        raise errors.InputError("kappa must be strictly between 0 and 1")
    floored = max(p, P_FLOOR)
    if level == DEFAULT_KAPPA:
        return 0.5 / math.sqrt(floored)
    result: float = level * floored ** (level - 1.0)
    return result


def rational(value: Any, name: str) -> Fraction:
    """Reads a level in [0, 1] as an exact fraction.

    Args:
        value: An int, float or `Fraction`; a float is read as its shortest
            decimal (not its binary value) and a bool is refused.
        name: Field name used in the error message.

    Returns:
        The exact value.

    Raises:
        ValueError: If the value is outside [0, 1]. A value that is not a
            finite number is refused by `budget.exact`.
    """
    amount = budget.exact(value, name)
    if not 0 <= amount <= 1:
        raise errors.InputError(f"{name} must be within [0, 1]")
    return amount


class Tester(abc.ABC):
    """Base class: a level function plus a decision and wealth bookkeeping.

    Attributes:
        name: The rule's name.
        guarantee: What the rule promises and under which assumptions.
        alpha: Target FDR level, as an exact fraction.
        discount: The discount sequence.
    """

    name: ClassVar[str]
    guarantee: ClassVar[Guarantee]

    def __init__(self, alpha: Any, discount: Discount | None = None) -> None:
        """Creates the rule.

        Args:
            alpha: Target FDR level in [0, 1].
            discount: The discount sequence (default ``1 / (j (j + 1))``).
        """
        self.alpha = rational(alpha, "alpha")
        self.discount = discount if discount is not None else Discount()

    def initial(self) -> State:
        """Gives the state before any hypothesis.

        Returns:
            A state with no tests, rejections or candidates and nothing
            spent.
        """
        return State()

    @abc.abstractmethod
    def level(self, state: State) -> Fraction:
        """Computes the level of the next hypothesis from the past only.

        Args:
            state: The state after the previous hypothesis.

        Returns:
            The test level ``alpha_t`` of the next hypothesis.
        """

    @abc.abstractmethod
    def wealth(self, state: State) -> Fraction:
        """Computes the wealth in a state.

        Args:
            state: The state to measure.

        Returns:
            The wealth, as the rule defines it.
        """

    @abc.abstractmethod
    def decide(
        self, level: Fraction, evidence: Evidence
    ) -> tuple[bool, bool, float | None]:
        """Decides one test.

        Args:
            level: The level assigned to the hypothesis.
            evidence: Its p-value or e-value.

        Returns:
            Whether it is rejected, whether it is a candidate, and the
            e-value used (None for the p-value rules).
        """

    def expend(
        self,
        state: State,
        level: Fraction,
        candidate: bool,  # noqa: ARG002 -- the default charges every level; subclasses that discount candidates override it with this signature
    ) -> Fraction:
        """Computes the level spent after a test.

        The default charges every level.

        Args:
            state: The state before the test.
            level: The level the test was given.
            candidate: Whether the test was a candidate.

        Returns:
            The total level spent after the test.
        """
        return state.spent + level

    def step(self, state: State, evidence: Evidence) -> Step:
        """Tests the next hypothesis.

        Args:
            state: State after the previous hypothesis.
            evidence: The p-value or e-value of the next hypothesis.

        Returns:
            The level, the decision, the wealth and the new state. A rule
            that needs a kind of evidence the item lacks refuses it with a
            `ValueError` (see `decide`).

        Raises:
            InvariantError: If the wealth would turn negative.
        """
        level = self.level(state)
        rejected, candidate, e_value = self.decide(level, evidence)
        time = state.tests + 1
        moved = State(
            tests=time,
            rejections=(*state.rejections, time)
            if rejected
            else state.rejections,
            candidates=(*state.candidates, time)
            if candidate
            else state.candidates,
            spent=self.expend(state, level, candidate),
        )
        after = self.wealth(moved)
        if after < 0:
            raise InvariantError(f"wealth would be negative at test {time}")
        return Step(
            level=level,
            rejected=rejected,
            candidate=candidate,
            e_value=e_value,
            wealth_before=self.wealth(state),
            wealth_after=after,
            state=moved,
        )

    def credit(self, state: State) -> Fraction:  # noqa: ARG002 -- the default earns no credit; LORD++ and SAFFRON override it with this signature
        """Computes the credit earned, used by LORD++ and SAFFRON.

        Args:
            state: The state to measure.

        Returns:
            Zero in the base class; the p-value rules return the initial
            wealth plus a reward per rejection.
        """
        return Fraction(0)
