"""Online false-discovery-rate rules as exact state machines.

Hypotheses arrive one at a time. A rule turns the history of past decisions
into a test level for the next hypothesis, so each rule is a pure function
``(state, evidence) -> (step, new state)``. All levels, thresholds and wealth
are exact `fractions.Fraction` values and every comparison is exact, so a
replay of the same evidence reproduces every level and decision bit for bit on
any platform and Python version (a float p-value is read as its exact binary
value). Floating point enters only when a p-value is calibrated into an
e-value, and the ledger stores the e-value it used.

Verified statements (read from the primary sources on 2026-11-03).
"Verified" means the statement was read in the cited arXiv paper or in the
onlineFDR manual; it does not mean this code is proved correct. Journal and
conference venues are quoted as the manual or the paper gives them; the
AISTATS 2024 venue of e-LOND is from memory and was not checked (the arXiv
text read is v1, November 2023).

e-LOND (Xu and Ramdas, "Online multiple testing with e-values",
arXiv:2311.06412v1, Section 3.2, Theorem 1; proof in Section 8.1)
    Level ``alpha_t = alpha * gamma_t * (|R_{t-1}| + 1)`` where ``gamma_t``
    is a fixed nonnegative sequence with ``sum gamma_t <= 1``. Hypothesis
    ``t`` is rejected when ``e_t >= 1 / alpha_t``. If every true null has a
    valid e-value, ``E[e_t] <= 1`` (their equation 3), then
    ``FDR <= alpha`` at every time ``t`` under *arbitrary* dependence between
    the e-values. The proof uses only ``E[e_t] <= 1``; it needs neither
    independence nor a conditional version of the e-value condition. The
    hypotheses are indexed in advance (the index set of true nulls is fixed),
    so this module's ledger registers every hypothesis before it is tested and
    never lets an earlier result choose a later hypothesis.

p-to-e calibration (Vovk and Wang, "E-values: calibration and combination",
Annals of Statistics 49(3), 2021, doi:10.1214/20-AOS2020)
    A calibrator ``f: [0, 1] -> [0, inf)`` is nonincreasing, upper
    semicontinuous and has ``integral f <= 1``; ``f(P)`` is then an e-value
    for every super-uniform ``P``. The family used here is
    ``f_kappa(p) = kappa * p ** (kappa - 1)`` for ``kappa`` in (0, 1)
    (``integral = 1``). For a nonincreasing ``f`` and a super-uniform ``P``,
    ``E f(P) <= integral f`` (the layer-cake formula), so the calibrated
    value is a valid e-value; Xu and Ramdas (Corollary 1) use the same device
    to feed p-values to e-LOND. The cost is power: the rejection rule
    ``f(p) >= 1 / alpha_t`` is ``p <= (kappa * alpha_t) ** (1 / (1 - kappa))``.
    The default ``kappa = 1/2`` makes ``e = 1 / (2 * sqrt(p))``, whose only
    floating-point operation (a square root) is exactly rounded and so
    identical on every platform.

LORD++ (Ramdas, Yang, Wainwright and Jordan, "Online control of the false
discovery rate with decaying memory", NeurIPS 2017, arXiv:1710.00499,
equation (5) and Theorem 1)
    ``alpha_t = gamma_t W0 + (alpha - W0) gamma_{t - tau_1}
    + alpha * sum_{j >= 2, tau_j < t} gamma_{t - tau_j}`` with ``tau_j`` the
    time of the j-th rejection, ``0 <= W0 <= alpha`` and a nonincreasing
    positive ``gamma`` summing to one. Hypothesis ``t`` is rejected when
    ``p_t <= alpha_t``. The theorem gives ``FDR <= alpha`` for *independent*
    p-values. The onlineFDR manual (Bioconductor 3.23, version 2.20.0) adds
    that control also holds if only the true-null p-values are mutually
    independent and independent of the non-null p-values. There is no
    guarantee under dependence.

SAFFRON (Ramdas, Zrnic, Wainwright and Jordan, "SAFFRON: an adaptive
algorithm for online control of the false discovery rate", ICML 2018,
arXiv:1802.09098, Section 2.3 and Theorem 1)
    A p-value ``<= lambda`` is a candidate. ``alpha_t = min(lambda,
    (1 - lambda) * (W0 gamma_{t - C0+} + (alpha - W0) gamma_{t - tau_1 - C1+}
    + alpha * sum_{j >= 2} gamma_{t - tau_j - Cj+}))`` where ``Cj+`` counts
    the candidates in ``(tau_j, t - 1]``. FDR control at ``alpha`` holds for
    independent p-values (Theorem 1(d), which also needs the levels to be
    monotone functions of the past, as here); only mFDR control is claimed
    under conditional super-uniformity. The onlineFDR manual gives the same
    independence statement and the defaults ``lambda = 1/2``,
    ``W0 = alpha / 2``.

Not verified, so not claimed. R ``onlineFDR`` 2.20.0 has no e-LOND function,
and no R output could be generated here (R is not installed), so these rules
are checked against hand calculations from the formulas above and against
brute-force re-derivations of the levels from the definitions, not against R.
The package's default discount sequence is ``gamma_j`` proportional to
``j ** -1.6``, an irrational normalisation that cannot be held exactly, so the
default here is the exact ``1 / (j (j + 1))`` (sum one). Defaults for the
initial wealth follow the papers (``W0 = alpha / 2``); the onlineFDR LORD
default is ``alpha / 10``.

Bookkeeping of "wealth". For LORD++ and SAFFRON the wealth after ``t`` tests
is the credit earned (initial wealth plus a reward per rejection) minus the
level spent; the discount sequence uses each ``gamma`` index at most once per
credit source, so it never goes negative, and this module raises if it ever
did. For e-LOND there is no spending: the wealth reported is
``alpha * (1 - sum_{i <= t} gamma_i)``, the share of the level not yet
assigned to any hypothesis.
"""

import abc
from collections.abc import Mapping
import dataclasses
from fractions import Fraction
import math
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.contracts import qualification
from scmle.data.ledger import budget as budget_module

P_FLOOR = 1e-300
DEFAULT_KAPPA = 0.5
DEFAULT_LAMBDA = Fraction(1, 2)
ELOND = "e-lond"
LORD_PLUS_PLUS = "lord++"
SAFFRON = "saffron"
RULE_NAMES = (ELOND, LORD_PLUS_PLUS, SAFFRON)
DEFAULT_RULE = ELOND


class RuleInvariantError(RuntimeError):
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
        """Validates the horizon."""
        if self.horizon is not None:
            qualification.count(self.horizon, "horizon", minimum=1)

    def weight(self, index: int) -> Fraction:
        """`gamma_index` for ``index >= 1``."""
        if index < 1:
            raise ValueError("index must be at least 1")
        if self.horizon is None:
            return Fraction(1, index * (index + 1))
        return (
            Fraction(1, self.horizon) if index <= self.horizon else Fraction(0)
        )

    def cumulative(self, count: int) -> Fraction:
        """``sum_{j <= count} gamma_j`` in closed form."""
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
        """Validates the values."""
        if self.p is None and self.e is None:
            raise ValueError("evidence needs a p-value or an e-value")
        if self.p is not None:
            qualification.probability(self.p, "p")
        if self.e is not None:
            value = qualification.finite(self.e, "e")
            if value < 0.0:
                raise ValueError("e must be non-negative")


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
        raise ValueError("kappa must be strictly between 0 and 1")
    floored = max(p, P_FLOOR)
    if level == DEFAULT_KAPPA:
        return 0.5 / math.sqrt(floored)
    return level * floored ** (level - 1.0)


def exact_alpha(value: Any, name: str) -> Fraction:
    """A level in [0, 1] as an exact fraction."""
    amount = budget_module.exact(value, name)
    if not 0 <= amount <= 1:
        raise ValueError(f"{name} must be within [0, 1]")
    return amount


class OnlineRule(abc.ABC):
    """Base class: a level function plus a decision and wealth bookkeeping."""

    name: ClassVar[str]
    guarantee: ClassVar[Guarantee]

    def __init__(self, alpha: Any, discount: Discount | None = None) -> None:
        """Creates the rule.

        Args:
            alpha: Target FDR level in [0, 1].
            discount: The discount sequence (default ``1 / (j (j + 1))``).
        """
        self.alpha = exact_alpha(alpha, "alpha")
        self.discount = discount if discount is not None else Discount()

    def initial(self) -> State:
        """The state before any hypothesis."""
        return State()

    @abc.abstractmethod
    def level(self, state: State) -> Fraction:
        """Level of the next hypothesis, a function of the past only."""

    @abc.abstractmethod
    def wealth(self, state: State) -> Fraction:
        """Wealth in `state`."""

    @abc.abstractmethod
    def decide(
        self, level: Fraction, evidence: Evidence
    ) -> tuple[bool, bool, float | None]:
        """Returns (rejected, candidate, e-value used) for one test."""

    def update_spent(
        self, state: State, level: Fraction, candidate: bool
    ) -> Fraction:
        """Level spent after a test; the default charges every level."""
        return state.spent + level

    def step(self, state: State, evidence: Evidence) -> Step:
        """Tests the next hypothesis.

        Args:
            state: State after the previous hypothesis.
            evidence: The p-value or e-value of the next hypothesis.

        Returns:
            The level, the decision, the wealth and the new state.

        Raises:
            ValueError: If the rule needs a kind of evidence that is missing.
            RuleInvariantError: If the wealth would turn negative.
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
            spent=self.update_spent(state, level, candidate),
        )
        after = self.wealth(moved)
        if after < 0:
            raise RuleInvariantError(f"wealth would be negative at test {time}")
        return Step(
            level=level,
            rejected=rejected,
            candidate=candidate,
            e_value=e_value,
            wealth_before=self.wealth(state),
            wealth_after=after,
            state=moved,
        )

    def credit(self, state: State) -> Fraction:
        """Initial wealth plus rewards, shared by LORD++ and SAFFRON."""
        return Fraction(0)


class ELond(OnlineRule):
    """e-LOND: FDR control under arbitrary dependence from valid e-values."""

    name = ELOND
    guarantee = Guarantee(
        statement=(
            "FDR <= alpha at every time, under arbitrary dependence between "
            "the e-values"
        ),
        assumptions=(
            "every true null has an e-value with E[e] <= 1",
            "hypotheses are fixed in advance, not chosen from earlier results",
            "p-values are converted with a calibrator and must be "
            "super-uniform: P(p <= s) <= s",
            "the discount sequence is fixed and sums to at most one",
        ),
        citation=(
            "Xu and Ramdas, Online multiple testing with e-values, "
            "arXiv:2311.06412v1, Theorem 1; calibrator: Vovk and Wang, "
            "Ann. Statist. 49(3), 2021"
        ),
        verified=True,
        dependence="arbitrary",
    )

    def __init__(
        self,
        alpha: Any,
        discount: Discount | None = None,
        kappa: float = DEFAULT_KAPPA,
    ) -> None:
        """Creates the rule; `kappa` is the p-to-e calibrator parameter."""
        super().__init__(alpha, discount)
        level = qualification.finite(kappa, "kappa")
        if not 0.0 < level < 1.0:
            raise ValueError("kappa must be strictly between 0 and 1")
        self.kappa = level

    def level(self, state: State) -> Fraction:
        """``alpha * gamma_t * (|R_{t-1}| + 1)``."""
        return (
            self.alpha
            * self.discount.weight(state.tests + 1)
            * (len(state.rejections) + 1)
        )

    def wealth(self, state: State) -> Fraction:
        """``alpha * (1 - sum_{i <= t} gamma_i)``."""
        return self.alpha * (1 - self.discount.cumulative(state.tests))

    def decide(
        self, level: Fraction, evidence: Evidence
    ) -> tuple[bool, bool, float | None]:
        """Rejects when ``e >= 1 / level`` (exactly, ``e * level >= 1``)."""
        if evidence.e is not None:
            e_value = evidence.e
        elif evidence.p is not None:
            e_value = calibrate(evidence.p, self.kappa)
        else:  # pragma: no cover - Evidence forbids this
            raise ValueError("evidence needs a p-value or an e-value")
        return Fraction(e_value) * level >= 1, False, e_value


class PValueRule(OnlineRule):
    """Shared parts of the p-value rules (initial wealth `w0`)."""

    def __init__(
        self,
        alpha: Any,
        discount: Discount | None = None,
        w0: Any = None,
    ) -> None:
        """Creates the rule; `w0` defaults to ``alpha / 2`` (the papers)."""
        super().__init__(alpha, discount)
        start = self.alpha / 2 if w0 is None else exact_alpha(w0, "w0")
        if not 0 <= start <= self.alpha:
            raise ValueError("w0 must be within [0, alpha]")
        self.w0 = start

    def credit(self, state: State) -> Fraction:
        """``W0 + (alpha - W0) 1[R >= 1] + alpha * max(R - 1, 0)``."""
        rejected = len(state.rejections)
        first = (self.alpha - self.w0) if rejected >= 1 else Fraction(0)
        return self.w0 + first + self.alpha * max(rejected - 1, 0)

    def require_p(self, evidence: Evidence) -> Fraction:
        """The exact p-value, which these rules need."""
        if evidence.p is None:
            raise ValueError(f"{self.name} needs a p-value")
        return Fraction(evidence.p)


class LordPlusPlus(PValueRule):
    """LORD++: FDR control for independent p-values."""

    name = LORD_PLUS_PLUS
    guarantee = Guarantee(
        statement="FDR <= alpha at every time for independent p-values",
        assumptions=(
            "the true-null p-values are mutually independent and independent "
            "of the non-null p-values",
            "each true-null p-value is super-uniform",
            "no guarantee under dependence",
        ),
        citation=(
            "Ramdas, Yang, Wainwright and Jordan, NeurIPS 2017, "
            "arXiv:1710.00499, eq. (5) and Theorem 1; onlineFDR 2.20.0 manual"
        ),
        verified=True,
        dependence="independent",
    )

    def level(self, state: State) -> Fraction:
        """Equation (5) of the paper."""
        time = state.tests + 1
        total = self.w0 * self.discount.weight(time)
        for position, moment in enumerate(state.rejections):
            gap = self.discount.weight(time - moment)
            total += (
                self.alpha - self.w0 if position == 0 else self.alpha
            ) * gap
        return total

    def wealth(self, state: State) -> Fraction:
        """Credit earned minus the level spent."""
        return self.credit(state) - state.spent

    def decide(
        self, level: Fraction, evidence: Evidence
    ) -> tuple[bool, bool, float | None]:
        """Rejects when ``p <= level``."""
        return self.require_p(evidence) <= level, False, None


class Saffron(PValueRule):
    """SAFFRON: adaptive FDR control for independent p-values."""

    name = SAFFRON
    guarantee = Guarantee(
        statement="FDR <= alpha at every time for independent p-values",
        assumptions=(
            "the true-null p-values are mutually independent and independent "
            "of the non-null p-values",
            "each true-null p-value is super-uniform",
            "levels are monotone functions of past rejections and candidates",
            "no guarantee under dependence",
        ),
        citation=(
            "Ramdas, Zrnic, Wainwright and Jordan, ICML 2018, "
            "arXiv:1802.09098, Section 2.3 and Theorem 1(d); onlineFDR "
            "2.20.0 manual"
        ),
        verified=True,
        dependence="independent",
    )

    def __init__(
        self,
        alpha: Any,
        discount: Discount | None = None,
        w0: Any = None,
        lam: Any = DEFAULT_LAMBDA,
    ) -> None:
        """Creates the rule; `lam` is the candidate threshold in (0, 1)."""
        super().__init__(alpha, discount, w0)
        threshold = budget_module.exact(lam, "lambda")
        if not 0 < threshold < 1:
            raise ValueError("lambda must be strictly between 0 and 1")
        self.lam = threshold

    def level(self, state: State) -> Fraction:
        """``min(lambda, (1 - lambda) * (...))`` from Section 2.3."""
        time = state.tests + 1
        seen = state.candidates
        total = self.w0 * self.discount.weight(time - len(seen))
        for position, moment in enumerate(state.rejections):
            after = sum(1 for c in seen if c > moment)
            gap = self.discount.weight(time - moment - after)
            total += (
                self.alpha - self.w0 if position == 0 else self.alpha
            ) * gap
        return min(self.lam, (1 - self.lam) * total)

    def wealth(self, state: State) -> Fraction:
        """``(1 - lambda) * credit`` minus the level spent on non-candidates."""
        return (1 - self.lam) * self.credit(state) - state.spent

    def decide(
        self, level: Fraction, evidence: Evidence
    ) -> tuple[bool, bool, float | None]:
        """Candidate when ``p <= lambda``; rejected when ``p <= level``."""
        p = self.require_p(evidence)
        return p <= level, p <= self.lam, None

    def update_spent(
        self, state: State, level: Fraction, candidate: bool
    ) -> Fraction:
        """Candidates cost nothing; other tests cost their level."""
        return state.spent if candidate else state.spent + level


@dataclasses.dataclass(frozen=True, slots=True)
class RuleConfig:
    """Serializable choice of rule and parameters.

    Attributes:
        rule: `e-lond`, `lord++` or `saffron`.
        alpha: Target FDR level (the exploration wealth `W0` of the budget).
        horizon: Discount horizon, or None for ``1 / (j (j + 1))``.
        w0: Initial wealth of LORD++/SAFFRON (None means ``alpha / 2``).
        lam: SAFFRON candidate threshold.
        kappa: e-LOND calibrator parameter.
    """

    rule: str
    alpha: Fraction
    horizon: int | None = None
    w0: Fraction | None = None
    lam: Fraction = DEFAULT_LAMBDA
    kappa: float = DEFAULT_KAPPA

    def __post_init__(self) -> None:
        """Validates by building the rule."""
        self.build()

    def build(self) -> OnlineRule:
        """The rule this configuration describes.

        Raises:
            ValueError: On an unknown rule or invalid parameters.
        """
        discount = Discount(self.horizon)
        if self.rule == ELOND:
            return ELond(self.alpha, discount, self.kappa)
        if self.rule == LORD_PLUS_PLUS:
            return LordPlusPlus(self.alpha, discount, self.w0)
        if self.rule == SAFFRON:
            return Saffron(self.alpha, discount, self.w0, self.lam)
        raise ValueError(f"rule must be one of {RULE_NAMES}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record; fractions are written as ``"p/q"``."""
        return {
            "rule": self.rule,
            "alpha": str(self.alpha),
            "horizon": self.horizon,
            "w0": None if self.w0 is None else str(self.w0),
            "lambda": str(self.lam),
            "kappa": self.kappa,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the configuration, rejecting unknown or missing fields."""
        names = {"rule", "alpha", "horizon", "w0", "lambda", "kappa"}
        if set(record) != names:
            raise ValueError("unexpected or missing rule fields")
        w0 = record["w0"]
        return cls(
            rule=record["rule"],
            alpha=budget_module.parse_fraction(record["alpha"], "alpha"),
            horizon=record["horizon"],
            w0=None if w0 is None else budget_module.parse_fraction(w0, "w0"),
            lam=budget_module.parse_fraction(record["lambda"], "lambda"),
            kappa=record["kappa"],
        )


def replay(config: RuleConfig, evidence: list[Evidence]) -> list[Step]:
    """Runs a rule over a fixed evidence sequence from the start.

    Args:
        config: The rule and its parameters.
        evidence: One item per hypothesis, in order.

    Returns:
        One step per hypothesis.
    """
    rule = config.build()
    state = rule.initial()
    steps: list[Step] = []
    for item in evidence:
        step = rule.step(state, item)
        steps.append(step)
        state = step.state
    return steps
