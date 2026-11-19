"""e-LOND, LORD++ and SAFFRON, their settings and a replay."""

from collections.abc import Mapping
import dataclasses
from fractions import Fraction
from typing import Any
from typing import Self

from scmle.assurance.ledger import budget
from scmle.assurance.ledger.online_fdr import tester
from scmle.contracts import errors
from scmle.contracts import qualification


class Lond(tester.Tester):
    """e-LOND: FDR control under arbitrary dependence from valid e-values."""

    name = tester.ELOND
    guarantee = tester.Guarantee(
        statement=(
            "FDR <= alpha at every time, under arbitrary dependence between "
            "the e-values"
        ),
        assumptions=(
            "every true null has an e-value with E[e] <= 1",
            "hypotheses are fixed in advance, not chosen from earlier results",
            (
                "p-values are converted with a calibrator and must be "
                "super-uniform: P(p <= s) <= s"
            ),
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
        discount: tester.Discount | None = None,
        kappa: float = tester.DEFAULT_KAPPA,
    ) -> None:
        """Creates the rule.

        Args:
            alpha: Target FDR level in [0, 1].
            discount: The discount sequence (default ``1 / (j (j + 1))``).
            kappa: The p-to-e calibrator parameter, strictly between 0 and 1.

        Raises:
            ValueError: If `kappa` is not strictly between 0 and 1.
        """
        super().__init__(alpha, discount)
        level = qualification.finite(kappa, "kappa")
        if not 0.0 < level < 1.0:
            raise errors.InputError("kappa must be strictly between 0 and 1")
        self.kappa = level

    def level(self, state: tester.State) -> Fraction:
        """Computes ``alpha * gamma_t * (|R_{t-1}| + 1)``.

        Args:
            state: The state after the previous hypothesis.

        Returns:
            The level of the next hypothesis.
        """
        return (
            self.alpha
            * self.discount.weight(state.tests + 1)
            * (len(state.rejections) + 1)
        )

    def wealth(self, state: tester.State) -> Fraction:
        """Computes ``alpha * (1 - sum_{i <= t} gamma_i)``.

        Args:
            state: The state after ``t`` tests.

        Returns:
            The share of the level not yet assigned to any hypothesis.
        """
        return self.alpha * (1 - self.discount.cumulative(state.tests))

    def decide(
        self, level: Fraction, evidence: tester.Evidence
    ) -> tuple[bool, bool, float | None]:
        """Rejects when ``e >= 1 / level`` (exactly, ``e * level >= 1``).

        Args:
            level: The level assigned to the hypothesis.
            evidence: Its evidence; a p-value is calibrated into an e-value
                only when no e-value is given.

        Returns:
            Whether it is rejected, False (e-LOND has no candidates) and the
            e-value used.

        Raises:
            ValueError: If the evidence holds neither value (`Evidence`
                forbids this).
        """
        if evidence.e is not None:
            e_value = evidence.e
        elif evidence.p is not None:
            e_value = tester.calibrate(evidence.p, self.kappa)
        else:  # pragma: no cover - Evidence forbids this
            raise errors.InputError("evidence needs a p-value or an e-value")
        return Fraction(e_value) * level >= 1, False, e_value


class PValueTester(tester.Tester):
    """Shared parts of the p-value rules (initial wealth `w0`)."""

    def __init__(
        self,
        alpha: Any,
        discount: tester.Discount | None = None,
        w0: Any = None,
    ) -> None:
        """Creates the rule.

        Args:
            alpha: Target FDR level in [0, 1].
            discount: The discount sequence (default ``1 / (j (j + 1))``).
            w0: Initial wealth; defaults to ``alpha / 2`` (the papers).

        Raises:
            ValueError: If `w0` is outside [0, alpha].
        """
        super().__init__(alpha, discount)
        start = self.alpha / 2 if w0 is None else tester.rational(w0, "w0")
        if not 0 <= start <= self.alpha:
            raise errors.InputError("w0 must be within [0, alpha]")
        self.w0 = start

    def credit(self, state: tester.State) -> Fraction:
        """Computes ``w0 + (alpha - w0) 1[R >= 1] + alpha * max(R - 1, 0)``.

        Args:
            state: The state to measure; ``R`` is its number of rejections.

        Returns:
            The initial wealth plus the reward for the rejections so far.
        """
        rejected = len(state.rejections)
        first = (self.alpha - self.w0) if rejected >= 1 else Fraction(0)
        return self.w0 + first + self.alpha * max(rejected - 1, 0)

    def read(self, evidence: tester.Evidence) -> Fraction:
        """Reads the exact p-value, which these rules need.

        Args:
            evidence: The evidence of the hypothesis.

        Returns:
            The p-value as an exact fraction.

        Raises:
            ValueError: If the evidence has no p-value.
        """
        if evidence.p is None:
            raise errors.InputError(f"{self.name} needs a p-value")
        return Fraction(evidence.p)


class Lord(PValueTester):
    """LORD++: FDR control for independent p-values."""

    name = tester.LORD_PLUS_PLUS
    guarantee = tester.Guarantee(
        statement="FDR <= alpha at every time for independent p-values",
        assumptions=(
            (
                "the true-null p-values are mutually independent and "
                "independent of the non-null p-values"
            ),
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

    def level(self, state: tester.State) -> Fraction:
        """Computes the level from equation (5) of the paper.

        Args:
            state: The state after the previous hypothesis.

        Returns:
            The level of the next hypothesis.
        """
        time = state.tests + 1
        total = self.w0 * self.discount.weight(time)
        for position, moment in enumerate(state.rejections):
            gap = self.discount.weight(time - moment)
            total += (
                self.alpha - self.w0 if position == 0 else self.alpha
            ) * gap
        return total

    def wealth(self, state: tester.State) -> Fraction:
        """Computes the credit earned minus the level spent.

        Args:
            state: The state to measure.

        Returns:
            The wealth in `state`.
        """
        return self.credit(state) - state.spent

    def decide(
        self, level: Fraction, evidence: tester.Evidence
    ) -> tuple[bool, bool, float | None]:
        """Rejects when ``p <= level``.

        Args:
            level: The level assigned to the hypothesis.
            evidence: Its evidence, which must hold a p-value (a missing one
                is refused with a `ValueError` by `read`).

        Returns:
            Whether it is rejected, False (LORD++ has no candidates) and
            None (no e-value is used).
        """
        return self.read(evidence) <= level, False, None


class Saffron(PValueTester):
    """SAFFRON: adaptive FDR control for independent p-values."""

    name = tester.SAFFRON
    guarantee = tester.Guarantee(
        statement="FDR <= alpha at every time for independent p-values",
        assumptions=(
            (
                "the true-null p-values are mutually independent and "
                "independent of the non-null p-values"
            ),
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
        discount: tester.Discount | None = None,
        w0: Any = None,
        lam: Any = tester.DEFAULT_LAMBDA,
    ) -> None:
        """Creates the rule.

        Args:
            alpha: Target FDR level in [0, 1].
            discount: The discount sequence (default ``1 / (j (j + 1))``).
            w0: Initial wealth; defaults to ``alpha / 2``.
            lam: The candidate threshold, strictly between 0 and 1.

        Raises:
            ValueError: If `lam` is not strictly between 0 and 1.
        """
        super().__init__(alpha, discount, w0)
        threshold = budget.exact(lam, "lambda")
        if not 0 < threshold < 1:
            raise errors.InputError("lambda must be strictly between 0 and 1")
        self.lam = threshold

    def level(self, state: tester.State) -> Fraction:
        """Computes ``min(lambda, (1 - lambda) * (...))`` from Section 2.3.

        Args:
            state: The state after the previous hypothesis.

        Returns:
            The level of the next hypothesis.
        """
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

    def wealth(self, state: tester.State) -> Fraction:
        """Computes ``(1 - lambda) * credit`` minus the level spent.

        Only non-candidates are charged to the level spent.

        Args:
            state: The state to measure.

        Returns:
            The wealth in `state`.
        """
        return (1 - self.lam) * self.credit(state) - state.spent

    def decide(
        self, level: Fraction, evidence: tester.Evidence
    ) -> tuple[bool, bool, float | None]:
        """Marks a candidate when ``p <= lambda``, a rejection ``p <= level``.

        Args:
            level: The level assigned to the hypothesis.
            evidence: Its evidence, which must hold a p-value (a missing one
                is refused with a `ValueError` by `read`).

        Returns:
            Whether it is rejected, whether it is a candidate, and None (no
            e-value is used).
        """
        p = self.read(evidence)
        return p <= level, p <= self.lam, None

    def expend(
        self, state: tester.State, level: Fraction, candidate: bool
    ) -> Fraction:
        """Computes the level spent; candidates cost nothing.

        Args:
            state: The state before the test.
            level: The level the test was given.
            candidate: Whether the test was a candidate.

        Returns:
            The spent total unchanged for a candidate, otherwise increased by
            the level.
        """
        return state.spent if candidate else state.spent + level


@dataclasses.dataclass(frozen=True, slots=True)
class Settings:
    """Serializable choice of rule and parameters.

    Attributes:
        rule: `e-lond`, `lord++` or `saffron`.
        alpha: Target FDR level (the exploration wealth of the budget).
        horizon: Discount horizon, or None for ``1 / (j (j + 1))``.
        w0: Initial wealth of LORD++/SAFFRON (None means ``alpha / 2``).
        lam: SAFFRON candidate threshold.
        kappa: e-LOND calibrator parameter.
    """

    rule: str
    alpha: Fraction
    horizon: int | None = None
    w0: Fraction | None = None
    lam: Fraction = tester.DEFAULT_LAMBDA
    kappa: float = tester.DEFAULT_KAPPA

    def __post_init__(self) -> None:
        """Validates by building the rule.

        A rule name or parameter that `build` refuses raises the same
        `ValueError` here.
        """
        self.build()

    def build(self) -> tester.Tester:
        """Builds the rule this configuration describes.

        Returns:
            The rule object, with the discount sequence and parameters of
            this configuration.

        Raises:
            ValueError: On an unknown rule or invalid parameters.
        """
        discount = tester.Discount(self.horizon)
        if self.rule == tester.ELOND:
            return Lond(self.alpha, discount, self.kappa)
        if self.rule == tester.LORD_PLUS_PLUS:
            return Lord(self.alpha, discount, self.w0)
        if self.rule == tester.SAFFRON:
            return Saffron(self.alpha, discount, self.w0, self.lam)
        raise errors.InputError(f"rule must be one of {tester.RULE_NAMES}")

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record.

        Returns:
            A mapping of every parameter, with fractions written as
            ``"p/q"`` strings (``alpha``, ``w0``, ``lambda``), the SAFFRON
            threshold stored under ``lambda`` and ``horizon`` and ``kappa``
            unchanged.
        """
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
        """Rebuilds the configuration, rejecting unknown or missing fields.

        Args:
            record: A mapping as written by `to_dict`.

        Returns:
            The configuration described by the record.

        Raises:
            ValueError: If the record has unexpected or missing fields. A
                fraction that does not parse, an unknown rule or an invalid
                parameter is also refused with a `ValueError`.
        """
        names = {"rule", "alpha", "horizon", "w0", "lambda", "kappa"}
        if set(record) != names:
            raise errors.InputError("unexpected or missing rule fields")
        w0 = record["w0"]
        return cls(
            rule=record["rule"],
            alpha=budget.parse(record["alpha"], "alpha"),
            horizon=record["horizon"],
            w0=None if w0 is None else budget.parse(w0, "w0"),
            lam=budget.parse(record["lambda"], "lambda"),
            kappa=record["kappa"],
        )


def replay(
    config: Settings, evidence: list[tester.Evidence]
) -> list[tester.Step]:
    """Runs a rule over a fixed evidence sequence from the start.

    Args:
        config: The rule and its parameters.
        evidence: One item per hypothesis, in order.

    Returns:
        One step per hypothesis.
    """
    rule = config.build()
    state = rule.initial()
    steps: list[tester.Step] = []
    for item in evidence:
        step = rule.step(state, item)
        steps.append(step)
        state = step.state
    return steps
