"""Selective prediction: when to stop and ask a human (F13, E1).

Cases are routed to a human when a score ``g`` is below a threshold ``tau``.
Coverage is ``phi(tau) = P(g >= tau)`` and selective risk
``R(tau) = E[loss | g >= tau]``; the trade-off between them is the object of
El-Yaniv and Wiener (JMLR 11, 2010) and, for deep classifiers, Geifman and
El-Yaniv (arXiv:1705.08500). Both were read on 2026-11-05: coverage is the
mean of the selection function, the selective risk is the risk over the
accepted region, and Geifman and El-Yaniv's guarantee is that the selective
risk is at most the target ``r*`` with probability at least ``1 - delta`` over
an i.i.d. labelled sample (their Theorem 3.2; an i.i.d. assumption, not only
exchangeability).

Score. `set_size_score` turns the F12 conformal set size and the novelty flag
into ``g = 1 / |set|`` (0 for an empty or unbounded set, or a novel case). The
set carries a finite-sample marginal coverage statement; a softmax margin does
not.

Threshold. `select_threshold` picks ``tau`` on matured calibration cases by
fixed-sequence testing over a grid fixed in advance: from the most
conservative ``tau`` (answer only the top score) to the least, test
``H: R(tau) > r*`` with the exact binomial p-value
``P(Bin(n, r*) <= errors)``, stop at the first non-rejection and take the
least conservative ``tau`` whose hypothesis was rejected. That controls
``P(R(tau_hat) > r*) <= delta`` when the calibration cases are i.i.d. draws
from the deployment distribution and the losses are 0 or 1. This is the
fixed-sequence procedure of "Learn then Test" (Angelopoulos, Bates, Candes,
Jordan and Lei, arXiv:2110.01052, read on 2026-11-05): test hypotheses in a
fixed order from the most promising value and stop at the first
non-rejection (Section 2.3.1, Proposition 3: family-wise error control),
with, for selective classification, the exact binomial tail as p-value
(Section 4). The paper states an i.i.d. calibration set; the random number of
answered cases (the denominator of the selective risk) is how the paper's
selective-classification experiment is set up, and this module repeats it
without re-deriving it. With weights the
sample is replaced by its Kish effective size and the weighted error count is
rounded up; that is an approximation and is reported as such. With no ``tau``
rejected the answer is ``+inf``: everything is escalated.

Capacity. `route` never raises ``tau`` to fit the human capacity ``c_max``:
escalations beyond the budget are not answered by the model, they receive the
declared safe default action and a backlog entry.

Under shift the guarantee lapses; the loop watches the selective risk of
answered cases with an anytime-valid alarm (`scmle.monitoring.
sequential_tests.RiskAlarm`) and halts when it excludes ``r*``.
"""

from collections.abc import Sequence
import dataclasses
import math
from typing import Any

import numpy as np
from scipy import stats

ANSWERED = "answered"
ESCALATED = "escalated"
SAFE_DEFAULT = "safe_default"
DEFAULT_GRID = 101


def set_size_score(set_sizes: Any, novel: Any = None) -> np.ndarray:
    """Score ``g`` from conformal set sizes and a novelty flag.

    Args:
        set_sizes: Size of each case's prediction set; 0, NaN or infinity
            (an unbounded regression margin) mean "no usable set".
        novel: Optional boolean per case, True for out-of-support cases.

    Returns:
        ``1 / size`` in ``(0, 1]``, or 0 for a case with no usable set or a
        novel case.
    """
    sizes = np.asarray(set_sizes, dtype=float)
    usable = np.isfinite(sizes) & (sizes >= 1.0)
    score = np.where(usable, 1.0 / np.where(usable, sizes, 1.0), 0.0)
    if novel is not None:
        flags = np.asarray(novel, dtype=bool)
        if flags.shape != score.shape:
            raise ValueError("novel must be aligned with set_sizes")
        score = np.where(flags, 0.0, score)
    return score


def binomial_p_value(errors: int, cases: int, r_star: float) -> float:
    """Exact p-value of ``H: R > r*``: ``P(Bin(cases, r*) <= errors)``."""
    if cases <= 0:
        return 1.0
    return float(stats.binom.cdf(errors, cases, r_star))


@dataclasses.dataclass(frozen=True, slots=True)
class Step:
    """One tested threshold of the fixed sequence.

    Attributes:
        tau: Threshold.
        cases: Answered cases at `tau` (effective size when weighted).
        errors: Error count used by the test.
        p_value: Exact binomial p-value of ``H: R(tau) > r*``.
        rejected: Whether `p_value <= delta`.
    """

    tau: float
    cases: int
    errors: int
    p_value: float
    rejected: bool


@dataclasses.dataclass(frozen=True, slots=True)
class ThresholdResult:
    """The selected threshold and the sequence that led to it.

    Attributes:
        tau: Selected threshold; ``+inf`` when none was rejected.
        coverage: Share of calibration cases answered at `tau`.
        risk: Observed selective risk at `tau` (NaN when nothing is answered).
        r_star: Target risk.
        delta: Level of the guarantee.
        weighted: Whether Kish effective sizes were used (approximate).
        steps: The tested sequence.
    """

    tau: float
    coverage: float
    risk: float
    r_star: float
    delta: float
    weighted: bool
    steps: tuple[Step, ...]

    @property
    def everything_escalates(self) -> bool:
        """Whether no threshold passed (``tau = +inf``)."""
        return math.isinf(self.tau)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record (``+inf`` is written as ``null``)."""
        return {
            "tau": None if self.everything_escalates else self.tau,
            "coverage": self.coverage,
            "risk": None if math.isnan(self.risk) else self.risk,
            "r_star": self.r_star,
            "delta": self.delta,
            "weighted": self.weighted,
            "steps": [dataclasses.asdict(s) for s in self.steps],
        }


def validate_calibration(
    score: Any, loss: Any, weights: Any = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Checks a calibration set and returns it as arrays."""
    g = np.asarray(score, dtype=float)
    l_values = np.asarray(loss, dtype=float)
    if g.shape != l_values.shape or g.ndim != 1:
        raise ValueError("score and loss must be aligned one-dimensional")
    if np.any(~np.isfinite(g)) or np.any(g < 0) or np.any(g > 1):
        raise ValueError("score must be finite and within [0, 1]")
    if np.any(~np.isfinite(l_values)) or np.any(l_values < 0):
        raise ValueError("loss must be finite and non-negative")
    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape != g.shape or np.any(~np.isfinite(w)) or np.any(w <= 0):
            raise ValueError("weights must be positive and aligned")
    return g, l_values, w


def select_threshold(
    score: Any,
    loss: Any,
    r_star: float,
    delta: float,
    weights: Any = None,
    grid: Sequence[float] | None = None,
    minimum: int = 1,
) -> ThresholdResult:
    """Fixed-sequence selection of the least conservative safe ``tau``.

    Args:
        score: Score ``g`` of each matured calibration case, in ``[0, 1]``.
        loss: Loss of each case; the test is exact for losses in {0, 1}.
        r_star: Target selective risk, in ``(0, 1)``.
        delta: Level, in ``(0, 1)``: the chance that the selected threshold
            has true risk above `r_star` is at most `delta`.
        weights: Optional Horvitz-Thompson weights ``1 / pi``.
        grid: Thresholds in the order from most to least conservative,
            fixed before looking at the data (default: 101 points from 1 to
            0). It must not depend on the calibration losses.
        minimum: Smallest answered sample worth testing. Thresholds whose
            answered (effective) size is below the larger of `minimum` and
            the size at which a zero-error sample could reject
            (``ceil(log(delta) / log(1 - r*))``) are skipped: they can never
            be rejected, and skipping them depends on the scores and weights
            only, not on the losses.

    Returns:
        The result. ``tau = +inf`` when even the most conservative threshold
        is not rejected (for example too few calibration cases for `r_star`
        at `delta`).

    Raises:
        ValueError: On invalid inputs or a grid that is not decreasing.
    """
    if not 0.0 < r_star < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("r_star and delta must be strictly between 0 and 1")
    g, losses, w = validate_calibration(score, loss, weights)
    points = (
        np.linspace(1.0, 0.0, DEFAULT_GRID)
        if grid is None
        else np.asarray(grid, dtype=float)
    )
    if points.size == 0 or np.any(np.diff(points) >= 0):
        raise ValueError("grid must be strictly decreasing")
    steps: list[Step] = []
    chosen = math.inf
    needed = max(minimum, math.ceil(math.log(delta) / math.log1p(-r_star)))
    for tau in points:
        answered = g >= tau
        count = int(answered.sum())
        if w is None:
            used = count
        elif count:
            weight = w[answered]
            used = math.floor(
                float(weight.sum()) ** 2 / float((weight**2).sum())
            )
        else:
            used = 0
        if used < needed:
            steps.append(Step(float(tau), used, 0, 1.0, False))
            continue
        if w is None:
            errors = math.ceil(float(losses[answered].sum()) - 1e-12)
        else:
            rate = float(
                (weight * losses[answered]).sum() / float(weight.sum())
            )
            errors = math.ceil(rate * used - 1e-12)
        p = binomial_p_value(errors, used, r_star)
        rejected = p <= delta
        steps.append(Step(float(tau), used, errors, p, rejected))
        if not rejected:
            break
        chosen = float(tau)
    if math.isinf(chosen):
        return ThresholdResult(
            chosen, 0.0, math.nan, r_star, delta, w is not None, tuple(steps)
        )
    answered = g >= chosen
    risk = float(losses[answered].mean())
    return ThresholdResult(
        chosen,
        float(answered.mean()),
        risk,
        r_star,
        delta,
        w is not None,
        tuple(steps),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class RiskCoverage:
    """The empirical risk-coverage curve and its area.

    Attributes:
        tau: Thresholds, from the highest score down.
        coverage: Share answered at each threshold.
        risk: Selective risk at each threshold.
        aurc: Area under the risk-coverage curve (lower is better).
    """

    tau: np.ndarray
    coverage: np.ndarray
    risk: np.ndarray
    aurc: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "tau": self.tau.tolist(),
            "coverage": self.coverage.tolist(),
            "risk": self.risk.tolist(),
            "aurc": self.aurc,
        }


def risk_coverage(
    score: Any, loss: Any, weights: Any = None, points: int = DEFAULT_GRID
) -> RiskCoverage:
    """Risk-coverage curve on a fixed grid, optionally weighted."""
    g, losses, w = validate_calibration(score, loss, weights)
    grid = np.linspace(1.0, 0.0, points)
    weight = np.ones_like(g) if w is None else w
    coverage = np.empty(points)
    risk = np.empty(points)
    total = float(weight.sum())
    for index, tau in enumerate(grid):
        answered = g >= tau
        mass = float(weight[answered].sum())
        coverage[index] = mass / total
        risk[index] = (
            float((weight[answered] * losses[answered]).sum() / mass)
            if mass > 0
            else math.nan
        )
    valid = ~np.isnan(risk)
    area = (
        float(np.trapezoid(risk[valid], coverage[valid]))
        if valid.sum() > 1
        else math.nan
    )
    return RiskCoverage(grid, coverage, risk, area)


@dataclasses.dataclass(frozen=True, slots=True)
class ChowCheck:
    """Comparison of the threshold with Chow's rule.

    Attributes:
        cut: Chow's cut on ``1 - max_y p(y|x)``, equal to ``c_h / c_e``.
        disagreement: Share of cases on which the two rules decide
            differently (escalate under one, answer under the other).
        consistent: Whether `disagreement` is at most `tolerance`.
    """

    cut: float
    disagreement: float
    consistent: bool


def chow_check(
    confidence: Any,
    score: Any,
    tau: float,
    review_cost: float,
    error_cost: float,
    tolerance: float = 0.1,
) -> ChowCheck:
    """Cross-checks ``tau`` against Chow's rule.

    Chow (1957, 1970) was not available to the author. What El-Yaniv and
    Wiener (JMLR 11, Section 10, read) say of it is that it is the Bayes-optimal
    reject rule under 0-1 loss when the distribution is completely known, that
    it rejects when no posterior is sufficiently distinct, and that with
    posteriors estimated with error it need not be optimal (Fumera et al.
    2000). The reading ``c_h / c_e`` of the cut as a normalised reject cost is
    this module's own and was not checked against Chow.

    With calibrated probabilities, abstain when ``1 - max_y p(y|x) > c_h /
    c_e``. A large disagreement with the score rule means the calibration or
    the declared costs are off, and is reported.

    Args:
        confidence: ``max_y p(y|x)`` of each case.
        score: Score ``g`` of each case.
        tau: The selected threshold.
        review_cost: Cost of a human review, `c_h`.
        error_cost: Cost of an error, `c_e`.
        tolerance: Disagreement tolerated before it is called inconsistent.
    """
    if review_cost <= 0 or error_cost <= 0:
        raise ValueError("costs must be positive")
    p = np.asarray(confidence, dtype=float)
    g = np.asarray(score, dtype=float)
    if p.shape != g.shape:
        raise ValueError("confidence and score must be aligned")
    cut = review_cost / error_cost
    by_chow = (1.0 - p) > cut
    by_score = g < tau
    share = float(np.mean(by_chow != by_score))
    return ChowCheck(cut, share, share <= tolerance)


@dataclasses.dataclass(frozen=True, slots=True)
class Routing:
    """Where each case of a batch goes.

    Attributes:
        status: Per case, `answered`, `escalated` or `safe_default`.
        backlog: Number of cases that beat capacity and got the safe default.
    """

    status: np.ndarray
    backlog: int

    @property
    def answered(self) -> np.ndarray:
        """Cases the model answers."""
        return self.status == ANSWERED

    @property
    def coverage(self) -> float:
        """Share of the batch the model answers."""
        return float(self.answered.mean()) if self.status.size else 0.0


def route(
    score: Any,
    tau: float,
    c_max: float,
    halted: bool = False,
    blocked: Any = None,
) -> Routing:
    """Routes a batch: answer, escalate to a human, or safe default.

    Args:
        score: Score ``g`` of each case.
        tau: Threshold; cases below it are escalated.
        c_max: Largest share of the batch the humans can take.
        halted: When True every case is escalated (coverage 0).
        blocked: Optional boolean per case that must not be answered by the
            model (for example out-of-support cases with a high-impact
            action); they are escalated like low-score cases.

    Returns:
        The routing. Escalations beyond ``floor(c_max * n)`` go to the safe
        default, lowest scores first to the humans; `tau` is never changed.
    """
    g = np.asarray(score, dtype=float)
    if not 0.0 <= c_max <= 1.0:
        raise ValueError("c_max must be within [0, 1]")
    keep = np.zeros(g.shape, dtype=bool) if halted else g >= tau
    if blocked is not None:
        keep &= ~np.asarray(blocked, dtype=bool)
    status = np.where(keep, ANSWERED, ESCALATED).astype(object)
    room = math.floor(c_max * g.size + 1e-9)
    escalated = np.flatnonzero(~keep)
    if escalated.size > room:
        order = escalated[np.argsort(g[escalated], kind="stable")]
        overflow = order[room:]
        status[overflow] = SAFE_DEFAULT
        return Routing(status.astype(str), int(overflow.size))
    return Routing(status.astype(str), 0)
