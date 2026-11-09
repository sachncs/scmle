"""Cost-sensitive thresholding with a calibration gate.

For a calibrated probability `p = P(y = 1 | x)` the expected cost of acting
is `(1 - p) c_FP` and of not acting `p c_FN`, so acting is optimal iff

    p >= t* = c_FP / (c_FP + c_FN).

That is optimal only if `p` is calibrated. This module therefore checks
calibration on development data first, with a binned chi-square test (equal
count bins; the statistic compares observed events to the sum of the
predicted probabilities, with the Bernoulli variance, and has `bins` degrees
of freedom because the probabilities are out of fold). If the check rejects,
a Platt recalibration fitted on development data is tried and re-checked on
cross-fitted recalibrated probabilities; if that also fails the grid search of
`scmle.models.thresholds` runs and the result is `unverified`. The false-alarm
rate of the check is measured by the certificates. Nothing here reads the
final evaluation set.
"""

import dataclasses
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy import optimize
from scipy import stats

from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.framing import costs
from scmle.decision import netbenefit
from scmle.models import thresholds as grid_thresholds

BINS = 10
FOLDS = 5
CLIP = 1e-6
PATHS = ("bayes", "recalibrated", "grid")


def bayes_threshold(c_fp: float, c_fn: float) -> float:
    """Returns the Bayes threshold `t* = c_FP / (c_FP + c_FN)`.

    Args:
        c_fp: Cost of a false positive; positive.
        c_fn: Cost of a false negative; positive.

    Returns:
        The probability above which acting has the lower expected cost, in
        (0, 1).

    Raises:
        ValueError: If either cost is not positive. A cost that is not a
            finite number is rejected by `qualification.finite` with a
            `ValueError`.
    """
    fp = qualification.finite(c_fp, "c_fp")
    fn = qualification.finite(c_fn, "c_fn")
    if fp <= 0 or fn <= 0:
        raise ValueError("c_fp and c_fn must both be positive")
    return fp / (fp + fn)


@dataclasses.dataclass(frozen=True, slots=True)
class CalibrationCheck:
    """Result of the binned calibration test.

    Attributes:
        statistic: The chi-square statistic.
        degrees: Its degrees of freedom.
        p_value: Upper-tail probability.
        level: The level the check was run at.
        calibrated: Whether the check did not reject.
    """

    statistic: float
    degrees: int
    p_value: float
    level: float
    calibrated: bool


def probabilities_and_labels(
    p: ArrayLike, y: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validated probabilities and 0/1 labels of equal length.

    Args:
        p: Predicted probabilities.
        y: Labels, 0 or 1.

    Returns:
        The probabilities and the labels as float arrays.

    Raises:
        ValueError: If the shapes differ, or a probability is not finite or
            lies outside [0, 1]. Labels that are not a non-empty
            one-dimensional array of 0 and 1 are rejected by
            `costs.binary_vector` with a `ValueError`.
    """
    risk = np.asarray(p, dtype=float)
    labels = costs.binary_vector(y, "y")
    if risk.shape != labels.shape:
        raise ValueError("p and y must have the same length")
    if not np.isfinite(risk).all() or (risk < 0).any() or (risk > 1).any():
        raise ValueError("p must lie in [0, 1]")
    return risk, labels


def calibration_check(
    p: ArrayLike, y: ArrayLike, level: float = 0.01, bins: int = BINS
) -> CalibrationCheck:
    """Binned chi-square test that `E[y | p] = p`.

    Args:
        p: Out-of-fold probabilities.
        y: Labels.
        level: Level of the test, in (0, 1).
        bins: Equal-count bins by probability.

    Returns:
        The statistic, p-value and whether the probabilities were accepted.

    Raises:
        ValueError: If `level` is not strictly between 0 and 1, or there
            are fewer than `5 * bins` rows. Invalid `p` or `y`, or `bins`
            below 2, are also rejected with a `ValueError` by the helpers
            `probabilities_and_labels` and `qualification.count`.
    """
    risk, labels = probabilities_and_labels(p, y)
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between 0 and 1")
    qualification.count(bins, "bins", minimum=2)
    if len(risk) < 5 * bins:
        raise ValueError("too few rows for the calibration check")
    order = np.argsort(risk, kind="stable")
    statistic = 0.0
    for part in np.array_split(order, bins):
        expected = float(risk[part].sum())
        variance = float((risk[part] * (1.0 - risk[part])).sum())
        observed = float(labels[part].sum())
        statistic += (observed - expected) ** 2 / max(variance, 1e-12)
    p_value = float(stats.chi2.sf(statistic, bins))
    return CalibrationCheck(
        statistic=statistic,
        degrees=bins,
        p_value=p_value,
        level=level,
        calibrated=p_value >= level,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Recalibrator:
    """Platt scaling `sigmoid(a + b logit(p))`, fitted on development data.

    Attributes:
        intercept: The intercept `a`.
        slope: The slope `b` on the logit of the probability.
    """

    intercept: float
    slope: float

    def apply(self, p: ArrayLike) -> NDArray[np.float64]:
        """Recalibrates probabilities.

        Args:
            p: Probabilities; clipped to `[CLIP, 1 - CLIP]` first.

        Returns:
            The recalibrated probabilities, same shape as `p`.
        """
        risk = np.clip(np.asarray(p, dtype=float), CLIP, 1.0 - CLIP)
        score = self.intercept + self.slope * np.log(risk / (1.0 - risk))
        result: NDArray[np.float64] = 1.0 / (1.0 + np.exp(-score))
        return result


def fit_recalibrator(p: ArrayLike, y: ArrayLike) -> Recalibrator:
    """Maximum-likelihood Platt scaling (two parameters, no penalty).

    Args:
        p: Probabilities to recalibrate; clipped to `[CLIP, 1 - CLIP]`.
        y: Labels, 0 or 1.

    Returns:
        The fitted intercept and slope.

    Invalid `p` or `y` are rejected with a `ValueError` by
    `probabilities_and_labels`.
    """
    risk, labels = probabilities_and_labels(p, y)
    clipped = np.clip(risk, CLIP, 1.0 - CLIP)
    feature = np.log(clipped / (1.0 - clipped))

    def loss(theta: NDArray[np.float64]) -> float:
        score = theta[0] + theta[1] * feature
        return float(np.sum(np.logaddexp(0.0, score) - labels * score))

    def gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        score = theta[0] + theta[1] * feature
        residual = 1.0 / (1.0 + np.exp(-score)) - labels
        return np.array([residual.sum(), (residual * feature).sum()])

    fitted = optimize.minimize(
        loss, np.array([0.0, 1.0]), jac=gradient, method="BFGS"
    )
    return Recalibrator(float(fitted.x[0]), float(fitted.x[1]))


def cross_fitted(p: ArrayLike, y: ArrayLike, seed: int) -> NDArray[np.float64]:
    """Recalibrated probabilities where each row's map never saw its label.

    Rows are split at random into `FOLDS` folds; the map applied to a fold is
    fitted on the other folds.

    Args:
        p: Probabilities to recalibrate.
        y: Labels, 0 or 1.
        seed: Seed of the fold assignment.

    Returns:
        The cross-fitted probabilities, one per row, in input order.

    Invalid `p` or `y` are rejected with a `ValueError` by
    `probabilities_and_labels`.
    """
    risk, labels = probabilities_and_labels(p, y)
    rng = randomness.component_rng(seed, "decision.threshold.folds", 0)
    fold = rng.permutation(len(risk)) % FOLDS
    out = np.empty_like(risk)
    for k in range(FOLDS):
        held = fold == k
        out[held] = fit_recalibrator(risk[~held], labels[~held]).apply(
            risk[held]
        )
    return out


@dataclasses.dataclass(frozen=True, slots=True)
class ThresholdRule:
    """A frozen decision rule: act iff the (recalibrated) probability >= t.

    Attributes:
        threshold: The cut on the (recalibrated) probability.
        path: `bayes`, `recalibrated` or `grid`.
        recalibrator: The Platt map when `path` is `recalibrated`.
    """

    threshold: float
    path: str
    recalibrator: Recalibrator | None = None

    def act(self, p: ArrayLike) -> NDArray[np.float64]:
        """Applies the rule.

        Args:
            p: Raw probabilities.

        Returns:
            1.0 where the (recalibrated) probability reaches the threshold,
            otherwise 0.0, one per input.
        """
        risk = np.asarray(p, dtype=float)
        if self.recalibrator is not None:
            risk = self.recalibrator.apply(risk)
        return (risk >= self.threshold).astype(float)

    @property
    def raw_threshold(self) -> float | None:
        """The equivalent cut on the raw probability, or None.

        A Platt map with a positive slope is increasing, so acting when the
        recalibrated probability reaches `t` is acting when the raw one
        reaches the returned cut. A non-increasing map has no such cut.
        """
        r = self.recalibrator
        if r is None:
            return self.threshold
        if r.slope <= 0:
            return None
        score = np.log(self.threshold / (1.0 - self.threshold))
        return float(1.0 / (1.0 + np.exp(-(score - r.intercept) / r.slope)))

    @property
    def digest(self) -> str:
        """The hash identifying the frozen rule (not its actions)."""
        r = self.recalibrator
        label = f"{self.path}|{r.intercept!r}|{r.slope!r}" if r else self.path
        return netbenefit.rule_hash(label, self.threshold)


@dataclasses.dataclass(frozen=True, slots=True)
class ThresholdDecision:
    """What the thresholding step produced on development data.

    Attributes:
        rule: The frozen rule.
        tier: `attested` when every cost came from a person and the Bayes
            or recalibrated path ran, otherwise `unverified`.
        check: The calibration check on the raw probabilities.
        recheck: The check on cross-fitted recalibrated probabilities, or
            None when the raw probabilities passed.
        expected_cost: Mean cost per row of the rule on the development rows.
        cost_interval: Bootstrap interval of that cost (cluster bootstrap
            when clusters are given).
        stable_range: Span of `t` inside the chartered range on which the
            rule's development net benefit beats treating all and none, or
            None.
    """

    rule: ThresholdRule
    tier: str
    check: CalibrationCheck
    recheck: CalibrationCheck | None
    expected_cost: float
    cost_interval: tuple[float, float]
    stable_range: tuple[float, float] | None

    def to_dict(self) -> dict[str, Any]:
        """Returns the JSON-ready record.

        Returns:
            The threshold, path, rule hash, tier, both checks, expected
            cost, cost interval and stable range, with tuples as lists.
        """
        return {
            "threshold": self.rule.threshold,
            "path": self.rule.path,
            "rule_hash": self.rule.digest,
            "tier": self.tier,
            "check": dataclasses.asdict(self.check),
            "recheck": (
                None
                if self.recheck is None
                else dataclasses.asdict(self.recheck)
            ),
            "expected_cost": self.expected_cost,
            "cost_interval": list(self.cost_interval),
            "stable_range": (
                None if self.stable_range is None else list(self.stable_range)
            ),
        }


def cost_interval(
    y: NDArray[np.float64],
    acts: NDArray[np.float64],
    model: costs.DecisionCostModel,
    clusters: ArrayLike | None,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    """95% percentile interval of the mean cost per row.

    Args:
        y: Labels, 0 or 1.
        acts: Actions of the rule, 0 or 1, aligned with `y`.
        model: The cost model giving the cost of each (action, outcome).
        clusters: Cluster label per row, or None to resample rows.
        replicates: Number of bootstrap replicates.
        seed: Seed of the bootstrap stream.

    Returns:
        The 2.5% and 97.5% quantiles of the bootstrapped mean cost per row.
        With clusters, whole clusters are resampled. Clusters that are not
        one-dimensional are rejected with a `ValueError` by
        `netbenefit.factorize`.
    """
    cell = np.where(
        acts == 1.0,
        np.where(y == 1.0, model.c_11, model.c_10),
        np.where(y == 1.0, model.c_01, model.c_00),
    )
    groups = netbenefit.factorize(clusters)
    rng = randomness.component_rng(seed, "decision.threshold.cost", 0)
    if groups is None:
        draws = rng.integers(0, len(cell), size=(replicates, len(cell)))
        means = cell[draws].mean(axis=1)
    else:
        size = int(np.max(groups)) + 1
        sums = np.bincount(groups, weights=cell, minlength=size)
        rows = np.bincount(groups, minlength=size).astype(float)
        picks = rng.integers(0, size, size=(replicates, size))
        means = sums[picks].sum(axis=1) / rows[picks].sum(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def stable_span(
    p: NDArray[np.float64],
    y: NDArray[np.float64],
    rule: ThresholdRule,
    model: costs.DecisionCostModel,
    points: int = 41,
) -> tuple[float, float] | None:
    """Span of `t` in the chartered range where the frozen rule beats both.

    Args:
        p: Raw probabilities.
        y: Labels, 0 or 1.
        rule: The frozen rule.
        model: The cost model giving the chartered range of `t`.
        points: Number of grid points across the range.

    Returns:
        The first and last grid value of the longest run of thresholds at
        which the rule's net benefit exceeds both treating everyone and
        treating no one, or None if there is none.
    """
    acts = rule.act(p)
    grid = np.linspace(model.t_lo, model.t_hi, points)
    good = np.array(
        [
            costs.net_benefit(y, acts, t)
            > max(costs.net_benefit_all(y, t), 0.0)
            for t in grid
        ]
    )
    return netbenefit.useful_span(grid, good)


def choose_threshold(
    p: ArrayLike,
    y: ArrayLike,
    model: costs.DecisionCostModel,
    *,
    level: float = 0.01,
    clusters: ArrayLike | None = None,
    replicates: int = 500,
    seed: int = 0,
) -> ThresholdDecision:
    """Chooses the operating threshold on development data only.

    Args:
        p: Out-of-fold development probabilities.
        y: Development labels.
        model: The chartered cost model.
        level: Level of the calibration checks.
        clusters: Entity labels when the split is grouped.
        replicates: Bootstrap replicates for the cost interval.
        seed: Seed of the streams.

    Returns:
        The frozen rule and its development evidence.
    """
    risk, labels = probabilities_and_labels(p, y)
    first = calibration_check(risk, labels, level)
    recheck = None
    target = bayes_threshold(model.c_fp, model.c_fn)
    if first.calibrated:
        rule = ThresholdRule(target, "bayes")
    else:
        recheck = calibration_check(
            cross_fitted(risk, labels, seed), labels, level
        )
        platt = fit_recalibrator(risk, labels)
        if recheck.calibrated and platt.slope > 0:
            rule = ThresholdRule(target, "recalibrated", platt)
        else:
            cut, ignored = grid_thresholds.optimize_threshold(
                risk,
                labels.astype(bool),
                grid_thresholds.CostObjective(model.c_fp, model.c_fn),
            )
            rule = ThresholdRule(cut, "grid")
    acts = rule.act(risk)
    tier = (
        "attested"
        if rule.path != "grid" and not model.unverified_entries
        else "unverified"
    )
    return ThresholdDecision(
        rule=rule,
        tier=tier,
        check=first,
        recheck=recheck,
        expected_cost=costs.expected_cost(labels, acts, model),
        cost_interval=cost_interval(
            labels, acts, model, clusters, replicates, seed
        ),
        stable_range=stable_span(risk, labels, rule, model),
    )
