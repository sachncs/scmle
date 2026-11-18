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

import numpy
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy import optimize
from scipy import stats

from scmle.assurance.framing import costs
from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.decision import netbenefit
from scmle.models import thresholds
from scmle.runtime import randomness

BINS = 10
FOLDS = 5
CLIP = 1e-6
PATHS = ("bayes", "recalibrated", "grid")


def breakeven(c_fp: float, c_fn: float) -> float:
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
        raise errors.InputError("c_fp and c_fn must both be positive")
    return fp / (fp + fn)


@dataclasses.dataclass(frozen=True, slots=True)
class Calibration:
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


def pair(
    p: ArrayLike, y: ArrayLike
) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
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
            `costs.dichotomous` with a `ValueError`.
    """
    risk = numpy.asarray(p, dtype=float)
    labels = costs.dichotomous(y, "y")
    if risk.shape != labels.shape:
        raise errors.InputError("p and y must have the same length")
    if not numpy.isfinite(risk).all() or (risk < 0).any() or (risk > 1).any():
        raise errors.InputError("p must lie in [0, 1]")
    return risk, labels


def hosmer(
    p: ArrayLike, y: ArrayLike, level: float = 0.01, bins: int = BINS
) -> Calibration:
    """Binned chi-square test that `E[y | p] = p`.

    Args:
        p: Out-of-fold probabilities.
        y: Labels.
        level: Level of the test, in (0, 1).
        bins: Equal-count bins by probability.

    Returns:
        The check: its statistic, `bins` degrees of freedom, upper-tail
        p-value, `level` and whether the p-value reached `level`
        (`calibrated`).

    Raises:
        ValueError: If `level` is not strictly between 0 and 1, or there
            are fewer than `5 * bins` rows. Invalid `p` or `y`, or `bins`
            below 2, are also rejected with a `ValueError` by the helpers
            `pair` and `qualification.count`.
    """
    risk, labels = pair(p, y)
    if not 0.0 < level < 1.0:
        raise errors.InputError("level must be strictly between 0 and 1")
    qualification.count(bins, "bins", minimum=2)
    if len(risk) < 5 * bins:
        raise errors.InputError("too few rows for the calibration check")
    order = numpy.argsort(risk, kind="stable")
    statistic = 0.0
    for part in numpy.array_split(order, bins):
        expected = float(risk[part].sum())
        variance = float((risk[part] * (1.0 - risk[part])).sum())
        observed = float(labels[part].sum())
        statistic += (observed - expected) ** 2 / max(variance, 1e-12)
    p_value = float(stats.chi2.sf(statistic, bins))
    return Calibration(
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

    def apply(self, p: ArrayLike) -> NDArray[numpy.float64]:
        """Recalibrates probabilities.

        Args:
            p: Probabilities; clipped to `[CLIP, 1 - CLIP]` first.

        Returns:
            The recalibrated probabilities, same shape as `p`.
        """
        risk = numpy.clip(numpy.asarray(p, dtype=float), CLIP, 1.0 - CLIP)
        score = self.intercept + self.slope * numpy.log(risk / (1.0 - risk))
        result: NDArray[numpy.float64] = 1.0 / (1.0 + numpy.exp(-score))
        return result


def calibrator(p: ArrayLike, y: ArrayLike) -> Recalibrator:
    """Maximum-likelihood Platt scaling (two parameters, no penalty).

    Args:
        p: Probabilities to recalibrate; clipped to `[CLIP, 1 - CLIP]`.
        y: Labels, 0 or 1.

    Returns:
        The fitted intercept and slope.

    Invalid `p` or `y` are rejected with a `ValueError` by
    `pair`.
    """
    risk, labels = pair(p, y)
    clipped = numpy.clip(risk, CLIP, 1.0 - CLIP)
    feature = numpy.log(clipped / (1.0 - clipped))

    def loss(theta: NDArray[numpy.float64]) -> float:
        score = theta[0] + theta[1] * feature
        return float(numpy.sum(numpy.logaddexp(0.0, score) - labels * score))

    def gradient(theta: NDArray[numpy.float64]) -> NDArray[numpy.float64]:
        score = theta[0] + theta[1] * feature
        residual = 1.0 / (1.0 + numpy.exp(-score)) - labels
        return numpy.array([residual.sum(), (residual * feature).sum()])

    fitted = optimize.minimize(
        loss, numpy.array([0.0, 1.0]), jac=gradient, method="BFGS"
    )
    return Recalibrator(float(fitted.x[0]), float(fitted.x[1]))


def crossfit(p: ArrayLike, y: ArrayLike, seed: int) -> NDArray[numpy.float64]:
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
    `pair`.
    """
    risk, labels = pair(p, y)
    rng = randomness.component_rng(seed, "decision.threshold.folds", 0)
    fold = rng.permutation(len(risk)) % FOLDS
    out = numpy.empty_like(risk)
    for k in range(FOLDS):
        held = fold == k
        out[held] = calibrator(risk[~held], labels[~held]).apply(risk[held])
    return out


@dataclasses.dataclass(frozen=True, slots=True)
class Threshold:
    """A frozen decision rule: act iff the (recalibrated) probability >= t.

    Attributes:
        threshold: The cut on the (recalibrated) probability.
        path: `bayes`, `recalibrated` or `grid`.
        recalibrator: The Platt map when `path` is `recalibrated`.
    """

    threshold: float
    path: str
    recalibrator: Recalibrator | None = None

    def act(self, p: ArrayLike) -> NDArray[numpy.float64]:
        """Applies the rule.

        Args:
            p: Raw probabilities.

        Returns:
            1.0 where the (recalibrated) probability reaches the threshold,
            otherwise 0.0, one per input.
        """
        risk = numpy.asarray(p, dtype=float)
        if self.recalibrator is not None:
            risk = self.recalibrator.apply(risk)
        return (risk >= self.threshold).astype(float)

    @property
    def uncalibrated(self) -> float | None:
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
        score = numpy.log(self.threshold / (1.0 - self.threshold))
        return float(1.0 / (1.0 + numpy.exp(-(score - r.intercept) / r.slope)))

    @property
    def digest(self) -> str:
        """The hash identifying the frozen rule (not its actions)."""
        r = self.recalibrator
        label = f"{self.path}|{r.intercept!r}|{r.slope!r}" if r else self.path
        return netbenefit.rule_hash(label, self.threshold)


@dataclasses.dataclass(frozen=True, slots=True)
class Thresholding:
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
        stable_range: First and last value of the longest run of grid points
            of `t` inside the chartered range at which the rule's development
            net benefit beats treating all and treating none, or None.
    """

    rule: Threshold
    tier: str
    check: Calibration
    recheck: Calibration | None
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
    y: NDArray[numpy.float64],
    acts: NDArray[numpy.float64],
    model: costs.Tariff,
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
    cell = numpy.where(
        acts == 1.0,
        numpy.where(y == 1.0, model.c_11, model.c_10),
        numpy.where(y == 1.0, model.c_01, model.c_00),
    )
    groups = netbenefit.factorize(clusters)
    rng = randomness.component_rng(seed, "decision.threshold.cost", 0)
    if groups is None:
        draws = rng.integers(0, len(cell), size=(replicates, len(cell)))
        means = cell[draws].mean(axis=1)
    else:
        size = int(numpy.max(groups)) + 1
        sums = numpy.bincount(groups, weights=cell, minlength=size)
        rows = numpy.bincount(groups, minlength=size).astype(float)
        picks = rng.integers(0, size, size=(replicates, size))
        means = sums[picks].sum(axis=1) / rows[picks].sum(axis=1)
    return float(numpy.quantile(means, 0.025)), float(
        numpy.quantile(means, 0.975)
    )


def span(
    p: NDArray[numpy.float64],
    y: NDArray[numpy.float64],
    rule: Threshold,
    model: costs.Tariff,
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
    grid = numpy.linspace(model.low, model.high, points)
    good = numpy.array(
        [
            costs.net_benefit(y, acts, t)
            > max(costs.net_benefit_all(y, t), 0.0)
            for t in grid
        ]
    )
    return netbenefit.stretch(grid, good)


def operate(
    p: ArrayLike,
    y: ArrayLike,
    model: costs.Tariff,
    *,
    level: float = 0.01,
    clusters: ArrayLike | None = None,
    replicates: int = 500,
    seed: int = 0,
) -> Thresholding:
    """Chooses the operating threshold on development data only.

    If the probabilities pass a calibration check, the rule is the Bayes
    threshold of the cost model. Otherwise the probabilities are recalibrated
    with Platt scaling when a cross-fitted recheck passes and the slope is
    positive; failing that, the threshold is the cost-minimising cut found
    on a grid. A grid rule, or a cost model with unverified entries, is
    labelled `unverified`; otherwise the tier is `attested`.

    Args:
        p: Out-of-fold development probabilities.
        y: Development labels, 0 or 1.
        model: The chartered cost model.
        level: Significance level, in (0, 1), of the calibration checks.
        clusters: Entity labels when the split is grouped, so the cost
            interval resamples whole entities; None resamples rows.
        replicates: Bootstrap replicates for the cost interval.
        seed: Seed of the cross-fitting and of the bootstrap.

    Returns:
        The frozen rule with its tier, calibration checks, expected cost,
        cost interval and stable threshold range (None if there is none).
        Invalid inputs are rejected with a `ValueError` by the helpers.
    """
    risk, labels = pair(p, y)
    first = hosmer(risk, labels, level)
    recheck = None
    target = breakeven(model.c_fp, model.c_fn)
    if first.calibrated:
        rule = Threshold(target, "bayes")
    else:
        recheck = hosmer(crossfit(risk, labels, seed), labels, level)
        platt = calibrator(risk, labels)
        if recheck.calibrated and platt.slope > 0:
            rule = Threshold(target, "recalibrated", platt)
        else:
            cut = thresholds.argmax(
                risk,
                labels.astype(bool),
                thresholds.Costs(model.c_fp, model.c_fn),
            )[0]
            rule = Threshold(cut, "grid")
    acts = rule.act(risk)
    tier = (
        "attested"
        if rule.path != "grid" and not model.unreviewed
        else "unverified"
    )
    return Thresholding(
        rule=rule,
        tier=tier,
        check=first,
        recheck=recheck,
        expected_cost=costs.expected_cost(labels, acts, model),
        cost_interval=cost_interval(
            labels, acts, model, clusters, replicates, seed
        ),
        stable_range=span(risk, labels, rule, model),
    )
