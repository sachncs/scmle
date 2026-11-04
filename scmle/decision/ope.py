"""Off-policy evaluation of a target policy from a randomized log (F11, C).

Setting: rows `(x_i, a_i, y_i)` logged under a behaviour policy `mu(a | x)`
with recorded probabilities, and a target policy `pi(a | x)` fixed before the
outcomes of the evaluation log are read. The value is `V(pi) = E[Y(pi)]`.
With `w_i = pi(a_i | x_i) / mu(a_i | x_i)` (clipped at `M` when declared):

    IPW   = mean(w y)
    SNIPS = sum(w y) / sum(w)
    DR    = mean( sum_a pi(a | x) q(x, a) + w (y - q(x, a_i)) )

`q` is an outcome model fitted on development data and frozen. The intervals
are normal with the sample variance of the influence function (the per-row
term minus its mean), clustered when clusters are given. Clipping at `M`
trades bias for variance; the clipped IPW of non-negative rewards is a lower
bound on the value, and the estimate is labelled as clipped.

Three gates decide whether a value is issued, all declared in the charter:
support (the target may put mass only where `mu >= eps`), the Kish effective
sample size over `n`, and the presence of recorded propensities. A refusal
returns no value. Observational logs, whose propensities are estimated, are
accepted only with a statement from an eligible identification strategy
(F10); otherwise the result is labelled `unverified`.
"""

import dataclasses
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy import stats

from scmle.contracts import qualification
from scmle.evaluation import sequential

ESTIMATORS = ("ipw", "snips", "dr")
ISSUED = "issued"
REFUSED = "refused"


@dataclasses.dataclass(frozen=True, slots=True)
class LoggedBandit:
    """A logged decision problem on `n` rows and `K` actions.

    Attributes:
        actions: Logged action index per row, in `0 .. K-1`.
        rewards: Logged reward per row.
        behaviour: `mu(a | x)` for every action, shape `(n, K)`; rows sum to
            one.
        target: `pi(a | x)` for every action, shape `(n, K)`.
        outcome_model: `q(x, a)` for every action, shape `(n, K)`, from a
            model fitted on development data; None for IPW and SNIPS only.
        clusters: Cluster label per row, or None.
        randomized: Whether `behaviour` was recorded by a randomization
            (True) or estimated from observational data (False).
    """

    actions: NDArray[np.intp]
    rewards: NDArray[np.float64]
    behaviour: NDArray[np.float64]
    target: NDArray[np.float64]
    outcome_model: NDArray[np.float64] | None = None
    clusters: NDArray[Any] | None = None
    randomized: bool = True

    def __post_init__(self) -> None:
        """Validates shapes and probabilities."""
        n = len(self.actions)
        if self.rewards.shape != (n,) or n == 0:
            raise ValueError("rewards must align with actions")
        for name in ("behaviour", "target"):
            matrix = getattr(self, name)
            if matrix.ndim != 2 or matrix.shape[0] != n:
                raise ValueError(f"{name} must have shape (n, K)")
            if (matrix < 0).any() or not np.allclose(matrix.sum(axis=1), 1.0):
                raise ValueError(f"{name} rows must be probabilities")
        k = self.behaviour.shape[1]
        if self.target.shape[1] != k:
            raise ValueError("behaviour and target need the same K")
        if ((self.actions < 0) | (self.actions >= k)).any():
            raise ValueError("an action index is out of range")
        if self.outcome_model is not None and self.outcome_model.shape != (
            n,
            k,
        ):
            raise ValueError("outcome_model must have shape (n, K)")
        if self.clusters is not None and self.clusters.shape != (n,):
            raise ValueError("clusters must align with actions")

    @property
    def rows(self) -> int:
        """Number of logged rows."""
        return len(self.actions)

    def taken(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """The entry of `matrix` at each row's logged action."""
        result: NDArray[np.float64] = matrix[np.arange(self.rows), self.actions]
        return result


@dataclasses.dataclass(frozen=True, slots=True)
class ValueEstimate:
    """One estimate of a policy value, or the reason there is none.

    Attributes:
        status: `issued` or `refused`.
        estimator: One of `ESTIMATORS`.
        value: The estimate, or None when refused.
        interval: Two-sided normal interval from the influence function.
        ess_ratio: Kish effective sample size over `n`.
        clipped_share: Share of rows whose weight was clipped.
        reasons: Why the value was refused or labelled.
        tier: `confirmed`, `attested` or `unverified`.
        clipped: Whether weights were clipped at the declared level.
    """

    status: str
    estimator: str
    value: float | None
    interval: tuple[float, float] | None
    ess_ratio: float
    clipped_share: float
    reasons: tuple[str, ...]
    tier: str
    clipped: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)


def weights(
    log: LoggedBandit, clip: float | None
) -> tuple[NDArray[np.float64], float]:
    """Importance weights at the logged actions and the clipped share."""
    ratio = log.taken(log.target) / log.taken(log.behaviour)
    if clip is None:
        return ratio, 0.0
    return np.minimum(ratio, clip), float(np.mean(ratio > clip))


def support_violations(log: LoggedBandit, min_propensity: float) -> int:
    """Rows where the target puts mass on an action with `mu < eps`."""
    bad = (log.target > 0) & (log.behaviour < min_propensity)
    return int(bad.any(axis=1).sum())


def terms(
    log: LoggedBandit, estimator: str, clip: float | None
) -> tuple[NDArray[np.float64], float]:
    """Per-row influence terms whose mean is the estimate."""
    w, share = weights(log, clip)
    if estimator == "ipw":
        return w * log.rewards, share
    if estimator == "snips":
        value = float(np.sum(w * log.rewards) / np.sum(w))
        return w * (log.rewards - value) / float(np.mean(w)) + value, share
    if log.outcome_model is None:
        raise ValueError("dr needs an outcome model")
    direct = (log.target * log.outcome_model).sum(axis=1)
    resid = log.rewards - log.taken(log.outcome_model)
    return direct + w * resid, share


def mean_interval(
    values: NDArray[np.float64],
    clusters: NDArray[Any] | None,
    level: float,
) -> tuple[float, tuple[float, float]]:
    """Mean and normal interval from the (cluster-)influence variance."""
    n = len(values)
    centre = float(values.mean())
    if clusters is None:
        se = float(values.std(ddof=1) / np.sqrt(n))
    else:
        labels = np.unique(clusters, return_inverse=True)[1]
        g = int(np.max(labels)) + 1
        sums = np.bincount(labels, weights=values - centre, minlength=g)
        se = float(np.sqrt(g / (g - 1) * np.sum(sums**2)) / n)
    z = float(stats.norm.ppf(1 - (1 - level) / 2))
    return centre, (centre - z * se, centre + z * se)


def estimate_value(
    log: LoggedBandit,
    estimator: str = "dr",
    *,
    clip: float | None = None,
    min_propensity: float = 0.01,
    ess_floor: float = 0.2,
    level: float = 0.95,
    eligible_strategy: bool = False,
) -> ValueEstimate:
    """Estimates `V(pi)` for a frozen target policy, or refuses.

    Args:
        log: The logged data, target and frozen outcome model.
        estimator: `ipw`, `snips` or `dr`.
        clip: Declared weight clip, or None.
        min_propensity: Declared support floor.
        ess_floor: Declared floor for the effective sample size ratio.
        level: Confidence level of the interval.
        eligible_strategy: For a non-randomized log, whether an eligible F10
            identification strategy supports the propensities.

    Returns:
        The estimate; refused when the support or ESS gate fails.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between 0 and 1")
    qualification.finite(min_propensity, "min_propensity")
    raw, ignored = weights(log, None)
    ess = sequential.effective_sample_size(raw) / log.rows
    reasons = []
    violations = support_violations(log, min_propensity)
    if violations:
        reasons.append(
            f"support: {violations} rows put target mass where the logged "
            f"probability is below {min_propensity}"
        )
    if ess < ess_floor:
        reasons.append(
            f"effective sample size ratio {ess:.3f} is below {ess_floor}"
        )
    if reasons:
        return ValueEstimate(
            REFUSED,
            estimator,
            None,
            None,
            ess,
            0.0,
            tuple(reasons),
            qualification.Tier.UNVERIFIED.value,
            clip is not None,
        )
    row_terms, share = terms(log, estimator, clip)
    centre, interval = mean_interval(row_terms, log.clusters, level)
    notes = []
    if clip is not None:
        notes.append(f"weights clipped at {clip}: {share:.1%} of rows")
    if log.randomized:
        tier = qualification.Tier.CONFIRMED.value
    elif eligible_strategy:
        tier = qualification.Tier.ATTESTED.value
        notes.append("propensities estimated; tier inherits the F10 tier")
    else:
        tier = qualification.Tier.UNVERIFIED.value
        notes.append("propensities estimated and no eligible F10 strategy")
    return ValueEstimate(
        ISSUED,
        estimator,
        centre,
        interval,
        ess,
        share,
        tuple(notes),
        tier,
        clip is not None,
    )


def difference_value(
    log: LoggedBandit,
    baseline: ArrayLike,
    estimator: str = "dr",
    *,
    clip: float | None = None,
    min_propensity: float = 0.01,
    ess_floor: float = 0.2,
    level: float = 0.95,
    eligible_strategy: bool = False,
) -> ValueEstimate:
    """`V(pi) - V(pi0)` for a baseline policy given as a `(n, K)` matrix.

    The two policies are evaluated on the same rows, so the interval comes
    from the influence function of the paired difference.
    """
    other = dataclasses.replace(log, target=np.asarray(baseline, dtype=float))
    main = estimate_value(
        log,
        estimator,
        clip=clip,
        min_propensity=min_propensity,
        ess_floor=ess_floor,
        level=level,
        eligible_strategy=eligible_strategy,
    )
    base = estimate_value(
        other,
        estimator,
        clip=clip,
        min_propensity=min_propensity,
        ess_floor=ess_floor,
        level=level,
        eligible_strategy=eligible_strategy,
    )
    if main.status == REFUSED:
        return main
    if base.status == REFUSED:
        return base
    first, ignored = terms(log, estimator, clip)
    second, ignored = terms(other, estimator, clip)
    centre, interval = mean_interval(first - second, log.clusters, level)
    return dataclasses.replace(
        main,
        value=centre,
        interval=interval,
        ess_ratio=min(main.ess_ratio, base.ess_ratio),
    )


def fit_outcome_model(
    features: ArrayLike,
    actions: ArrayLike,
    rewards: ArrayLike,
    n_actions: int,
    ridge: float = 1.0,
) -> Any:
    """Per-action ridge regression of the reward, fitted on development data.

    Returns a function mapping a feature matrix to `q(x, a)` of shape
    `(n, K)`. Fit it on development rows only and freeze it before the
    evaluation log is read.
    """
    x = np.asarray(features, dtype=float)
    a = np.asarray(actions)
    r = np.asarray(rewards, dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coefficients = []
    for k in range(n_actions):
        rows = a == k
        if not rows.any():
            raise ValueError(f"action {k} never appears in the fitting log")
        d = design[rows]
        coefficients.append(np.linalg.solve(d.T @ d + penalty, d.T @ r[rows]))
    fitted = np.stack(coefficients, axis=1)

    def predict(new: ArrayLike) -> NDArray[np.float64]:
        z = np.asarray(new, dtype=float)
        result: NDArray[np.float64] = (
            np.column_stack([np.ones(len(z)), z]) @ fitted
        )
        return result

    return predict


@dataclasses.dataclass(frozen=True, slots=True)
class ThresholdPolicy:
    """Treat iff the estimated effect exceeds `cut` (a one-parameter class).

    Attributes:
        cut: The cut on `tau_hat(x)`.
        objective: Penalised DR value on the fitting log.
        penalty: Per-treated-row penalty used in the fit.
    """

    cut: float
    objective: float
    penalty: float

    def probabilities(self, tau: ArrayLike) -> NDArray[np.float64]:
        """`pi(a | x)` for actions (0, 1) as an `(n, 2)` matrix."""
        treat = (np.asarray(tau, dtype=float) > self.cut).astype(float)
        return np.column_stack([1.0 - treat, treat])


def fit_threshold_policy(
    log: LoggedBandit,
    tau: ArrayLike,
    *,
    penalty: float = 0.0,
    clip: float | None = None,
) -> ThresholdPolicy:
    """Chooses the cut on development data by penalised DR value.

    The class has one parameter, so the complexity penalty is the declared
    per-treated-row cost `penalty` (treatment cost plus any charter
    penalty). The held-out value, not this objective, is the evidence.
    """
    effect = np.asarray(tau, dtype=float)
    if log.outcome_model is None or log.behaviour.shape[1] != 2:
        raise ValueError("policy fitting needs a binary action and a q model")
    ratio = 1.0 / log.taken(log.behaviour)
    if clip is not None:
        ratio = np.minimum(ratio, clip)
    resid = log.rewards - log.taken(log.outcome_model)
    gain = (
        log.outcome_model[:, 1] - log.outcome_model[:, 0]
    ) + ratio * resid * (2.0 * (log.actions == 1) - 1.0)
    best: ThresholdPolicy | None = None
    for cut in np.unique(np.quantile(effect, np.linspace(0, 1, 101))):
        treat = effect > cut
        objective = float(np.mean(np.where(treat, gain - penalty, 0.0)))
        if best is None or objective > best.objective:
            best = ThresholdPolicy(float(cut), objective, penalty)
    assert best is not None
    none = ThresholdPolicy(float(np.max(effect)), 0.0, penalty)
    return best if best.objective > none.objective else none
