"""Anytime-valid evidence for a bounded mean (feature F11, part D and E).

For a stream `z_1, z_2, ...` with values in `[low, high]` and conditional mean
`m`, rescale to `u = (z - low) / (high - low)` in `[0, 1]` and let `m0` be the
rescaled null. To reject `H0: m <= m0` bet on the mean being larger: for a
`lam` in `(0, 1 / m0)` the wealth

    K_t(lam) = prod_{i <= t} (1 + lam (u_i - m0))

is a nonnegative martingale under `H0` with `K_0 = 1` (each factor is
positive and has conditional mean at most one). A convex combination of such
martingales is again a nonnegative supermartingale, so the equal-weight
mixture over a fixed grid of `lam` values,

    E_t = mean_k K_t(lam_k),

is an e-process, and by Ville's inequality `P(sup_t E_t >= 1 / alpha) <=
alpha` under `H0` for every stopping rule, including peeking after every
unit. The grid is fixed in advance and does not depend on the data, so the
guarantee needs no appeal to a result about plug-in betting. The idea of
testing a bounded mean by betting is the one in Waudby-Smith and Ramdas
(2024, arXiv:2010.09686); the mixture over a fixed grid below is the
construction used here and its error rate is measured by the certificates
(`benchmarks/qualification/f11_certify.py`), not taken from that paper. The
price of fixing the grid is that the e-process is slower than the best single
`lam` for the true effect; the price against a fixed-horizon test is
measured (B3).

To reject `H0: m >= m0` the stream is reflected (`u -> 1 - u`).

For a randomized experiment with challenger probability `q` and outcome `y`
in `[0, 1]`, the pseudo-outcome

    z_i = a_i y_i / q - (1 - a_i) y_i / (1 - q)

has mean equal to the mean difference and lies in `[-1/(1 - q), 1/q]`. It is
bounded because the assignment probability is known, and its mean is the
effect because assignment is randomized.
"""

import dataclasses
import math

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy import special

from scmle.contracts import qualification
from scmle.data.ledger import online_fdr

GRID_SIZE = 14
GRID_TOP = 0.9
MAX_LOG = 700.0
DIRECTIONS = ("greater", "less")


@dataclasses.dataclass(frozen=True, slots=True)
class Stream:
    """A bounded stream and the null it is tested against.

    Attributes:
        low: Smallest value the stream can take.
        high: Largest value it can take.
        null: The null mean, strictly inside `(low, high)`.
        direction: `greater` rejects `mean <= null`; `less` rejects
            `mean >= null`.
    """

    low: float
    high: float
    null: float
    direction: str = "greater"

    def __post_init__(self) -> None:
        """Validates the stream."""
        low = qualification.finite(self.low, "low")
        high = qualification.finite(self.high, "high")
        null = qualification.finite(self.null, "null")
        if not low < high:
            raise ValueError("low must be below high")
        if not low < null < high:
            raise ValueError("null must lie strictly inside (low, high)")
        if self.direction not in DIRECTIONS:
            raise ValueError("direction must be 'greater' or 'less'")

    def rescaled_null(self) -> float:
        """The null in the orientation where larger values are evidence."""
        position = (self.null - self.low) / (self.high - self.low)
        return position if self.direction == "greater" else 1.0 - position

    def rescale(self, values: ArrayLike) -> NDArray[np.float64]:
        """Values in the orientation where larger values are evidence."""
        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            raise ValueError("values must be one-dimensional")
        if not np.isfinite(array).all():
            raise ValueError("values must be finite")
        tolerance = 1e-9 * (self.high - self.low)
        if (array < self.low - tolerance).any() or (
            array > self.high + tolerance
        ).any():
            raise ValueError("a value lies outside [low, high]")
        unit = np.clip((array - self.low) / (self.high - self.low), 0.0, 1.0)
        return unit if self.direction == "greater" else 1.0 - unit


def bet_grid(null_unit: float, size: int = GRID_SIZE) -> NDArray[np.float64]:
    """Fixed bet sizes `GRID_TOP / null_unit * 2 ** -k` for `k < size`."""
    if not 0.0 < null_unit < 1.0:
        raise ValueError("null_unit must be strictly between 0 and 1")
    qualification.count(size, "size", minimum=1)
    grid: NDArray[np.float64] = (GRID_TOP / null_unit) * 0.5 ** np.arange(
        size, dtype=float
    )
    return grid


def log_wealth(
    unit: NDArray[np.float64], null_unit: float, size: int = GRID_SIZE
) -> NDArray[np.float64]:
    """Log of the mixture e-process after every observation.

    Args:
        unit: Rescaled observations, shape `(T,)` or `(R, T)`; larger values
            are evidence against the null.
        null_unit: The rescaled null in `(0, 1)`.
        size: Number of bet sizes in the mixture.

    Returns:
        An array of the same shape as `unit`.
    """
    grid = bet_grid(null_unit, size)
    centred = unit - null_unit
    total: NDArray[np.float64] | None = None
    for lam in grid:
        path = np.cumsum(np.log1p(lam * centred), axis=-1)
        total = path if total is None else np.logaddexp(total, path)
    assert total is not None
    result: NDArray[np.float64] = total - math.log(len(grid))
    return result


def log_wealth_paths(
    values: ArrayLike, stream: Stream, size: int = GRID_SIZE
) -> NDArray[np.float64]:
    """Log e-process paths for raw values, shape `(T,)` or `(R, T)`."""
    array = np.asarray(values, dtype=float)
    flat = stream.rescale(array.reshape(-1)).reshape(array.shape)
    return log_wealth(flat, stream.rescaled_null(), size)


@dataclasses.dataclass(slots=True)
class EProcess:
    """A running mixture e-process.

    Attributes:
        stream: The bounded stream and its null.
        size: Number of bet sizes in the mixture.
        count: Observations so far.
        log_e: Log of the current e-value.
        max_log_e: Largest log e-value at any observation so far.
        totals: Log wealth of each bet size.
        path: Log e-value after each observation of the last batch.
    """

    stream: Stream
    size: int = GRID_SIZE
    count: int = 0
    log_e: float = 0.0
    max_log_e: float = 0.0
    totals: NDArray[np.float64] | None = None
    path: NDArray[np.float64] = dataclasses.field(
        default_factory=lambda: np.zeros(0)
    )

    def __post_init__(self) -> None:
        """Starts every bet at wealth one."""
        grid = bet_grid(self.stream.rescaled_null(), self.size)
        if self.totals is None:
            self.totals = np.zeros(len(grid))

    def update(self, values: ArrayLike) -> float:
        """Adds observations in order and returns the current log e-value.

        The running maximum is taken at every single observation, so peeking
        after each unit is covered even when values arrive in batches.
        """
        unit = self.stream.rescale(values)
        self.path = np.zeros(0)
        if unit.size == 0:
            return self.log_e
        null_unit = self.stream.rescaled_null()
        grid = bet_grid(null_unit, self.size)
        assert self.totals is not None
        steps = np.log1p(np.outer(unit - null_unit, grid))
        paths = self.totals + np.cumsum(steps, axis=0)
        mixed = special.logsumexp(paths, axis=1) - math.log(len(grid))
        self.totals = paths[-1].copy()
        self.path = mixed
        self.count += int(unit.size)
        self.log_e = float(mixed[-1])
        self.max_log_e = max(self.max_log_e, float(mixed.max()))
        return self.log_e

    def crossed(self, level: float) -> bool:
        """Whether the e-value is at least `1 / level` now."""
        return self.log_e >= -math.log(positive_level(level))

    def ever_crossed(self, level: float) -> bool:
        """Whether it was at least `1 / level` at any observation."""
        return self.max_log_e >= -math.log(positive_level(level))

    def evidence(self) -> online_fdr.Evidence:
        """The current e-value as ledger evidence.

        An e-process evaluated at a stopping time is an e-value, so the
        value may enter an online FDR rule once the experiment has stopped.
        """
        return online_fdr.Evidence(e=math.exp(min(self.log_e, MAX_LOG)))


def positive_level(level: float) -> float:
    """A level strictly between 0 and 1."""
    value = qualification.finite(level, "level")
    if not 0.0 < value < 1.0:
        raise ValueError("level must be strictly between 0 and 1")
    return value


def pseudo_outcomes(
    arm: ArrayLike, outcome: ArrayLike, assignment_prob: float
) -> NDArray[np.float64]:
    """`z = a y / q - (1 - a) y / (1 - q)`, mean difference per unit.

    Args:
        arm: 1 for the challenger, 0 for the champion.
        outcome: The metric rescaled to `[0, 1]`.
        assignment_prob: The recorded probability `q` of the challenger arm.

    Returns:
        Pseudo-outcomes in `[-1 / (1 - q), 1 / q]`.
    """
    q = qualification.finite(assignment_prob, "assignment_prob")
    if not 0.0 < q < 1.0:
        raise ValueError("assignment_prob must be strictly between 0 and 1")
    a = np.asarray(arm, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if a.shape != y.shape or a.ndim != 1:
        raise ValueError("arm and outcome must be 1-d arrays of equal length")
    if not np.isin(a, (0.0, 1.0)).all():
        raise ValueError("arm must hold only 0 and 1")
    if (y < 0).any() or (y > 1).any() or not np.isfinite(y).all():
        raise ValueError("outcome must lie in [0, 1]")
    return a * y / q - (1.0 - a) * y / (1.0 - q)


def difference_stream(
    assignment_prob: float, null: float, direction: str = "greater"
) -> Stream:
    """The stream of pseudo-outcomes for an effect on a `[0, 1]` metric."""
    q = assignment_prob
    return Stream(-1.0 / (1.0 - q), 1.0 / q, null, direction)


def bound_stream(
    bound: float, null: float = 0.0, direction: str = "greater"
) -> Stream:
    """The stream of paired differences bounded in `[-bound, bound]`."""
    b = qualification.finite(bound, "bound")
    if b <= 0:
        raise ValueError("bound must be positive")
    return Stream(-b, b, null, direction)


def confidence_bound(
    values: ArrayLike,
    stream: Stream,
    alpha: float,
    size: int = GRID_SIZE,
    steps: int = 40,
) -> float:
    """Time-uniform one-sided bound on the mean, valid at every sample size.

    For `direction='greater'` this is a lower bound `L_t` with
    `P(exists t: mean < L_t) <= alpha`; for `'less'` an upper bound. It is
    the boundary null at which the e-process at time `t` equals `1 / alpha`,
    found by bisection (the e-value is monotone in the null).
    """
    level = positive_level(alpha)
    array = np.asarray(values, dtype=float)
    target = -math.log(level)
    inside = (stream.low, stream.high)

    def exceeds(null: float) -> bool:
        moved = dataclasses.replace(stream, null=null)
        if array.size == 0:
            return False
        return float(log_wealth_paths(array, moved, size)[-1]) >= target

    span = inside[1] - inside[0]
    eps = 1e-9 * span
    if stream.direction == "greater":
        lo, hi = inside[0] + eps, inside[1] - eps
        if not exceeds(lo):
            return stream.low
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            if exceeds(mid):
                lo = mid
            else:
                hi = mid
        return lo
    lo, hi = inside[0] + eps, inside[1] - eps
    if not exceeds(hi):
        return stream.high
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if exceeds(mid):
            hi = mid
        else:
            lo = mid
    return hi
