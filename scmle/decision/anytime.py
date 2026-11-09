"""Anytime-valid evidence for a bounded mean.

For a stream `z_1, z_2, ...` with values in `[low, high]` and conditional mean
`m`, rescale to `u = (z - low) / (high - low)` in `[0, 1]` and let `m0` be the
rescaled null. To reject `H_0: m <= m0` bet on the mean being larger: for a
`lam` in `(0, 1 / m0)` the wealth

    K_t(lam) = prod_{i <= t} (1 + lam (u_i - m0))

is a nonnegative martingale under `H_0` with `K_0 = 1` (each factor is
positive and has conditional mean at most one). A convex combination of such
martingales is again a nonnegative supermartingale, so the equal-weight
mixture over a fixed grid of `lam` values,

    E_t = mean_k K_t(lam_k),

is an e-process, and by Ville's inequality `P(sup_t E_t >= 1 / alpha) <=
alpha` under `H_0` for every stopping rule, including peeking after every
unit. The grid is fixed in advance and does not depend on the data, so the
guarantee needs no appeal to a result about plug-in betting. The idea of
testing a bounded mean by betting is the one in Waudby-Smith and Ramdas
(2024, arXiv:2010.09686); the mixture over a fixed grid below is the
construction used here and its error rate is measured by the certificates,
not taken from that paper. The
price of fixing the grid is that the e-process is slower than the best single
`lam` for the true effect; the price against a fixed-horizon test is
measured.

To reject `H_0: m >= m0` the stream is reflected (`u -> 1 - u`).

For a randomized experiment with challenger probability `q` and outcome `y`
in `[0, 1]`, the pseudo-outcome

    z_i = a_i y_i / q - (1 - a_i) y_i / (1 - q)

has mean equal to the mean difference and lies in `[-1/(1 - q), 1/q]`. It is
bounded because the assignment probability is known, and its mean is the
effect because assignment is randomized.

The raw pseudo-outcome carries the level of `y` in its variance, which makes
the e-process slow for a metric whose mean is far from zero. Subtracting any
centre `c_i` that is fixed before unit `i`'s outcome (for example the running
mean of earlier outcomes) leaves the mean unchanged, because
`E[a c / q - (1 - a) c / (1 - q)] = c - c = 0` when `a` is Bernoulli(`q`)
independent of the past, and with `y` and `c` in `[0, 1]` the centred value
lies in `[-M, M]`, `M = max(1/q, 1/(1 - q))`. `running_center` and
`centered_stream` implement it; the uncentred pair stays for comparison.
"""

import dataclasses
import math

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy import special

from scmle.contracts import errors
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
        """Validates the stream.

        Raises:
            ValueError: If `low` is not below `high`, `null` does not lie
                strictly inside `(low, high)`, or `direction` is not
                `greater` or `less`. A bound or null that is not a finite
                number is rejected with a `ValueError` by
                `qualification.finite`.
        """
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
        """Returns the null in the orientation where larger values are evidence.

        Returns:
            The null as a position in `(0, 1)` after rescaling to the unit
            interval, reflected when `direction` is `less`.
        """
        position = (self.null - self.low) / (self.high - self.low)
        return position if self.direction == "greater" else 1.0 - position

    def rescale(self, values: ArrayLike) -> NDArray[np.float64]:
        """Rescales values so that larger values are evidence.

        Args:
            values: Observations of the stream, one-dimensional.

        Returns:
            The values rescaled to `[0, 1]` and reflected when `direction`
            is `less`. Values outside `[low, high]` by no more than 1e-9 of
            the range `high - low` are clipped.

        Raises:
            ValueError: If `values` is not one-dimensional, is not finite,
                or has a value outside `[low, high]` beyond the tolerance.
        """
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
    """Builds the fixed bet sizes `GRID_TOP / null_unit * 2 ** -k`, `k < size`.

    Args:
        null_unit: The rescaled null, strictly between 0 and 1.
        size: Number of bet sizes; at least 1.

    Returns:
        The bet sizes, largest first.

    Raises:
        ValueError: If `null_unit` is not strictly between 0 and 1. A
            `size` below 1 is rejected with a `ValueError` by
            `qualification.count`.
    """
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
    """Computes the log of the mixture e-process after every observation.

    Args:
        unit: Rescaled observations, shape `(T,)` or `(R, T)`; larger values
            are evidence against the null.
        null_unit: The rescaled null in `(0, 1)`.
        size: Number of bet sizes in the mixture.

    Returns:
        An array of the same shape as `unit`. A `null_unit` outside `(0, 1)`
        or a `size` below 1 is rejected by `bet_grid` with a `ValueError`.

    Raises:
        ScmleError: If the bet grid is empty, which `bet_grid` already
            prevents.
    """
    grid = bet_grid(null_unit, size)
    centred = unit - null_unit
    total: NDArray[np.float64] | None = None
    for lam in grid:
        path = np.cumsum(np.log1p(lam * centred), axis=-1)
        total = path if total is None else np.logaddexp(total, path)
    if total is None:
        raise errors.ScmleError("the bet grid is empty")
    result: NDArray[np.float64] = total - math.log(len(grid))
    return result


def log_wealth_paths(
    values: ArrayLike, stream: Stream, size: int = GRID_SIZE
) -> NDArray[np.float64]:
    """Computes log e-process paths for raw values, `(T,)` or `(R, T)`.

    Args:
        values: Raw observations of the stream; `rescale` is applied to the
            flattened values, so they must lie in the stream's range.
        stream: The bounded stream and its null.
        size: Number of bet sizes in the mixture.

    Returns:
        The log e-value after each observation, same shape as `values`. Values
        that are not finite or lie outside the stream's range are refused
        with a `ValueError` by `Stream.rescale`.
    """
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
        max_log_e: Largest log e-value so far, including the starting value
            0 (an e-value of one) before any observation.
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
        """Starts every bet at wealth one.

        When no totals were supplied, the log wealth of each bet size is set
        to zero. A `size` below 1 is refused with a `ValueError` by
        `bet_grid`.
        """
        grid = bet_grid(self.stream.rescaled_null(), self.size)
        if self.totals is None:
            self.totals = np.zeros(len(grid))

    def update(self, values: ArrayLike) -> float:
        """Adds observations in order and returns the current log e-value.

        The running maximum is taken at every single observation, so peeking
        after each unit is covered even when values arrive in batches.

        Args:
            values: New observations of the stream, one-dimensional and in
                the stream's range.

        Returns:
            The log e-value after the last observation, or the unchanged
            current value if `values` is empty.

        Raises:
            ScmleError: If the bet totals are not initialised. Values that
                are not one-dimensional, not finite or outside the stream's
                range are refused with a `ValueError` by `Stream.rescale`.
        """
        unit = self.stream.rescale(values)
        self.path = np.zeros(0)
        if unit.size == 0:
            return self.log_e
        null_unit = self.stream.rescaled_null()
        grid = bet_grid(null_unit, self.size)
        if self.totals is None:
            raise errors.ScmleError("bet totals are not initialised")
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
        """Checks whether the current e-value is at least `1 / level`.

        Args:
            level: Level of the test, strictly between 0 and 1.

        Returns:
            True if the current e-value has reached `1 / level`.
        """
        return self.log_e >= -math.log(positive_level(level))

    def ever_crossed(self, level: float) -> bool:
        """Checks whether the e-value was at least `1 / level` at any point.

        Args:
            level: Level of the test, strictly between 0 and 1.

        Returns:
            True if the e-value reached `1 / level` at some observation or at
        the start (the starting value of one never does for a level below 1).
        """
        return self.max_log_e >= -math.log(positive_level(level))

    def evidence(self) -> online_fdr.Evidence:
        """Returns the current e-value as ledger evidence.

        An e-process evaluated at a stopping time is an e-value, so the
        value may enter an online FDR rule once the experiment has stopped.

        Returns:
            The e-value (capped at `exp(MAX_LOG)`) as ledger evidence.
        """
        return online_fdr.Evidence(e=math.exp(min(self.log_e, MAX_LOG)))


def positive_level(level: float) -> float:
    """Validates a level that must lie strictly between 0 and 1.

    Args:
        level: The candidate level.

    Returns:
        `level` as a float.

    Raises:
        ValueError: If `level` is not strictly between 0 and 1. A `level`
            that is not a finite number is rejected with a `ValueError` by
            `qualification.finite`.
    """
    value = qualification.finite(level, "level")
    if not 0.0 < value < 1.0:
        raise ValueError("level must be strictly between 0 and 1")
    return value


def pseudo_outcomes(
    arm: ArrayLike,
    outcome: ArrayLike,
    assignment_prob: float,
    center: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Computes `z = a (y - c) / q - (1 - a) (y - c) / (1 - q)`.

    The mean of `z` is the mean difference between the arms.

    Args:
        arm: 1 for the challenger, 0 for the champion.
        outcome: The metric rescaled to `[0, 1]`.
        assignment_prob: The recorded probability `q` of the challenger arm.
        center: Per-unit centres in `[0, 1]` fixed before each outcome, or
            None for no centring.

    Returns:
        Pseudo-outcomes with the shape of `outcome` and the mean difference
        as their mean; without a centre they lie in `[-1 / (1 - q), 1 / q]`.

    Raises:
        ValueError: If `assignment_prob` is not strictly between 0 and 1,
            `arm` and `outcome` differ in shape or are not 1-d or 2-d, `arm`
            holds a value other than 0 and 1, `outcome` is not finite or
            lies outside `[0, 1]`, or `center` differs in shape from
            `outcome` or lies outside `[0, 1]` (a non-finite `center` is not
            checked).
    """
    q = qualification.finite(assignment_prob, "assignment_prob")
    if not 0.0 < q < 1.0:
        raise ValueError("assignment_prob must be strictly between 0 and 1")
    a = np.asarray(arm, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if a.shape != y.shape or a.ndim not in (1, 2):
        raise ValueError(
            "arm and outcome must be arrays of equal shape (1-d or 2-d)"
        )
    if not np.isin(a, (0.0, 1.0)).all():
        raise ValueError("arm must hold only 0 and 1")
    if (y < 0).any() or (y > 1).any() or not np.isfinite(y).all():
        raise ValueError("outcome must lie in [0, 1]")
    c = np.zeros_like(y) if center is None else np.asarray(center, dtype=float)
    if c.shape != y.shape or (c < 0).any() or (c > 1).any():
        raise ValueError("center must lie in [0, 1] and match the outcome")
    shifted = y - c
    result: NDArray[np.float64] = a * shifted / q - (1.0 - a) * shifted / (
        1.0 - q
    )
    return result


def running_center(
    outcome: ArrayLike,
    prior: float = 0.5,
    strength: float = 2.0,
    carry: tuple[float, int] = (0.0, 0),
) -> NDArray[np.float64]:
    """Computes centres from earlier outcomes only: a shrunken running mean.

    `c_i = (prior * strength + sum_{j < i} y_j) / (strength + (i - 1))`, along
    the last axis; `carry` is the `(sum, count)` of outcomes seen in earlier
    batches. Each centre is fixed before its own unit is observed, which is
    all the validity argument needs.

    Args:
        outcome: The metric rescaled to `[0, 1]`.
        prior: Value the centre shrinks toward.
        strength: Weight of `prior`, in pseudo-observations. It is not
            validated; zero with no carried outcomes gives NaN for the
            first centre.
        carry: The `(sum, count)` of outcomes seen in earlier batches.

    Returns:
        The centre for each outcome, same shape as `outcome`.
    """
    y = np.asarray(outcome, dtype=float)
    total, seen = carry
    before = np.cumsum(y, axis=-1) - y + total
    count = np.arange(y.shape[-1], dtype=float) + seen
    result: NDArray[np.float64] = (prior * strength + before) / (
        strength + count
    )
    return result


def centered_stream(
    assignment_prob: float, null: float, direction: str = "greater"
) -> Stream:
    """Builds the stream of centred pseudo-outcomes, bounded in `[-M, M]`.

    Args:
        assignment_prob: The recorded probability `q` of the challenger,
            expected strictly between 0 and 1 (not validated here; a Python
            float of exactly 0 or 1 raises `ZeroDivisionError`).
        null: The null mean of the centred difference.
        direction: `greater` or `less`; any other value is rejected by
            `Stream` with a `ValueError`, as is a `null` outside `(-M, M)`.

    Returns:
        The stream on `[-M, M]` with `M = max(1/q, 1/(1 - q))`.
    """
    q = assignment_prob
    bound = max(1.0 / q, 1.0 / (1.0 - q))
    return Stream(-bound, bound, null, direction)


def centered_paths(
    arm: ArrayLike,
    outcome: ArrayLike,
    assignment_prob: float,
    null: float,
    direction: str = "greater",
    sign: float = 1.0,
) -> NDArray[np.float64]:
    """Computes log e-process paths of the centred pseudo-outcomes.

    The result has shape `(T,)` or `(R, T)`, like `outcome`; the centre of
    each unit is `running_center` of the outcomes before it.

    Args:
        arm: 1 for the challenger, 0 for the champion.
        outcome: The metric rescaled to `[0, 1]`.
        assignment_prob: The recorded probability of the challenger arm.
        null: The null mean of the (signed) difference in rescaled units.
        direction: `greater` or `less`.
        sign: `-1` to test the negated difference (a lower-is-worse metric).

    Returns:
        The log e-value after each observation, with the shape of
        `outcome`.
    """
    y = np.asarray(outcome, dtype=float)
    z = sign * pseudo_outcomes(arm, y, assignment_prob, running_center(y))
    return log_wealth_paths(
        z, centered_stream(assignment_prob, null, direction)
    )


def difference_stream(
    assignment_prob: float, null: float, direction: str = "greater"
) -> Stream:
    """Builds the stream of pseudo-outcomes for an effect on a `[0, 1]` metric.

    Args:
        assignment_prob: The recorded probability `q` of the challenger,
            expected strictly between 0 and 1 (not validated here).
        null: The null mean difference.
        direction: `greater` or `less`.

    Returns:
        The stream on `[-1/(1 - q), 1/q]`.
    """
    q = assignment_prob
    return Stream(-1.0 / (1.0 - q), 1.0 / q, null, direction)


def bound_stream(
    bound: float, null: float = 0.0, direction: str = "greater"
) -> Stream:
    """Builds the stream of paired differences bounded in `[-bound, bound]`.

    Args:
        bound: Largest absolute difference; positive.
        null: The null mean difference.
        direction: `greater` or `less`.

    Returns:
        The stream on `[-bound, bound]`.

    Raises:
        ValueError: If `bound` is not positive. A `bound` that is not a
            finite number is rejected with a `ValueError` by
            `qualification.finite`.
    """
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
    """Computes a time-uniform one-sided bound on the mean.

    For `direction='greater'` this is a lower bound `L_t` with
    `P(exists t: mean < L_t) <= alpha`; for `'less'` an upper bound. It is
    the boundary null at which the e-process at time `t` equals `1 / alpha`,
    found by bisection (the e-value is monotone in the null).

    Args:
        values: Observations of the stream in its range; may be empty.
        stream: The bounded stream; its null is ignored.
        alpha: Error level, strictly between 0 and 1.
        size: Number of bet sizes in the mixture.
        steps: Bisection steps.

    Returns:
        The bound. When the e-process never reaches `1 / alpha`, even for a
        null at the edge of the range (and for empty `values`), the edge
        itself: `stream.low` for `greater`, `stream.high` for `less`. The
        edge is tested at 1e-9 of the range inside it. Otherwise the bisection
        returns its conservative end (the lower end for `greater`, the upper
        end for `less`). An `alpha` outside `(0, 1)` is refused by
        `positive_level` and values that are not finite or lie outside the
        stream's range by `Stream.rescale`, both with a `ValueError`.
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
        for ignored in range(steps):
            mid = 0.5 * (lo + hi)
            if exceeds(mid):
                lo = mid
            else:
                hi = mid
        return lo
    lo, hi = inside[0] + eps, inside[1] - eps
    if not exceeds(hi):
        return stream.high
    for ignored in range(steps):
        mid = 0.5 * (lo + hi)
        if exceeds(mid):
            hi = mid
        else:
            lo = mid
    return hi
