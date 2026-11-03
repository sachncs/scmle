"""Bivariate association tests the ledger can run, with a code-path hash.

Each statistic takes a feature and a target and returns a p-value, a test
statistic and an effect size. The same function runs on `X` during
exploration and on `C` during confirmation, so a hypothesis is a (column,
statistic) pair rather than a result.

Validity. `spearman` uses the t approximation with ``n - 2`` degrees of
freedom (the one `scipy.stats.spearmanr` uses) and `kruskal` the chi-square
approximation of `scipy.stats.kruskal`. Both assume independent rows. On
grouped or serially dependent data their p-values are not valid, which is
exactly what the negative controls in `controls` measure; a user with such
data should read the control, not the p-values. Pairs with a missing value are
dropped, and an untestable pair (fewer than three pairs, or a constant
variable) gets ``p = 1`` so it cannot be rejected.
"""

from collections.abc import Callable
import dataclasses
import hashlib
import inspect

import numpy as np
from scipy import stats

TWO_SIDED = "two-sided"
GREATER = "greater"
LESS = "less"
DIRECTIONS = (TWO_SIDED, GREATER, LESS)
MIN_PAIRS = 3


@dataclasses.dataclass(frozen=True, slots=True)
class Outcome:
    """Result of one statistic.

    Attributes:
        p: P-value in [0, 1] (1 when the pair is untestable).
        statistic: The test statistic (Spearman's rho, or Kruskal's H).
        effect: Effect size (rho, or H / (n - 1)).
        n: Rows used.
        note: Why the pair was untestable, or an empty string.
    """

    p: float
    statistic: float
    effect: float
    n: int
    note: str = ""


def untestable(n: int, note: str) -> Outcome:
    """The outcome of a pair that cannot be tested."""
    return Outcome(p=1.0, statistic=0.0, effect=0.0, n=n, note=note)


def complete_pairs(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Both arrays restricted to rows where neither value is missing."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.shape != ys.shape or xs.ndim != 1:
        raise ValueError("x and y must be one-dimensional and equally long")
    keep = np.isfinite(xs) & np.isfinite(ys)
    return xs[keep], ys[keep]


def tail_probability(t_value: float, freedom: int, direction: str) -> float:
    """P-value of a t statistic for the requested alternative."""
    if direction == TWO_SIDED:
        return float(2.0 * stats.t.sf(abs(t_value), freedom))
    if direction == GREATER:
        return float(stats.t.sf(t_value, freedom))
    return float(stats.t.cdf(t_value, freedom))


def spearman(
    x: np.ndarray, y: np.ndarray, direction: str = TWO_SIDED
) -> Outcome:
    """Spearman rank correlation with the t approximation.

    Args:
        x: Feature values.
        y: Target values (class codes for a binary target).
        direction: `two-sided`, `greater` (rho > 0) or `less` (rho < 0).

    Returns:
        The outcome; ``statistic`` and ``effect`` are both rho.

    Raises:
        ValueError: On an unknown direction or arrays of different shape.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    xs, ys = complete_pairs(x, y)
    n = len(xs)
    if n < MIN_PAIRS:
        return untestable(n, "fewer than three complete pairs")
    if np.ptp(xs) == 0.0 or np.ptp(ys) == 0.0:
        return untestable(n, "a variable is constant")
    rx = stats.rankdata(xs)
    ry = stats.rankdata(ys)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rho = max(-1.0, min(1.0, rho))
    if abs(rho) >= 1.0:
        limit = {
            TWO_SIDED: 0.0,
            GREATER: 0.0 if rho > 0 else 1.0,
            LESS: 0.0 if rho < 0 else 1.0,
        }
        return Outcome(limit[direction], rho, rho, n)
    t_value = rho * float(np.sqrt((n - 2) / (1.0 - rho * rho)))
    p = min(1.0, max(0.0, tail_probability(t_value, n - 2, direction)))
    return Outcome(p, rho, rho, n)


def kruskal(
    x: np.ndarray, labels: np.ndarray, direction: str = TWO_SIDED
) -> Outcome:
    """Kruskal-Wallis H test of a feature across the classes of a target.

    Args:
        x: Feature values.
        labels: Class codes of the target.
        direction: Only `two-sided` exists for a test across classes.

    Returns:
        The outcome; ``statistic`` is H and ``effect`` is ``H / (n - 1)``.

    Raises:
        ValueError: If `direction` is not `two-sided`.
    """
    if direction != TWO_SIDED:
        raise ValueError("kruskal supports only the two-sided alternative")
    xs, ls = complete_pairs(x, labels)
    n = len(xs)
    groups = [xs[ls == code] for code in np.unique(ls)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2 or n < MIN_PAIRS:
        return untestable(n, "fewer than two classes or three rows")
    if np.ptp(xs) == 0.0:
        return untestable(n, "a variable is constant")
    result = stats.kruskal(*groups)
    h = float(result.statistic)
    return Outcome(float(result.pvalue), h, h / (n - 1), n)


STATISTICS: dict[str, Callable[..., Outcome]] = {
    "spearman": spearman,
    "kruskal": kruskal,
}


def code_hash(statistic: str) -> str:
    """SHA-256 of the source of a registered statistic function.

    It identifies the code path that produced a p-value. It does not cover
    scipy or numpy, whose versions the run records separately.

    Raises:
        ValueError: On an unknown statistic.
    """
    function = _lookup(statistic)
    source = inspect.getsource(function).replace("\r\n", "\n")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _lookup(statistic: str) -> Callable[..., Outcome]:
    """The registered function, or a clear error."""
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {sorted(STATISTICS)}")
    return STATISTICS[statistic]


def run_statistic(
    statistic: str, x: np.ndarray, y: np.ndarray, direction: str = TWO_SIDED
) -> Outcome:
    """Runs a registered statistic.

    Args:
        statistic: `spearman` or `kruskal`.
        x: Feature values.
        y: Target values or class codes.
        direction: The alternative.

    Returns:
        The outcome.
    """
    return _lookup(statistic)(x, y, direction)
