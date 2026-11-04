"""Net benefit of a frozen rule with simultaneous intervals (feature F11, B).

For a rule that acts on row `i` with `a_i`, an outcome `y_i` and a threshold
probability `t` with weight `w = t / (1 - t)`,

    NB(t) = mean(a y) - w mean(a (1 - y)).

The baselines are treat none (`NB = 0`), treat all
(`NB = P - (1 - P) w`) and, when a column of past decisions `b_i` is given,
the incumbent. Each difference `D_b(t) = NB_rule(t) - NB_b(t)` is a ratio of
row means, so a cluster bootstrap resamples whole clusters (or single rows
when there are none). The band is simultaneous over the baselines and a grid
of `t` in the chartered range: the critical value `c` is the `1 - alpha`
quantile of `max_{b, t} (D*_b(t) - D_b(t)) / sd_b(t)` over the bootstrap
replicates, and the lower bound is `D_b(t) - c sd_b(t)`. The claim `useful`
needs the lower bound above zero everywhere on the grid. The coverage of the
band is a statement about the frozen rule on the population of the table; its
finite-sample coverage is measured by the certificates and is asymptotic in
the number of clusters.
"""

import dataclasses
import hashlib
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.framing import costs

GRID_POINTS = 21
MIN_SD = 1e-12
USEFUL = "useful"
NOT_USEFUL = "not_useful_in_range"
BASELINE_NAMES = ("treat_none", "treat_all", "incumbent")


class SecondEvaluationError(RuntimeError):
    """A frozen rule was evaluated on the final table a second time."""


def rule_hash(label: str, threshold: float) -> str:
    """SHA-256 of a frozen rule's label and threshold.

    The rule is identified by what fixed it on development data, never by
    its actions on the final table.
    """
    payload = f"{label}|{qualification.finite(threshold, 'threshold')!r}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def columns(
    y: NDArray[np.float64],
    acts: NDArray[np.float64],
    incumbent: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Row statistics `[a y, a (1-y), y, 1-y, b y, b (1-y)]`."""
    base = incumbent if incumbent is not None else np.zeros_like(y)
    return np.column_stack(
        [acts * y, acts * (1 - y), y, 1 - y, base * y, base * (1 - y)]
    )


def diffs(
    means: NDArray[np.float64],
    weights: NDArray[np.float64],
    with_incumbent: bool,
) -> NDArray[np.float64]:
    """Differences against each baseline; shape `(R, baselines, grid)`.

    Args:
        means: Row-mean statistics, shape `(R, 6)`.
        weights: The odds `t / (1 - t)` on the grid.
        with_incumbent: Whether the incumbent is a baseline.
    """
    rule = means[:, [0]] - means[:, [1]] * weights
    everyone = means[:, [2]] - means[:, [3]] * weights
    rows = [rule, rule - everyone]
    if with_incumbent:
        rows.append(rule - (means[:, [4]] - means[:, [5]] * weights))
    return np.stack(rows, axis=1)


def resample_means(
    stats: NDArray[np.float64],
    groups: NDArray[np.intp] | None,
    replicates: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Bootstrap row means, resampling clusters or rows; shape `(B, 6)`."""
    n = stats.shape[0]
    if groups is None:
        types, inverse, counts = np.unique(
            stats, axis=0, return_inverse=True, return_counts=True
        )
        del inverse
        draws = rng.multinomial(n, counts / n, size=replicates)
        sums = draws @ types
        result: NDArray[np.float64] = sums / n
        return result
    size = int(np.max(groups)) + 1
    sums_by = np.zeros((size, stats.shape[1]))
    np.add.at(sums_by, groups, stats)
    rows_by = np.bincount(groups, minlength=size).astype(float)
    draws = rng.multinomial(size, np.full(size, 1.0 / size), size=replicates)
    totals: NDArray[np.float64] = (draws @ sums_by) / (draws @ rows_by)[:, None]
    return totals


def factorize(clusters: ArrayLike | None) -> NDArray[np.intp] | None:
    """Cluster labels as integers `0 .. G-1`, or None."""
    if clusters is None:
        return None
    labels = np.asarray(clusters)
    if labels.ndim != 1:
        raise ValueError("clusters must be one-dimensional")
    return np.unique(labels, return_inverse=True)[1].astype(np.intp)


@dataclasses.dataclass(frozen=True, slots=True)
class NetBenefitReport:
    """Net benefit of a frozen rule over a range of threshold probabilities.

    Attributes:
        thresholds: The grid of `t`.
        rule: Net benefit of the rule on the grid.
        treat_all: Net benefit of acting on every row.
        incumbent: Net benefit of the incumbent, or None.
        baselines: Names of the baselines compared, in order.
        difference: Point differences, shape `(baselines, grid)`.
        lower: Simultaneous lower bounds, same shape.
        critical: The bootstrap critical value `c`.
        alpha: One-sided level of the band.
        replicates: Bootstrap replicates.
        rows: Rows in the table.
        clusters: Clusters (equal to `rows` when rows are independent).
        claim: `useful` or `not_useful_in_range`.
        useful_range: Largest `(t_lo, t_hi)` grid span on which the lower
            bounds are all positive and that contains the first positive
            point, or None.
    """

    thresholds: tuple[float, ...]
    rule: tuple[float, ...]
    treat_all: tuple[float, ...]
    incumbent: tuple[float, ...] | None
    baselines: tuple[str, ...]
    difference: tuple[tuple[float, ...], ...]
    lower: tuple[tuple[float, ...], ...]
    critical: float
    alpha: float
    replicates: int
    rows: int
    clusters: int
    claim: str
    useful_range: tuple[float, float] | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)


def useful_span(
    grid: NDArray[np.float64], positive: NDArray[np.bool_]
) -> tuple[float, float] | None:
    """The longest run of grid points where every lower bound is positive."""
    best: tuple[int, int] | None = None
    start: int | None = None
    for index in range(len(grid) + 1):
        good = index < len(grid) and bool(positive[index])
        if good and start is None:
            start = index
        if not good and start is not None:
            if best is None or index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    if best is None:
        return None
    return float(grid[best[0]]), float(grid[best[1] - 1])


def net_benefit_band(
    y: ArrayLike,
    acts: ArrayLike,
    t_range: tuple[float, float],
    *,
    incumbent: ArrayLike | None = None,
    clusters: ArrayLike | None = None,
    alpha: float = 0.05,
    replicates: int = 500,
    seed: int = 0,
    points: int = GRID_POINTS,
) -> NetBenefitReport:
    """Net benefit of a frozen rule with a simultaneous lower band.

    Args:
        y: Outcomes, 0 or 1.
        acts: The frozen rule's actions on these rows, 0 or 1.
        t_range: Chartered range `(t_lo, t_hi)` of threshold probabilities.
        incumbent: The incumbent's past decisions, or None.
        clusters: Cluster label per row, or None for independent rows.
        alpha: One-sided level of the band.
        replicates: Bootstrap replicates (the charter's `permutations_b`).
        seed: Seed of the bootstrap stream.
        points: Grid points on the range.

    Returns:
        The report, including the claim `useful` or `not_useful_in_range`.
    """
    outcomes, actions = costs.pair(y, acts)
    base = None if incumbent is None else costs.binary_vector(incumbent, "b")
    if base is not None and base.shape != outcomes.shape:
        raise ValueError("incumbent must have the length of y")
    low, high = t_range
    if not 0.0 < low <= high < 1.0:
        raise ValueError("t_range must satisfy 0 < t_lo <= t_hi < 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    qualification.count(replicates, "replicates", minimum=20)
    grid = np.linspace(low, high, 1 if low == high else points)
    weights = grid / (1.0 - grid)
    stats = columns(outcomes, actions, base)
    groups = factorize(clusters)
    if groups is not None and groups.shape != outcomes.shape:
        raise ValueError("clusters must have the length of y")
    rng = randomness.component_rng(seed, "decision.netbenefit", 0)
    point = diffs(stats.mean(axis=0, keepdims=True), weights, base is not None)[
        0
    ]
    boot = diffs(
        resample_means(stats, groups, replicates, rng),
        weights,
        base is not None,
    )
    sd = np.maximum(boot.std(axis=0, ddof=1), MIN_SD)
    sup = ((boot - point) / sd).reshape(replicates, -1).max(axis=1)
    critical = float(np.quantile(sup, 1.0 - alpha))
    lower = point - critical * sd
    positive = (lower > 0.0).all(axis=0)
    mean_row = stats.mean(axis=0)
    names = BASELINE_NAMES if base is not None else BASELINE_NAMES[:2]
    n_clusters = (
        int(np.max(groups)) + 1 if groups is not None else len(outcomes)
    )
    claim = USEFUL if bool(positive.all()) else NOT_USEFUL
    return NetBenefitReport(
        thresholds=tuple(float(t) for t in grid),
        rule=tuple(float(v) for v in mean_row[0] - mean_row[1] * weights),
        treat_all=tuple(float(v) for v in mean_row[2] - mean_row[3] * weights),
        incumbent=(
            None
            if base is None
            else tuple(float(v) for v in mean_row[4] - mean_row[5] * weights)
        ),
        baselines=names,
        difference=tuple(tuple(float(v) for v in row) for row in point),
        lower=tuple(tuple(float(v) for v in row) for row in lower),
        critical=critical,
        alpha=alpha,
        replicates=replicates,
        rows=len(outcomes),
        clusters=n_clusters,
        claim=claim,
        useful_range=useful_span(grid, positive),
    )


class FinalNetBenefit:
    """Evaluates one frozen rule on the final table exactly once.

    The rule is registered by its hash before the final table is read; the
    second call raises, so the interval is valid for the frozen rule and one
    look (canon P2).
    """

    def __init__(self, frozen: str) -> None:
        """Registers the hash of the frozen rule."""
        self.frozen = qualification_text(frozen)
        self.used = False

    def evaluate(
        self,
        frozen: str,
        y: ArrayLike,
        acts: ArrayLike,
        t_range: tuple[float, float],
        **options: Any,
    ) -> NetBenefitReport:
        """Runs the band once; a changed rule or a second call is refused."""
        if frozen != self.frozen:
            raise SecondEvaluationError(
                "the rule differs from the one registered before the final "
                "table was read"
            )
        if self.used:
            raise SecondEvaluationError(
                "the final table was already evaluated for this rule"
            )
        self.used = True
        return net_benefit_band(y, acts, t_range, **options)


def qualification_text(value: str) -> str:
    """A non-blank string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("the rule hash must be a non-blank string")
    return value
