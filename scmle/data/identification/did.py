"""Difference in differences with group-time effects and an event study.

Estimator in the spirit of Callaway and Sant'Anna (2021, Journal of
Econometrics): for each adoption group `g` and period `t`, the effect is the
change of the group's mean outcome from its base period (the last period
before adoption) to `t`, minus the same change among never-treated units.
Never-treated units are the only comparison group, so units that are already
treated never serve as controls. No covariates are used (the unconditional
parallel-trends version). Aggregations use group-size weights; their variance
follows from the unit-level decomposition below, which is exact for fixed
weights:

    theta = sum_g mean_{i in g}(a_i^g) - mean_{i never}(c_i),
    var   = sum_g var(a^g) / n_g + var(c) / n_c.

Sensitivity. Pre-trend tests are low powered (Roth 2022), so the pre-trend
Wald test is descriptive. The gating quantity is a simplified relative-
magnitudes bound in the spirit of Rambachan and Roth (2023, Review of
Economic Studies 90(5)): the post-treatment trend difference may change per
period by at most `M` times the largest pre-period change, so the bias at
event time `e >= 0` is at most `(e + 1) M J` with `J` the largest absolute
change between consecutive pre-period coefficients (the coefficient at -1 is
zero by construction). The reported set is the confidence interval widened by
the average bound. This is NOT the HonestDiD conditional or fixed-length
procedure: it ignores the noise in `J`; its coverage under planted violations
is measured in the certificate, and the literature details above were not
re-verified against the papers in this environment.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import itertools
from typing import Any

import numpy
import pandas
from scipy import stats

MIN_PERIODS = 3
MIN_NEVER_TREATED = 2


@dataclasses.dataclass(frozen=True)
class Estimate:
    """Group-time difference in differences.

    Attributes:
        estimate: Overall ATT (group-size-weighted post cells).
        se: Its standard error.
        low: Lower interval limit.
        high: Upper interval limit.
        level: Confidence level.
        groups: Adoption period to unit count.
        n_never: Never-treated units.
        events: Event times, in increasing order.
        coefficients: Event-study coefficient per event time.
        ses: Standard error per event time.
        covariance: Covariance matrix of the coefficients.
        pretrend_p: Wald p-value of the pre-period coefficients, if any.
        max_pre_change: `J`, largest consecutive pre-period change, or None
            when fewer than two pre-periods exist.
        mean_horizon: Group-size-weighted mean of `e + 1` over the post
            cells (bias multiplier).
    """

    estimate: float
    se: float
    low: float
    high: float
    level: float
    groups: dict[str, int]
    n_never: int
    events: tuple[int, ...]
    coefficients: tuple[float, ...]
    ses: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    pretrend_p: float | None
    max_pre_change: float | None
    mean_horizon: float

    def sensitivity(self, m: float) -> tuple[float, float]:
        """Widens the interval by the relative-magnitudes bound at `m`.

        Args:
            m: The multiple of the largest pre-period change that the
                post-treatment trend may change by per period.

        Returns:
            The lower and upper limits, each moved outward by
            ``m * max_pre_change * mean_horizon``.

        Raises:
            ValueError: If fewer than two pre-periods exist.
        """
        if self.max_pre_change is None:
            raise ValueError("sensitivity needs at least two pre-periods")
        bias = m * self.max_pre_change * self.mean_horizon
        return self.low - bias, self.high + bias

    def breakdown(self) -> float | None:
        """Finds the smallest `m` at which the widened interval contains zero.

        Returns:
            None when fewer than two pre-periods exist; 0.0 when the
            interval already contains zero; infinity when the bound does
            not grow with `m`; otherwise the distance from zero to the
            nearer interval limit divided by ``max_pre_change *
            mean_horizon``.
        """
        if self.max_pre_change is None:
            return None
        gap = min(abs(self.low), abs(self.high))
        if self.low <= 0 <= self.high:
            return 0.0
        scale = self.max_pre_change * self.mean_horizon
        return float("inf") if scale == 0 else gap / scale

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record.

        Returns:
            A mapping of every field.
        """
        return dataclasses.asdict(self)


def wide_panel(
    frame: pandas.DataFrame, unit: str, time: str, outcome: str
) -> pandas.DataFrame:
    """Reshapes a balanced panel to one row per unit and column per period.

    Args:
        frame: Long panel data.
        unit: Unit column.
        time: Period column.
        outcome: Outcome column.

    Returns:
        The outcomes with units as rows and periods as columns, columns in
        increasing period order.

    Raises:
        ValueError: If any unit-period outcome is missing or a unit-period
            pair occurs more than once.
    """
    table = frame.pivot(  # noqa: PD010 -- pivot_table would average repeated unit-period rows; pivot raises on them, as documented
        index=unit, columns=time, values=outcome
    )
    if table.isna().any().any():
        raise ValueError("the panel must be balanced with no missing outcomes")
    return table.sort_index(axis=1)


def adoption_of(
    frame: pandas.DataFrame, unit: str, adoption: str
) -> pandas.Series:
    """Finds the first treated period of each unit.

    Args:
        frame: Long panel data.
        unit: Unit column.
        adoption: Column holding the unit's adoption period (constant within
            a unit; missing for never-treated units).

    Returns:
        The adoption period per unit (NaN for never treated).

    Raises:
        ValueError: If the adoption value varies within a unit.
    """
    per_unit = frame.groupby(unit)[adoption]
    if (per_unit.nunique(dropna=False) > 1).any():  # noqa: PD101 -- a per-unit count on a SeriesGroupBy, not a check that one Series is constant
        raise ValueError("adoption must be constant within a unit")
    return per_unit.first()


def event_arms(
    y: numpy.ndarray,
    index: Mapping[Any, int],
    n_periods: int,
    groups: Sequence[Any],
    sizes: Mapping[Any, int],
    members: Mapping[Any, numpy.ndarray],
    never: numpy.ndarray,
    events: Sequence[int],
) -> tuple[dict[int, dict[Any, numpy.ndarray]], dict[int, numpy.ndarray]]:
    """Builds the per-unit outcome changes of each event time.

    Args:
        y: Outcomes, units by periods.
        index: Position of each period.
        n_periods: Number of periods.
        groups: Adoption periods, one per cohort.
        sizes: Number of units of each cohort.
        members: Mask of the units of each cohort.
        never: Mask of the never-treated units.
        events: Event times, in periods from adoption.

    Returns:
        For each event time, the cohort-weighted changes of the treated
        units of each cohort, and the cohort-weighted changes of the
        never-treated units; changes are taken from the period before
        adoption.
    """
    treated_vars: dict[int, dict[Any, numpy.ndarray]] = {}
    control_vars: dict[int, numpy.ndarray] = {}
    for e in events:
        cells = [g for g in groups if 0 <= index[g] + e < n_periods]
        total = sum(sizes[g] for g in cells)
        treated_vars[e] = {}
        control = numpy.zeros(int(never.sum()))
        for g in cells:
            base = index[g] - 1
            delta = y[:, index[g] + e] - y[:, base]
            weight = sizes[g] / total
            treated_vars[e][g] = weight * delta[members[g]]
            control += weight * delta[never]
        control_vars[e] = control
    return treated_vars, control_vars


def event_coefficients(
    events: Sequence[int],
    treated_vars: Mapping[int, Mapping[Any, numpy.ndarray]],
    control_vars: Mapping[int, numpy.ndarray],
) -> list[float]:
    """Computes the event-study coefficients.

    Args:
        events: Event times.
        treated_vars: Treated changes per event time and cohort.
        control_vars: Never-treated changes per event time.

    Returns:
        For each event time, the mean treated change minus the mean
        never-treated change.
    """
    coefficients = []
    for e in events:
        mean_treated = sum(v.mean() for v in treated_vars[e].values())
        coefficients.append(float(mean_treated - control_vars[e].mean()))
    return coefficients


def event_covariance(
    events: Sequence[int],
    groups: Sequence[Any],
    sizes: Mapping[Any, int],
    treated_vars: Mapping[int, Mapping[Any, numpy.ndarray]],
    control_vars: Mapping[int, numpy.ndarray],
    n_never: int,
) -> numpy.ndarray:
    """Estimates the covariance of the event-study coefficients.

    Args:
        events: Event times.
        groups: Adoption periods, one per cohort.
        sizes: Number of units of each cohort.
        treated_vars: Treated changes per event time and cohort.
        control_vars: Never-treated changes per event time.
        n_never: Number of never-treated units.

    Returns:
        The events by events covariance matrix.
    """
    covariance = numpy.zeros((len(events), len(events)))
    for a, e in enumerate(events):
        for b, f in enumerate(events):
            value = 0.0
            for g in groups:
                if g in treated_vars[e] and g in treated_vars[f]:
                    value += (
                        numpy.cov(treated_vars[e][g], treated_vars[f][g])[0, 1]
                        / sizes[g]
                        if sizes[g] > 1
                        else 0.0
                    )
            value += numpy.cov(control_vars[e], control_vars[f])[0, 1] / n_never
            covariance[a, b] = value
    return covariance


def overall_att(
    y: numpy.ndarray,
    index: Mapping[Any, int],
    post_cells: Sequence[tuple[Any, int]],
    weights: numpy.ndarray,
    members: Mapping[Any, numpy.ndarray],
    sizes: Mapping[Any, int],
    never: numpy.ndarray,
) -> tuple[float, float]:
    """Aggregates the post-adoption cells into one ATT and its standard error.

    Args:
        y: Outcomes, units by periods.
        index: Position of each period.
        post_cells: Pairs of adoption period and post-adoption period
            position.
        weights: Weight of each cell, summing to one.
        members: Mask of the units of each cohort.
        sizes: Number of units of each cohort.
        never: Mask of the never-treated units.

    Returns:
        The weighted ATT and its standard error.
    """
    n_never = int(never.sum())
    theta_parts = []
    treated_unit: dict[Any, numpy.ndarray] = {}
    control_unit = numpy.zeros(n_never)
    for (g, j), w in zip(post_cells, weights, strict=True):
        delta = y[:, j] - y[:, index[g] - 1]
        treated_unit[g] = treated_unit.get(g, 0.0) + w * delta[members[g]]
        control_unit += w * delta[never]
        theta_parts.append(w * (delta[members[g]].mean() - delta[never].mean()))
    theta = float(sum(theta_parts))
    variance = (
        sum(
            treated_unit[g].var(ddof=1) / sizes[g]
            for g in treated_unit
            if sizes[g] > 1
        )
        + control_unit.var(ddof=1) / n_never
    )
    return theta, float(numpy.sqrt(variance))


def pretrend_p_value(
    events: Sequence[int],
    coefficients: Sequence[float],
    covariance: numpy.ndarray,
) -> float | None:
    """Tests that the pre-period event-study coefficients are jointly zero.

    Args:
        events: Event times.
        coefficients: The event-study coefficients.
        covariance: Their covariance matrix.

    Returns:
        The chi-squared p-value, or None when no event time is before the
        reference period.
    """
    pre = [k for k, e in enumerate(events) if e < -1]
    if not pre:
        return None
    beta = numpy.array([coefficients[k] for k in pre])
    cov = covariance[numpy.ix_(pre, pre)]
    stat = float(beta @ numpy.linalg.pinv(cov) @ beta)
    return float(stats.chi2.sf(stat, len(pre)))


def max_pre_event_change(
    events: Sequence[int], coefficients: Sequence[float]
) -> float | None:
    """Finds the largest change between consecutive pre-period coefficients.

    Args:
        events: Event times.
        coefficients: The event-study coefficients.

    Returns:
        The largest absolute change along the pre-period chain that ends at
        the reference period (whose coefficient is 0), or None when there
        is no pre-period besides the reference one.
    """
    pre_events = [e for e in events if e < 0]
    if not [e for e in pre_events if e < -1]:
        return None
    chain = {e: coefficients[events.index(e)] for e in pre_events}
    chain[-1] = 0.0
    ordered = sorted(chain)
    jumps = [abs(chain[b] - chain[a]) for a, b in itertools.pairwise(ordered)]
    return float(max(jumps))


def estimate(
    frame: pandas.DataFrame,
    unit: str,
    time: str,
    adoption: str,
    outcome: str,
    level: float = 0.95,
) -> Estimate:
    """Estimates group-time ATT against never-treated units, aggregated.

    Args:
        frame: Long panel data.
        unit: Unit column.
        time: Period column.
        adoption: Column with the unit's first treated period (missing for
            never-treated units).
        outcome: Outcome column.
        level: Confidence level.

    Returns:
        The overall ATT with its interval, the event-study coefficients,
        their covariance and the quantities the sensitivity bound uses.

    Raises:
        ValueError: If the panel is unbalanced or repeats a unit-period
            pair, has fewer than three periods, has fewer than two
            never-treated units, the adoption value varies within a unit, or
            no adoption group has a pre-period (groups adopting in the first
            period or in a period not observed are dropped).
    """
    wide = wide_panel(frame, unit, time, outcome)
    periods = list(wide.columns)
    if len(periods) < MIN_PERIODS:
        raise ValueError(f"at least {MIN_PERIODS} periods are needed")
    adopt = adoption_of(frame, unit, adoption).reindex(wide.index)
    never = adopt.isna().to_numpy()
    if never.sum() < MIN_NEVER_TREATED:
        raise ValueError("at least two never-treated units are needed")
    y = wide.to_numpy(dtype=float)
    index = {p: i for i, p in enumerate(periods)}
    groups = sorted(g for g in adopt.dropna().unique() if g in index)
    groups = [g for g in groups if index[g] >= 1]
    if not groups:
        raise ValueError("no adoption group has a pre-period")
    members = {g: (adopt == g).to_numpy() for g in groups}
    sizes = {g: int(members[g].sum()) for g in groups}
    events = sorted(
        {
            j - index[g]
            for g in groups
            for j in range(len(periods))
            if j != index[g] - 1
        }
    )
    treated_vars, control_vars = event_arms(
        y, index, len(periods), groups, sizes, members, never, events
    )
    coefficients = event_coefficients(events, treated_vars, control_vars)
    n_never = int(never.sum())
    covariance = event_covariance(
        events, groups, sizes, treated_vars, control_vars, n_never
    )
    post_cells = [(g, j) for g in groups for j in range(index[g], len(periods))]
    weights = numpy.array([sizes[cell[0]] for cell in post_cells], dtype=float)
    weights /= weights.sum()
    theta, se = overall_att(
        y, index, post_cells, weights, members, sizes, never
    )
    z = float(stats.norm.ppf(0.5 + level / 2))
    pretrend_p = pretrend_p_value(events, coefficients, covariance)
    max_change = max_pre_event_change(events, coefficients)
    horizon = float(
        numpy.average(
            [j - index[g] + 1 for g, j in post_cells], weights=weights
        )
    )
    return Estimate(
        estimate=theta,
        se=se,
        low=theta - z * se,
        high=theta + z * se,
        level=level,
        groups={str(g): sizes[g] for g in groups},
        n_never=n_never,
        events=tuple(int(e) for e in events),
        coefficients=tuple(coefficients),
        ses=tuple(
            float(numpy.sqrt(max(covariance[k, k], 0.0)))
            for k in range(len(events))
        ),
        covariance=tuple(tuple(float(v) for v in row) for row in covariance),
        pretrend_p=pretrend_p,
        max_pre_change=max_change,
        mean_horizon=horizon,
    )
