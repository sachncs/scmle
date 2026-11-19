"""Rolling-origin, horizon-aware evaluation of forecast families on panels.

A forecast is judged where it will be used: from an origin, on the steps
after it, having seen only what came before. `origins` places the origins
back from the end of the panel one stride apart (by default one horizon,
so the evaluated windows do not overlap); at each origin every family is
fitted afresh on the history and forecasts each series `horizon` rows
ahead (Tashman 2000, "Out-of-sample tests of forecasting accuracy", IJF
16:437-450).

Each step is scored by its mean absolute error and by MASE, the error over
the series' in-sample naive error on the history before the origin
(Hyndman and Koehler 2006, the scale of `proper_scores.mase`; the seasonal
naive error when a period is given). A series whose history never changes
has no such scale: it counts in the error, is counted in `undefined`, and
is left out of MASE only. Each family is compared with the baseline over
the (origin, series) pairs by a one-sided Wilcoxon signed-rank test.
"""

from collections.abc import Sequence
import copy
import dataclasses
import math

import numpy
import pandas
from scipy import stats

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.models.forecast import families
from scmle.models.forecast.panel import Panel

ALPHA = 0.05
"""One-sided error rate at which a family is said to beat the baseline."""

PAIRS = math.ceil(math.log2(1 / ALPHA))
"""Fewest non-tied (origin, series) pairs a comparison needs: the smallest
one-sided p-value the exact signed-rank test can give `n` pairs is `2**-n`,
which is below `ALPHA` only from `n = ceil(log2(1 / ALPHA)) = 5` on. With
fewer the comparison is left unassessed (None), never called a loss."""

HISTORY = 2
"""Fewest time points before the first origin: a naive scale needs one
change, so two values."""


@dataclasses.dataclass(frozen=True)
class Score:
    """One family's backtest.

    Attributes:
        name: The family.
        mae: Mean absolute error at each step ahead.
        mase: Mean scaled error at each step ahead (NaN when no pair at the
            step has a scale).
        overall: Mean scaled error over every step.
        pvalue: One-sided Wilcoxon p-value that the family's per-pair error
            is below the baseline's; None for the baseline itself or with
            fewer than `PAIRS` non-tied pairs.
        beats: Whether `pvalue` is below `ALPHA`; None when unassessed.
    """

    name: str
    mae: tuple[float, ...]
    mase: tuple[float, ...]
    overall: float
    pvalue: float | None
    beats: bool | None


@dataclasses.dataclass(frozen=True)
class Backtest:
    """A rolling-origin comparison of forecast families.

    Attributes:
        horizon: Rows forecast ahead from each origin.
        origins: The origins, oldest first, as text.
        entities: Series in the panel.
        baseline: The family the others are compared with.
        counts: Forecasts scored at each step (pairs with an actual value).
        undefined: (origin, series) pairs with no naive scale, scored by
            MAE only.
        scores: One score per family, in the order given.
    """

    horizon: int
    origins: tuple[str, ...]
    entities: int
    baseline: str
    counts: tuple[int, ...]
    undefined: int
    scores: tuple[Score, ...]

    def best(self) -> str:
        """The family with the lowest overall scaled error.

        Returns:
            Its name; ties go to the earlier family.
        """
        return min(self.scores, key=lambda s: s.overall).name


def origins(
    panel: Panel, horizon: int, count: int, stride: int | None = None
) -> numpy.ndarray:
    """Places the forecast origins back from the end of the panel.

    Args:
        panel: The full panel.
        horizon: Rows forecast ahead.
        count: Origins wanted.
        stride: Time points between origins; None uses `horizon`.

    Returns:
        `count` times of the panel, oldest first; the latest leaves
        `horizon` later time points to score.

    Raises:
        BlockedError: If the panel has fewer than
            `horizon + (count - 1) * stride + HISTORY` time points.
    """
    moments = panel.moments()
    step = stride or horizon
    needed = horizon + (count - 1) * step + HISTORY
    if len(moments) < needed:
        raise errors.BlockedError(
            f"{count} origins {step} apart with a horizon of {horizon} needs "
            f"{needed} time points; the panel has {len(moments)}: use fewer "
            "origins (validation.folds) or a shorter horizon"
        )
    last = len(moments) - 1 - horizon
    return moments[[last - k * step for k in reversed(range(count))]]


def spread(values: numpy.ndarray, period: int | None) -> float:
    """The in-sample (seasonal) naive mean absolute error of a history.

    Args:
        values: The history.
        period: Rows per season, or None.

    Returns:
        The mean absolute difference of values `period` rows apart (one row
        apart without a period or with too short a history); NaN with fewer
        than two values.
    """
    lag = period if period and len(values) > period else 1
    if len(values) <= lag:
        return math.nan
    return float(numpy.abs(values[lag:] - values[:-lag]).mean())


def pad(values: numpy.ndarray, horizon: int) -> numpy.ndarray:
    """A forecast or actual stretched to `horizon` entries with NaN."""
    found = numpy.full(horizon, numpy.nan)
    found[: len(values)] = values[:horizon]
    return found


def compare(mine: numpy.ndarray, theirs: numpy.ndarray) -> float | None:
    """One-sided signed-rank p-value that `mine` is below `theirs`.

    Args:
        mine: Per-pair errors of a family.
        theirs: Per-pair errors of the baseline.

    Returns:
        The p-value, or None with fewer than `PAIRS` non-tied finite pairs.
    """
    gaps = mine - theirs
    gaps = gaps[numpy.isfinite(gaps) & (gaps != 0)]
    if len(gaps) < PAIRS:
        return None
    return float(stats.wilcoxon(gaps, alternative="less").pvalue)


def mean(values: numpy.ndarray) -> float:
    """Mean of the finite entries; NaN when there are none."""
    finite = values[numpy.isfinite(values)]
    return float(finite.mean()) if len(finite) else math.nan


def gather(
    panel: Panel,
    moments: numpy.ndarray,
    horizon: int,
    models: Sequence[families.Forecaster],
    period: int | None,
) -> tuple[dict[str, list[numpy.ndarray]], numpy.ndarray]:
    """Forecasts from every origin and records each pair's misses.

    Args:
        panel: The full panel.
        moments: The origins.
        horizon: Rows forecast ahead.
        models: The families, each refitted on a copy at every origin.
        period: Rows per season for the naive scale, or None.

    Returns:
        Family name to one array of `horizon` absolute errors per (origin,
        series) pair (NaN where no actual exists), and each pair's naive
        scale on the history before its origin.

    Raises:
        BlockedError: If a family cannot learn from the history before an
            origin (its fit raised DataError, for example too few rows for
            a boosting step): the panel is too short to backtest it.
    """
    misses: dict[str, list[numpy.ndarray]] = {m.name: [] for m in models}
    scales = []
    for moment in moments:
        past = panel.history(moment)
        truth = panel.ahead(moment, horizon)
        guesses = []
        for model in models:
            try:
                fitted = copy.deepcopy(model).fit(past, horizon)
            except errors.DataError as exc:
                raise errors.BlockedError(f"at origin {moment}: {exc}") from exc
            guesses.append(fitted.predict())
        for key, values in past.series().items():
            actual = pad(truth[key], horizon)
            scales.append(spread(values, period))
            for model, guess in zip(models, guesses, strict=True):
                misses[model.name].append(
                    numpy.abs(actual - pad(guess[key], horizon))
                )
    return misses, numpy.asarray(scales)


def backtest(
    panel: Panel,
    horizon: int,
    models: Sequence[families.Forecaster],
    count: int = 5,
    stride: int | None = None,
    period: int | None = None,
    baseline: str | None = None,
) -> Backtest:
    """Rolling-origin backtest of forecast families.

    Args:
        panel: The full panel.
        horizon: Rows forecast ahead from each origin.
        models: The families, each refitted on a copy at every origin.
        count: Origins; 5 by default, so a single series still gives the
            `PAIRS` pairs a comparison needs.
        stride: Time points between origins; None uses `horizon`.
        period: Rows per season for the MASE scale, or None.
        baseline: Family compared against; None takes the first.

    Returns:
        The per-step errors of each family and its comparison with the
        baseline.

    Raises:
        InputError: If two families share a name or the baseline is not one
            of them.
        BlockedError: If the panel is too short for the origins, or for a
            family to learn from the history before an origin.
    """
    names = [model.name for model in models]
    if len(set(names)) != len(names):
        raise errors.InputError(f"backtest families must be distinct: {names}")
    chosen = baseline or names[0]
    if chosen not in names:
        raise errors.InputError(f"baseline '{chosen}' is not one of {names}")
    moments = origins(panel, horizon, count, stride)
    misses, scale = gather(panel, moments, horizon, models, period)
    usable = numpy.isfinite(scale) & (scale > 0)
    reference = numpy.vstack(misses[chosen])
    scored = numpy.isfinite(reference)
    scores = []
    for name in names:
        table = numpy.vstack(misses[name])
        scaled = table[usable] / scale[usable][:, None]
        pvalue = None
        if name != chosen:
            pvalue = compare(averages(table), averages(reference))
        scores.append(
            Score(
                name,
                tuple(mean(column) for column in table.T),
                tuple(mean(column) for column in scaled.T),
                mean(scaled.ravel()),
                pvalue,
                None if pvalue is None else pvalue < ALPHA,
            )
        )
    return Backtest(
        horizon,
        tuple(str(moment) for moment in moments),
        len(panel.series()),
        chosen,
        tuple(int(c) for c in scored.sum(axis=0)),
        int((~usable & scored.any(axis=1)).sum()),
        tuple(scores),
    )


def averages(table: numpy.ndarray) -> numpy.ndarray:
    """Each pair's mean error over its scored steps (NaN when none)."""
    counts = numpy.isfinite(table).sum(axis=1)
    totals = numpy.nansum(table, axis=1)
    return numpy.where(counts > 0, totals / numpy.maximum(counts, 1), numpy.nan)


def contenders(spec: tasks.Config) -> list[families.Forecaster]:
    """The forecast families a project configuration describes.

    Args:
        spec: The project configuration.

    Returns:
        The seasonal-naive baseline with `data.seasonal_period`, simple
        exponential smoothing, and lag boosting with `data.forecast_lags`
        and `data.forecast_windows` (else the family defaults), the period
        and `validation.seed`.

    Raises:
        InputError: If a lag, window or period is out of range.
    """
    data = spec.data
    period = data.seasonal_period
    return [
        families.Seasonal(period),
        families.Smoothing(),
        families.Boosted(
            lags=data.forecast_lags or families.LAGS,
            windows=data.forecast_windows or families.WINDOWS,
            period=period,
            seed=spec.validation.seed,
        ),
    ]


def spacing(spec: tasks.Config, built: Panel) -> int | None:
    """Time points between origins from `validation.horizon_rows`.

    The rolling splitter counts `horizon_rows` in rows of the whole table,
    every series' rows together, while origins are counted in time points;
    one time point holds a row of each series, so the rows are divided by
    the number of series.

    Args:
        spec: The project configuration.
        built: The panel.

    Returns:
        `horizon_rows // series` (at least 1), or None (one horizon) when
        `horizon_rows` is not set.
    """
    rows = spec.validation.horizon_rows
    if rows is None:
        return None
    return max(1, rows // max(1, len(built.series())))


def run(spec: tasks.Config, frame: pandas.DataFrame) -> Backtest:
    """Backtests the forecast families a project configuration describes.

    The panel is `data.group_column` (entities), `data.time_column` and the
    value (`data.value_column`, else the target) of the raw table, before
    the future-value targets are added. The families are `contenders`; the
    `validation.folds` origins are `spacing` apart (one horizon when
    `validation.horizon_rows` is not set).

    Args:
        spec: The project configuration.
        frame: The raw rows.

    Returns:
        The backtest.

    Raises:
        InputError: If the configuration has no time column or horizon.
        DataError: If the panel's columns are unusable.
        BlockedError: If the panel is too short for the origins or for a
            family to learn from the history before an origin.
    """
    data = spec.data
    if not data.time_column or data.horizon is None:
        raise errors.InputError(
            "a forecast backtest needs data.time_column and data.horizon"
        )
    value = data.value_column or data.target
    built = Panel.build(frame, data.time_column, value, data.group_column)
    return backtest(
        built,
        data.horizon,
        contenders(spec),
        count=spec.validation.folds,
        stride=spacing(spec, built),
        period=data.seasonal_period,
    )
