"""Backtest of a forecaster: error at every horizon step against baselines.

For each step ahead the model's mean absolute error is set beside two
baselines a practitioner would try first: repeating the current value, and
repeating the value from the same point of the last season. A model is only
worth deploying if it beats both at the steps that matter.
"""

from collections.abc import Sequence
from typing import Any

import numpy
import pandas

from scmle.evaluation import forecast


def seasonal_naive(
    frame: pandas.DataFrame,
    positions: numpy.ndarray,
    value: str,
    group: str | None,
    steps: int,
    period: int,
) -> numpy.ndarray:
    """Seasonal-naive forecasts for the rows at `positions` of an ordered frame.

    Args:
        frame: Rows ordered in time (within entity when `group` is given).
        positions: Integer positions of the forecast origins in `frame`.
        value: Column holding the series value.
        group: Column identifying the entity, or None for a single series.
        steps: Number of steps ahead.
        period: Length of a season in rows.

    Returns:
        An array `(rows, steps)`; step `k` repeats the value `lag` rows back
        within the entity, where the lag puts it one or more whole seasons
        before the forecast time. An entry is NaN where the entity has no
        row that far back.
    """
    series = frame.groupby(group, sort=False)[value] if group else frame[value]
    columns = []
    for step in range(1, steps + 1):
        # The smallest non-negative lag that puts the value a whole number of
        # seasons before the forecast target (0 when `step` is a multiple of
        # `period`).
        seasons = -(-step // period)  # Ceiling division.
        lag = seasons * period - step
        shifted = series.shift(lag) if lag else frame[value]
        columns.append(shifted.to_numpy(float)[positions])
    return numpy.column_stack(columns)


def per_step_errors(truth: numpy.ndarray, guess: numpy.ndarray) -> list[float]:
    """Mean absolute error at each step ahead.

    Args:
        truth: Observed values, shape `(rows, steps)`.
        guess: Forecast values, same shape.

    Returns:
        One mean absolute error per step (column), in step order.
    """
    return [float(v) for v in numpy.abs(truth - guess).mean(axis=0)]


def report(
    estimator: Any,
    features: pandas.DataFrame,
    primary: pandas.Series,
    seasonal: numpy.ndarray | None,
) -> list[dict[str, float | None]]:
    """Per-step comparison of the model with the baselines on given rows.

    Args:
        estimator: A fitted forecaster exposing `predict`, `target_columns`
            and `value_column`.
        features: Rows to forecast from; they must also hold the estimator's
            other target columns (the later steps).
        primary: The first-step target for those rows.
        seasonal: Seasonal-naive forecasts of shape `(rows, steps)`, or None
            when no season is known.

    Returns:
        One record per step ahead, with `step` (starting at 1),
        `model_mae`, `last_value_mae` and `seasonal_naive_mae` (None when
        `seasonal` is None).
    """
    truth = forecast.truth_matrix(estimator, features, primary)
    steps = truth.shape[1]
    model = per_step_errors(
        truth, numpy.asarray(estimator.predict(features), dtype=float)
    )
    last = per_step_errors(
        truth, forecast.last_value(estimator, features, steps)
    )
    season = None if seasonal is None else per_step_errors(truth, seasonal)
    return [
        {
            "step": step + 1,
            "model_mae": model[step],
            "last_value_mae": last[step],
            "seasonal_naive_mae": None if season is None else season[step],
        }
        for step in range(steps)
    ]


def origins(
    fitted: Sequence[tuple[Any, pandas.DataFrame, pandas.Series]],
) -> list[dict[str, Any]]:
    """Per-origin step errors for models fitted on each fold's past.

    Args:
        fitted: One (estimator, features, primary target) triple per
            origin, in order.

    Returns:
        One record per origin, numbered from 1, with the row count and the
        per-step `model_mae` and `last_value_mae` lists.
    """
    found = []
    for number, (estimator, features, primary) in enumerate(fitted, start=1):
        rows = report(estimator, features, primary, None)
        found.append(
            {
                "origin": number,
                "rows": len(features),
                "model_mae": [r["model_mae"] for r in rows],
                "last_value_mae": [r["last_value_mae"] for r in rows],
            }
        )
    return found
