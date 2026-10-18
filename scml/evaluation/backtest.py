"""Backtest of a forecaster: error at every horizon step against baselines.

For each step ahead the model's mean absolute error is set beside two
baselines a practitioner would try first: repeating the current value, and
repeating the value from the same point of the last season. A model is only
worth deploying if it beats both at the steps that matter.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from scml.evaluation import forecast


def seasonal_lag(step: int, period: int) -> int:
    """Rows back to the value at the same season point as `t + step`."""
    seasons = -(-step // period)  # Ceiling division.
    return seasons * period - step


def seasonal_naive(
    frame: pd.DataFrame,
    positions: np.ndarray,
    value: str,
    group: str | None,
    steps: int,
    period: int,
) -> np.ndarray:
    """Seasonal-naive forecasts for the rows at `positions` of an ordered frame.

    Returns:
        An array `(rows, steps)`; step `k` repeats the value `lag` rows back
        within the entity, where the lag puts it one or more whole seasons
        before the forecast time.
    """
    series = frame.groupby(group, sort=False)[value] if group else frame[value]
    columns = []
    for step in range(1, steps + 1):
        lag = seasonal_lag(step, period)
        shifted = series.shift(lag) if lag else frame[value]
        columns.append(shifted.to_numpy(float)[positions])
    return np.column_stack(columns)


def per_step_errors(truth: np.ndarray, guess: np.ndarray) -> list[float]:
    """Mean absolute error at each step ahead."""
    return [float(v) for v in np.abs(truth - guess).mean(axis=0)]


def report(
    estimator: Any,
    features: pd.DataFrame,
    primary: pd.Series,
    seasonal: np.ndarray | None,
) -> list[dict[str, float | None]]:
    """Per-step comparison of the model with the baselines on given rows."""
    truth = forecast.truth_matrix(estimator, features, primary)
    steps = truth.shape[1]
    model = per_step_errors(
        truth, np.asarray(estimator.predict(features), dtype=float)
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
    fitted: Sequence[tuple[Any, pd.DataFrame, pd.Series]],
) -> list[dict[str, Any]]:
    """Per-origin step errors for models fitted on each fold's past."""
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
