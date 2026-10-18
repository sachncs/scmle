"""Turning a time series into a supervised forecasting table.

A forecast of `horizon` steps is `horizon` targets: the value one step ahead,
two steps ahead, and so on. Each target is the series shifted back within its
entity, so row `t` is paired with what happened at `t+1 .. t+horizon`. Rows
near the end of a series, whose future is unknown, are dropped from training.
"""

import pandas as pd

from scml.contracts import tasks


def target_names(value: str, horizon: int) -> list[str]:
    """Names of the future-value targets: `<value>__h1 .. <value>__h<H>`."""
    return [
        f"{value}{tasks.HORIZON_SEPARATOR}{k}" for k in range(1, horizon + 1)
    ]


def add_targets(frame: pd.DataFrame, data: tasks.DataSpec) -> pd.DataFrame:
    """Sorts the series and adds the future-value target columns.

    Args:
        frame: The raw table.
        data: Data specification (value, time, group and horizon).

    Returns:
        A new frame ordered by entity and time, with `<value>__h1 ..` columns,
        without the rows whose future is not fully known.
    """
    value, horizon = data.value_column, data.horizon
    if value is None or horizon is None:
        raise ValueError("a forecasting spec needs value_column and horizon")
    keys = [c for c in (data.group_column, data.time_column) if c]
    ordered = frame.sort_values(keys, kind="stable").reset_index(drop=True)
    series = (
        ordered.groupby(data.group_column, sort=False)[value]
        if data.group_column
        else ordered[value]
    )
    names = tasks.horizon_names(value, horizon)
    for step, name in enumerate(names, start=1):
        ordered[name] = series.shift(-step)
    return ordered.dropna(subset=names).reset_index(drop=True)
