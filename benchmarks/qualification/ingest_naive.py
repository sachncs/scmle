"""Naive point-in-time feature construction, used as a reference oracle.

Everything is a plain Python loop. Times are floats measured in days.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import math
from typing import Any

import pandas as pd

PRIMITIVES = (
    "count",
    "sum",
    "mean",
    "min",
    "max",
    "std",
    "nunique",
    "last",
    "time_since_last",
    "slope",
)

NAN = float("nan")


def is_missing(value: Any) -> bool:
    """Return True for None or NaN cells.

    Args:
        value: The cell value.

    Returns:
        True if the value is None or compares unequal to itself (NaN); False
        when the comparison itself fails.
    """
    if value is None:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def records(frame: pd.DataFrame) -> list[Mapping[str, Any]]:
    """Return the frame rows as dicts keyed by column name.

    Args:
        frame: The table.

    Returns:
        One dict per row, with column names converted to text.
    """
    out: list[Mapping[str, Any]] = []
    for rec in frame.to_dict("records"):
        out.append({str(k): v for k, v in rec.items()})
    return out


def known_time(
    row: Mapping[str, Any],
    event_col: str,
    known_col: str | None,
    delta_rec_days: float,
) -> float:
    """Return when a row became known.

    Args:
        row: The table row.
        event_col: Event time column.
        known_col: Known time column, or None.
        delta_rec_days: Recording delay used when the cell is absent.

    Returns:
        The known time cell, else ``event_time + delta_rec_days``.
    """
    if (
        known_col is not None
        and known_col in row
        and not is_missing(row[known_col])
    ):
        return float(row[known_col])
    return float(row[event_col]) + delta_rec_days


def scope_rows(
    rows: Sequence[Mapping[str, Any]],
    key_col: str,
    event_col: str,
    known_col: str | None,
    delta_rec_days: float,
    entity: Any,
    cutoff: float,
    window_days: float | None,
) -> list[Mapping[str, Any]]:
    """Return visible, in-window rows in original order.

    Args:
        rows: Table rows.
        key_col: Entity key column.
        event_col: Event time column.
        known_col: Known time column, or None.
        delta_rec_days: Recording delay fallback.
        entity: Entity of the target.
        cutoff: Cutoff time of the target.
        window_days: Window length, or None for no window.

    Returns:
        The rows in scope.
    """
    out: list[Mapping[str, Any]] = []
    for row in rows:
        if row[key_col] != entity:
            continue
        event = float(row[event_col])
        if event > cutoff:
            continue
        if known_time(row, event_col, known_col, delta_rec_days) > cutoff:
            continue
        if window_days is not None and not event > cutoff - window_days:
            continue
        out.append(row)
    return out


def primitive(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    value_col: str | None,
    event_col: str,
    known_col: str | None,
    delta_rec_days: float,
    cutoff: float,
) -> float:
    """Apply a primitive to rows already in scope.

    Args:
        name: One of ``PRIMITIVES``.
        rows: Scope rows in table order.
        value_col: Value column; None means every value is missing.
        event_col: Event time column.
        known_col: Known time column, or None.
        delta_rec_days: Recording delay fallback.
        cutoff: Cutoff time of the target.

    Returns:
        The primitive value as a float.

    Raises:
        ValueError: If the primitive name is unknown.
    """
    if name not in PRIMITIVES:
        raise ValueError(f"unknown primitive {name!r}")
    if name == "count":
        return float(len(rows))
    if name == "time_since_last":
        if not rows:
            return NAN
        latest = float(rows[0][event_col])
        for row in rows:
            latest = max(latest, float(row[event_col]))
        return cutoff - latest
    valid: list[Mapping[str, Any]] = []
    for row in rows:
        if value_col is not None and not is_missing(row.get(value_col)):
            valid.append(row)
    vals: list[float] = []
    if value_col is not None:
        for row in valid:
            vals.append(float(row[value_col]))
    n = len(vals)
    if name == "sum":
        total = 0.0
        for v in vals:
            total += v
        return total
    if name == "nunique":
        seen: list[float] = []
        for v in vals:
            if v not in seen:
                seen.append(v)
        return float(len(seen))
    if n == 0:
        return NAN
    if name == "min":
        low = vals[0]
        for v in vals:
            if v < low:
                low = v
        return low
    if name == "max":
        high = vals[0]
        for v in vals:
            if v > high:
                high = v
        return high
    if name == "last":
        best = 0
        for i in range(1, n):
            cand = (
                float(valid[i][event_col]),
                known_time(valid[i], event_col, known_col, delta_rec_days),
            )
            cur = (
                float(valid[best][event_col]),
                known_time(valid[best], event_col, known_col, delta_rec_days),
            )
            if cand >= cur:
                best = i
        return vals[best]
    total = 0.0
    for v in vals:
        total += v
    mean = total / n
    if name == "mean":
        return mean
    if name == "std":
        if n < 2:
            return NAN
        ss = 0.0
        for v in vals:
            ss += (v - mean) ** 2
        return math.sqrt(ss / (n - 1))
    # slope
    if n < 2:
        return NAN
    xs = [float(row[event_col]) for row in valid]
    xsum = 0.0
    for x in xs:
        xsum += x
    xbar = xsum / n
    sxx = 0.0
    sxy = 0.0
    for i in range(n):
        sxx += (xs[i] - xbar) ** 2
        sxy += (xs[i] - xbar) * (vals[i] - mean)
    if sxx == 0.0:
        return NAN
    return sxy / sxx


def aggregate_feature(
    targets: pd.DataFrame,
    key_col: str,
    moment_col: str,
    child: pd.DataFrame,
    child_key_col: str,
    event_col: str,
    known_col: str | None,
    value_col: str | None,
    primitive_name: str,
    window_days: float | None,
    lag_days: float,
    delta_rec_days: float,
) -> list[float]:
    """Compute one aggregate feature per target row, in order.

    Args:
        targets: Target frame.
        key_col: Entity column of the targets.
        moment_col: Prediction moment column (float days).
        child: Child table.
        child_key_col: Entity column of the child.
        event_col: Child event time column.
        known_col: Child known time column, or None.
        value_col: Child value column, or None.
        primitive_name: One of ``PRIMITIVES``.
        window_days: Window length, or None.
        lag_days: Gap between cutoff and prediction moment.
        delta_rec_days: Recording delay fallback.

    Returns:
        One value per target row.
    """
    rows = records(child)
    out: list[float] = []
    for target in records(targets):
        cutoff = float(target[moment_col]) - lag_days
        scope = scope_rows(
            rows,
            child_key_col,
            event_col,
            known_col,
            delta_rec_days,
            target[key_col],
            cutoff,
            window_days,
        )
        out.append(
            primitive(
                primitive_name,
                scope,
                value_col,
                event_col,
                known_col,
                delta_rec_days,
                cutoff,
            )
        )
    return out


def nested_feature(
    targets: pd.DataFrame,
    key_col: str,
    moment_col: str,
    mid: pd.DataFrame,
    mid_key_col: str,
    mid_id_col: str,
    mid_event_col: str,
    mid_known_col: str | None,
    leaf: pd.DataFrame,
    leaf_link_col: str,
    leaf_event_col: str,
    leaf_known_col: str | None,
    leaf_value_col: str | None,
    inner_primitive: str,
    outer_primitive: str,
    inner_window_days: float | None,
    outer_window_days: float | None,
    lag_days: float,
    delta_rec_days: float,
) -> list[float]:
    """Compute a depth-2 nested feature per target row, in order.

    Args:
        targets: Target frame.
        key_col: Entity column of the targets.
        moment_col: Prediction moment column (float days).
        mid: Middle table.
        mid_key_col: Entity column of the middle table.
        mid_id_col: Identifier column of the middle table.
        mid_event_col: Middle event time column.
        mid_known_col: Middle known time column, or None.
        leaf: Leaf table.
        leaf_link_col: Leaf column referencing the middle identifier.
        leaf_event_col: Leaf event time column.
        leaf_known_col: Leaf known time column, or None.
        leaf_value_col: Leaf value column, or None.
        inner_primitive: Primitive over leaf rows per middle row.
        outer_primitive: Primitive over the inner values.
        inner_window_days: Window on leaf events, or None.
        outer_window_days: Window on middle events, or None.
        lag_days: Gap between cutoff and prediction moment.
        delta_rec_days: Recording delay fallback.

    Returns:
        One value per target row.
    """
    mid_rows = records(mid)
    leaf_rows = records(leaf)
    out: list[float] = []
    for target in records(targets):
        cutoff = float(target[moment_col]) - lag_days
        mids = scope_rows(
            mid_rows,
            mid_key_col,
            mid_event_col,
            mid_known_col,
            delta_rec_days,
            target[key_col],
            cutoff,
            outer_window_days,
        )
        pseudo: list[Mapping[str, Any]] = []
        for m in mids:
            leaves = scope_rows(
                leaf_rows,
                leaf_link_col,
                leaf_event_col,
                leaf_known_col,
                delta_rec_days,
                m[mid_id_col],
                cutoff,
                inner_window_days,
            )
            inner = primitive(
                inner_primitive,
                leaves,
                leaf_value_col,
                leaf_event_col,
                leaf_known_col,
                delta_rec_days,
                cutoff,
            )
            pseudo.append(
                {
                    "event": float(m[mid_event_col]),
                    "known": known_time(
                        m, mid_event_col, mid_known_col, delta_rec_days
                    ),
                    "value": inner,
                }
            )
        out.append(
            primitive(
                outer_primitive,
                pseudo,
                "value",
                "event",
                "known",
                delta_rec_days,
                cutoff,
            )
        )
    return out


def lookup_feature(
    targets: pd.DataFrame,
    key_col: str,
    moment_col: str,
    parent: pd.DataFrame,
    parent_key_col: str,
    event_col: str,
    known_col: str | None,
    value_col: str,
    lag_days: float,
    delta_rec_days: float,
) -> list[float]:
    """Latest visible parent value per target (``last``, no window).

    Args:
        targets: Target frame.
        key_col: Entity column of the targets.
        moment_col: Prediction moment column (float days).
        parent: Parent table.
        parent_key_col: Entity column of the parent.
        event_col: Parent event time column.
        known_col: Parent known time column, or None.
        value_col: Parent value column.
        lag_days: Gap between cutoff and prediction moment.
        delta_rec_days: Recording delay fallback.

    Returns:
        One value per target row.
    """
    return aggregate_feature(
        targets,
        key_col,
        moment_col,
        parent,
        parent_key_col,
        event_col,
        known_col,
        value_col,
        "last",
        None,
        lag_days,
        delta_rec_days,
    )
