"""Evaluates a proposed feature through the as-of views.

Every value is computed from the rows a view shows at each target row's
cutoff; nothing here reads a table directly or looks at the target.
"""

from collections.abc import Sequence
from typing import Any

import numpy
import pandas

from scmle.assurance.ingest import asof_view
from scmle.assurance.ingest.synthesize.grammar import COUNT
from scmle.assurance.ingest.synthesize.grammar import Feature
from scmle.assurance.ingest.synthesize.grammar import LAST
from scmle.assurance.ingest.synthesize.grammar import MAX
from scmle.assurance.ingest.synthesize.grammar import MEAN
from scmle.assurance.ingest.synthesize.grammar import MIN
from scmle.assurance.ingest.synthesize.grammar import Nest
from scmle.assurance.ingest.synthesize.grammar import NUNIQUE
from scmle.assurance.ingest.synthesize.grammar import PRIMITIVES
from scmle.assurance.ingest.synthesize.grammar import SLOPE
from scmle.assurance.ingest.synthesize.grammar import Source
from scmle.assurance.ingest.synthesize.grammar import STD
from scmle.assurance.ingest.synthesize.grammar import SUM
from scmle.assurance.ingest.synthesize.grammar import TIME_SINCE_LAST
from scmle.contracts import errors

MIN_SLOPE_ROWS = 2


def outcomes(
    targets: pandas.DataFrame, key: str, moment: str, lag_days: float, view: Any
) -> pandas.DataFrame:
    """Builds the target frame a view is asked about.

    Args:
        targets: One row per prediction case.
        key: Column of `targets` joined to the view's key.
        moment: Column holding each case's prediction moment.
        lag_days: Gap between the cutoff and the moment.
        view: The as-of view the frame is for.

    Returns:
        A frame with the view's key column, `TARGET_COLUMN` (the row
        position) and `CUTOFF_COLUMN` (moment minus lag, in days).
    """
    cutoff = asof_view.convert(targets[moment]) - lag_days
    return pandas.DataFrame(
        {
            view.key: targets[key].to_numpy(),
            asof_view.TARGET_COLUMN: numpy.arange(len(targets)),
            asof_view.CUTOFF_COLUMN: cutoff,
        }
    )


def fold(
    rows: pandas.DataFrame,
    primitive: str,
    value: str | None,
    window: float | None,
    count: int,
) -> pandas.Series:
    """One primitive per target row over visible rows.

    Args:
        rows: Visible rows with `TARGET_COLUMN`, `CUTOFF_COLUMN`,
            `EVENT_COLUMN`, `KNOWN_COLUMN` and the value column.
        primitive: One of `PRIMITIVES`.
        value: Value column; required unless the primitive is value-free.
        window: Window in days (rows older than `cutoff - window` leave), or
            None.
        count: Number of target rows; the result has one entry for each.

    Returns:
        A float (or object, for `last` of non-numeric values) series indexed
        by target row `0..count-1`.

    Raises:
        ValueError: If the primitive is unknown, or it needs a value column
            and `value` is None.
    """
    if primitive not in PRIMITIVES:
        raise errors.InputError(f"unknown primitive: {primitive}")
    if window is not None:
        rows = rows.loc[
            rows[asof_view.EVENT_COLUMN]
            > rows[asof_view.CUTOFF_COLUMN] - window
        ]
    index = pandas.RangeIndex(count)
    label = asof_view.TARGET_COLUMN
    if primitive == COUNT:
        return (
            rows.groupby(label)
            .size()
            .reindex(index, fill_value=0)
            .astype(float)
        )
    if primitive == TIME_SINCE_LAST:
        latest = rows.groupby(label)[asof_view.EVENT_COLUMN].max()
        cutoff = rows.groupby(label)[asof_view.CUTOFF_COLUMN].first()
        return (cutoff - latest).reindex(index).astype(float)
    if value is None:
        raise errors.InputError(f"{primitive} needs a value column")
    if primitive == LAST:
        valid = rows.loc[rows[value].notna()]
        ordered = valid.sort_values(
            [asof_view.EVENT_COLUMN, asof_view.KNOWN_COLUMN], kind="stable"
        )
        return ordered.groupby(label)[value].last().reindex(index)
    valid = rows.loc[rows[value].notna()]
    if primitive == NUNIQUE:
        grouped = valid.groupby(label)[value]
        return grouped.nunique().reindex(index, fill_value=0).astype(float)
    return aggregate(valid, primitive, value, index)


def aggregate(
    valid: pandas.DataFrame,
    primitive: str,
    value: str,
    index: pandas.RangeIndex,
) -> pandas.Series:
    """One numeric primitive per target row over rows that have a value.

    Args:
        valid: Visible rows whose value is present, with `TARGET_COLUMN`
            and the value column.
        primitive: One of `sum`, `mean`, `min`, `max`, `std` or `slope`.
        value: Value column, converted to float here.
        index: Target rows `0..count-1`.

    Returns:
        A float series indexed by target row.

    Raises:
        ValueError: If the primitive is not one of those listed.
    """
    label = asof_view.TARGET_COLUMN
    numbers = valid.assign(**{value: valid[value].astype(float)})
    grouped = numbers.groupby(label)[value]
    if primitive == SUM:
        return grouped.sum().reindex(index, fill_value=0.0).astype(float)
    if primitive == MEAN:
        return grouped.mean().reindex(index).astype(float)
    if primitive == MIN:
        return grouped.min().reindex(index).astype(float)
    if primitive == MAX:
        return grouped.max().reindex(index).astype(float)
    if primitive == STD:
        return grouped.std(ddof=1).reindex(index).astype(float)
    if primitive == SLOPE:
        return slope(numbers, value).reindex(index).astype(float)
    raise errors.InputError(f"unknown primitive: {primitive}")


def slope(rows: pandas.DataFrame, value: str) -> pandas.Series:
    """Fits the OLS slope of a value on event time per target row.

    Args:
        rows: Visible rows with `TARGET_COLUMN`, `EVENT_COLUMN` and the
            value column.
        value: Numeric value column.

    Returns:
        The slope per target row, in value units per day; NaN for a target
        with fewer than two rows or no spread in event time.
    """
    label = asof_view.TARGET_COLUMN
    x = rows[asof_view.EVENT_COLUMN]
    y = rows[value]
    xbar = x.groupby(rows[label]).transform("mean")
    ybar = y.groupby(rows[label]).transform("mean")
    dx = x - xbar
    sxy = (dx * (y - ybar)).groupby(rows[label]).sum()
    sxx = (dx * dx).groupby(rows[label]).sum()
    size = rows.groupby(label).size()
    spread = x.groupby(rows[label]).max() - x.groupby(rows[label]).min()
    result = sxy / sxx.where(spread > 0.0)
    return result.where(size >= MIN_SLOPE_ROWS)


def evaluate(
    spec: Feature,
    targets: pandas.DataFrame,
    moment: str,
    sources: Sequence[Source],
    nests: Sequence[Nest],
    lag_days: float,
) -> pandas.Series:
    """Computes one feature for every target row.

    Args:
        spec: The feature.
        targets: One row per prediction case.
        moment: Column holding each case's prediction moment.
        sources: Related tables as views.
        nests: Depth-2 routes.
        lag_days: Gap between the cutoff and the moment.

    Returns:
        The feature values, one per target row in order.

    Raises:
        ScmleError: If a `nested` spec lacks its nest or inner primitive.
    """
    source = sources[spec.source]
    view = source.view
    count = len(targets)
    asked = outcomes(targets, source.anchor, moment, lag_days, view)
    rows = view.visible(asked)
    if spec.kind in ("aggregate", "lookup"):
        return fold(rows, spec.primitive, spec.value, spec.window, count)
    if spec.nest is None or spec.inner is None:
        raise errors.ScmleError("nested feature lacks nest or inner")
    nest = nests[spec.nest]
    if spec.window is not None:
        rows = rows.loc[
            rows[asof_view.EVENT_COLUMN]
            > rows[asof_view.CUTOFF_COLUMN] - spec.window
        ]
    pairs = rows.reset_index(drop=True)
    pair_targets = pandas.DataFrame(
        {
            nest.leaf.key: pairs[nest.link].to_numpy(),
            asof_view.TARGET_COLUMN: numpy.arange(len(pairs)),
            asof_view.CUTOFF_COLUMN: pairs[asof_view.CUTOFF_COLUMN].to_numpy(),
        }
    )
    leaves = nest.leaf.visible(pair_targets)
    inner = fold(leaves, spec.inner, spec.value, spec.window, len(pairs))
    outer_rows = pairs.assign(inner_value=inner.to_numpy())
    return fold(outer_rows, spec.primitive, "inner_value", None, count)
