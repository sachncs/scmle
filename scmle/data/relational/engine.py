"""Point-in-time joins and windowed aggregates over related tables.

Each target row has a cutoff, its moment less a declared lag. A related row
is visible to it when the row's key matches, its event time is at or before
the cutoff and its known time (see `schema.Table`) is at or before the
cutoff, so no feature ever reads a row from the target row's future or one
recorded after it. A row without an event time is never visible in a timed
table, since nothing places it before the cutoff.

* A `many_to_one` link joins each target row to the latest visible version
  of the row its key names (latest event time, then latest known time, then
  later stored row): its columns are added as `<link>__<column>`, nulls of
  that version included. Later links may join from those columns.
* A `one_to_many` link feeds `Aggregate`s: a primitive over the visible rows
  whose event time lies in `(cutoff - window, cutoff]`, or over all visible
  rows without a window.

Primitives: `count` (0 without rows), `sum` (0 without values), `mean`,
`min`, `max`, `last` (the value of the latest row that has one) and `trend`
(the least-squares slope of the value on event time, per day); each is NaN
without the values it needs. Times are text (ISO 8601), datetimes or plain
numbers, taken as days.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses

import numpy
import pandas

from scmle.contracts import errors
from scmle.data.relational import schema
from scmle.data.relational import stores

COUNT = "count"
SUM = "sum"
MEAN = "mean"
MIN = "min"
MAX = "max"
LAST = "last"
TREND = "trend"
PRIMITIVES = (COUNT, SUM, MEAN, MIN, MAX, LAST, TREND)
# Primitives that read no value column.
FREE = (COUNT,)
# Primitives that need numbers; `last` takes any value.
NUMERIC = (SUM, MEAN, MIN, MAX, TREND)
# A least-squares line needs two points with distinct times: one point
# fixes no slope, so fewer rows give NaN (a definition, not a tuning).
LINE = 2
# Windows of the default grid: a week, a month, a quarter, and all rows.
WINDOWS: tuple[float | None, ...] = (7.0, 30.0, 90.0, None)
SECONDS = 86_400.0
EPOCH = pandas.Timestamp("1970-01-01")
# Working columns; a table holding one of these names is refused.
ROW = "relational:row"
CUTOFF = "relational:cutoff"
EVENT = "relational:event"
KNOWN = "relational:known"
RESERVED = (ROW, CUTOFF, EVENT, KNOWN)


def days(values: pandas.Series) -> numpy.ndarray:
    """Times as float days since 1970-01-01.

    Numbers are taken as days already; datetimes are measured from the
    epoch, timezone-aware ones in UTC; text is ISO 8601 per value, else
    one format pandas infers for the column. Missing values become NaN.

    Args:
        values: The column of times.

    Returns:
        A float64 array, one entry per value.

    Raises:
        InputError: If text times have no single reading.
    """
    if pandas.api.types.is_bool_dtype(values):
        raise errors.InputError(f"column {values.name!r} holds booleans")
    if pandas.api.types.is_numeric_dtype(values):
        return values.to_numpy(dtype="float64")
    stamps = values
    if not pandas.api.types.is_datetime64_any_dtype(values):
        stamps = pandas.to_datetime(
            values, format="ISO8601", utc=True, errors="coerce"
        )
        if bool((stamps.isna() & values.notna()).any()):
            stamps = pandas.to_datetime(values, utc=True, errors="coerce")
        unread = stamps.isna() & values.notna()
        if bool(unread.any()):
            first = values[unread].iloc[0]
            raise errors.InputError(
                f"column {values.name!r} holds {first!r}, not a time"
            )
    if getattr(stamps.dt, "tz", None) is not None:
        stamps = stamps.dt.tz_convert("UTC").dt.tz_localize(None)
    seconds = (stamps - EPOCH).dt.total_seconds()
    return seconds.to_numpy(dtype="float64") / SECONDS


@dataclasses.dataclass(frozen=True, slots=True)
class Aggregate:
    """One windowed aggregate of a `one_to_many` link.

    Attributes:
        link: The link's name.
        primitive: One of `PRIMITIVES`.
        value: Value column, or None for `count`.
        window: Window in days, or None for every visible row.
    """

    link: str
    primitive: str
    value: str | None = None
    window: float | None = None

    def __post_init__(self) -> None:
        """Validates the primitive, value and window.

        Raises:
            ConfigError: If the primitive is unknown, a value is missing or
                given where none is read, or the window is not positive.
        """
        if self.primitive not in PRIMITIVES:
            raise errors.ConfigError(
                f"unknown primitive {self.primitive!r}; use {PRIMITIVES}"
            )
        if (self.value is None) != (self.primitive in FREE):
            raise errors.ConfigError(
                f"{self.primitive} takes "
                f"{'no' if self.primitive in FREE else 'a'} value column"
            )
        if self.window is not None and not self.window > 0:
            raise errors.ConfigError(f"window {self.window} is not positive")

    @property
    def name(self) -> str:
        """The feature's column name.

        Returns:
            `<link>__<primitive>__<value or *>__<window>d`, or `__all` for
            no window.
        """
        span = "all" if self.window is None else f"{self.window:g}d"
        return f"{self.link}__{self.primitive}__{self.value or '*'}__{span}"


def grid(
    link: str,
    values: Sequence[str],
    primitives: Sequence[str] = PRIMITIVES,
    windows: Sequence[float | None] = WINDOWS,
) -> tuple[Aggregate, ...]:
    """Every primitive of every value over every window.

    Args:
        link: The `one_to_many` link's name.
        values: Value columns.
        primitives: Primitives; `count` is made once per window.
        windows: Windows in days; None for every visible row.

    Returns:
        The aggregates, window by window.
    """
    found: list[Aggregate] = []
    for window in windows:
        for primitive in primitives:
            picks: Sequence[str | None] = (
                [None] if primitive in FREE else values
            )
            found.extend(
                Aggregate(link, primitive, pick, window) for pick in picks
            )
    return tuple(found)


def prepare(frame: pandas.DataFrame, table: schema.Table) -> pandas.DataFrame:
    """A table with its event and known days added.

    Args:
        frame: The table's rows in stored order.
        table: Its declaration.

    Returns:
        The rows with `EVENT` and `KNOWN` (minus infinity for a static
        table; event plus delay where no recorded time is given).

    Raises:
        InputError: If the table holds a reserved column name.
    """
    clash = [c for c in RESERVED if c in frame.columns]
    if clash:
        raise errors.InputError(
            f"{table.name!r} holds reserved columns {clash}"
        )
    rows = frame.reset_index(drop=True)
    if table.event is None:
        return rows.assign(**{EVENT: -numpy.inf, KNOWN: -numpy.inf})
    event = days(rows[table.event])
    known = event + table.delay
    if table.known is not None:
        recorded = days(rows[table.known])
        known = numpy.where(numpy.isnan(recorded), known, recorded)
    return rows.assign(**{EVENT: event, KNOWN: known})


def visible(
    asked: pandas.DataFrame, rows: pandas.DataFrame, link: schema.Link
) -> pandas.DataFrame:
    """Pairs each target row with the related rows visible at its cutoff.

    Args:
        asked: `ROW`, `CUTOFF` and the link's target column.
        rows: The related table from `prepare`.
        link: The link.

    Returns:
        One row per (target row, visible related row), in target then
        stored order, with every related column plus `ROW` and `CUTOFF`.

    Raises:
        InputError: If the target and key columns are not both numbers or
            both not numbers.
    """
    left = pandas.DataFrame(
        {
            link.key: asked[link.target].to_numpy(),
            ROW: asked[ROW].to_numpy(),
            CUTOFF: asked[CUTOFF].to_numpy(),
        }
    )
    sides = (left[link.key], rows[link.key])
    if len({pandas.api.types.is_numeric_dtype(s) for s in sides}) > 1:
        raise errors.InputError(
            f"link {link.name!r} matches {link.target!r} "
            f"({sides[0].dtype}) to {link.key!r} ({sides[1].dtype})"
        )
    left = left.loc[left[link.key].notna()]
    pairs = left.merge(rows, on=link.key, how="inner", sort=False)
    keep = (pairs[EVENT] <= pairs[CUTOFF]) & (pairs[KNOWN] <= pairs[CUTOFF])
    return pairs.loc[keep].reset_index(drop=True)


def latest(pairs: pandas.DataFrame) -> pandas.DataFrame:
    """The latest visible row per target row.

    Args:
        pairs: Output of `visible`, in stored order within a target row.

    Returns:
        One row per target row that sees any, by event time, then known
        time, then stored order; indexed by `ROW`.
    """
    ordered = pairs.sort_values([ROW, EVENT, KNOWN], kind="stable")
    return ordered.drop_duplicates(ROW, keep="last").set_index(ROW)


def trend(rows: pandas.DataFrame, value: str) -> pandas.Series:
    """The least-squares slope of a value on event time per target row.

    Args:
        rows: Rows with `ROW`, `EVENT` and a float value.
        value: The value column.

    Returns:
        The slope per day, NaN under `LINE` rows or with one event time.
    """
    group = rows[ROW]
    x, y = rows[EVENT], rows[value]
    dx = x - x.groupby(group).transform("mean")
    dy = y - y.groupby(group).transform("mean")
    sxy = (dx * dy).groupby(group).sum()
    sxx = (dx * dx).groupby(group).sum()
    spread = x.groupby(group).max() - x.groupby(group).min()
    size = rows.groupby(ROW).size()
    return (sxy / sxx.where(spread > 0.0)).where(size >= LINE)


FOLDS: Mapping[str, Callable[[pandas.DataFrame, str], pandas.Series]] = {
    SUM: lambda rows, value: rows.groupby(ROW)[value].sum(),
    MEAN: lambda rows, value: rows.groupby(ROW)[value].mean(),
    MIN: lambda rows, value: rows.groupby(ROW)[value].min(),
    MAX: lambda rows, value: rows.groupby(ROW)[value].max(),
    LAST: lambda rows, value: latest(rows)[value],
    TREND: trend,
}


def compute(
    pairs: pandas.DataFrame, aggregate: Aggregate, count: int
) -> pandas.Series:
    """One aggregate per target row.

    Args:
        pairs: Output of `visible` for the aggregate's link.
        aggregate: The aggregate.
        count: Number of target rows.

    Returns:
        A series indexed `0..count-1`: float, or the value's own type for
        `last` of a non-numeric column.

    Raises:
        InputError: If the value column is absent, or not numeric for a
            primitive that needs numbers.
    """
    index = pandas.RangeIndex(count)
    rows = pairs
    if aggregate.window is not None:
        rows = rows.loc[rows[EVENT] > rows[CUTOFF] - aggregate.window]
    if aggregate.value is None:
        return rows.groupby(ROW).size().reindex(index, fill_value=0) * 1.0
    value = aggregate.value
    if value not in rows.columns:
        raise errors.InputError(f"{aggregate.link!r} has no column {value!r}")
    rows = rows.loc[rows[value].notna()]
    if aggregate.primitive in NUMERIC:
        column = rows[value]
        if pandas.api.types.is_bool_dtype(column) or not (
            pandas.api.types.is_numeric_dtype(column)
        ):
            raise errors.InputError(
                f"{aggregate.primitive} needs numbers; {aggregate.link}."
                f"{value} is {column.dtype}"
            )
        rows = rows.assign(**{value: column.astype("float64")})
    found = FOLDS[aggregate.primitive](rows, value)
    empty = 0.0 if aggregate.primitive == SUM else numpy.nan
    filled = found.reindex(index, fill_value=empty)
    return filled if aggregate.primitive == LAST else filled.astype(float)


def ask(targets: pandas.DataFrame, moment: str, lag: float) -> pandas.DataFrame:
    """The target rows with their positions and cutoffs.

    Args:
        targets: One row per prediction case.
        moment: Column of each row's prediction moment.
        lag: Days between the cutoff and the moment.

    Returns:
        The rows indexed from zero, with `ROW` and `CUTOFF` added.

    Raises:
        ConfigError: If the lag is negative.
        InputError: If a row has no moment.
    """
    if not lag >= 0:
        raise errors.ConfigError(f"lag {lag} must be non-negative")
    asked = targets.reset_index(drop=True)
    cutoff = days(asked[moment]) - lag
    missing = int(numpy.isnan(cutoff).sum())
    if missing:
        raise errors.InputError(
            f"column {moment!r} has {missing} missing moments"
        )
    return asked.assign(**{ROW: numpy.arange(len(asked)), CUTOFF: cutoff})


def agree(left: pandas.Series, right: pandas.Series, tolerance: float) -> bool:
    """Whether two feature columns hold the same values.

    Args:
        left: One column.
        right: The other, of the same length and row order.
        tolerance: Largest relative and absolute difference of two numbers.

    Returns:
        Numbers equal within `tolerance`, missing where both are missing;
        any other values equal one by one or missing in both.
    """
    if all(pandas.api.types.is_numeric_dtype(c) for c in (left, right)):
        return bool(
            numpy.allclose(
                left.to_numpy(dtype="float64"),
                right.to_numpy(dtype="float64"),
                rtol=tolerance,
                atol=tolerance,
                equal_nan=True,
            )
        )
    same = left.to_numpy(dtype=object) == right.to_numpy(dtype=object)
    absent = left.isna().to_numpy() & right.isna().to_numpy()
    return bool((same | absent).all())


class Engine:
    """Features of target rows from declared, related tables.

    Attributes:
        schema: The declared tables and links.
        frames: Each declared table, prepared with its event and known
            days.
    """

    def __init__(
        self,
        declared: schema.Schema,
        frames: Mapping[str, pandas.DataFrame],
    ) -> None:
        """Checks the tables against the schema and prepares them.

        Args:
            declared: The schema.
            frames: Each declared table by name, in stored order.
        """
        declared.check(frames)
        self.schema = declared
        self.frames = {
            t.name: prepare(frames[t.name], t)
            for t in declared.tables
            if t.name in frames
        }

    @classmethod
    def open(cls, declared: schema.Schema, store: stores.Store) -> "Engine":
        """Reads every declared table from a store.

        Args:
            declared: The schema.
            store: Where the tables live.

        Returns:
            The engine.
        """
        return cls(
            declared, {t.name: store.read(t.name) for t in declared.tables}
        )

    def features(
        self,
        targets: pandas.DataFrame,
        moment: str,
        aggregates: Sequence[Aggregate] = (),
        lag: float = 0.0,
    ) -> pandas.DataFrame:
        """The joined columns and aggregates of every target row.

        Args:
            targets: One row per prediction case, holding `moment` and the
                target column of each first-level link.
            moment: Column of each row's prediction moment.
            aggregates: Aggregates of `one_to_many` links.
            lag: Days between the cutoff and the moment.

        Returns:
            Indexed like `targets`: the columns of every `many_to_one`
            link, in link order, then one column per aggregate.

        Raises:
            ConfigError: If an aggregate names a link that is not
                `one_to_many`, or the lag is negative.
            InputError: If a link's target column is absent, or a target
                row has no moment: without a cutoff nothing places a related
                row before it, and a count of 0 would claim it had none.
        """
        asked = ask(targets, moment, lag)
        added: dict[str, pandas.Series] = {}
        for link in self.schema.links:
            if link.target not in asked.columns:
                raise errors.InputError(
                    f"link {link.name!r} needs column {link.target!r}"
                )
            if link.cardinality != schema.ONE:
                continue
            found = latest(visible(asked, self.frames[link.table], link))
            for column in self.frames[link.table].columns:
                if column in RESERVED or column == link.key:
                    continue
                joined = found[column].reindex(asked[ROW])
                name = f"{link.name}__{column}"
                added[name] = pandas.Series(joined.to_numpy(), name=name)
                asked[name] = added[name].to_numpy()
        pairs: dict[str, pandas.DataFrame] = {}
        for aggregate in aggregates:
            link = self.schema.link(aggregate.link)
            if link.cardinality != schema.MANY:
                raise errors.ConfigError(
                    f"aggregate {aggregate.name!r} needs a one_to_many link"
                )
            if link.name not in pairs:
                pairs[link.name] = visible(asked, self.frames[link.table], link)
            added[aggregate.name] = compute(
                pairs[link.name], aggregate, len(asked)
            ).rename(aggregate.name)
        frame = pandas.DataFrame(added, index=pandas.RangeIndex(len(asked)))
        return frame.set_axis(targets.index)
