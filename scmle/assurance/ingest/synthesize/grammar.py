"""The declared feature grammar: primitives, windows, sources and proposals.

Proposals are made from the declared sources, nests and windows alone,
before any row is read, so the family size is fixed in advance.
"""

from collections.abc import Sequence
import dataclasses

import pandas

from scmle.assurance.ingest import asof_view
from scmle.contracts import errors

COUNT = "count"


SUM = "sum"


MEAN = "mean"


MIN = "min"


MAX = "max"


STD = "std"


NUNIQUE = "nunique"


LAST = "last"


TIME_SINCE_LAST = "time_since_last"


SLOPE = "slope"


PRIMITIVES = (
    COUNT,
    SUM,
    MEAN,
    MIN,
    MAX,
    STD,
    NUNIQUE,
    LAST,
    TIME_SINCE_LAST,
    SLOPE,
)


VALUE_FREE = (COUNT, TIME_SINCE_LAST)


NUMERIC_ONLY = (SUM, MEAN, MIN, MAX, STD, SLOPE)


WINDOWS_DAYS = (7.0, 30.0, 90.0)


NESTED_INNER = (COUNT, SUM, MEAN)


NESTED_OUTER = (MEAN, MAX, SUM)


CHILD = "child"


PARENT = "parent"


@dataclasses.dataclass(frozen=True, slots=True)
class Source:
    """A related table and how the target rows reach it.

    Attributes:
        view: The as-of view of the table; the only way to read it.
        anchor: Column of the target frame joined to `view.key`.
        role: `child` (one-to-many, aggregated) or `parent` (many-to-one
            lookup of the latest visible version).
    """

    view: asof_view.View
    anchor: str
    role: str = CHILD

    def __post_init__(self) -> None:
        """Validates the role.

        Raises:
            ValueError: If `role` is neither `child` nor `parent`.
        """
        if self.role not in (CHILD, PARENT):
            raise errors.InputError("role must be child or parent")


@dataclasses.dataclass(frozen=True, slots=True)
class Nest:
    """A depth-2 route: target to `mid` rows, each to its `leaf` rows.

    Attributes:
        mid: The child table of the target.
        leaf: As-of view of the grandchild table, joined on its `key`.
        link: Column of the mid table the leaf rows link to.
    """

    mid: Source
    leaf: asof_view.View
    link: str


@dataclasses.dataclass(frozen=True, slots=True)
class Feature:
    """One feature of the grammar.

    Attributes:
        name: Column name of the feature.
        kind: `aggregate`, `lookup` or `nested`.
        source: Index into the sources (the mid table for `nested`).
        nest: Index into the nests, for `nested`.
        primitive: Primitive over the table (outer primitive for `nested`).
        inner: Inner primitive of a `nested` feature.
        value: Value column, or None for a value-free primitive.
        window: Window in days, or None for all visible rows.
    """

    name: str
    kind: str
    source: int
    primitive: str
    value: str | None
    window: float | None
    nest: int | None = None
    inner: str | None = None

    def expression(self) -> str:
        """Writes the feature as text, for the manifest.

        Returns:
            The primitive, value column and window as a readable expression.
        """
        window = "all" if self.window is None else f"{self.window:g}d"
        if self.kind == "nested":
            return (
                f"{self.primitive}(per-row {self.inner}({self.value}) "
                f"in {window})"
            )
        return f"{self.primitive}({self.value or '*'}) over {window}"


def caption(window: float | None) -> str:
    """Writes a window as text for feature names.

    Args:
        window: Window in days, or None for all visible rows.

    Returns:
        `all` for None, otherwise the days followed by `d`, e.g. `30d`.
    """
    return "all" if window is None else f"{window:g}d"


def sift(source: Source) -> tuple[list[str], list[str]]:
    """Finds the value columns of a source, excluding structure.

    Key, time, reserved and identifier-like (`id`, `*_id`) columns are
    skipped.

    Args:
        source: The related table.

    Returns:
        The numeric non-boolean columns and the other columns, each in
        table order.
    """
    view = source.view
    frame = view.reader()
    skip = {
        view.key,
        view.time,
        view.rule.recorded,
        *asof_view.RESERVED,
    }
    numeric, other = [], []
    for name in view.columns:
        if name in skip or name.lower() == "id" or name.lower().endswith("_id"):
            continue
        if pandas.api.types.is_numeric_dtype(frame[name]) and not (
            pandas.api.types.is_bool_dtype(frame[name])
        ):
            numeric.append(name)
        else:
            other.append(name)
    return numeric, other


def roll(
    index: int,
    table: str,
    numeric: Sequence[str],
    other: Sequence[str],
    windows: Sequence[float | None],
) -> list[Feature]:
    """Lists the windowed aggregates of a child source.

    Args:
        index: Position of the source.
        table: Name of the source's view.
        numeric: The numeric candidate columns.
        other: The other candidate columns.
        windows: Windows in days; None means all visible rows.

    Returns:
        For each window, the value-free primitives, the numeric primitives
        of each numeric column and the distinct count of every column.
    """
    specs: list[Feature] = []
    for window in windows:
        label = caption(window)
        specs.extend(
            Feature(
                f"{table}__{primitive}__{label}",
                "aggregate",
                index,
                primitive,
                None,
                window,
            )
            for primitive in VALUE_FREE
        )
        specs.extend(
            Feature(
                f"{table}__{primitive}__{column}__{label}",
                "aggregate",
                index,
                primitive,
                column,
                window,
            )
            for column in numeric
            for primitive in (*NUMERIC_ONLY, LAST)
        )
        specs.extend(
            Feature(
                f"{table}__{NUNIQUE}__{column}__{label}",
                "aggregate",
                index,
                NUNIQUE,
                column,
                window,
            )
            for column in (*numeric, *other)
        )
    return specs


def nest(
    sources: Sequence[Source],
    nests: Sequence[Nest],
    windows: Sequence[float | None],
) -> list[Feature]:
    """Lists the depth-2 aggregates of the nests.

    Args:
        sources: Related tables as views; each nest's `mid` is one of them.
        nests: Depth-2 routes.
        windows: Windows in days; None means all visible rows.

    Returns:
        For each nest and window, every outer primitive over every inner
        primitive of the leaf's numeric columns (of no column, for a count).
    """
    specs: list[Feature] = []
    for position, nest in enumerate(nests):
        mid_index = next(k for k, s in enumerate(sources) if s is nest.mid)
        leaf = Source(nest.leaf, nest.leaf.key)
        numeric, others = sift(leaf)
        del others
        for window in windows:
            label = caption(window)
            for inner in NESTED_INNER:
                for outer in NESTED_OUTER:
                    picks: list[str | None] = (
                        [None] if inner == COUNT else list(numeric)
                    )
                    specs.extend(
                        Feature(
                            f"{nest.mid.view.name}__{outer}__of__"
                            f"{nest.leaf.name}__{inner}__"
                            f"{pick or '*'}__{label}",
                            "nested",
                            mid_index,
                            outer,
                            pick,
                            window,
                            nest=position,
                            inner=inner,
                        )
                        for pick in picks
                    )
    return specs


def propose(
    sources: Sequence[Source],
    nests: Sequence[Nest] = (),
    windows: Sequence[float | None] = WINDOWS_DAYS,
) -> list[Feature]:
    """Lists every feature of the grammar for these sources.

    Args:
        sources: Related tables as views; a `parent` source gives lookups,
            a `child` source gives windowed aggregates.
        nests: Depth-2 routes; each `mid` must be one of `sources`.
        windows: Windows in days; None means all visible rows.

    Returns:
        The feature specifications in a fixed order.
    """
    specs: list[Feature] = []
    for index, source in enumerate(sources):
        table = source.view.name
        numeric, other = sift(source)
        if source.role == PARENT:
            specs.extend(
                Feature(
                    f"{table}__lookup__{column}",
                    "lookup",
                    index,
                    LAST,
                    column,
                    None,
                )
                for column in (*numeric, *other)
            )
            continue
        specs.extend(roll(index, table, numeric, other, windows))
    specs.extend(nest(sources, nests, windows))
    return specs
