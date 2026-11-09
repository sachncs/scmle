"""Feature synthesis through the as-of view.

A fixed grammar, Deep Feature Synthesis style: ten primitives over the rows of
a related table in declared windows, parent-attribute lookups, and depth-2
nesting (an aggregate of per-row aggregates of a grandchild table). The
grammar, the windows and the cap `B_feat` are declared; the family size is
fixed before exploration and every kept feature is registered in the ledger
as a `synthesized` hypothesis, so the multiplicity the ledger corrects for
is the multiplicity that exists.

This module never touches a table. It receives `AsOfView` objects and asks
them for the rows visible at each target row's cutoff; `tests` hold it to
that by checking its imports. Label-free filters (constant, mostly missing,
duplicate of an earlier feature) run first; they use no labels, and nothing
here reads the target.
"""

from collections.abc import Sequence
import dataclasses
import datetime
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.data.admissibility import provenance
from scmle.data.ingest import asof_view
from scmle.data.ledger import ledger as ledger_module

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
FAMILY = "synthesized"
STATISTIC = "spearman"
CHILD = "child"
PARENT = "parent"


class MultiplicityError(ValueError):
    """The declared family size differs from the registered hypotheses."""


@dataclasses.dataclass(frozen=True, slots=True)
class Source:
    """A related table and how the target rows reach it.

    Attributes:
        view: The as-of view of the table; the only way to read it.
        target_key: Column of the target frame joined to `view.key`.
        role: `child` (one-to-many, aggregated) or `parent` (many-to-one
            lookup of the latest visible version).
    """

    view: asof_view.AsOfView
    target_key: str
    role: str = CHILD

    def __post_init__(self) -> None:
        """Validates the role.

        Raises:
            ValueError: If `role` is neither `child` nor `parent`.
        """
        if self.role not in (CHILD, PARENT):
            raise ValueError("role must be child or parent")


@dataclasses.dataclass(frozen=True, slots=True)
class Nest:
    """A depth-2 route: target to `mid` rows, each to its `leaf` rows.

    Attributes:
        mid: The child table of the target.
        leaf: As-of view of the grandchild table, joined on its `key`.
        mid_id: Column of the mid table the leaf rows link to.
    """

    mid: Source
    leaf: asof_view.AsOfView
    mid_id: str


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One feature of the grammar.

    Attributes:
        name: Column name of the feature.
        kind: `aggregate`, `lookup` or `nested`.
        source: Index into the sources (the mid table for `nested`).
        nest: Index into the nests, for `nested`.
        primitive: Primitive over the table (outer primitive for `nested`).
        inner: Inner primitive of a `nested` feature.
        value: Value column, or None for a value-free primitive.
        window_days: Window in days, or None for all visible rows.
    """

    name: str
    kind: str
    source: int
    primitive: str
    value: str | None
    window_days: float | None
    nest: int | None = None
    inner: str | None = None

    def expression(self) -> str:
        """Writes the feature as text, for the manifest.

        Returns:
            The primitive, value column and window as a readable expression.
        """
        window = "all" if self.window_days is None else f"{self.window_days:g}d"
        if self.kind == "nested":
            return (
                f"{self.primitive}(per-row {self.inner}({self.value}) "
                f"in {window})"
            )
        return f"{self.primitive}({self.value or '*'}) over {window}"


def window_label(window: float | None) -> str:
    """Writes a window as text for feature names.

    Args:
        window: Window in days, or None for all visible rows.

    Returns:
        `all` for None, otherwise the days followed by `d`, e.g. `30d`.
    """
    return "all" if window is None else f"{window:g}d"


def candidate_columns(source: Source) -> tuple[list[str], list[str]]:
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
        view.event_time,
        view.rule.known_column,
        *asof_view.RESERVED,
    }
    numeric, other = [], []
    for name in view.columns:
        if name in skip or name.lower() == "id" or name.lower().endswith("_id"):
            continue
        if pd.api.types.is_numeric_dtype(frame[name]) and not (
            pd.api.types.is_bool_dtype(frame[name])
        ):
            numeric.append(name)
        else:
            other.append(name)
    return numeric, other


def propose(
    sources: Sequence[Source],
    nests: Sequence[Nest] = (),
    windows: Sequence[float | None] = WINDOWS_DAYS,
) -> list[FeatureSpec]:
    """Lists every feature of the grammar for these sources.

    Args:
        sources: Related tables as views; a `parent` source gives lookups,
            a `child` source gives windowed aggregates.
        nests: Depth-2 routes; each `mid` must be one of `sources`.
        windows: Windows in days; None means all visible rows.

    Returns:
        The feature specifications in a fixed order.
    """
    specs: list[FeatureSpec] = []
    for index, source in enumerate(sources):
        table = source.view.name
        numeric, other = candidate_columns(source)
        if source.role == PARENT:
            for column in (*numeric, *other):
                specs.append(
                    FeatureSpec(
                        f"{table}__lookup__{column}",
                        "lookup",
                        index,
                        LAST,
                        column,
                        None,
                    )
                )
            continue
        for window in windows:
            label = window_label(window)
            for primitive in VALUE_FREE:
                specs.append(
                    FeatureSpec(
                        f"{table}__{primitive}__{label}",
                        "aggregate",
                        index,
                        primitive,
                        None,
                        window,
                    )
                )
            for column in numeric:
                for primitive in (*NUMERIC_ONLY, LAST):
                    specs.append(
                        FeatureSpec(
                            f"{table}__{primitive}__{column}__{label}",
                            "aggregate",
                            index,
                            primitive,
                            column,
                            window,
                        )
                    )
            for column in (*numeric, *other):
                specs.append(
                    FeatureSpec(
                        f"{table}__{NUNIQUE}__{column}__{label}",
                        "aggregate",
                        index,
                        NUNIQUE,
                        column,
                        window,
                    )
                )
    for position, nest in enumerate(nests):
        mid_index = next(k for k, s in enumerate(sources) if s is nest.mid)
        leaf = Source(nest.leaf, nest.leaf.key)
        numeric, others = candidate_columns(leaf)
        del others
        for window in windows:
            label = window_label(window)
            for inner in NESTED_INNER:
                for outer in NESTED_OUTER:
                    picks: list[str | None] = (
                        [None] if inner == COUNT else list(numeric)
                    )
                    for pick in picks:
                        specs.append(
                            FeatureSpec(
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
                        )
    return specs


def target_frame(
    targets: pd.DataFrame, key: str, moment: str, lag_days: float, view: Any
) -> pd.DataFrame:
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
    cutoff = asof_view.to_days(targets[moment]) - lag_days
    return pd.DataFrame(
        {
            view.key: targets[key].to_numpy(),
            asof_view.TARGET_COLUMN: np.arange(len(targets)),
            asof_view.CUTOFF_COLUMN: cutoff,
        }
    )


def reduce_rows(
    rows: pd.DataFrame,
    primitive: str,
    value: str | None,
    window: float | None,
    count: int,
) -> pd.Series:
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
        raise ValueError(f"unknown primitive: {primitive}")
    if window is not None:
        rows = rows.loc[
            rows[asof_view.EVENT_COLUMN]
            > rows[asof_view.CUTOFF_COLUMN] - window
        ]
    index = pd.RangeIndex(count)
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
        raise ValueError(f"{primitive} needs a value column")
    if primitive == LAST:
        valid = rows.loc[rows[value].notna()]
        ordered = valid.sort_values(
            [asof_view.EVENT_COLUMN, asof_view.KNOWN_COLUMN], kind="stable"
        )
        return ordered.groupby(label)[value].last().reindex(index)
    valid = rows.loc[rows[value].notna()]
    grouped = valid.groupby(label)[value]
    if primitive == NUNIQUE:
        return grouped.nunique().reindex(index, fill_value=0).astype(float)
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
    raise ValueError(f"unknown primitive: {primitive}")


def slope(rows: pd.DataFrame, value: str) -> pd.Series:
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
    return result.where(size >= 2)


def evaluate(
    spec: FeatureSpec,
    targets: pd.DataFrame,
    moment: str,
    sources: Sequence[Source],
    nests: Sequence[Nest],
    lag_days: float,
) -> pd.Series:
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
    asked = target_frame(targets, source.target_key, moment, lag_days, view)
    rows = view.visible(asked)
    if spec.kind in ("aggregate", "lookup"):
        return reduce_rows(
            rows, spec.primitive, spec.value, spec.window_days, count
        )
    if spec.nest is None or spec.inner is None:
        raise errors.ScmleError("nested feature lacks nest or inner")
    nest = nests[spec.nest]
    if spec.window_days is not None:
        rows = rows.loc[
            rows[asof_view.EVENT_COLUMN]
            > rows[asof_view.CUTOFF_COLUMN] - spec.window_days
        ]
    pairs = rows.reset_index(drop=True)
    pair_targets = pd.DataFrame(
        {
            nest.leaf.key: pairs[nest.mid_id].to_numpy(),
            asof_view.TARGET_COLUMN: np.arange(len(pairs)),
            asof_view.CUTOFF_COLUMN: pairs[asof_view.CUTOFF_COLUMN].to_numpy(),
        }
    )
    leaves = nest.leaf.visible(pair_targets)
    inner = reduce_rows(
        leaves, spec.inner, spec.value, spec.window_days, len(pairs)
    )
    outer_rows = pairs.assign(inner_value=inner.to_numpy())
    return reduce_rows(outer_rows, spec.primitive, "inner_value", None, count)


def tables_read(
    spec: FeatureSpec, sources: Sequence[Source], nests: Sequence[Nest]
) -> list[asof_view.AsOfView]:
    """Lists the views a feature reads.

    Args:
        spec: The feature.
        sources: Related tables as views.
        nests: Depth-2 routes.

    Returns:
        The source view and, for a nested feature, the leaf view.
    """
    views = [sources[spec.source].view]
    if spec.nest is not None:
        views.append(nests[spec.nest].leaf)
    return views


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureSet:
    """Kept features, the manifest, and the declared family size.

    Attributes:
        frame: One column per kept feature, indexed like the targets.
        manifest: One record per proposed feature, kept or not, with the
            reason it was dropped.
        declared_family_size: Number of kept features. Fixed here, before any
            feature meets the target.
    """

    frame: pd.DataFrame
    manifest: tuple[dict[str, Any], ...]
    declared_family_size: int

    def kept(self) -> list[dict[str, Any]]:
        """Selects the manifest records of the kept features.

        Returns:
            The records whose `kept` flag is true, in manifest order.
        """
        return [m for m in self.manifest if m["kept"]]

    def to_dict(self) -> dict[str, Any]:
        """Serializes the set, for `feature_manifest.json`.

        Returns:
            A JSON-ready mapping with the declared family size, the number
            of proposals and the manifest.
        """
        return {
            "declared_family_size": self.declared_family_size,
            "proposed": len(self.manifest),
            "features": list(self.manifest),
        }


def fingerprint(column: pd.Series) -> str:
    """Digests a feature column.

    Args:
        column: The feature values.

    Returns:
        A SHA-256 hex digest, equal for identical values (NaN equals NaN).
    """
    values = column.to_numpy()
    if values.dtype == object:
        text = repr(list(values)).encode("utf-8")
    else:
        text = np.nan_to_num(values.astype(float), nan=-1.2345e300).tobytes()
    return hashlib.sha256(text).hexdigest()


def synthesize(
    targets: pd.DataFrame,
    moment: str,
    sources: Sequence[Source],
    nests: Sequence[Nest] = (),
    lag_days: float = 0.0,
    windows: Sequence[float | None] = WINDOWS_DAYS,
    max_features: int = 200,
    max_null_rate: float = 0.95,
    strict: bool = False,
) -> FeatureSet:
    """Builds the feature family for `targets` through the as-of views.

    Args:
        targets: One row per prediction case.
        moment: Column holding each case's prediction moment.
        sources: Related tables as views; the keys are columns of `targets`.
        nests: Depth-2 routes; each `mid` must be one of `sources`.
        lag_days: Gap between the cutoff and the moment (`t_i - lag`).
        windows: Declared windows in days; None means all visible rows.
        max_features: Cap on proposals. Proposals beyond it are dropped in
            grammar order and listed.
        max_null_rate: Features missing more often are dropped.
        strict: Drop features whose weakest source tier is `unverified`.

    Returns:
        The feature set. Every dropped proposal is in the manifest with its
        reason: `cap`, `no as-of rule`, `unverified (strict)`, `constant`,
        `mostly missing` or `duplicate of <name>`.

    Raises:
        ValueError: If `lag_days` is negative.
    """
    if lag_days < 0:
        raise ValueError("lag_days must be non-negative")
    specs = propose(sources, nests, windows)
    columns: dict[str, pd.Series] = {}
    manifest: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for position, spec in enumerate(specs):
        views = tables_read(spec, sources, nests)
        tier = asof_view.weakest([v.tier for v in views])
        record: dict[str, Any] = {
            "name": spec.name,
            "kind": spec.kind,
            "expression": spec.expression(),
            "tables": [v.name for v in views],
            "cutoff_rule": f"known_time <= moment - {lag_days:g}d",
            "as_of": [v.evidence() for v in views],
            "tier": tier.value,
            "kept": False,
            "dropped": "",
        }
        manifest.append(record)
        if position >= max_features:
            record["dropped"] = "cap"
            continue
        blocked = [
            v.name
            for v in views
            if v.rule.mode == asof_view.NON_VERSIONED and not v.permissive
        ]
        if blocked:
            record["dropped"] = f"no as-of rule: {blocked}"
            continue
        if strict and tier is qualification.Tier.UNVERIFIED:
            record["dropped"] = "unverified (strict)"
            continue
        values = evaluate(spec, targets, moment, sources, nests, lag_days)
        values.index = targets.index
        values.name = spec.name
        if values.isna().mean() > max_null_rate:
            record["dropped"] = "mostly missing"
            continue
        if values.nunique(dropna=False) <= 1:
            record["dropped"] = "constant"
            continue
        digest = fingerprint(values)
        if digest in seen:
            record["dropped"] = f"duplicate of {seen[digest]}"
            continue
        seen[digest] = spec.name
        record["kept"] = True
        columns[spec.name] = values
    frame = pd.DataFrame(columns, index=targets.index)
    return FeatureSet(frame, tuple(manifest), len(columns))


def feature_provenance(
    features: FeatureSet,
    lag_days: float,
    views: Sequence[asof_view.AsOfView],
    today: datetime.date | None = None,
) -> provenance.ProvenanceTable:
    """Builds the provenance table of the kept features.

    Each feature inherits the weakest source among the tables it reads. A
    feature whose weakest source is unverified gets an unknown as-of rule, so
    admissibility tiers it `unverified` and strict mode excludes it.

    Args:
        features: The feature set.
        lag_days: Lag used in synthesis.
        views: Every view that was used, to look up each table's rule.
        today: Day on which attestations must be valid.

    Returns:
        One record per kept feature, in manifest order.
    """
    by_name = {v.name: v for v in views}
    records = []
    for entry in features.kept():
        sources = [by_name[t] for t in entry["tables"]]
        weakest_view = min(
            sources, key=lambda v: asof_view.TIER_ORDER[v.rule.tier(today)]
        )
        rule = weakest_view.rule
        tier = rule.tier(today)
        history = (f"{entry['expression']}; {entry['cutoff_rule']}",)
        attestation = rule.attestation
        if tier is qualification.Tier.CONFIRMED:
            as_of = provenance.AsOf(
                provenance.AsOfKind.TIMESTAMP, lag_days * 86_400.0
            )
            source = provenance.AsOfSource.LOG
        elif tier is qualification.Tier.ATTESTED:
            as_of = provenance.AsOf(
                provenance.AsOfKind.LAG, lag_days * 86_400.0
            )
            source = provenance.AsOfSource.ATTESTED
        else:
            as_of = provenance.AsOf.unknown()
            source = provenance.AsOfSource.NONE
            attestation = None
        records.append(
            provenance.FieldProvenance(
                name=entry["name"],
                origin=provenance.Origin.DERIVED,
                writer="scmle.data.ingest.synthesize",
                written_at=provenance.WrittenAt.UNKNOWN,
                overwritten=provenance.Overwritten.NEVER,
                definition_history=history,
                as_of=as_of,
                as_of_source=source,
                available_at_serving=provenance.Availability.UNKNOWN,
                attestation=attestation,
                log_check=rule.log_check
                if tier is qualification.Tier.CONFIRMED
                else None,
            )
        )
    return provenance.ProvenanceTable(tuple(records))


def register_family(
    ledger: ledger_module.Ledger, features: FeatureSet
) -> list[ledger_module.Hypothesis]:
    """Registers every kept feature as a `synthesized` hypothesis.

    If the number registered then differs from the declared size,
    `check_multiplicity` raises `MultiplicityError`, which propagates.

    Args:
        ledger: The hypothesis ledger.
        features: The feature set whose kept features are registered.

    Returns:
        The registered hypotheses, in manifest order.
    """
    registered = []
    for entry in features.kept():
        registered.append(
            ledger.register(
                ledger_module.HypothesisSpec.create(
                    name=f"{entry['name']}~target",
                    family=FAMILY,
                    statistic=STATISTIC,
                    null=(
                        f"{entry['name']} is not monotonically associated "
                        "with the target"
                    ),
                    column=entry["name"],
                )
            )
        )
    check_multiplicity(features.declared_family_size, ledger)
    return registered


def check_multiplicity(declared: int, ledger: ledger_module.Ledger) -> int:
    """Requires the registered `synthesized` count to equal the declared one.

    Args:
        declared: The declared family size.
        ledger: The hypothesis ledger.

    Returns:
        The number of `synthesized` hypotheses registered.

    Raises:
        MultiplicityError: On a mismatch; the run must fail.
    """
    registered = sum(
        h.spec.family == FAMILY for h in ledger.hypotheses.values()
    )
    if registered != declared:
        raise MultiplicityError(
            f"declared family size {declared} but {registered} "
            f"{FAMILY} hypotheses are registered"
        )
    return registered
