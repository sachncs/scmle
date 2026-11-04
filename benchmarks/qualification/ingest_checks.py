"""Point-in-time checks for F9: differential, metamorphic, late arrival.

* Differential: the vectorised synthesis equals `ingest_naive`, a row-by-row
  reference written separately from the code under test, on random schemas.
  Floating sums are compared to a relative 1e-9 (the two add in different
  orders); counts and distinct counts are compared exactly by the same test
  because they are integers in a float.
* Metamorphic: rewriting every child row that is not yet known at a target's
  cutoff (its values, its event time, even its key) leaves that target's
  feature unchanged.
* Late arrival: on worlds with late arrivals and backfills, an event-time-only
  filter leaks rows recorded after the cutoff, and the view does not.

All three draw only from `randomness.component_rng`.
"""

import dataclasses

import numpy as np
import pandas as pd

from benchmarks.qualification import ingest_naive
from benchmarks.qualification import ingest_worlds
from scmle.core import randomness
from scmle.data.ingest import asof_view
from scmle.data.ingest import synthesize

PRIMITIVES = synthesize.PRIMITIVES
RTOL = 1e-9


@dataclasses.dataclass
class Schema:
    """A random parent/child/leaf schema and the feature to compare."""

    targets: pd.DataFrame
    orders: pd.DataFrame
    items: pd.DataFrame
    known: bool
    leaf_known: bool
    delta: float
    lag: float
    window: float | None
    primitive: str
    inner: str
    outer: str
    nested: bool


def random_schema(seed: int, name: str = "ingest.check.schema") -> Schema:
    """One random schema, with ties, gaps, late rows and shared keys."""
    rng = randomness.component_rng(seed, name, 0)
    n_targets = int(rng.integers(3, 12))
    keys = int(rng.integers(2, 6))
    targets = pd.DataFrame(
        {
            "cid": rng.integers(0, keys, n_targets),
            "moment": rng.choice(
                [rng.uniform(20, 100), float(rng.integers(20, 100))], n_targets
            ),
        }
    )
    rows = int(rng.integers(0, 40))
    event = np.round(rng.uniform(0, 100, rows), int(rng.integers(0, 2)))
    value = rng.normal(size=rows).round(int(rng.integers(0, 3)))
    value[rng.random(rows) < 0.15] = np.nan
    recorded = event + rng.exponential(2.0, rows).round(1)
    recorded[rng.random(rows) < 0.1] = np.nan
    orders = pd.DataFrame(
        {
            "cid": rng.integers(0, keys, rows),
            "oid": np.arange(rows),
            "ev": event,
            "rec": recorded,
            "val": value,
        }
    )
    leaf_rows = int(rng.integers(0, 60))
    leaf_event = np.round(
        rng.uniform(0, 100, leaf_rows), int(rng.integers(0, 2))
    )
    leaf_value = rng.normal(size=leaf_rows).round(2)
    leaf_value[rng.random(leaf_rows) < 0.15] = np.nan
    leaf_recorded = leaf_event + rng.exponential(3.0, leaf_rows).round(1)
    leaf_recorded[rng.random(leaf_rows) < 0.1] = np.nan
    items = pd.DataFrame(
        {
            "oid": rng.integers(0, max(rows, 1), leaf_rows),
            "ev": leaf_event,
            "rec": leaf_recorded,
            "val": leaf_value,
        }
    )
    return Schema(
        targets,
        orders,
        items,
        known=bool(rng.random() < 0.5),
        leaf_known=bool(rng.random() < 0.5),
        delta=float(rng.choice([0.0, 1.0, 2.5])),
        lag=float(rng.choice([0.0, 1.5, 3.0])),
        window=(None, 7.0, 30.0)[int(rng.integers(0, 3))],
        primitive=str(rng.choice(PRIMITIVES)),
        inner=str(rng.choice(synthesize.NESTED_INNER)),
        outer=str(rng.choice(["mean", "max", "sum", "count", "last", "std"])),
        nested=bool(rng.random() < 0.3),
    )


def views(schema: Schema) -> tuple[synthesize.Source, asof_view.AsOfView]:
    """The order source and the item view of a schema."""

    def rule(known: bool) -> asof_view.TimeRule:
        if known:
            return asof_view.TimeRule(
                asof_view.TRANSACTION, "rec", schema.delta
            )
        return asof_view.TimeRule(asof_view.EVENT_DELAY, None, schema.delta)

    orders = asof_view.AsOfView(
        schema.orders, "cid", rule(schema.known), "ev", "orders"
    )
    items = asof_view.AsOfView(
        schema.items, "oid", rule(schema.leaf_known), "ev", "items"
    )
    return synthesize.Source(orders, "cid"), items


def fast_feature(schema: Schema) -> pd.Series:
    """The feature computed by the code under test."""
    source, items = views(schema)
    if schema.nested:
        spec = synthesize.FeatureSpec(
            "f",
            "nested",
            0,
            schema.outer,
            None if schema.inner == synthesize.COUNT else "val",
            schema.window,
            nest=0,
            inner=schema.inner,
        )
        nest = synthesize.Nest(source, items, "oid")
        return synthesize.evaluate(
            spec, schema.targets, "moment", [source], [nest], schema.lag
        )
    value = None if schema.primitive in synthesize.VALUE_FREE else "val"
    spec = synthesize.FeatureSpec(
        "f", "aggregate", 0, schema.primitive, value, schema.window
    )
    return synthesize.evaluate(
        spec, schema.targets, "moment", [source], [], schema.lag
    )


def naive_feature(schema: Schema) -> list[float]:
    """The same feature by the reference implementation."""
    known = "rec" if schema.known else None
    leaf_known = "rec" if schema.leaf_known else None
    if schema.nested:
        return ingest_naive.nested_feature(
            schema.targets,
            "cid",
            "moment",
            schema.orders,
            "cid",
            "oid",
            "ev",
            known,
            schema.items,
            "oid",
            "ev",
            leaf_known,
            None if schema.inner == synthesize.COUNT else "val",
            schema.inner,
            schema.outer,
            schema.window,
            schema.window,
            schema.lag,
            schema.delta,
        )
    value = None if schema.primitive in synthesize.VALUE_FREE else "val"
    return ingest_naive.aggregate_feature(
        schema.targets,
        "cid",
        "moment",
        schema.orders,
        "cid",
        "ev",
        known,
        value,
        schema.primitive,
        schema.window,
        schema.lag,
        schema.delta,
    )


def same(left: np.ndarray, right: np.ndarray) -> bool:
    """Equal within the floating tolerance, NaN equal to NaN."""
    return bool(
        np.allclose(
            left.astype(float),
            right.astype(float),
            rtol=RTOL,
            atol=RTOL,
            equal_nan=True,
        )
    )


def differential_equal(seed: int) -> bool:
    """Whether fast and naive agree on the schema of this seed."""
    schema = random_schema(seed)
    fast = fast_feature(schema).to_numpy()
    return same(fast, np.array(naive_feature(schema)))


def metamorphic_unchanged(seed: int) -> bool:
    """Whether rewriting not-yet-known rows leaves target 0 unchanged.

    The rows rewritten are those whose known time exceeds target 0's cutoff.
    Their value and key are replaced at random, and their event time too when
    they have a recorded time; the recorded time column is untouched, because
    the rule reads it.
    """
    schema = random_schema(seed, "ingest.check.meta")
    rng = randomness.component_rng(seed, "ingest.check.meta.noise", 0)
    schema.targets = schema.targets.iloc[:1].reset_index(drop=True)
    schema.known = True
    schema.leaf_known = True
    cutoff = float(schema.targets["moment"].iloc[0]) - schema.lag
    before = fast_feature(schema).to_numpy()
    for frame in (schema.orders, schema.items):
        recorded = frame["rec"].to_numpy()
        fallback = frame["ev"].to_numpy() + schema.delta
        known = np.where(np.isnan(recorded), fallback, recorded)
        later = known > cutoff
        count = int(later.sum())
        frame.loc[later, "val"] = rng.normal(size=count) * 1000.0
        # With no recorded time the rule reads the event time, so that one
        # stays; every other field of an unknown row is free to change.
        free = later & ~np.isnan(recorded)
        frame.loc[free, "ev"] = rng.uniform(-50, 200, int(free.sum()))
        if "cid" in frame.columns:
            frame.loc[later, "cid"] = rng.integers(0, 6, count)
    after = fast_feature(schema).to_numpy()
    return same(before, after)


@dataclasses.dataclass(frozen=True)
class Leaks:
    """Rows read that were recorded after their cutoff, in one world."""

    baseline: int
    view: int


def late_leaks(seed: int) -> Leaks:
    """Leaked rows of an event-time-only filter and of the view."""
    world = ingest_worlds.late_world(seed)
    cutoff = world.targets["moment"].to_numpy() - world.lag_days
    asked = pd.DataFrame(
        {
            "cid": world.targets["cid"].to_numpy(),
            asof_view.TARGET_COLUMN: np.arange(len(world.targets)),
            asof_view.CUTOFF_COLUMN: cutoff,
        }
    )
    view = asof_view.AsOfView(
        world.child,
        "cid",
        asof_view.TimeRule(asof_view.TRANSACTION, "recorded", 0.0),
        "event_time",
        "events",
    )
    seen = view.visible(asked)
    view_leaks = int((seen["recorded"] > seen[asof_view.CUTOFF_COLUMN]).sum())
    joined = asked.merge(world.child, on="cid")
    by_event = joined.loc[
        joined["event_time"] <= joined[asof_view.CUTOFF_COLUMN]
    ]
    baseline = int(
        (by_event["recorded"] > by_event[asof_view.CUTOFF_COLUMN]).sum()
    )
    return Leaks(baseline, view_leaks)
