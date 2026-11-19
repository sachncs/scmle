"""The relational engine against a naive reference on five real datasets.

The datasets are the five public relational datasets the C5 evidence
(`benchmarks.evidence.relational`) pins by checksum and caches. The engine
reads each through a store (the SQLite databases as they are, the CSV set
as it is, the Stack Exchange XML converted to a Parquet folder, since no
store reads XML); the reference reads the same files with the standard
library alone and loops row by row (`benchmarks.qualification.ingest_naive`).
A dataset absent from the cache is skipped by name, never downloaded here.

Aggregates: every primitive of every measure over every window, under the
recorded-time rule and the delay rule, at two lags, for entities from the
lightest to the heaviest at the evidence's hard cutoffs (ties with event
and recorded times, a window's edge, before the first row, after the last),
against the naive reference and against the as-of synthesizer
(`scmle.assurance.ingest.synthesize`) on all five datasets.
Joins: each child row's table joined many-to-one by its own identifier,
at cutoffs that tie its event and recorded times, on the four datasets
whose child rows carry an identifier (baseball's do not).
"""

import math
import pathlib

import numpy
import pandas
import pytest

from benchmarks import acceptance
from benchmarks.evidence import relational
from benchmarks.qualification import ingest_naive
from scmle.assurance.ingest import asof_view
from scmle.assurance.ingest import synthesize
from scmle.contracts import errors
from scmle.data.relational import engine
from scmle.data.relational import schema
from scmle.data.relational import stores

TOLERANCE = acceptance.load()["reference_rel_tol"].value
NAMES = {d.name: d for d in relational.DATASETS}
# Rules compared: (recorded time used, recording delay in days).
RULES = ((True, 0.0), (False, 1.0))
LAGS = (0.0, 30.0)
NAIVE = {"trend": "slope"}


def cached(dataset):
    folder = relational.CACHE / dataset.name
    if not all((folder / f.name).exists() for f in dataset.files):
        pytest.skip(f"{dataset.name} is not in the cache {relational.CACHE}")
    return folder


def opened(dataset, folder, tmp_path):
    """The dataset's store and the store name of its child table."""
    if dataset.form == "sqlite":
        return stores.Database(folder / dataset.files[0].name), (
            dataset.child.name
        )
    if dataset.form == "csv":
        return stores.Csv(folder), pathlib.Path(dataset.child.name).stem
    converted = tmp_path / "parquet"
    converted.mkdir()
    for file in dataset.files:
        frame = pandas.read_xml(folder / file.name, parser="etree")
        frame.to_parquet(converted / f"{pathlib.Path(file.name).stem}.parquet")
    return stores.Parquet(converted), pathlib.Path(dataset.child.name).stem


def reference(dataset, folder):
    rows = relational.raw(dataset, folder, dataset.child)
    return relational.reference(rows, dataset.child, dataset.numeric)


def targets(dataset, keys, cutoffs, lag):
    """Product targets (key, moment) for reference cutoffs in days."""
    return pandas.DataFrame(
        {
            "key": pandas.Series(list(keys)),
            "moment": [
                relational.instant(c + lag, dataset.numeric) for c in cutoffs
            ],
        }
    )


def declared(dataset, name, known, delay, key, cardinality):
    child = dataset.child
    table = schema.Table(
        name, child.event, child.known if known else None, delay
    )
    link = schema.Link(name, key, "key", cardinality, "rows")
    return schema.Schema((table,), (link,))


def synthesizer(dataset, frame, name, known, delay):
    """The child table as the as-of synthesizer reads it, under one rule."""
    child = dataset.child
    rule = (
        asof_view.Timing(asof_view.TRANSACTION, child.known, delay)
        if known
        else asof_view.Timing(asof_view.EVENT_DELAY, None, delay)
    )
    view = asof_view.View(frame, child.key, rule, child.event, name)
    return synthesize.Source(view, "key")


def compare(got, want):
    got = numpy.asarray(got, dtype=float)
    want = numpy.asarray(want, dtype=float)
    return numpy.allclose(
        got, want, rtol=TOLERANCE, atol=TOLERANCE, equal_nan=True
    )


@pytest.mark.parametrize("name", sorted(NAMES))
def test_aggregates_equal_the_naive_reference(name, tmp_path):
    dataset = NAMES[name]
    folder = cached(dataset)
    store, table = opened(dataset, folder, tmp_path)
    frame = store.read(table)
    rows = reference(dataset, folder)
    keys = relational.entities(rows)
    window = engine.WINDOWS[0]
    asked = [
        (key, cut)
        for key in keys
        for cut in relational.cutoffs(rows.loc[rows["key"] == key], window)
    ]
    aggregates = engine.grid("rows", list(dataset.child.measures))
    unequal, compared, synthesized = [], 0, 0
    rules = [r for r in RULES if dataset.child.known or not r[0]]
    for known, delay in rules:
        layout = declared(
            dataset, table, known, delay, dataset.child.key, schema.MANY
        )
        built = engine.Engine(layout, {table: frame})
        source = synthesizer(dataset, frame, table, known, delay)
        for lag in LAGS:
            cases = targets(
                dataset, [a[0] for a in asked], [a[1] for a in asked], lag
            )
            found = built.features(cases, "moment", aggregates, lag)
            for aggregate in aggregates:
                feature = synthesize.Feature(
                    aggregate.name,
                    "aggregate",
                    0,
                    NAIVE.get(aggregate.primitive, aggregate.primitive),
                    aggregate.value,
                    aggregate.window,
                )
                other = synthesize.evaluate(
                    feature, cases, "moment", [source], [], lag
                )
                synthesized += 1
                if not compare(found[aggregate.name], other):
                    unequal.append((aggregate.name, known, delay, lag, "as-of"))
            moments = pandas.DataFrame(
                {
                    "key": [a[0] for a in asked],
                    "moment": [a[1] + lag for a in asked],
                },
                dtype=object,
            )
            for aggregate in aggregates:
                want = numpy.full(len(moments), numpy.nan)
                for key, group in moments.groupby("key", sort=False):
                    own = rows.loc[rows["key"] == key]
                    want[group.index.to_numpy()] = (
                        ingest_naive.aggregate_feature(
                            group,
                            "key",
                            "moment",
                            own,
                            "key",
                            relational.EVENT,
                            relational.KNOWN if known else None,
                            aggregate.value,
                            NAIVE.get(aggregate.primitive, aggregate.primitive),
                            aggregate.window,
                            lag,
                            delay,
                        )
                    )
                compared += 1
                if not compare(found[aggregate.name], want):
                    unequal.append((aggregate.name, known, delay, lag))
    assert compared == len(rules) * len(LAGS) * len(aggregates)
    assert synthesized == compared
    assert unequal == []


JOINABLE = [n for n, d in sorted(NAMES.items()) if d.child.ident and d.leaf]


@pytest.mark.parametrize("name", JOINABLE)
def test_a_many_to_one_join_equals_the_naive_lookup(name, tmp_path):
    dataset = NAMES[name]
    folder = cached(dataset)
    store, table = opened(dataset, folder, tmp_path)
    frame = store.read(table)
    rows = reference(dataset, folder)
    idents = sorted(rows["ident"])
    picks = numpy.linspace(0, len(idents) - 1, relational.ENTITIES).round()
    chosen = [idents[int(i)] for i in dict.fromkeys(picks)]
    asked = []
    for ident in chosen:
        own = rows.loc[rows["ident"] == ident].iloc[0]
        times = [own[relational.EVENT], own[relational.KNOWN]]
        present = [t for t in times if not math.isnan(t)]
        asked.extend(
            (ident, cut)
            for cut in (present[0] - 1.0, *present, max(present) + 1.0)
        )
    unequal, compared = [], 0
    rules = [r for r in RULES if dataset.child.known or not r[0]]
    for known, delay in rules:
        layout = declared(
            dataset, table, known, delay, dataset.child.ident, schema.ONE
        )
        built = engine.Engine(layout, {table: frame})
        for lag in LAGS:
            cases = targets(
                dataset, [a[0] for a in asked], [a[1] for a in asked], lag
            )
            found = built.features(cases, "moment", (), lag)
            moments = pandas.DataFrame(
                {
                    "key": [a[0] for a in asked],
                    "moment": [a[1] + lag for a in asked],
                },
                dtype=object,
            )
            for measure in dataset.child.measures:
                want = ingest_naive.lookup_feature(
                    moments, "key", "moment", rows, "ident", relational.EVENT,
                    relational.KNOWN if known else None, measure, lag, delay,
                )  # fmt: skip
                compared += 1
                got = found[f"rows__{measure}"]
                if not compare(got, want):
                    unequal.append((measure, known, delay, lag))
            # Every chosen row is joined at least at its last cutoff.
            seen = found[f"rows__{dataset.child.event}"].notna()
            assert seen.sum() >= len(chosen)
    assert compared == len(rules) * len(LAGS) * len(dataset.child.measures)
    assert unequal == []


def test_a_key_that_repeats_is_refused_as_many_to_one(tmp_path):
    dataset = NAMES["baseball"]
    folder = cached(dataset)
    store, table = opened(dataset, folder, tmp_path)
    layout = declared(dataset, table, False, 0.0, dataset.child.key, schema.ONE)
    with pytest.raises(errors.SchemaError, match="declared many_to_one"):
        engine.Engine(layout, {table: store.read(table)})


@pytest.mark.parametrize(
    "name", [n for n, d in sorted(NAMES.items()) if d.form == "sqlite"]
)
def test_a_database_table_gives_the_same_features_from_parquet_and_csv(
    name, tmp_path
):
    dataset = NAMES[name]
    folder = cached(dataset)
    store, table = opened(dataset, folder, tmp_path)
    frame = store.read(table)
    for kind in ("parquet", "csv"):
        (tmp_path / kind).mkdir()
    frame.to_parquet(tmp_path / "parquet" / f"{table}.parquet")
    frame.to_csv(tmp_path / "csv" / f"{table}.csv", index=False)
    rows = reference(dataset, folder)
    keys = relational.entities(rows)
    cuts = [
        cut
        for key in keys
        for cut in relational.cutoffs(rows.loc[rows["key"] == key], 7.0)
    ]
    owners = [
        key
        for key in keys
        for ignored in relational.cutoffs(rows.loc[rows["key"] == key], 7.0)
    ]
    cases = targets(dataset, owners, cuts, 0.0)
    layout = declared(dataset, table, True, 0.0, dataset.child.key, schema.MANY)
    aggregates = engine.grid("rows", list(dataset.child.measures))
    found = [
        engine.Engine.open(layout, s).features(cases, "moment", aggregates)
        for s in (
            store,
            stores.locate(tmp_path / "parquet"),
            stores.locate(tmp_path / "csv"),
        )
    ]
    assert found[0]["rows__count__*__all"].sum() > 0
    for other in found[1:]:
        pandas.testing.assert_frame_equal(other, found[0])
