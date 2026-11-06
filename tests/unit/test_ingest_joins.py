"""Join discovery by exact inclusion with conservative acceptance."""

import numpy as np
import pandas as pd

from benchmarks.qualification import ingest_worlds as worlds
from scmle.data.ingest import budget
from scmle.data.ingest import grain
from scmle.data.ingest import joins
from scmle.data.ingest import profile


def graph_of(db, tables, declared=(), eps_fk=0.0):
    source = db.source()
    catalog = profile.profile_catalog(source, tables)
    scans = {t.name: grain.infer_keys(source, t) for t in catalog.tables}
    return (
        source,
        catalog,
        scans,
        joins.discover(source, catalog, scans, declared, eps_fk),
    )


def two_tables(child_col="customer_id", child_values=None, parents=40):
    db = worlds.new_database("j")
    worlds.load(db, "customer", pd.DataFrame({"id": np.arange(1, parents + 1)}))
    values = (
        child_values
        if child_values is not None
        else np.arange(1, 101) % parents + 1
    )
    worlds.load(
        db,
        "fact",
        pd.DataFrame({"id": np.arange(100), child_col: values}),
    )
    return db


def find(graph, column):
    return [link for link in graph.links if link.child_columns == (column,)]


def test_named_exact_containment_is_accepted():
    db = two_tables()
    graph = graph_of(db, ["customer", "fact"])[3]
    link = find(graph, "customer_id")[0]
    assert link.decision == joins.ACCEPTED
    assert link.cardinality == joins.MANY_TO_ONE
    assert link.containment == 1.0


def test_coincidental_inclusion_stays_a_candidate():
    db = two_tables(child_col="qty")
    graph = graph_of(db, ["customer", "fact"])[3]
    link = find(graph, "qty")[0]
    assert link.decision == joins.CANDIDATE
    assert not link.joinable
    assert graph.questions()


def test_partial_containment_is_candidate_or_absent():
    values = np.arange(1, 101) % 60 + 1  # 41..60 have no parent
    db = two_tables(child_values=values)
    strict = graph_of(db, ["customer", "fact"], eps_fk=0.0)[3]
    assert find(strict, "customer_id") == []
    loose = graph_of(db, ["customer", "fact"], eps_fk=0.5)[3]
    link = find(loose, "customer_id")[0]
    assert link.decision == joins.CANDIDATE
    assert 0.5 < link.containment < 1.0


def test_declared_key_is_accepted_and_violation_is_flagged():
    declared = (joins.DeclaredKey("fact", ("qty",), "customer", ("id",)),)
    db = two_tables(child_col="qty")
    graph = graph_of(db, ["customer", "fact"], declared)[3]
    link = find(graph, "qty")[0]
    assert link.declared and link.decision == joins.ACCEPTED
    bad = two_tables(child_values=np.arange(1, 101) + 30)
    broken = (joins.DeclaredKey("fact", ("customer_id",), "customer", ("id",)),)
    graph = graph_of(bad, ["customer", "fact"], broken)[3]
    link = find(graph, "customer_id")[0]
    assert link.decision == joins.VIOLATED


def test_twin_parents_are_resolved_by_table_name_or_asked():
    db = worlds.new_database("twin")
    for name in ("user", "account"):
        worlds.load(db, name, pd.DataFrame({"id": np.arange(1, 51)}))
    worlds.load(
        db,
        "fact",
        pd.DataFrame(
            {
                "id": np.arange(100),
                "user_id": np.arange(100) % 50 + 1,
                "owner_id": np.arange(100) % 50 + 1,
            }
        ),
    )
    graph = graph_of(db, ["user", "account", "fact"])[3]
    named = find(graph, "user_id")
    assert {link.parent: link.decision for link in named} == {
        "user": joins.ACCEPTED,
        "account": joins.CANDIDATE,
    }
    assert all(
        link.decision == joins.CANDIDATE for link in find(graph, "owner_id")
    )


def test_many_to_many_is_reported_and_never_accepted():
    db = worlds.new_database("m2m")
    worlds.load(db, "a", pd.DataFrame({"id": range(30), "tag": [0, 1, 2] * 10}))
    worlds.load(db, "b", pd.DataFrame({"id": range(30), "tag": [0, 1, 2] * 10}))
    source, catalog, scans, graph = graph_of(db, ["a", "b"])
    found = joins.many_to_many(source, catalog, scans)
    assert [link.cardinality for link in found] == [joins.MANY_TO_MANY]
    assert not found[0].joinable
    assert all(link.child_columns != ("tag",) for link in graph.accepted())


def many_to_many_world():
    db = worlds.new_database("m2m-budget")
    for name in ("a", "b", "c"):
        worlds.load(
            db,
            name,
            pd.DataFrame({"id": range(30), "tag": [0, 1, 2] * 10}),
        )
    return db


def test_many_to_many_charges_the_budget_per_pair_and_stops_at_a_boundary():
    db = many_to_many_world()
    plain = db.source()
    catalog = profile.profile_catalog(plain, ["a", "b", "c"])
    scans = {t.name: grain.infer_keys(plain, t) for t in catalog.tables}
    free = joins.many_to_many(plain, catalog, scans)
    assert len(free) == 3

    # room for exactly one pair (60 rows scanned, two queries)
    meter = budget.BudgetMeter(budget.IngestBudget(max_rows_scanned=60))
    limited = profile.SqlSource(db.factory, meter)
    found = joins.many_to_many(limited, catalog, scans)
    assert len(found) == 1
    assert meter.rows == 60
    assert [stop.stage for stop in meter.stops] == ["many_to_many"]
    assert meter.stops[0].reason == "rows"
    assert meter.partial


def test_many_to_many_without_a_meter_or_with_room_is_unchanged():
    db = many_to_many_world()
    plain = db.source()
    catalog = profile.profile_catalog(plain, ["a", "b", "c"])
    scans = {t.name: grain.infer_keys(plain, t) for t in catalog.tables}
    meter = budget.BudgetMeter(budget.IngestBudget())
    roomy = profile.SqlSource(db.factory, meter)
    assert len(joins.many_to_many(roomy, catalog, scans)) == 3
    assert not meter.partial


def test_paths_go_up_and_down_to_depth_two_without_revisiting():
    jw = worlds.surrogate_world(4)
    source = jw.db.source()
    catalog = profile.profile_catalog(source, jw.tables)
    scans = {t.name: grain.infer_keys(source, t) for t in catalog.tables}
    declared = tuple(
        joins.DeclaredKey(c, (col,), p, ("id",)) for c, col, p in jw.truth
    )
    graph = joins.discover(source, catalog, scans, declared)
    found = joins.paths(graph, "fact", 2)
    assert found
    assert all(len(p) <= 2 for p in found)
    for path in found:
        seen = ["fact"]
        for step in path:
            assert step.source == seen[-1]
            seen.append(step.target)
        assert len(set(seen)) == len(seen)
    assert joins.paths(graph, "fact", 0) == []
    assert {p[0].direction for p in found} == {"up"}
    assert {p[0].direction for p in joins.paths(graph, "customer", 1)} == {
        "down"
    }


def test_surrogate_worlds_have_no_spurious_accepts():
    spurious = 0
    for seed in range(30):
        jw = worlds.surrogate_world(seed)
        graph = graph_of(jw.db, jw.tables)[3]
        spurious += sum(
            (link.child, link.child_columns[0], link.parent) not in jw.truth
            for link in graph.accepted()
        )
    assert spurious == 0


def test_name_supports_rules():
    assert joins.name_supports(("customer_id",), "customer", ("id",))
    assert joins.name_supports(("customer_id",), "customers", ("id",))
    assert joins.name_supports(("user_id",), "accounts", ("user_id",))
    assert not joins.name_supports(("qty",), "customer", ("id",))
    assert not joins.name_supports(("label",), "account", ("label",))
    assert joins.name_supports(("a", "b"), "t", ("a", "b"))
