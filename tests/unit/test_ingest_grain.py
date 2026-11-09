"""Grain and key inference, and the unit verdict."""

import numpy
import pandas
import pytest

from benchmarks.qualification import ingest_worlds
from scmle.data.ingest import budget
from scmle.data.ingest import grain
from scmle.data.ingest import profile


def scan_of(frame, eps_key=0.0, eps_null=0.0, size=3, meter=None):
    db = ingest_worlds.new_database("g")
    ingest_worlds.load(db, "t", frame)
    source = profile.SqlSource(db.factory, meter)
    table = profile.profile_table(source, "t")
    return (
        source,
        table,
        grain.infer_keys(source, table, eps_key, eps_null, size),
    )


def test_unit_text_parsing():
    assert grain.unit_components("customer-month") == ("customer", "month")
    assert grain.unit_components("one row per customers per Month") == (
        "customer",
        "month",
    )
    assert grain.name_tokens("customerID") == ("customer", "id")
    assert grain.name_tokens("order_items") == ("order", "item")


def test_single_and_composite_keys_are_found_minimal():
    n = 60
    frame = pandas.DataFrame(
        {
            "a": numpy.repeat(numpy.arange(10), 6),
            "b": numpy.tile(numpy.arange(6), 10),
            "c": numpy.arange(n) % 3,
        }
    )
    scan = scan_of(frame)[2]
    assert [k.columns for k in scan.keys] == [("a", "b")]
    assert scan.keys[0].g3 == 0.0
    assert scan.levels_done == 3


def test_three_column_key_needs_level_three():
    frame = pandas.DataFrame(
        [(a, b, c) for a in range(4) for b in range(4) for c in range(4)],
        columns=["a", "b", "c"],
    )
    scan = scan_of(frame, size=3)[2]
    assert [k.columns for k in scan.keys] == [("a", "b", "c")]
    short = scan_of(frame, size=2)[2]
    assert short.keys == []


@pytest.mark.parametrize("duplicates", [9, 10, 11, 12])
def test_near_key_flips_at_the_threshold_to_one_row(duplicates):
    n = 1000
    values = numpy.arange(n)
    values[:duplicates] = values[n - duplicates :]
    scan = scan_of(pandas.DataFrame({"k": values}), eps_key=0.01)[2]
    assert bool(scan.keys) == (duplicates <= 10)
    measured_g3 = scan.measured[("k",)][0]
    assert measured_g3 == pytest.approx(duplicates / n)


def test_null_rate_disqualifies_a_key():
    frame = pandas.DataFrame({"k": [1.0, 2.0, 3.0, None, 5.0]})
    strict = scan_of(frame, eps_key=0.5, eps_null=0.0)[2]
    assert strict.keys == []
    loose = scan_of(frame, eps_key=0.5, eps_null=0.3)[2]
    assert [k.columns for k in loose.keys] == [("k",)]


def test_ties_are_listed_not_broken():
    frame = pandas.DataFrame(
        {"a": numpy.arange(20), "b": numpy.arange(20)[::-1]}
    )
    scan = scan_of(frame)[2]
    assert scan.ties() == [[("a",), ("b",)]]
    assert all(k.small_n for k in scan.keys)


def test_hopeless_sets_cost_no_query():
    frame = pandas.DataFrame({"a": [0, 1] * 50, "b": [0, 1, 2, 3] * 25})
    source, table = scan_of(frame)[:2]
    before = source.queries
    assert not grain.bound_ok(table, ("a", "b"), 0.0)
    grain.infer_keys(source, table)
    assert source.queries == before


def test_budget_stops_the_lattice_at_a_level_boundary():
    frame = pandas.DataFrame(
        {c: numpy.tile(numpy.arange(5), 40) for c in "abc"}
        | {"d": numpy.repeat(numpy.arange(40), 5)}
    )
    meter = budget.BudgetMeter(budget.Budget(max_queries=100))
    db = ingest_worlds.new_database("gb")
    ingest_worlds.load(db, "t", frame)
    source = profile.SqlSource(db.factory, meter)
    table = profile.profile_table(source, "t")
    tight = budget.BudgetMeter(budget.Budget(max_queries=0))
    source.meter = tight
    scan = grain.infer_keys(source, table)
    assert scan.partial
    assert scan.levels_done == 1
    assert tight.stops[0].unit == "level 2"


@pytest.mark.parametrize("unit", ingest_worlds.UNITS)
@pytest.mark.parametrize("shape", ingest_worlds.SHAPES)
def test_unit_verdict_matches_construction(unit, shape):
    db, table = ingest_worlds.grain_world(7, shape)
    source = db.source()
    profiled = profile.profile_table(source, table)
    scan = grain.infer_keys(source, profiled)
    verdict = grain.unit_verdict(source, profiled, unit, scan)
    assert verdict.verdict == ingest_worlds.expected_verdict(unit, shape)


def test_unmapped_unit_words_are_unknown_and_asked():
    db, table = ingest_worlds.grain_world(7, "customers")
    source = db.source()
    profiled = profile.profile_table(source, table)
    scan = grain.infer_keys(source, profiled)
    verdict = grain.unit_verdict(source, profiled, "warehouse-shift", scan)
    assert verdict.verdict == grain.UNKNOWN
    assert verdict.blocks
    assert len(verdict.questions) == 2


def test_no_key_at_tolerance_is_unknown():
    frame = pandas.DataFrame({"customer_id": [1, 1, 2, 2], "x": [0, 0, 1, 1]})
    source, table, scan = scan_of(frame)
    verdict = grain.unit_verdict(source, table, "customer", scan)
    assert verdict.verdict == grain.UNKNOWN


def test_coarser_is_never_match_across_tolerances():
    for eps in (0.0, 0.01, 0.05, 0.2):
        for seed in range(8):
            db, table = ingest_worlds.grain_world(seed, "customers")
            source = db.source()
            profiled = profile.profile_table(source, table)
            scan = grain.infer_keys(source, profiled, eps_key=eps)
            verdict = grain.unit_verdict(
                source, profiled, "customer-month", scan
            )
            assert verdict.verdict != grain.MATCH


def test_invalid_tolerances_are_rejected():
    source, table = scan_of(pandas.DataFrame({"a": [1, 2]}))[:2]
    with pytest.raises(ValueError):
        grain.infer_keys(source, table, eps_key=1.5)
    with pytest.raises(ValueError):
        grain.infer_keys(source, table, max_size=0)
