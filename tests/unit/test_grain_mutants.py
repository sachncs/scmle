"""Targeted tests for grain behaviours the main suite left unpinned."""

import numpy
import pandas
import pytest

from benchmarks.qualification import ingest_worlds
from scmle.data.ingest import budget
from scmle.data.ingest import grain
from scmle.data.ingest import profile
from tests.unit.test_ingest_grain import scan_of


class ScriptedSource:
    """A source that records each statement and answers from a script."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.statements = []

    def scalar(self, sql, stage="", unit=""):
        self.statements.append(sql)
        return self.answers.pop(0)


def column(name, distinct, non_null, kind="number"):
    return profile.ColumnProfile(name, kind, non_null, distinct, None, None)


def table_of(rows, *columns):
    return profile.TableProfile("t", rows, tuple(columns))


def empty_scan(**changes):
    fields = {
        "table": "t",
        "rows": 10,
        "keys": [],
        "measured": {},
        "levels_done": 1,
        "partial": False,
        "eps_key": 0.0,
        "eps_null": 0.0,
    }
    fields.update(changes)
    return grain.KeyScan(**fields)


def candidate(columns, g3=0.0, nulls=0.0):
    return grain.KeyCandidate(columns, g3, nulls, 5, 10, True)


def test_stem_keeps_short_words_and_double_s():
    assert grain.stem("bus") == "bus"
    assert grain.stem("gas") == "gas"
    assert grain.stem("boss") == "boss"
    assert grain.stem("class") == "class"
    assert grain.stem("dogs") == "dog"
    assert grain.stem("customers") == "customer"


def test_ties_group_only_equal_ranks_and_need_two_members():
    scan = empty_scan(
        keys=[candidate(("a",)), candidate(("b",)), candidate(("c", "d"))]
    )
    assert scan.ties() == [[("a",), ("b",)]]
    alone = empty_scan(keys=[candidate(("a",)), candidate(("c", "d"))])
    assert alone.ties() == []
    unequal = empty_scan(keys=[candidate(("a",), 0.0), candidate(("b",), 0.1)])
    assert unequal.ties() == []


def test_measure_issues_the_two_grouped_queries_in_upper_case():
    source = ScriptedSource([7, 3])
    result = grain.measure(source, "tbl", ("a", "b"), 10)
    assert result == (pytest.approx(0.3), pytest.approx(0.3))
    assert source.statements == [
        (
            'SELECT COUNT(*) FROM (SELECT DISTINCT "a", "b" FROM "tbl" '
            'WHERE "a" IS NOT NULL AND "b" IS NOT NULL)'
        ),
        'SELECT COUNT(*) FROM "tbl" WHERE "a" IS NULL OR "b" IS NULL',
    ]


def test_measure_of_an_empty_table_is_all_duplicates_and_no_nulls():
    assert grain.measure(ScriptedSource([0, 0]), "t", ("a",), 0) == (1.0, 0.0)
    assert grain.measure(ScriptedSource([1, 0]), "t", ("a",), 1) == (0.0, 0.0)


def test_measure_treats_missing_counts_as_zero():
    result = grain.measure(ScriptedSource([None, None]), "t", ("a",), 4)
    assert result == (1.0, 0.0)
    assert grain.measure(ScriptedSource([0, 4]), "t", ("a",), 4) == (1.0, 1.0)
    assert grain.measure(ScriptedSource([4, 1]), "t", ("a",), 4) == (0.0, 0.25)


def test_measure_on_a_real_table_counts_nulls_and_distinct_rows():
    frame = pandas.DataFrame(
        {"a": [1.0, 1.0, 2.0, None, 3.0], "b": [1, 1, 2, 3, None]}
    )
    source, table = scan_of(frame)[:2]
    g3, nulls = grain.measure(source, "t", ("a", "b"), table.rows)
    assert g3 == pytest.approx((5 - 2) / 5)
    assert nulls == pytest.approx(2 / 5)


def test_bound_ok_uses_the_distinct_product_and_the_smallest_count():
    table = table_of(
        100, column("a", 5, 100), column("b", 10, 100), column("c", 3, 40)
    )
    assert grain.bound_ok(table, ("a", "b"), 0.5)
    assert not grain.bound_ok(table, ("a", "b"), 0.4)
    assert not grain.bound_ok(table, ("a", "c"), 0.0)
    assert grain.bound_ok(table, ("a", "b"), 0.0) is False
    wide = table_of(100, column("a", 100, 100), column("c", 2, 40))
    assert not grain.bound_ok(wide, ("a", "c"), 0.0)
    assert grain.bound_ok(wide, ("a", "c"), 0.6)


def test_bound_ok_needs_a_row_budget_that_scales_with_the_tolerance():
    table = table_of(10, column("a", 6, 10), column("b", 2, 10))
    assert grain.bound_ok(table, ("a", "b"), 0.0)
    assert grain.bound_ok(table, ("a", "b"), 0.2)
    assert grain.bound_ok(table, ("a",), 0.4)
    assert not grain.bound_ok(table, ("a",), 0.3)
    assert not grain.bound_ok(table, ("b",), 0.7)
    assert grain.bound_ok(table, ("b",), 0.8)


def test_bound_ok_counts_a_constant_column_as_one_value_not_two():
    table = table_of(2, column("c", 1, 2))
    assert not grain.bound_ok(table, ("c",), 0.0)
    nothing = table_of(2, column("z", 0, 0))
    assert not grain.bound_ok(nothing, ("z",), 0.0)


def test_single_column_scan_of_an_empty_table():
    table = table_of(0, column("a", 0, 0))
    scan = empty_scan(rows=0)
    nonkey = grain.scan_single_columns(table, ["a"], scan)
    assert scan.measured == {("a",): (1.0, 0.0)}
    assert scan.keys == []
    assert nonkey == {("a",)}


def test_single_column_keys_carry_every_measure():
    table = table_of(1000, column("k", 1000, 1000), column("w", 400, 1000))
    scan = empty_scan(rows=1000)
    nonkey = grain.scan_single_columns(table, ["k", "w"], scan)
    key = scan.keys[0]
    assert key == grain.KeyCandidate(("k",), 0.0, 0.0, 1000, 1000, False)
    assert nonkey == {("w",)}
    small = empty_scan(rows=999)
    grain.scan_single_columns(
        table_of(999, column("k", 999, 999)), ["k"], small
    )
    assert small.keys == [grain.KeyCandidate(("k",), 0.0, 0.0, 999, 999, True)]


def test_single_column_with_too_many_nulls_cannot_join_a_key():
    table = table_of(10, column("n", 5, 6), column("m", 5, 10))
    scan = empty_scan(rows=10, eps_null=0.3)
    nonkey = grain.scan_single_columns(table, ["n", "m"], scan)
    assert scan.measured[("n",)] == (pytest.approx(0.5), pytest.approx(0.4))
    assert nonkey == {("m",)}
    looser = empty_scan(rows=10, eps_null=0.4)
    assert grain.scan_single_columns(table, ["n", "m"], looser) == {
        ("n",),
        ("m",),
    }


def test_level_candidates_follow_the_usable_column_order():
    names = ["z", "y", "x"]
    table = table_of(10, *(column(name, 9, 10) for name in names))
    nonkey = {("z",), ("y",), ("x",)}
    candidates, runnable = grain.level_candidates(names, nonkey, 2, table, 0.0)
    assert candidates == [("z", "y"), ("z", "x"), ("y", "x")]
    assert runnable == candidates
    only = {("y",), ("x",)}
    assert grain.level_candidates(names, only, 2, table, 0.0)[0] == [("y", "x")]
    third = grain.level_candidates(
        names, {("z", "y"), ("z", "x"), ("y", "x")}, 3, table, 0.0
    )
    assert third[0] == [("z", "y", "x")]


def test_a_level_candidate_needs_every_smaller_subset_to_be_a_non_key():
    names = ["a", "b", "c"]
    table = table_of(10, *(column(name, 9, 10) for name in names))
    nonkey = {("a", "b"), ("a", "c")}
    candidates, runnable = grain.level_candidates(names, nonkey, 3, table, 0.0)
    assert candidates == []
    assert runnable == []


def test_a_non_key_pair_that_was_queried_still_reaches_level_three():
    n = 24
    frame = pandas.DataFrame(
        {
            "a": numpy.arange(n) % 4,
            "b": numpy.arange(n) % 6,
            "c": numpy.arange(n) // 12,
        }
    )
    table, scan = scan_of(frame)[1:]
    assert grain.bound_ok(table, ("a", "b"), 0.0)
    assert [k.columns for k in scan.keys] == [("a", "b", "c")]
    assert scan.measured[("a", "b")] == (pytest.approx(0.5), 0.0)
    assert scan.levels_done == 3


def test_composite_key_candidates_record_distinct_rows_and_smallness():
    n = 20
    frame = pandas.DataFrame(
        {"a": numpy.arange(n) % 4, "b": numpy.arange(n) // 4 % 5}
    )
    frame.loc[0, "b"] = frame.loc[4, "b"]
    scan = scan_of(frame, eps_key=0.1)[2]
    assert [k.columns for k in scan.keys] == [("a", "b")]
    key = scan.keys[0]
    assert key.distinct == 19
    assert key.rows == n
    assert key.small_n is True
    assert key.g3 == pytest.approx(0.05)
    assert key.null_rate == 0.0


def test_a_composite_key_of_exactly_the_small_table_limit_is_not_small():
    n = 1000
    frame = pandas.DataFrame(
        {"a": numpy.arange(n) % 40, "b": numpy.arange(n) // 40}
    )
    scan = scan_of(frame, size=2)[2]
    assert [k.columns for k in scan.keys] == [("a", "b")]
    assert scan.keys[0].small_n is False
    assert scan.keys[0].distinct == 1000
    below = scan_of(frame.iloc[:999], size=2)[2]
    assert below.keys[0].small_n is True


def test_infer_keys_records_the_table_and_the_search_state():
    frame = pandas.DataFrame({"k": numpy.arange(5), "z": [3, 3, 3, 3, 3]})
    scan = scan_of(frame)[2]
    assert scan.table == "t"
    assert scan.rows == 5
    assert scan.partial is False
    assert scan.eps_key == 0.0
    assert scan.eps_null == 0.0
    assert scan.levels_done == 2
    assert list(scan.measured) == [("k",)]


def test_infer_keys_ignores_constant_and_empty_columns():
    frame = pandas.DataFrame(
        {
            "k": numpy.arange(6),
            "two": [0, 1] * 3,
            "const": [4] * 6,
            "empty": [None] * 6,
        }
    )
    scan = scan_of(frame)[2]
    assert set(scan.measured) >= {("k",), ("two",)}
    assert ("const",) not in scan.measured
    assert ("empty",) not in scan.measured
    assert set(scan.measured) == {("k",), ("two",)}


def test_a_column_with_two_values_is_usable():
    frame = pandas.DataFrame({"a": [0, 1, 0, 1], "b": [0, 0, 1, 1]})
    scan = scan_of(frame)[2]
    assert [k.columns for k in scan.keys] == [("a", "b")]


def test_the_default_null_tolerance_is_zero():
    frame = pandas.DataFrame({"k": [1.0, 2.0, 3.0, None, 5.0]})
    source, table = scan_of(frame)[:2]
    scan = grain.infer_keys(source, table, eps_key=0.5)
    assert scan.keys == []
    assert scan.eps_null == 0.0


def test_tolerances_may_be_exactly_one_and_the_size_exactly_one():
    source, table = scan_of(pandas.DataFrame({"a": [1, 2, 3]}))[:2]
    scan = grain.infer_keys(source, table, 1.0, 1.0, 1)
    assert scan.levels_done == 1
    assert [k.columns for k in scan.keys] == [("a",)]
    assert grain.infer_keys(source, table, 0.0, 1.0).eps_null == 1.0


def test_invalid_tolerance_messages_name_the_argument():
    source, table = scan_of(pandas.DataFrame({"a": [1, 2]}))[:2]
    with pytest.raises(ValueError, match=r"^eps_key must be in \[0, 1\]$"):
        grain.infer_keys(source, table, eps_key=-0.1)
    with pytest.raises(ValueError, match=r"^eps_null must be in \[0, 1\]$"):
        grain.infer_keys(source, table, eps_null=1.1)
    with pytest.raises(ValueError, match=r"^max_size must be at least 1$"):
        grain.infer_keys(source, table, max_size=0)


def build_budgeted(frame):
    meter = budget.BudgetMeter(budget.Budget(max_queries=1000))
    db = ingest_worlds.new_database("gm")
    ingest_worlds.load(db, "t", frame)
    source = profile.SqlSource(db.factory, meter)
    table = profile.profile_table(source, "t")
    return source, table, meter


def test_each_level_charges_two_queries_and_two_scans_per_runnable_set():
    n = 24
    frame = pandas.DataFrame(
        {
            "a": numpy.arange(n) % 4,
            "b": numpy.arange(n) % 6,
            "c": numpy.arange(n) // 12,
        }
    )
    source, table, meter = build_budgeted(frame)
    queries_before, rows_before = meter.queries, meter.rows
    grain.infer_keys(source, table)
    assert meter.queries - queries_before == 4
    assert meter.rows - rows_before == 96


def test_a_stopped_level_names_the_stage_and_the_level():
    frame = pandas.DataFrame(
        {c: numpy.tile(numpy.arange(5), 40) for c in "abc"}
        | {"d": numpy.repeat(numpy.arange(40), 5)}
    )
    source, table = build_budgeted(frame)[:2]
    source.meter = budget.BudgetMeter(budget.Budget(max_queries=0))
    scan = grain.infer_keys(source, table)
    stop = source.meter.stops[0]
    assert (stop.stage, stop.unit, stop.reason) == (
        "grain",
        "level 2",
        "queries",
    )
    assert scan.partial is True
    other = build_budgeted(frame)
    other[0].meter = budget.BudgetMeter(budget.Budget(max_queries=0))
    grain.infer_keys(other[0], other[1], stage="custom")
    assert other[0].meter.stops[0].stage == "custom"


def test_the_level_charge_is_two_queries_for_each_runnable_set():
    frame = pandas.DataFrame(
        {c: numpy.tile(numpy.arange(5), 40) for c in "abc"}
        | {"d": numpy.repeat(numpy.arange(40), 5)}
    )
    source, table = build_budgeted(frame)[:2]
    spend = budget.BudgetMeter(budget.Budget(max_queries=1000))
    source.meter = spend
    scan = grain.infer_keys(source, table, max_size=2)
    runnable = [
        cols
        for cols in scan.measured
        if len(cols) == 2 and ("d" in cols or cols == ("a", "b"))
    ]
    assert spend.queries == 2 * len(runnable)
    assert spend.rows == 200 * 2 * len(runnable)


def customers_table():
    frame = pandas.DataFrame(
        {
            "customer_id": [1, 1, 2, 2],
            "customer_name": ["a", "a", "b", "b"],
            "month": [1, 2, 1, 2],
            "order_no": [10, 11, 12, 13],
        }
    )
    source, table, scan = scan_of(frame)
    return source, table, scan


def test_a_word_with_several_columns_prefers_the_identifier():
    table = customers_table()[1]
    columns, questions = grain.match_columns("customer", table)
    assert columns == ("customer_id",)
    assert questions == []


def test_match_columns_are_returned_in_table_order():
    table = customers_table()[1]
    columns, questions = grain.match_columns("month-customer", table)
    assert columns == ("customer_id", "month")
    assert questions == []


def test_match_columns_ask_about_missing_and_ambiguous_words():
    table = customers_table()[1]
    columns, questions = grain.match_columns("warehouse customer", table)
    assert columns == ()
    assert questions == ["which column is 'warehouse'? found no column"]
    names = profile.TableProfile(
        "t", 4, (column("a_xy", 2, 4), column("b_xy", 2, 4), column("c", 4, 4))
    )
    assert grain.match_columns("xy", names) == (
        (),
        ["which column is 'xy'? found columns ['a_xy', 'b_xy']"],
    )


def test_match_columns_without_grain_words_say_so():
    table = customers_table()[1]
    assert grain.match_columns("the one row", table) == (
        (),
        ["the unit text has no grain words"],
    )


def test_unmapped_words_give_a_complete_unknown_verdict():
    source, table, scan = customers_table()
    verdict = grain.unit_verdict(source, table, "warehouse", scan)
    assert verdict.verdict == grain.UNKNOWN
    assert verdict.unit == "warehouse"
    assert verdict.unit_columns == ()
    assert verdict.reason == "unit words do not map to columns"
    assert verdict.questions == (
        "which column is 'warehouse'? found no column",
    )
    assert verdict.to_dict()["blocks"] is True


def test_a_matching_unit_reports_its_columns_and_reason():
    source, table, scan = customers_table()
    verdict = grain.unit_verdict(source, table, "customer month", scan)
    assert verdict.verdict == grain.MATCH
    assert verdict.unit == "customer month"
    assert verdict.unit_columns == ("customer_id", "month")
    assert verdict.reason == "unit columns are a key"
    assert verdict.questions == ()
    assert not verdict.blocks


def test_a_finer_table_names_the_first_key_it_found():
    source, table, scan = customers_table()
    verdict = grain.unit_verdict(source, table, "customer", scan)
    assert verdict.verdict == grain.FINER
    assert verdict.unit == "customer"
    assert verdict.unit_columns == ("customer_id",)
    assert verdict.reason == (
        "unit columns repeat; the table is keyed by "
        f"{list(scan.keys[0].columns)}, so aggregation is needed"
    )
    assert len(scan.keys) > 1
    assert not verdict.blocks


def test_a_table_without_any_key_gives_the_full_unknown_verdict():
    frame = pandas.DataFrame({"customer_id": [1, 1, 2, 2], "x": [0, 0, 1, 1]})
    source, table, scan = scan_of(frame)
    verdict = grain.unit_verdict(source, table, "customer", scan)
    assert verdict.verdict == grain.UNKNOWN
    assert verdict.unit == "customer"
    assert verdict.unit_columns == ("customer_id",)
    assert verdict.reason == "no key found at the declared tolerance"
    assert verdict.questions == ("which columns identify a row?",)
    assert verdict.blocks


def test_a_coarser_table_names_the_subset_that_is_a_key():
    frame = pandas.DataFrame(
        {"customer_id": [1, 2, 3, 4], "month": [1, 1, 2, 2]}
    )
    source, table, scan = scan_of(frame)
    verdict = grain.unit_verdict(source, table, "customer month", scan)
    assert verdict.verdict == grain.COARSER
    assert verdict.unit == "customer month"
    assert verdict.unit_columns == ("customer_id", "month")
    assert verdict.reason == (
        "['customer_id'] is already a key, so rows are coarser than the unit"
    )
    assert verdict.questions == ()
    assert verdict.blocks


def test_subset_verdict_stops_with_a_question_when_the_budget_does():
    verdict = grain.subset_verdict("u", ("a", "b"), lambda cols: None)
    assert verdict == grain.UnitVerdict(
        grain.UNKNOWN,
        "u",
        ("a", "b"),
        "budget stopped the unit check",
        ("raise the ingest budget",),
    )
    assert grain.subset_verdict("u", ("a", "b"), lambda cols: False) is None
    assert grain.subset_verdict("u", ("a",), lambda cols: True) is None


def test_subset_verdict_reports_the_first_subset_in_size_order():
    seen = []

    def key_like(cols):
        seen.append(cols)
        return cols == ("c",)

    verdict = grain.subset_verdict("u", ("a", "b", "c"), key_like)
    assert verdict.verdict == grain.COARSER
    assert verdict.unit == "u"
    assert verdict.unit_columns == ("a", "b", "c")
    assert verdict.reason == (
        "['c'] is already a key, so rows are coarser than the unit"
    )
    assert seen == [("a",), ("b",), ("c",)]


def test_the_unit_check_charges_the_unit_stage_and_stops_on_budget():
    frame = pandas.DataFrame(
        {
            "customer_id": [1, 1, 2, 2],
            "month": [1, 2, 1, 2],
            "x": [0, 1, 2, 3],
        }
    )
    db = ingest_worlds.new_database("uv")
    ingest_worlds.load(db, "t", frame)
    source = profile.SqlSource(db.factory)
    table = profile.profile_table(source, "t")
    scan = grain.infer_keys(source, table, max_size=1)
    source.meter = budget.BudgetMeter(budget.Budget(max_queries=0))
    verdict = grain.unit_verdict(source, table, "customer month", scan)
    assert verdict.verdict == grain.UNKNOWN
    assert verdict.reason == "budget stopped the unit check"
    assert verdict.questions == ("raise the ingest budget",)
    assert verdict.unit == "customer month"
    assert verdict.unit_columns == ("customer_id", "month")
    stop = source.meter.stops[0]
    assert (stop.stage, stop.unit) == ("unit", "customer_id,month")


def test_the_unit_check_measures_unseen_sets_once_and_charges_for_them():
    frame = pandas.DataFrame(
        {
            "customer_id": [1, 1, 2, 2],
            "month": [1, 2, 1, 2],
            "x": [0, 1, 2, 3],
        }
    )
    db = ingest_worlds.new_database("uw")
    ingest_worlds.load(db, "t", frame)
    source = profile.SqlSource(db.factory)
    table = profile.profile_table(source, "t")
    scan = grain.infer_keys(source, table, max_size=1)
    spend = budget.BudgetMeter(budget.Budget(max_queries=100))
    source.meter = spend
    before = source.queries
    verdict = grain.unit_verdict(source, table, "customer month", scan)
    assert verdict.verdict == grain.MATCH
    assert ("customer_id", "month") in scan.measured
    assert source.queries - before == 2
    assert spend.queries == 2
    assert spend.rows == 8


def test_a_set_that_cannot_be_a_key_is_never_queried_by_the_unit_check():
    frame = pandas.DataFrame(
        {"customer_id": [1, 1, 2, 2], "month": [1, 2, 1, 2]}
    )
    db = ingest_worlds.new_database("ux")
    ingest_worlds.load(db, "t", frame)
    source = profile.SqlSource(db.factory)
    table = profile.profile_table(source, "t")
    scan = grain.infer_keys(source, table, max_size=1)
    before = source.queries
    verdict = grain.unit_verdict(source, table, "customer", scan)
    assert source.queries == before
    assert verdict.verdict == grain.UNKNOWN


def test_match_columns_follow_the_table_order_not_the_alphabet():
    table = profile.TableProfile(
        "t", 4, (column("zone", 2, 4), column("alpha", 2, 4))
    )
    assert grain.match_columns("alpha zone", table) == (("zone", "alpha"), [])


def test_a_verdict_serializes_every_field_and_whether_it_blocks():
    verdict = grain.UnitVerdict(
        grain.UNKNOWN, "u", ("a", "b"), "why", ("q1", "q2")
    )
    assert verdict.to_dict() == {
        "verdict": "unknown",
        "unit": "u",
        "unit_columns": ["a", "b"],
        "reason": "why",
        "questions": ["q1", "q2"],
        "blocks": True,
    }
    match = grain.UnitVerdict(grain.MATCH, "u", ("a",), "ok")
    assert match.to_dict()["questions"] == []
    assert match.to_dict()["blocks"] is False
    assert grain.UnitVerdict(grain.FINER, "u", (), "r").blocks is False
    assert grain.UnitVerdict(grain.COARSER, "u", (), "r").blocks is True


def test_a_scan_serializes_keys_ties_and_state():
    scan = empty_scan(
        keys=[candidate(("a",)), candidate(("b",))], levels_done=2
    )
    document = scan.to_dict()
    assert document["table"] == "t"
    assert document["rows"] == 10
    assert document["keys"][0] == {
        "columns": ["a"],
        "g3": 0.0,
        "null_rate": 0.0,
        "distinct": 5,
        "rows": 10,
        "small_n": True,
    }
    assert document["ties"] == [[["a"], ["b"]]]
    assert document["levels_done"] == 2
    assert document["partial"] is False
    assert document["eps_key"] == 0.0
    assert document["eps_null"] == 0.0
