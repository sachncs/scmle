"""Unit tests of the feature grammar and its evaluation, by full name."""

import math

import pandas
import pytest

from scmle.assurance.ingest import asof_view
from scmle.assurance.ingest.synthesize import evaluation
from scmle.assurance.ingest.synthesize import grammar
from scmle.contracts import errors


def events_view():
    frame = pandas.DataFrame(
        {
            "cid": [1, 1, 1, 2],
            "ev": [1.0, 5.0, 9.0, 3.0],
            "rec": [1.0, 5.0, 9.0, 3.0],
            "val": [10.0, 20.0, 30.0, 40.0],
            "kind": ["a", "b", "a", "c"],
            "order_id": [100, 101, 102, 103],
            "flag": [True, False, True, True],
        }
    )
    rule = asof_view.Timing(asof_view.TRANSACTION, "rec")
    return asof_view.View(frame, "cid", rule, "ev", "events")


def leaf_view():
    frame = pandas.DataFrame(
        {
            "order_id": [100, 100, 101, 102, 103],
            "ev": [1.0, 2.0, 5.0, 9.0, 3.0],
            "rec": [1.0, 2.0, 5.0, 9.0, 3.0],
            "qty": [1.0, 3.0, 5.0, 7.0, 9.0],
        }
    )
    rule = asof_view.Timing(asof_view.TRANSACTION, "rec")
    return asof_view.View(frame, "order_id", rule, "ev", "items")


def targets():
    return pandas.DataFrame({"customer": [1, 2, 3], "at": [10.0, 10.0, 10.0]})


# ---- grammar ---------------------------------------------------------------


def test_the_grammar_constants_have_their_values():
    assert grammar.PRIMITIVES == (
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
    assert grammar.VALUE_FREE == ("count", "time_since_last")
    assert grammar.NUMERIC_ONLY == ("sum", "mean", "min", "max", "std", "slope")
    assert grammar.WINDOWS_DAYS == (7.0, 30.0, 90.0)
    assert grammar.NESTED_INNER == ("count", "sum", "mean")
    assert grammar.NESTED_OUTER == ("mean", "max", "sum")
    assert (grammar.CHILD, grammar.PARENT) == ("child", "parent")


def test_a_source_refuses_an_unknown_role():
    with pytest.raises(errors.InputError, match="role must be child or parent"):
        grammar.Source(events_view(), "customer", "sibling")
    assert grammar.Source(events_view(), "customer").role == "child"


def test_feature_expressions_read_as_text():
    plain = grammar.Feature("f", "aggregate", 0, "sum", "val", 7.0)
    assert plain.expression() == "sum(val) over 7d"
    free = grammar.Feature("f", "aggregate", 0, "count", None, None)
    assert free.expression() == "count(*) over all"
    nested = grammar.Feature(
        "f", "nested", 0, "max", "qty", 30.0, nest=0, inner="sum"
    )
    assert nested.expression() == "max(per-row sum(qty) in 30d)"


def test_caption_writes_windows():
    assert grammar.caption(None) == "all"
    assert grammar.caption(7.0) == "7d"
    assert grammar.caption(2.5) == "2.5d"


def test_sift_skips_structure_and_identifiers():
    numeric, other = grammar.sift(grammar.Source(events_view(), "customer"))
    assert numeric == ["val"]
    assert other == ["kind", "flag"]


def test_roll_lists_aggregates_per_window_in_order():
    specs = grammar.roll(2, "t", ["v"], ["k"], [7.0, None])
    names = [s.name for s in specs]
    assert len(specs) == 2 * (2 + 7 + 2)
    assert names[:3] == [
        "t__count__7d",
        "t__time_since_last__7d",
        "t__sum__v__7d",
    ]
    assert names[8] == "t__last__v__7d"
    assert names[9:11] == ["t__nunique__v__7d", "t__nunique__k__7d"]
    assert names[11] == "t__count__all"
    assert {s.source for s in specs} == {2}
    assert specs[0].value is None
    assert specs[11].window_days is None


def test_propose_gives_lookups_for_a_parent():
    source = grammar.Source(events_view(), "customer", grammar.PARENT)
    specs = grammar.propose([source])
    assert [s.name for s in specs] == [
        "events__lookup__val",
        "events__lookup__kind",
        "events__lookup__flag",
    ]
    assert {(s.kind, s.primitive) for s in specs} == {("lookup", "last")}


def test_propose_counts_child_and_nested_features():
    mid = grammar.Source(events_view(), "customer")
    nest = grammar.Nest(mid, leaf_view(), "order_id")
    specs = grammar.propose([mid], [nest], [None])
    child = [s for s in specs if s.kind == "aggregate"]
    nested = [s for s in specs if s.kind == "nested"]
    assert len(child) == 2 + 7 + 3
    # count: three outers of no column; sum and mean: three outers of qty
    assert len(nested) == 3 + 3 + 3
    assert nested[0].name == "events__mean__of__items__count__*__all"
    assert nested[0].nest == 0
    assert nested[0].inner == "count"
    assert nested[-1].name == "events__sum__of__items__mean__qty__all"


# ---- evaluation ------------------------------------------------------------


def test_outcomes_builds_the_cutoff_frame():
    view = events_view()
    out = evaluation.outcomes(targets(), "customer", "at", 2.0, view)
    assert out.columns.tolist() == ["cid", "target_row", "cutoff_days"]
    assert out["cid"].tolist() == [1, 2, 3]
    assert out["target_row"].tolist() == [0, 1, 2]
    assert out["cutoff_days"].tolist() == [8.0, 8.0, 8.0]


def visible():
    view = events_view()
    return view.visible(
        evaluation.outcomes(targets(), "customer", "at", 0.0, view)
    )


def values(series):
    return [
        None if isinstance(v, float) and math.isnan(v) else v for v in series
    ]


@pytest.mark.parametrize(
    ("primitive", "window", "expected"),
    [
        ("count", None, [3.0, 1.0, 0.0]),
        ("count", 7.0, [2.0, 0.0, 0.0]),
        ("time_since_last", None, [1.0, 7.0, None]),
        ("sum", None, [60.0, 40.0, 0.0]),
        ("mean", None, [20.0, 40.0, None]),
        ("min", None, [10.0, 40.0, None]),
        ("max", None, [30.0, 40.0, None]),
        ("std", None, [10.0, None, None]),
        ("slope", None, [2.5, None, None]),
        ("last", None, [30.0, 40.0, None]),
        ("nunique", None, [2.0, 1.0, 0.0]),
    ],
)
def test_fold_computes_each_primitive(primitive, window, expected):
    column = "kind" if primitive == "nunique" else "val"
    out = evaluation.fold(visible(), primitive, column, window, 3)
    assert values(out.tolist()) == expected


def test_fold_refuses_unknown_primitives_and_missing_values():
    with pytest.raises(errors.InputError, match="unknown primitive: median"):
        evaluation.fold(visible(), "median", "val", None, 3)
    with pytest.raises(errors.InputError, match="sum needs a value column"):
        evaluation.fold(visible(), "sum", None, None, 3)


def test_aggregate_refuses_a_primitive_it_does_not_compute():
    with pytest.raises(errors.InputError, match="unknown primitive: count"):
        evaluation.aggregate(visible(), "count", "val", pandas.RangeIndex(3))


def test_slope_needs_two_rows_and_spread_in_time():
    rows = pandas.DataFrame(
        {
            "target_row": [0, 0, 1, 1, 2],
            "event_days": [0.0, 2.0, 1.0, 1.0, 4.0],
            "v": [1.0, 5.0, 3.0, 4.0, 9.0],
        }
    )
    assert values(evaluation.slope(rows, "v").tolist()) == [2.0, None, None]
    assert evaluation.MIN_SLOPE_ROWS == 2


def test_evaluate_aggregates_and_lookups():
    source = grammar.Source(events_view(), "customer")
    spec = grammar.Feature("f", "aggregate", 0, "sum", "val", None)
    out = evaluation.evaluate(spec, targets(), "at", [source], [], 0.0)
    assert out.tolist() == [60.0, 40.0, 0.0]
    early = evaluation.evaluate(spec, targets(), "at", [source], [], 6.0)
    assert early.tolist() == [10.0, 40.0, 0.0]


def test_evaluate_nested_features_through_both_views():
    mid = grammar.Source(events_view(), "customer")
    nest = grammar.Nest(mid, leaf_view(), "order_id")
    spec = grammar.Feature(
        "f", "nested", 0, "max", "qty", None, nest=0, inner="sum"
    )
    out = evaluation.evaluate(spec, targets(), "at", [mid], [nest], 0.0)
    # orders of customer 1 hold qty sums 4, 5 and 7; customer 2 holds 9
    assert values(out.tolist()) == [7.0, 9.0, None]
    windowed = grammar.Feature(
        "f", "nested", 0, "sum", None, 7.0, nest=0, inner="count"
    )
    out = evaluation.evaluate(windowed, targets(), "at", [mid], [nest], 0.0)
    assert out.tolist() == [2.0, 0.0, 0.0]


def test_evaluate_refuses_a_nested_feature_without_its_nest():
    source = grammar.Source(events_view(), "customer")
    spec = grammar.Feature("f", "nested", 0, "max", "qty", None, inner="sum")
    with pytest.raises(errors.ScmleError, match="lacks nest or inner"):
        evaluation.evaluate(spec, targets(), "at", [source], [], 0.0)
