"""Greedy counterfactual search over actionable columns, then pruning."""

import numpy
import pandas
import pytest

from scmle.explainability import counterfactual

BACKGROUND = pandas.DataFrame(
    {
        "income": [float(v) for v in range(0, 101, 5)],
        "age": [30.0] * 21,
        "city": ["north"] * 11 + ["south"] * 7 + ["east"] * 3,
    }
)


def score(frame):
    return (
        frame["income"].to_numpy(dtype=float)
        + 30.0 * (frame["city"] == "south").to_numpy()
    )


def recourse(threshold=60.0, max_changes=3, actionable=("income", "city")):
    return counterfactual.Recourse(
        progress=score,
        reached=lambda f: score(f) >= threshold,
        predict=lambda f: numpy.where(score(f) >= threshold, "yes", "no"),
        background=BACKGROUND,
        actionable=actionable,
        max_changes=max_changes,
    )


def test_the_search_constants_are_fixed():
    assert counterfactual.QUANTILE_STEPS == 21
    assert counterfactual.MAX_CATEGORIES == 30
    assert counterfactual.MIN_SCALE == 1e-9


def test_a_numeric_proposal_is_the_sorted_distinct_quantiles():
    values = counterfactual.Recourse.propose(
        pandas.Series([1.0, 1.0, 3.0, None])
    )
    grid = numpy.quantile([1.0, 1.0, 3.0], numpy.linspace(0, 1, 21))
    assert values == sorted({float(v) for v in grid})
    assert values[0] == 1.0 and values[-1] == 3.0


def test_a_categorical_proposal_is_most_frequent_first_and_capped():
    column = pandas.Series(["b", "a", "a", None, "c", "a", "b"])
    assert counterfactual.Recourse.propose(column) == ["a", "b", "c"]
    many = pandas.Series([f"v{i:02d}" for i in range(40)])
    assert len(counterfactual.Recourse.propose(many)) == 30
    flags = pandas.Series([True, False, True])
    assert counterfactual.Recourse.propose(flags) == [True, False]


def test_a_measure_is_the_larger_of_spread_and_iqr_or_one():
    column = pandas.Series([0.0, 0.0, 0.0, 10.0])
    assert counterfactual.Recourse.measure(column) == pytest.approx(
        column.std()
    )
    wide = pandas.Series([0.0, 10.0, 10.0, 10.0, 20.0] * 4)
    assert counterfactual.Recourse.measure(wide) == pytest.approx(
        max(10.0 - 10.0, wide.std())
    )
    iqr = pandas.Series([0.0, 0.0, 10.0, 10.0])
    low, high = numpy.quantile(iqr, [0.25, 0.75])
    assert counterfactual.Recourse.measure(iqr) == pytest.approx(
        max(high - low, iqr.std())
    )
    assert counterfactual.Recourse.measure(pandas.Series([5.0, 5.0])) == 1e-9
    assert counterfactual.Recourse.measure(pandas.Series(["a", "b"])) == 1.0
    assert counterfactual.Recourse.measure(pandas.Series([True, False])) == 1.0


def test_a_cost_is_scaled_for_numbers_and_one_otherwise():
    search = recourse()
    scale = search.scales["income"]
    assert search.cost("income", 10.0, 40.0) == pytest.approx(30.0 / scale)
    assert search.cost("income", 40, 10) == pytest.approx(30.0 / scale)
    assert search.cost("city", "north", "south") == 1.0
    assert search.cost("income", 0, True) == 1.0


def test_a_distance_sums_the_costs_of_differing_columns():
    search = recourse()
    original = {"income": 10.0, "age": 1.0, "city": "north"}
    current = {"income": 20.0, "age": 99.0, "city": "south"}
    assert search.distance(original, current) == pytest.approx(
        10.0 / search.scales["income"] + 1.0
    )
    assert search.distance(original, dict(original)) == 0.0


def test_candidates_skip_locked_columns_and_current_values():
    search = recourse()
    current = {"income": 0.0, "age": 30.0, "city": "north"}
    batch, moves = search.candidates(current, {"income"})
    assert moves == [("city", "south"), ("city", "east")]
    assert list(batch["city"]) == ["south", "east"]
    assert list(batch["income"]) == [0.0, 0.0]
    full, all_moves = search.candidates(current, set())
    assert len(all_moves) == 20 + 2
    assert ("income", 0.0) not in all_moves
    assert len(full) == len(all_moves)


def test_a_reached_row_is_returned_unchanged():
    search = recourse(threshold=0.0)
    row = {"income": 10.0, "age": 30.0, "city": "north"}
    found = search.search(row)
    assert found.found is True
    assert found.changed == {}
    assert found.edits == 0
    assert found.distance == 0.0
    assert found.row == row
    assert found.prediction == "yes"


def test_a_search_takes_the_best_move_then_prunes_needless_changes():
    search = recourse(threshold=60.0)
    row = {"income": 40.0, "age": 30.0, "city": "north"}
    found = search.search(row)
    assert found.found is True
    assert found.changed == {"income": (40.0, 100.0)}
    assert found.row == {"income": 100.0, "age": 30.0, "city": "north"}
    assert found.prediction == "yes"
    assert found.original == row
    assert found.distance == pytest.approx(60.0 / search.scales["income"])


def test_a_pruning_reverts_the_most_costly_change_that_is_not_needed():
    search = recourse(threshold=60.0)
    original = {"income": 40.0, "age": 30.0, "city": "north"}
    current = {"income": 100.0, "age": 30.0, "city": "south"}
    row, kept = search.prune(
        original, current, {"income": 100.0, "city": "south"}
    )
    assert kept == {"city": "south"}
    assert row == {"income": 40.0, "age": 30.0, "city": "south"}


def test_a_search_stops_at_the_change_limit():
    search = recourse(threshold=1000.0, max_changes=1)
    found = search.search({"income": 0.0, "age": 30.0, "city": "north"})
    assert found.found is False
    assert found.edits == 1
    assert found.changed == {"income": (0.0, 100.0)}
    assert found.prediction == "no"


def test_a_search_stops_when_no_move_improves():
    search = recourse(threshold=1000.0)
    found = search.search({"income": 100.0, "age": 30.0, "city": "south"})
    assert found.found is False
    assert found.changed == {}


def test_a_search_stops_when_no_candidate_is_left():
    search = recourse(threshold=1000.0, actionable=("city",))
    found = search.search({"income": 0.0, "age": 30.0, "city": "north"})
    assert found.found is False
    assert found.changed == {"city": ("north", "south")}
    assert found.distance == 1.0
