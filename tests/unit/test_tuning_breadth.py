"""Grid search and Hyperband brackets."""

import types

import pytest

from scmle.contracts import search
from scmle.contracts import tasks
from scmle.experimentation import tuning
from scmle.optimization import optimizers

SPACE = search.SearchSpace(
    (
        search.FloatParameter("rate", 0.01, 1.0, log=True),
        search.IntParameter("depth", 2, 10),
        search.ChoiceParameter("kind", ("a", "b")),
    )
)


def test_the_grid_covers_the_corners_and_the_middle_of_every_axis():
    grid = optimizers.create_optimizer("grid", SPACE, 0, True)

    points = [grid.ask() for count in range(3 * 3 * 2)]

    assert len({tuple(p.values()) for p in points}) == 18
    rates = sorted({round(p["rate"], 6) for p in points})
    assert rates == pytest.approx([0.01, 0.1, 1.0])
    assert {p["depth"] for p in points} == {2, 6, 10}
    assert {p["kind"] for p in points} == {"a", "b"}


def test_the_grid_order_depends_only_on_the_seed():
    first = optimizers.create_optimizer("grid", SPACE, 5, True)
    again = optimizers.create_optimizer("grid", SPACE, 5, True)
    other = optimizers.create_optimizer("grid", SPACE, 6, True)

    a = [first.ask() for count in range(6)]
    b = [again.ask() for count in range(6)]
    c = [other.ask() for count in range(6)]

    assert a == b and a != c


def test_after_the_grid_is_exhausted_random_draws_continue():
    grid = optimizers.create_optimizer("grid", SPACE, 0, True)
    for point in range(18):
        assert grid.ask() is not None, point

    extra = grid.ask()

    assert set(extra) == {"rate", "depth", "kind"}
    assert 0.01 <= extra["rate"] <= 1.0


def tuner_with(folds, eta):
    tuner = tuning.Tuner.__new__(tuning.Tuner)
    tuner.runner = types.SimpleNamespace(folds=list(range(folds)))
    tuner.search = tasks.Search(pruning="hyperband", eta=eta)
    return tuner


def test_hyperband_brackets_start_cheap_and_always_end_with_the_full_cost():
    schedule = tuner_with(9, 3).rung_schedule()

    assert schedule == [[1, 3, None], [3, None], [None]]


def test_with_few_folds_the_schedule_collapses():
    assert tuner_with(5, 3).rung_schedule() == [[1, None], [None]]
    assert tuner_with(1, 3).rung_schedule() == [[None]]
