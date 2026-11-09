"""One calibration score per entity."""

import numpy

from scmle.evaluation import conformal


def test_one_position_per_entity_in_order_of_first_appearance():
    entities = ["b", "a", "b", "c", "a", "b"]
    chosen = conformal.one_row_per_entity(entities, numpy.random.default_rng(0))
    assert len(chosen) == 3
    assert [entities[i] for i in chosen] == ["b", "a", "c"]


def test_the_choice_is_reproducible_and_covers_each_row_over_draws():
    entities = ["a"] * 4 + ["b"] * 4
    first = conformal.one_row_per_entity(entities, numpy.random.default_rng(3))
    again = conformal.one_row_per_entity(entities, numpy.random.default_rng(3))
    assert first == again
    seen = {
        conformal.one_row_per_entity(entities, numpy.random.default_rng(s))[0]
        for s in range(60)
    }
    assert seen == {0, 1, 2, 3}


def test_no_rows_give_no_positions():
    assert conformal.one_row_per_entity([], numpy.random.default_rng(0)) == []
