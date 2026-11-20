"""The fairness report's group boundary and its record keys."""

import dataclasses

import numpy
import pandas

from scmle.evaluation import fairness
from tests.unit.test_fairness import make_inputs


def labels(sizes: dict[str, int]) -> pandas.Series:
    return pandas.Series(numpy.repeat(list(sizes), list(sizes.values())))


def test_a_group_of_exactly_the_minimum_is_reported() -> None:
    least = fairness.MIN_GROUP_ROWS
    groups = labels({"a": least, "b": least - 1})
    assert fairness.Fairness().exclude(groups) == {"b": least - 1}
    truth = numpy.zeros(len(groups))
    found = fairness.Fairness().numeric(groups, truth, truth)
    assert list(found["groups"]) == ["a"]
    assert found["excluded_groups"] == {"b": least - 1}
    assert found["power_warning"] is True


def test_a_binary_group_reports_its_base_rate() -> None:
    least = fairness.MIN_GROUP_ROWS
    groups = labels({"a": least})
    truth = numpy.array([1] * 6 + [0] * (least - 6))
    found = fairness.Fairness().binary(groups, truth, truth, 1)
    assert found["groups"]["a"]["base_rate"] == 6 / least


def test_binary_intersections_report_classification_rates() -> None:
    inputs = make_inputs()
    rng = numpy.random.default_rng(1)
    frame = inputs.features.assign(
        region=rng.choice(["n", "s"], len(inputs.features))
    )
    found = fairness.Fairness().compute(
        dataclasses.replace(
            inputs, features=frame, sensitive=("group", "region")
        )
    )
    cells = found["columns"]["intersectional"]["groups"]
    assert cells
    assert all("base_rate" in cell for cell in cells.values())
