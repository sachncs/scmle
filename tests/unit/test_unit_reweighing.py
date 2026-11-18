"""Reweighing makes group and label independent in the weighted data."""

import numpy
import pandas
import pytest

from scmle.contracts import errors
from scmle.evaluation import reweighing


def test_a_weighing_gives_the_kamiran_calders_weights():
    groups = pandas.Series(["a", "a", "a", "b"])
    labels = pandas.Series([1, 1, 0, 0])
    weights = reweighing.weigh(groups, labels)
    raw = numpy.array(
        [0.75 * 0.5 / 0.5] * 2 + [0.75 * 0.5 / 0.25, 0.25 * 0.5 / 0.25]
    )
    numpy.testing.assert_allclose(weights, raw / raw.mean())
    assert weights.mean() == pytest.approx(1.0)


def test_independent_groups_and_labels_get_unit_weights():
    groups = pandas.Series(["a", "a", "b", "b"])
    labels = pandas.Series([0, 1, 0, 1])
    numpy.testing.assert_allclose(reweighing.weigh(groups, labels), 1.0)


def test_weighted_data_has_no_group_label_association():
    rng = numpy.random.default_rng(0)
    groups = pandas.Series(rng.choice(["x", "y"], 400))
    labels = pandas.Series(
        (rng.random(400) < numpy.where(groups == "x", 0.8, 0.2)).astype(int)
    )
    weights = reweighing.weigh(groups, labels)
    for group in ("x", "y"):
        mask = (groups == group).to_numpy()
        rate = (weights[mask] * labels[mask]).sum() / weights[mask].sum()
        assert rate == pytest.approx(labels.mean())


def test_a_weighing_ignores_the_series_index():
    groups = pandas.Series(["a", "b"], index=[5, 9])
    labels = pandas.Series([1, 1], index=[0, 1])
    numpy.testing.assert_allclose(reweighing.weigh(groups, labels), [1.0, 1.0])


def test_unequal_lengths_raise_an_input_error():
    with pytest.raises(errors.InputError, match="same non-zero length"):
        reweighing.weigh(pandas.Series(["a"]), pandas.Series([1, 0]))


def test_empty_inputs_raise_an_input_error():
    with pytest.raises(errors.InputError, match="same non-zero length"):
        reweighing.weigh(
            pandas.Series([], dtype=str), pandas.Series([], dtype=int)
        )
