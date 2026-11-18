"""Conformalized quantile bands widen by the finite-sample miss quantile."""

import math

import numpy
import pandas
import pytest

from scmle.contracts import errors
from scmle.evaluation import quantile


class Constant:
    """A stand-in model that predicts given values."""

    def __init__(self, values):
        self.values = numpy.asarray(values, dtype=float)

    def predict(self, features):
        return self.values[: len(features)]


def test_the_quantile_levels_are_five_and_ninety_five_percent():
    assert quantile.LOWER_LEVEL == 0.05
    assert quantile.UPPER_LEVEL == 0.95


def test_a_corridor_swaps_crossed_edges():
    frame = pandas.DataFrame({"a": [0, 0, 0]})
    low, high = quantile.corridor(
        Constant([1.0, 5.0, 2.0]), Constant([3.0, 4.0, 2.0]), frame
    )
    assert list(low) == [1.0, 4.0, 2.0]
    assert list(high) == [3.0, 5.0, 2.0]
    assert low.dtype == float


def test_a_band_scores_the_signed_distance_outside_the_edges():
    observed = [0.0, 5.0, 12.0] + [5.0] * 17
    lower = [1.0] + [0.0] * 19
    upper = [10.0] * 20
    band = quantile.Band.calibrate(observed, lower, upper)
    assert band.scores[:3] == (1.0, -5.0, 2.0)
    assert len(band.scores) == 20
    assert all(isinstance(s, float) for s in band.scores)


def test_a_band_with_too_few_rows_raises_an_input_error():
    with pytest.raises(errors.InputError, match="at least 20"):
        quantile.Band.calibrate([1.0] * 19, [0.0] * 19, [2.0] * 19)


def test_a_band_with_exactly_twenty_rows_calibrates():
    band = quantile.Band.calibrate([1.0] * 20, [0.0] * 20, [2.0] * 20)
    assert band.scores == (-1.0,) * 20


def test_a_margin_is_the_conformal_rank_of_the_misses():
    band = quantile.Band(tuple(float(i) for i in range(1, 21)))
    assert band.margin(0.5) == 11.0
    assert band.margin(0.9) == 19.0
    assert band.margin(0.95) == 20.0
    assert band.margin(0.96) == math.inf


@pytest.mark.parametrize("coverage", [0.0, 1.0, -0.1, 1.5])
def test_a_margin_outside_the_open_unit_interval_raises(coverage):
    band = quantile.Band(tuple(float(i) for i in range(20)))
    with pytest.raises(errors.InputError, match="strictly between 0 and 1"):
        band.margin(coverage)


def intervals():
    band = quantile.Band(tuple(float(i) for i in range(1, 21)))
    return quantile.QuantileIntervals(
        Constant([0.0, 10.0]), Constant([4.0, 6.0]), band
    )


def test_an_interval_widens_the_native_band_by_the_margin():
    frame = pandas.DataFrame({"a": [0, 0]})
    low, high = intervals().interval(frame, 0.5)
    assert list(low) == [-11.0, -5.0]
    assert list(high) == [15.0, 21.0]


def test_a_native_band_is_the_uncalibrated_corridor():
    frame = pandas.DataFrame({"a": [0, 0]})
    low, high = intervals().native(frame)
    assert list(low) == [0.0, 6.0]
    assert list(high) == [4.0, 10.0]


def test_a_covering_reports_nominal_empirical_and_width():
    frame = pandas.DataFrame({"a": [0, 0]})
    band = quantile.Band((-4.0,) * 20)
    model = quantile.QuantileIntervals(
        Constant([0.0, 0.0]), Constant([10.0, 10.0]), band
    )
    curve = model.covering(frame, [5.0, 9.5], [0.5])
    assert curve == [{"nominal": 0.5, "empirical": 0.5, "mean_width": 2.0}]


def test_a_covering_counts_values_on_the_edges_as_inside():
    frame = pandas.DataFrame({"a": [0, 0]})
    band = quantile.Band((0.0,) * 20)
    model = quantile.QuantileIntervals(
        Constant([0.0, 0.0]), Constant([10.0, 10.0]), band
    )
    curve = model.covering(frame, [0.0, 10.0], [0.5, 0.9])
    assert [c["empirical"] for c in curve] == [1.0, 1.0]
    assert [c["nominal"] for c in curve] == [0.5, 0.9]
    assert [c["mean_width"] for c in curve] == [10.0, 10.0]
