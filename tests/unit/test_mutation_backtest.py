"""Exact values of the forecaster backtest: baselines, scores and reports."""

import numpy
import pandas
import pytest

from scmle.evaluation import backtest


class Forecaster:
    """A fitted forecaster stub with fixed predictions.

    Attributes:
        target_columns: The horizon columns, the primary one first.
        value_column: The column holding the current value.
    """

    def __init__(self, predictions: list[list[str]]) -> None:
        self.target_columns = ["next", "after"]
        self.value_column = "now"
        self.predictions = predictions

    def predict(self, features: pandas.DataFrame) -> list[list[str]]:
        """Returns the fixed predictions, one row per feature row.

        Args:
            features: Rows to forecast from.

        Returns:
            The predictions as text, so only a float conversion reads them.
        """
        return self.predictions[: len(features)]


def frame() -> tuple[Forecaster, pandas.DataFrame, pandas.Series]:
    """Two rows whose errors differ by step and by baseline."""
    features = pandas.DataFrame({"now": [1.0, 2.0], "after": [5.0, 9.0]})
    primary = pandas.Series([3.0, 4.0])
    return Forecaster([["2.0", "6.0"], ["4.0", "7.0"]]), features, primary


def test_seasonal_naive_repeats_the_value_whole_seasons_back() -> None:
    values = pandas.DataFrame({"value": numpy.arange(10.0)})
    found = backtest.seasonal_naive(
        values, numpy.array([6, 7]), "value", None, 4, 3
    )
    expected = numpy.array([[4.0, 5.0, 6.0, 4.0], [5.0, 6.0, 7.0, 5.0]])
    numpy.testing.assert_array_equal(found, expected)


def test_seasonal_naive_shifts_within_each_entity() -> None:
    values = pandas.DataFrame(
        {"entity": ["a", "b"] * 4, "value": numpy.arange(8.0)}
    )
    found = backtest.seasonal_naive(
        values, numpy.array([2, 7]), "value", "entity", 2, 2
    )
    expected = numpy.array([[0.0, 2.0], [5.0, 7.0]])
    numpy.testing.assert_array_equal(found, expected)


def test_seasonal_naive_is_nan_before_the_entity_starts() -> None:
    values = pandas.DataFrame({"entity": ["a", "b"], "value": [1.0, 2.0]})
    found = backtest.seasonal_naive(
        values, numpy.array([0, 1]), "value", "entity", 1, 2
    )
    assert numpy.isnan(found).all()
    assert found.shape == (2, 1)


def test_score_is_the_mean_absolute_error_per_step() -> None:
    truth = numpy.array([[1.0, 2.0], [3.0, 4.0]])
    guess = numpy.array([[2.0, 0.0], [1.0, 4.0]])
    assert backtest.score(truth, guess) == [1.5, 1.0]


def test_report_compares_model_and_baselines_per_step() -> None:
    estimator, features, primary = frame()
    seasonal = numpy.array([[3.0, 5.0], [0.0, 9.0]])
    found = backtest.report(estimator, features, primary, seasonal)
    assert found == [
        {
            "step": 1,
            "model_mae": 0.5,
            "last_value_mae": 2.0,
            "seasonal_naive_mae": 2.0,
        },
        {
            "step": 2,
            "model_mae": 1.5,
            "last_value_mae": 5.5,
            "seasonal_naive_mae": 0.0,
        },
    ]


def test_report_without_a_season_has_no_seasonal_error() -> None:
    estimator, features, primary = frame()
    found = backtest.report(estimator, features, primary, None)
    assert [r["seasonal_naive_mae"] for r in found] == [None, None]
    assert [r["step"] for r in found] == [1, 2]


def test_origins_number_each_fold_from_one() -> None:
    estimator, features, primary = frame()
    found = backtest.origins(
        [(estimator, features, primary), (estimator, features[:1], primary[:1])]
    )
    assert found == [
        {
            "origin": 1,
            "rows": 2,
            "model_mae": [0.5, 1.5],
            "last_value_mae": [2.0, 5.5],
        },
        {
            "origin": 2,
            "rows": 1,
            "model_mae": [1.0, 1.0],
            "last_value_mae": [2.0, 4.0],
        },
    ]


def test_origins_of_no_fold_is_empty() -> None:
    assert backtest.origins([]) == []


@pytest.mark.parametrize("period", [1, 2, 5])
def test_seasonal_naive_has_one_column_per_step(period: int) -> None:
    values = pandas.DataFrame({"value": numpy.arange(12.0)})
    found = backtest.seasonal_naive(
        values, numpy.array([11]), "value", None, 3, period
    )
    assert found.shape == (1, 3)
