"""Learning-curve edges: the class-count bound and the flat tolerance."""

import numpy
import pandas
from sklearn import dummy

from scmle.evaluation import learning_curve

ROWS = 100


def scorer(fitted, features, target) -> float:
    return float(len(features))


def test_a_loss_of_exactly_the_tolerance_is_a_plateau() -> None:
    assert learning_curve.judge([0.0, 0.0, -0.002]) == (
        "plateau",
        -0.002,
        -0.002,
    )


def test_a_curve_record_lists_its_points() -> None:
    curve = learning_curve.Curve((10, 20), (0.5, 0.6), "plateau", 0.1, 0.1)
    assert curve.to_dict() == {
        "sizes": [10, 20],
        "scores": [0.5, 0.6],
        "verdict": "plateau",
        "last_gain": 0.1,
        "total_gain": 0.1,
    }


def test_twenty_classes_must_all_appear_in_a_prefix() -> None:
    positions = numpy.arange(ROWS)
    labels = numpy.where(positions < 20, positions % 2, positions % 20)
    features = pandas.DataFrame({"x": positions.astype(float)})
    found = learning_curve.compute(
        dummy.DummyClassifier(),
        features,
        pandas.Series(labels),
        positions,
        positions[:10],
        scorer,
        True,
    )
    assert found is not None
    assert found.sizes == (40, 60, 80, 100)
