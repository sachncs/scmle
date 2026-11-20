"""The multi-target wrapper's fit, predictions and label probabilities."""

import numpy
import pandas
from sklearn import compose
from sklearn import feature_selection
from sklearn import linear_model
from sklearn import neighbors
from sklearn import pipeline

from scmle.preparation import multitarget

FEATURES = ["x1", "x2"]
ROWS = 80
# The targets are 2 * x1 and 3 * x2 plus noise of standard deviation 0.01
# (noise keeps the F statistics finite), so a fit that selects x1 recovers
# the primary target within ten noise deviations.
NOISE = 0.01
TOLERANCE = 10 * NOISE


def table() -> tuple[pandas.DataFrame, numpy.ndarray]:
    """Two features; target `a` follows x1, the second target `b` x2."""
    rng = numpy.random.default_rng(0)
    first, second, noise, other = rng.normal(size=(4, ROWS))
    frame = pandas.DataFrame(
        {"x1": first, "x2": second, "b": 3 * second + NOISE * other}
    )
    return frame, 2 * first + NOISE * noise


def keep() -> compose.ColumnTransformer:
    """Passes the features through by name, dropping the target columns."""
    return compose.ColumnTransformer([("keep", "passthrough", FEATURES)])


def regressor() -> pipeline.Pipeline:
    """Selects the one feature that best explains the target it is given."""
    return pipeline.Pipeline(
        [
            ("keep", keep()),
            (
                "select",
                feature_selection.SelectKBest(
                    feature_selection.f_regression, k=1
                ),
            ),
            ("model", linear_model.LinearRegression()),
        ]
    )


def test_the_preprocessing_is_fitted_on_the_primary_target() -> None:
    frame, primary = table()
    template = regressor()
    fitted = multitarget.Multioutput(template, ("a", "b")).fit(frame, primary)
    assert fitted.estimator is template
    assert fitted.target_columns == ("a", "b")
    assert not hasattr(template.named_steps["model"], "coef_")
    assert list(fitted.preprocessing_.named_steps) == [
        "keep",
        "select",
    ]
    assert list(fitted.preprocessing_.named_steps["select"].get_support()) == [
        True,
        False,
    ]
    assert isinstance(fitted.model_, linear_model.LinearRegression)
    assert fitted.model_.coef_.shape == (2, 1)


def test_the_predictions_hold_one_column_per_target() -> None:
    frame, primary = table()
    fitted = multitarget.Multioutput(regressor(), ("a", "b")).fit(
        frame, primary
    )
    found = fitted.predict(frame[FEATURES])
    assert found.shape == (ROWS, 2)
    assert numpy.allclose(found[:, 0], primary, atol=TOLERANCE)
    expected = fitted.model_.predict(
        fitted.preprocessing_.transform(frame[FEATURES])
    )
    assert numpy.array_equal(found, expected)


def test_the_label_probabilities_come_from_the_fitted_model() -> None:
    frame = table()[0]
    labels = frame.assign(b=(frame["x2"] > 0).astype(int))
    primary = (frame["x1"] > 0).astype(int).to_numpy()
    template = pipeline.Pipeline(
        [("keep", keep()), ("model", neighbors.KNeighborsClassifier(3))]
    )
    fitted = multitarget.Multilabel(template, ("a", "b")).fit(labels, primary)
    found = fitted.predict_proba(frame[FEATURES])
    expected = fitted.model_.predict_proba(
        fitted.preprocessing_.transform(frame[FEATURES])
    )
    assert len(found) == 2
    for column, reference in zip(found, expected, strict=True):
        assert column.shape == (ROWS, 2)
        assert numpy.array_equal(column, reference)
    assert fitted.predict(frame[FEATURES]).shape == (ROWS, 2)
