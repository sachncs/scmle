"""Per-row weights reach the model and count by their size."""

import numpy as np
import pandas as pd
from sklearn import compose
from sklearn import linear_model
from sklearn import pipeline
from sklearn import preprocessing

from scml.models import weighting


def frame_with_conflict(rows=400):
    """Two halves with opposite labels for the same feature value."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=rows)
    first = np.arange(rows) < rows // 2
    return pd.DataFrame({"x": x, "w": np.where(first, 1.0, 0.0)}), np.where(
        first, 1.0 * x, -1.0 * x
    )


def scaler():
    return compose.ColumnTransformer(
        [("x", preprocessing.StandardScaler(), ["x"])], remainder="drop"
    )


def regression_pipeline():
    return pipeline.Pipeline(
        [
            ("scale", scaler()),
            ("model", linear_model.LinearRegression()),
        ]
    )


def test_zero_weight_rows_do_not_influence_a_model_that_takes_weights():
    features, target = frame_with_conflict()
    model = weighting.weighted(regression_pipeline(), "w", 0).fit(
        features, target
    )

    assert model.accepts_weights(model.estimator_)
    slope = model.estimator_.named_steps["model"].coef_[0]
    assert slope > 0


class NoWeights(linear_model.LinearRegression):
    """Model whose `fit` has no `sample_weight` parameter."""

    def fit(self, features, target):
        return super().fit(features, target)


def test_models_without_weight_support_are_fitted_on_a_weighted_draw():
    features, target = frame_with_conflict()
    plain = pipeline.Pipeline([("scale", scaler()), ("model", NoWeights())])
    model = weighting.weighted(plain, "w", 0).fit(features, target)

    assert not model.accepts_weights(model.estimator_)
    assert model.estimator_.named_steps["model"].coef_[0] > 0


def test_prediction_does_not_need_the_weight_column():
    features, target = frame_with_conflict()
    model = weighting.weighted(regression_pipeline(), "w", 0).fit(
        features, target
    )

    assert len(model.predict(features.drop(columns=["w"]))) == len(features)
