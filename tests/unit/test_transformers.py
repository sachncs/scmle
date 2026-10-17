"""Tests for fitted feature transformers."""

import numpy as np
from sklearn import base

from mml.data.curation import transformers


def make_features():
    rng = np.random.default_rng(0)
    skewed = rng.exponential(scale=3.0, size=300)
    symmetric = rng.normal(size=300)
    negative = -np.abs(rng.exponential(size=300))
    return np.column_stack([skewed, symmetric, negative])


def test_selects_only_skewed_non_negative_columns():
    model = transformers.SkewedLogTransformer().fit(make_features())
    assert list(model.columns_) == [0]


def test_transform_logs_selected_columns_and_keeps_others():
    features = make_features()
    result = (
        transformers.SkewedLogTransformer().fit(features).transform(features)
    )
    np.testing.assert_allclose(result[:, 0], np.log1p(features[:, 0]))
    np.testing.assert_array_equal(result[:, 1:], features[:, 1:])


def test_columns_are_learned_from_fit_data_only():
    train = make_features()
    model = transformers.SkewedLogTransformer().fit(train)
    shifted = train.copy()
    shifted[:, 1] = np.exp(shifted[:, 1] * 3)  # skewed only at transform time
    result = model.transform(shifted)
    np.testing.assert_array_equal(result[:, 1], shifted[:, 1])


def test_negative_values_at_inference_are_clipped_not_nan():
    model = transformers.SkewedLogTransformer().fit(make_features())
    result = model.transform(np.array([[-5.0, 0.0, -1.0]]))
    assert np.isfinite(result).all() and result[0, 0] == 0.0


def test_clone_and_params():
    model = transformers.SkewedLogTransformer(threshold=2.0)
    assert base.clone(model).get_params() == {"threshold": 2.0}
    constant = transformers.SkewedLogTransformer().fit(np.ones((20, 2)))
    assert constant.columns_.size == 0


def test_feature_names_pass_through():
    names = np.array(["a", "b"], dtype=object)
    model = transformers.SkewedLogTransformer().fit(np.ones((20, 2)))
    assert list(model.get_feature_names_out(names)) == ["a", "b"]


def test_asinh_target_round_trips_and_tames_tails():
    target = np.array([-5.0, 0.0, 1.0, 10.0, 1000.0, 1e6])
    transform = transformers.AsinhTarget().fit(target)

    squeezed = transform.transform(target)

    np.testing.assert_allclose(transform.inverse_transform(squeezed), target)
    assert squeezed.max() < 20 and squeezed.min() < 0
    huge = transform.inverse_transform(np.array([1e9, -1e9]))
    assert np.isfinite(huge).all()
