"""Tests for fitted feature transformers."""

import numpy
from sklearn import base

from scmle.data.curation import transformers


def make_features():
    rng = numpy.random.default_rng(0)
    skewed = rng.exponential(scale=3.0, size=300)
    symmetric = rng.normal(size=300)
    negative = -numpy.abs(rng.exponential(size=300))
    return numpy.column_stack([skewed, symmetric, negative])


def test_selects_only_skewed_non_negative_columns():
    model = transformers.SkewedLogTransformer().fit(make_features())
    assert list(model.columns_) == [0]


def test_transform_logs_selected_columns_and_keeps_others():
    features = make_features()
    result = (
        transformers.SkewedLogTransformer().fit(features).transform(features)
    )
    numpy.testing.assert_allclose(result[:, 0], numpy.log1p(features[:, 0]))
    numpy.testing.assert_array_equal(result[:, 1:], features[:, 1:])


def test_columns_are_learned_from_fit_data_only():
    train = make_features()
    model = transformers.SkewedLogTransformer().fit(train)
    shifted = train.copy()
    shifted[:, 1] = numpy.exp(
        shifted[:, 1] * 3
    )  # skewed only at transform time
    result = model.transform(shifted)
    numpy.testing.assert_array_equal(result[:, 1], shifted[:, 1])


def test_negative_values_at_inference_are_clipped_not_nan():
    model = transformers.SkewedLogTransformer().fit(make_features())
    result = model.transform(numpy.array([[-5.0, 0.0, -1.0]]))
    assert numpy.isfinite(result).all() and result[0, 0] == 0.0


def test_clone_and_params():
    model = transformers.SkewedLogTransformer(threshold=2.0)
    assert base.clone(model).get_params() == {"threshold": 2.0}
    constant = transformers.SkewedLogTransformer().fit(numpy.ones((20, 2)))
    assert constant.columns_.size == 0


def test_feature_names_pass_through():
    names = numpy.array(["a", "b"], dtype=object)
    model = transformers.SkewedLogTransformer().fit(numpy.ones((20, 2)))
    assert list(model.get_feature_names_out(names)) == ["a", "b"]


def test_asinh_target_round_trips_and_tames_tails():
    target = numpy.array([-5.0, 0.0, 1.0, 10.0, 1000.0, 1e6])
    transform = transformers.AsinhTarget().fit(target)

    squeezed = transform.transform(target)

    numpy.testing.assert_allclose(transform.inverse_transform(squeezed), target)
    assert squeezed.max() < 20 and squeezed.min() < 0
    huge = transform.inverse_transform(numpy.array([1e9, -1e9]))
    assert numpy.isfinite(huge).all()
