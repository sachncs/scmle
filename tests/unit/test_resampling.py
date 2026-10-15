"""Tests for training-fold resampling."""

import numpy as np
import pandas as pd
import pytest
from sklearn import base
from sklearn import linear_model
from sklearn import model_selection

from mml.models import resampling


def imbalanced(rows=400, minority=40, seed=0):
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    target = pd.Series([1] * minority + [0] * (rows - minority))
    features.loc[target == 1, "a"] += 2.0
    return features, target


def test_oversampling_equalizes_classes_with_only_existing_rows():
    features, target = imbalanced()
    sampled, labels = resampling.RandomOverSampler().resample(
        features, target, 0
    )
    assert labels.value_counts().to_dict() == {0: 360, 1: 360}
    original = {tuple(row) for row in features.to_numpy()}
    assert {tuple(row) for row in sampled.to_numpy()} <= original


def test_undersampling_equalizes_classes_without_duplicates():
    features, target = imbalanced()
    sampled, labels = resampling.RandomUnderSampler().resample(
        features, target, 0
    )
    assert labels.value_counts().to_dict() == {0: 40, 1: 40}
    assert not sampled.duplicated().any()


def test_multiclass_resampling_reaches_the_extreme_class_size():
    features = pd.DataFrame({"a": np.arange(60.0)})
    target = pd.Series([0] * 40 + [1] * 15 + [2] * 5)
    over = resampling.RandomOverSampler().resample(features, target, 0)[1]
    under = resampling.RandomUnderSampler().resample(features, target, 0)[1]
    assert set(over.value_counts()) == {40} and set(under.value_counts()) == {5}


def test_resampling_is_seeded():
    features, target = imbalanced()
    first = resampling.RandomOverSampler().resample(features, target, 3)[0]
    second = resampling.RandomOverSampler().resample(features, target, 3)[0]
    other = resampling.RandomOverSampler().resample(features, target, 4)[0]
    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(other)


def test_validation_fold_keeps_its_original_distribution():
    features, target = imbalanced()
    seen = []

    class Spy(linear_model.LogisticRegression):
        def fit(self, features, labels, **kwargs):
            seen.append(pd.Series(labels).value_counts().to_dict())
            return super().fit(features, labels, **kwargs)

    model = resampling.ResamplingClassifier(
        Spy(), resampling.RandomOverSampler()
    )
    folds = model_selection.StratifiedKFold(3).split(features, target)
    scores = [
        model.fit(features.iloc[tr], target.iloc[tr]).score(
            features.iloc[va], target.iloc[va]
        )
        for tr, va in folds
    ]
    assert all(counts[0] == counts[1] for counts in seen)  # train is rebalanced
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_wrapper_predicts_and_clones():
    features, target = imbalanced()
    model = resampling.ResamplingClassifier(
        linear_model.LogisticRegression(), resampling.RandomUnderSampler(), 1
    ).fit(features, target)
    assert model.predict(features).shape == (400,)
    assert model.predict_proba(features).shape == (400, 2)
    assert list(model.classes_) == [0, 1]
    twin = base.clone(model)
    assert not hasattr(twin, "estimator_") and twin.random_state == 1
    with pytest.raises(AttributeError):
        twin.predict(features)
