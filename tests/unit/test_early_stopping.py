"""Tests for holdout early stopping of boosting libraries."""

import numpy as np
import pytest

from mml.models.boosting import early_stopping


def noisy_problem(rows=1500, seed=0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, 6))
    target = (features[:, 0] + rng.normal(0, 1.5, rows) > 0).astype(int)
    return features, target


def test_lightgbm_stops_well_before_the_round_limit():
    lightgbm = pytest.importorskip("lightgbm")
    features, target = noisy_problem()
    model = early_stopping.HoldoutEarlyStopping(
        lightgbm.LGBMClassifier(
            n_estimators=600, learning_rate=0.2, verbose=-1, n_jobs=1
        ),
        rounds=15,
    ).fit(features, target)

    assert model.best_iteration_ is not None and model.best_iteration_ < 300
    assert model.predict_proba(features).shape == (len(target), 2)
    assert list(model.classes_) == [0, 1]


def test_xgboost_stops_early_and_still_predicts():
    xgboost = pytest.importorskip("xgboost")
    features, target = noisy_problem()
    model = early_stopping.HoldoutEarlyStopping(
        xgboost.XGBClassifier(n_estimators=600, learning_rate=0.3, n_jobs=1),
        rounds=15,
    ).fit(features, target)

    assert model.best_iteration_ is not None and model.best_iteration_ < 300
    assert len(model.predict(features)) == len(target)


def test_regression_with_lightgbm_stops_early():
    lightgbm = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(2)
    features = rng.normal(size=(1200, 4))
    target = features[:, 0] + rng.normal(0, 2.0, 1200)
    model = early_stopping.HoldoutEarlyStopping(
        lightgbm.LGBMRegressor(
            n_estimators=500, learning_rate=0.2, verbose=-1, n_jobs=1
        ),
        rounds=10,
    ).fit(features, target)

    assert model.best_iteration_ < 250
    assert model.predict(features).shape == (1200,)


def test_too_little_data_trains_without_a_holdout():
    lightgbm = pytest.importorskip("lightgbm")
    features, target = noisy_problem(rows=40)
    model = early_stopping.HoldoutEarlyStopping(
        lightgbm.LGBMClassifier(n_estimators=20, verbose=-1, n_jobs=1)
    ).fit(features, target)

    assert model.best_iteration_ is None
    assert len(model.predict(features)) == 40


def test_catboost_stops_early_on_its_holdout():
    catboost = pytest.importorskip("catboost")
    features, target = noisy_problem(rows=1200)
    model = early_stopping.HoldoutEarlyStopping(
        catboost.CatBoostClassifier(
            iterations=500,
            learning_rate=0.3,
            verbose=0,
            allow_writing_files=False,
            thread_count=1,
        ),
        rounds=15,
    ).fit(features, target)

    assert model.best_iteration_ is not None and model.best_iteration_ < 250
    assert model.predict_proba(features).shape == (1200, 2)
