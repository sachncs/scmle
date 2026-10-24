"""Optimizer, loss and accumulation options of the PyTorch networks."""

import importlib.util

import numpy as np
import pytest

from scmle.contracts import errors
from scmle.models.deeplearning import estimators

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch is not installed"
)

FAST = {"epochs": 60, "patience": 8}


def separable(rows=400, seed=0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, 4)).astype(np.float32)
    labels = np.where(features[:, 0] + features[:, 1] > 0, "yes", "no")
    return features, labels


def linear(rows=400, seed=1):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, 3)).astype(np.float32)
    return features, 4 + 3 * features[:, 0] - 2 * features[:, 1]


@pytest.mark.parametrize("optimizer", ["adam", "adamw", "sgd", "rmsprop"])
def test_every_first_order_optimizer_learns(optimizer):
    features, labels = separable()
    rate = 5e-2 if optimizer == "sgd" else 1e-2

    model = estimators.TorchClassifier(
        optimizer=optimizer, learning_rate=rate, **FAST
    ).fit(features, labels)

    assert (model.predict(features) == labels).mean() > 0.9
    assert model.training_report_["verdict"] in {"ok", "stalled"}


def test_lbfgs_takes_full_batch_second_order_steps():
    features, target = linear()

    model = estimators.TorchRegressor(
        optimizer="lbfgs", epochs=25, patience=8
    ).fit(features, target)

    residual = target - model.predict(features)
    assert 1 - residual.var() / target.var() > 0.9


def test_accumulating_gradients_trains_like_a_larger_batch():
    features, labels = separable()

    model = estimators.TorchClassifier(
        batch_size=32, accumulation_steps=4, **FAST
    ).fit(features, labels)

    assert (model.predict(features) == labels).mean() > 0.9


def test_focal_loss_and_label_smoothing_still_learn_and_soften_confidence():
    features, labels = separable()
    plain = estimators.TorchClassifier(**FAST).fit(features, labels)

    smooth = estimators.TorchClassifier(label_smoothing=0.2, **FAST).fit(
        features, labels
    )
    focal = estimators.TorchClassifier(loss="focal", **FAST).fit(
        features, labels
    )

    assert (smooth.predict(features) == labels).mean() > 0.9
    assert (focal.predict(features) == labels).mean() > 0.9
    assert (
        smooth.predict_proba(features).max(axis=1).mean()
        < plain.predict_proba(features).max(axis=1).mean()
    )


def test_huber_loss_resists_outliers_that_pull_the_squared_loss():
    features, target = linear()
    corrupted = target.copy()
    corrupted[:20] += 80.0

    squared = estimators.TorchRegressor(epochs=80, patience=10).fit(
        features, corrupted
    )
    huber = estimators.TorchRegressor(
        loss="huber", huber_delta=0.5, epochs=80, patience=10
    ).fit(features, corrupted)

    clean = slice(20, None)
    error_squared = np.abs(
        squared.predict(features)[clean] - target[clean]
    ).mean()
    error_huber = np.abs(huber.predict(features)[clean] - target[clean]).mean()
    assert error_huber < error_squared


@pytest.mark.parametrize(
    ("kind", "settings"),
    [
        (estimators.TorchClassifier, {"optimizer": "newton"}),
        (estimators.TorchClassifier, {"loss": "huber"}),
        (estimators.TorchRegressor, {"loss": "focal"}),
        (estimators.TorchClassifier, {"accumulation_steps": 0}),
    ],
)
def test_unknown_options_are_refused_before_training(kind, settings):
    features, labels = separable()

    with pytest.raises(Exception, match=r"unknown|accumulation") as caught:
        kind(**settings).fit(features, labels)

    assert "ConfigError" in type(caught.value).__name__ or isinstance(
        caught.value, errors.ScmleError
    )
