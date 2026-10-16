"""Tests for the PyTorch model families."""

import io

import joblib
import numpy as np
import pytest
from scipy import sparse

from mml.contracts import errors
from mml.contracts import tasks
from mml.models import conformance
from mml.models import registry as registry_module
from mml.models.deeplearning import estimators

torch = pytest.importorskip("torch")

REGISTRY = registry_module.ModelRegistry.with_builtins()
FAST = {"epochs": 60, "patience": 6}


def separable(rows=400, seed=0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, 4))
    labels = np.where(features[:, 0] + features[:, 1] > 0, "yes", "no")
    return features.astype(np.float32), labels


def test_classifier_learns_and_keeps_string_labels():
    features, labels = separable()
    model = estimators.TorchClassifier(**FAST).fit(features, labels)

    assert set(model.classes_) == {"no", "yes"}
    assert (model.predict(features) == labels).mean() > 0.9
    probabilities = model.predict_proba(features)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)


def test_regressor_recovers_a_linear_signal_in_original_units():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(400, 3)).astype(np.float32)
    target = 50 + 10 * features[:, 0] - 4 * features[:, 1]

    model = estimators.TorchRegressor(**FAST).fit(features, target)

    residual = target - model.predict(features)
    assert 1 - residual.var() / target.var() > 0.9


def test_residual_architecture_learns_too():
    features, labels = separable()
    model = estimators.TorchClassifier(
        architecture="residual", hidden_sizes=(32, 32), **FAST
    ).fit(features, labels)

    assert (model.predict(features) == labels).mean() > 0.9


def test_same_seed_gives_the_same_network():
    features, labels = separable()
    first = estimators.TorchClassifier(random_state=3, **FAST).fit(
        features, labels
    )
    second = estimators.TorchClassifier(random_state=3, **FAST).fit(
        features, labels
    )

    np.testing.assert_allclose(
        first.predict_proba(features), second.predict_proba(features)
    )


def test_pickling_keeps_predictions_and_leaves_the_live_model_intact():
    features, labels = separable()
    model = estimators.TorchClassifier(**FAST).fit(features, labels)
    before = model.predict_proba(features)

    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)

    np.testing.assert_allclose(restored.predict_proba(features), before)
    np.testing.assert_allclose(model.predict_proba(features), before)


def test_early_stopping_ends_training_before_the_epoch_limit():
    features, labels = separable()
    model = estimators.TorchClassifier(
        epochs=300, patience=3, learning_rate=1e-2
    ).fit(features, labels)

    assert model.n_iter_ < 300


def test_balanced_class_weight_raises_minority_probability():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(600, 3)).astype(np.float32)
    labels = (features[:, 0] + rng.normal(0, 1.5, 600) > 1.8).astype(int)
    plain = estimators.TorchClassifier(**FAST).fit(features, labels)
    balanced = estimators.TorchClassifier(class_weight="balanced", **FAST).fit(
        features, labels
    )

    assert (
        balanced.predict_proba(features)[:, 1].mean()
        > plain.predict_proba(features)[:, 1].mean()
    )


def test_sparse_input_and_sample_weights_are_accepted():
    features, labels = separable()
    weights = np.where(labels == "yes", 2.0, 1.0)

    model = estimators.TorchClassifier(**FAST).fit(
        sparse.csr_matrix(features), labels, sample_weight=weights
    )

    assert len(model.predict(sparse.csr_matrix(features))) == len(labels)


def test_unknown_architecture_is_a_configuration_error():
    features, labels = separable()
    with pytest.raises(errors.ConfigError, match="unknown architecture"):
        estimators.TorchClassifier(architecture="transformer").fit(
            features, labels
        )


@pytest.mark.parametrize("name", ["torch_mlp", "torch_residual"])
def test_adapters_pass_conformance_and_support_every_task(name):
    adapter = REGISTRY.get(name)

    conformance.check_conformance(adapter)

    assert adapter.capabilities.tasks == frozenset(tasks.TaskType)
    assert adapter.rejection_reason(tasks.TaskType.BINARY) is None
