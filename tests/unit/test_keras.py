"""The Keras family: trained in the PyTorch worker, pickled as numpy."""

import importlib.util
import io

import joblib
import numpy
import pytest

from scmle.contracts import tasks
from scmle.models import conformance
from scmle.models.deeplearning import estimators
import scmle.models.registry

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("keras") is None,
    reason="torch and keras are not installed",
)

FAST = {"epochs": 40, "patience": 5}


def separable(rows=400, seed=0):
    rng = numpy.random.default_rng(seed)
    features = rng.normal(size=(rows, 4)).astype(numpy.float32)
    labels = numpy.where(features[:, 0] + features[:, 1] > 0, "yes", "no")
    return features, labels


def test_the_keras_estimator_declares_its_dense_architecture_contract():
    model = estimators.KerasClassifier(**FAST)

    assert model.architecture == "mlp"


def test_the_classifier_learns_and_keeps_string_labels():
    features, labels = separable()

    model = estimators.KerasClassifier(**FAST).fit(features, labels)

    assert set(model.classes_) == {"no", "yes"}
    assert (model.predict(features) == labels).mean() > 0.9
    numpy.testing.assert_allclose(
        model.predict_proba(features).sum(axis=1), 1.0, rtol=1e-5
    )


def test_the_regressor_recovers_a_linear_signal_in_original_units():
    rng = numpy.random.default_rng(1)
    features = rng.normal(size=(400, 3)).astype(numpy.float32)
    target = 50 + 10 * features[:, 0] - 4 * features[:, 1]

    model = estimators.KerasRegressor(**FAST).fit(features, target)

    residual = target - model.predict(features)
    assert 1 - residual.var() / target.var() > 0.9


def test_sample_weights_reach_the_network():
    features, labels = separable()
    weights = numpy.where(labels == "yes", 1.0, 0.0)

    model = estimators.KerasClassifier(**FAST).fit(
        features, labels, sample_weight=weights
    )

    assert (model.predict(features) == "yes").mean() > 0.9


def test_a_pickled_model_predicts_the_same_in_a_new_session():
    features, labels = separable()
    model = estimators.KerasClassifier(**FAST).fit(features, labels)
    expected = model.predict_proba(features)

    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)

    numpy.testing.assert_allclose(restored.predict_proba(features), expected)
    assert restored.training_report_["verdict"] in {"ok", "stalled"}


def test_the_adapter_passes_conformance_for_every_standard_task():
    adapter = scmle.models.registry.ModelRegistry.with_builtins().get(
        "keras_mlp"
    )

    conformance.check_conformance(adapter)

    assert adapter.capabilities.tasks == tasks.STANDARD_TASKS


def test_no_standard_adapter_claims_survival_except_the_baseline_and_survival_models():
    registry = scmle.models.registry.ModelRegistry.with_builtins()

    claiming = {
        name
        for name, adapter in registry.adapters.items()
        if tasks.TaskType.SURVIVAL in adapter.capabilities.tasks
    }

    assert claiming == {"dummy", "cox_ph", "discrete_time_boosting"}
