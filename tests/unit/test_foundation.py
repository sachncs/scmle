"""Tests for the TabPFN adapter and engines.

The engines are real production classes whose only external dependency is the
`tabpfn` package, reached through `Learner.build`. The tests below
override that one documented factory with ordinary scikit-learn models, so the
store, rebuild, limit and worker paths run for real without the licensed
package. Tests against the real package run only when it is installed and
`TABPFN_TOKEN` is set.
"""

import importlib.util
import os
import pathlib

import numpy
import pytest
from scipy import sparse
from sklearn import linear_model

from scmle.contracts import tasks
from scmle.models import registry
from scmle.models.deeplearning import estimators
from scmle.models.deeplearning import foundation
from scmle.models.deeplearning import worker

HERE = str(pathlib.Path(__file__).resolve().parent)


class SklearnClassifierEngine(foundation.Classifier):
    """The classifier engine with scikit-learn behind `build`."""

    def build(self):
        """Returns logistic regression in TabPFN's place."""
        return linear_model.LogisticRegression(max_iter=500)


class SklearnRegressorEngine(foundation.Regressor):
    """The regressor engine with scikit-learn behind `build`."""

    def build(self):
        """Returns ridge regression in TabPFN's place."""
        return linear_model.Ridge()


class WorkerClassifier(estimators.TabPfnClassifier):
    """Worker-backed classifier whose engine lives in this module."""

    engine_path = "test_foundation.SklearnClassifierEngine"


def separable(rows=300):
    rng = numpy.random.default_rng(0)
    features = rng.normal(size=(rows, 3)).astype(numpy.float32)
    return features, (features[:, 0] > 0).astype(int)


def test_the_engine_stores_rows_and_rebuilds_from_its_state():
    features, labels = separable()
    engine = SklearnClassifierEngine().fit(features, labels)
    restored = SklearnClassifierEngine.__new__(SklearnClassifierEngine)
    restored.__setstate__(engine.__getstate__())

    assert list(engine.__getstate__()["classes_"]) == [0, 1]
    numpy.testing.assert_allclose(
        engine.predict_proba(features), restored.predict_proba(features)
    )
    assert (restored.predict(features) == labels).mean() > 0.9


def test_the_state_carries_settings_and_the_training_rows():
    features, labels = separable(60)
    engine = SklearnClassifierEngine(
        device="cpu", random_state=3, n_estimators=2
    ).fit(features, labels)

    state = engine.__getstate__()

    assert (state["device"], state["random_state"]) == ("cpu", 3)
    assert state["n_estimators"] == 2
    assert state["train_features_"].shape == (60, 3)
    assert state["train_features_"].dtype == numpy.float32


def test_sparse_rows_are_densified_for_fit_and_prediction():
    features, labels = separable(120)

    engine = SklearnClassifierEngine().fit(sparse.csr_matrix(features), labels)

    assert engine.predict(sparse.csr_matrix(features)).shape == (120,)
    assert engine.rows(sparse.csr_matrix(features)).dtype == numpy.float32


def test_data_beyond_tabpfns_row_and_feature_limits_is_refused():
    too_wide = numpy.zeros(
        (20, foundation.MAX_FEATURES + 1), dtype=numpy.float32
    )
    too_long = numpy.zeros((foundation.MAX_ROWS + 1, 2), dtype=numpy.float32)

    with pytest.raises(ValueError, match="supports up to"):
        SklearnClassifierEngine().fit(too_wide, numpy.zeros(20))
    with pytest.raises(ValueError, match="supports up to"):
        SklearnClassifierEngine().fit(
            too_long, numpy.zeros(foundation.MAX_ROWS + 1)
        )


def test_sample_weights_are_refused_rather_than_silently_ignored():
    features, labels = separable(50)

    with pytest.raises(ValueError, match="does not support sample weights"):
        SklearnClassifierEngine().fit(
            features, labels, sample_weight=numpy.ones(50)
        )


def test_the_regression_engine_predicts():
    rng = numpy.random.default_rng(1)
    features = rng.normal(size=(200, 2)).astype(numpy.float32)
    target = features[:, 0] * 3

    engine = SklearnRegressorEngine().fit(features, target)

    assert numpy.corrcoef(engine.predict(features), target)[0, 1] > 0.99


def test_the_default_build_uses_the_named_tabpfn_class(monkeypatch):
    """`build` asks the installed package for the engine's class."""
    seen = {}

    class Library:
        class TabPFNClassifier:
            def __init__(self, **settings):
                seen.update(settings)

    monkeypatch.setattr(
        foundation.importlib, "import_module", lambda name: Library
    )

    foundation.Classifier(device="cuda", random_state=5, n_estimators=4).build()

    assert seen == {"device": "cuda", "random_state": 5, "n_estimators": 4}


def test_the_adapter_is_rejected_without_the_package_and_explains_why():
    adapter = registry.Catalog.with_builtins().get("tabpfn")

    if adapter.is_available():
        pytest.skip("tabpfn is installed here")
    reason = adapter.rejection_reason(tasks.Task.BINARY)

    assert reason is not None and "missing dependency: tabpfn" in reason
    assert adapter.family == "foundation"
    assert adapter.capabilities.max_rows == foundation.MAX_ROWS


def test_the_adapter_stays_extension_only_when_its_package_is_available(
    monkeypatch,
):
    """Installing TabPFN must not silently promote it into the beta."""
    adapter = registry.Catalog.with_builtins().get("tabpfn")
    monkeypatch.setattr(adapter, "is_available", lambda: True)

    reason = adapter.rejection_reason(tasks.Task.BINARY)

    assert reason is not None and "extension-only" in reason


def test_the_proxy_trains_and_predicts_through_the_worker(monkeypatch):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    existing = os.environ.get("PYTHONPATH", "")
    worker.client().shutdown()
    monkeypatch.setenv(
        "PYTHONPATH", HERE + (os.pathsep + existing if existing else "")
    )
    try:
        features, labels = separable()
        model = WorkerClassifier().fit(features, labels)

        assert list(model.classes_) == [0, 1]
        assert (model.predict(features) == labels).mean() > 0.9
        assert model.predict_proba(features).shape == (300, 2)
    finally:
        worker.client().shutdown()


def real_tabpfn_ready():
    """Why the real package cannot be exercised here, or an empty string."""
    if importlib.util.find_spec("tabpfn") is None:
        return "tabpfn is not installed"
    if not os.environ.get("TABPFN_TOKEN"):
        return "TABPFN_TOKEN is not set"
    return ""


def test_real_tabpfn_classifies_through_the_engine():
    reason = real_tabpfn_ready()
    if reason:
        pytest.skip(reason)
    features, labels = separable(200)

    engine = foundation.Classifier(device="cpu").fit(features, labels)
    restored = foundation.Classifier.__new__(foundation.Classifier)
    restored.__setstate__(engine.__getstate__())

    assert (engine.predict(features) == labels).mean() > 0.9
    numpy.testing.assert_allclose(
        engine.predict_proba(features),
        restored.predict_proba(features),
        atol=1e-3,
    )


def test_real_tabpfn_regresses_through_the_engine():
    reason = real_tabpfn_ready()
    if reason:
        pytest.skip(reason)
    rng = numpy.random.default_rng(1)
    features = rng.normal(size=(200, 2)).astype(numpy.float32)
    target = features[:, 0] * 3

    engine = foundation.Regressor(device="cpu").fit(features, target)

    assert numpy.corrcoef(engine.predict(features), target)[0, 1] > 0.95
