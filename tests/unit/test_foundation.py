"""Tests for the TabPFN adapter plumbing (with a stand-in package)."""

import os
import pathlib
import sys

import numpy as np
import pytest

from scml.contracts import tasks
from scml.models import registry as registry_module
from scml.models.deeplearning import estimators
from scml.models.deeplearning import foundation
from scml.models.deeplearning import worker

STUBS = str(pathlib.Path(__file__).resolve().parents[1] / "stubs")


@pytest.fixture
def stub_package(monkeypatch):
    monkeypatch.syspath_prepend(STUBS)
    sys.modules.pop("tabpfn", None)
    yield
    sys.modules.pop("tabpfn", None)


def separable(rows=300):
    rng = np.random.default_rng(0)
    features = rng.normal(size=(rows, 3)).astype(np.float32)
    return features, (features[:, 0] > 0).astype(int)


def test_the_engine_stores_rows_and_rebuilds_from_its_state(stub_package):
    features, labels = separable()
    engine = foundation.ClassifierEngine().fit(features, labels)
    restored = foundation.ClassifierEngine.__new__(foundation.ClassifierEngine)
    restored.__setstate__(engine.__getstate__())

    assert list(engine.__getstate__()["classes_"]) == [0, 1]
    np.testing.assert_allclose(
        engine.predict_proba(features), restored.predict_proba(features)
    )
    assert (restored.predict(features) == labels).mean() > 0.9


def test_data_beyond_tabpfns_limits_is_refused(stub_package):
    too_wide = np.zeros((20, foundation.MAX_FEATURES + 1), dtype=np.float32)

    with pytest.raises(ValueError, match="supports up to"):
        foundation.ClassifierEngine().fit(too_wide, np.zeros(20))


def test_the_regression_engine_predicts(stub_package):
    rng = np.random.default_rng(1)
    features = rng.normal(size=(200, 2)).astype(np.float32)
    target = features[:, 0] * 3

    engine = foundation.RegressorEngine().fit(features, target)

    assert np.corrcoef(engine.predict(features), target)[0, 1] > 0.99


def test_the_adapter_is_rejected_without_the_package_and_explains_why():
    adapter = registry_module.ModelRegistry.with_builtins().get("tabpfn")

    if adapter.is_available():
        pytest.skip("tabpfn is installed here")
    reason = adapter.rejection_reason(tasks.TaskType.BINARY)

    assert reason is not None and "missing dependency: tabpfn" in reason
    assert adapter.family == "foundation"
    assert adapter.capabilities.max_rows == foundation.MAX_ROWS


def test_the_adapter_stays_extension_only_when_its_package_is_available(
    monkeypatch,
):
    """Installing TabPFN must not silently promote it into the beta."""
    adapter = registry_module.ModelRegistry.with_builtins().get("tabpfn")
    monkeypatch.setattr(adapter, "is_available", lambda: True)

    reason = adapter.rejection_reason(tasks.TaskType.BINARY)

    assert reason is not None and "extension-only" in reason


def test_the_proxy_trains_and_predicts_through_the_worker(monkeypatch):
    existing = os.environ.get("PYTHONPATH", "")
    worker.client().shutdown()
    monkeypatch.setenv(
        "PYTHONPATH", STUBS + (":" + existing if existing else "")
    )
    try:
        features, labels = separable()
        model = estimators.TabPfnClassifier().fit(features, labels)

        assert list(model.classes_) == [0, 1]
        assert (model.predict(features) == labels).mean() > 0.9
        assert model.predict_proba(features).shape == (300, 2)
    finally:
        worker.client().shutdown()
