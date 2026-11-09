"""Tests for the model registry, adapters and conformance suite."""

import pytest
from sklearn import dummy

from scmle.contracts import errors
from scmle.contracts import models
from scmle.contracts import tasks
from scmle.models import conformance
import scmle.models.registry
from scmle.models.sklearn import classical


@pytest.fixture
def registry():
    return scmle.models.registry.ModelRegistry.with_builtins()


def test_candidates_depend_on_task(registry):
    classifiers = registry.candidates(tasks.TaskType.BINARY)[0]
    regressors, rejected = registry.candidates(tasks.TaskType.REGRESSION)
    assert "logistic_regression" in {a.name for a in classifiers}
    assert "logistic_regression" not in {a.name for a in regressors}
    assert "does not support" in rejected["logistic_regression"]
    costs = [a.capabilities.cost for a in classifiers]
    assert costs == sorted(costs)


@pytest.mark.parametrize(
    "adapter_class", classical.BUILTIN_ADAPTERS, ids=lambda c: c.name
)
def test_builtins_pass_conformance(adapter_class):
    conformance.check_conformance(adapter_class())


def test_builtin_estimator_types(registry):
    assert type(
        registry.get("random_forest").build(tasks.TaskType.BINARY, 0)
    ).__name__ == ("RandomForestClassifier")
    assert type(
        registry.get("random_forest").build(tasks.TaskType.REGRESSION, 0)
    ).__name__ == ("RandomForestRegressor")
    assert (
        registry.get("random_forest")
        .build(tasks.TaskType.BINARY, 5)
        .random_state
        == 5
    )


class ThirdPartyAdapter(models.ModelAdapter):
    name = "third_party"
    family = "baseline"
    capabilities = models.Capabilities(frozenset({tasks.TaskType.BINARY}))

    def build(self, task, seed, params=None):
        return dummy.DummyClassifier(strategy="prior")


def test_third_party_registration_without_core_edits(registry):
    adapter = ThirdPartyAdapter()
    registry.register(adapter)
    conformance.check_conformance(adapter)
    assert "third_party" in {
        a.name for a in registry.candidates(tasks.TaskType.BINARY)[0]
    }
    with pytest.raises(errors.ConfigError):
        registry.register(adapter)
    registry.register(adapter, overwrite=True)
    registry.unregister("third_party")
    assert "third_party" not in registry.adapters


def test_unknown_model(registry):
    with pytest.raises(errors.ConfigError, match="unknown model"):
        registry.get("nope")


def test_rejects_missing_dependency_and_oversized_data(registry):
    class NeedsPackage(ThirdPartyAdapter):
        name = "needs_package"
        capabilities = models.Capabilities(
            frozenset({tasks.TaskType.BINARY}), max_rows=10
        )
        requires = ("definitely_not_installed_pkg",)

    registry.register(NeedsPackage())
    rejected = registry.candidates(tasks.TaskType.BINARY, row_count=5)[1]
    assert "missing dependency" in rejected["needs_package"]

    class Small(ThirdPartyAdapter):
        name = "small"
        capabilities = models.Capabilities(
            frozenset({tasks.TaskType.BINARY}), max_rows=10
        )

    registry.register(Small())
    rejected = registry.candidates(tasks.TaskType.BINARY, row_count=500)[1]
    assert "too large" in rejected["small"]


def test_conformance_catches_bad_adapters():
    class NoProba:
        def fit(self, features, target):
            return self

        def predict(self, features):
            return [0] * len(features)

        def get_params(self, deep=True):
            return {}

        def set_params(self, **params):
            return self

    class Bad(ThirdPartyAdapter):
        name = "bad"

        def build(self, task, seed, params=None):
            return NoProba()

    with pytest.raises(errors.ConfigError, match="predict_proba"):
        conformance.check_conformance(Bad())

    class Empty(ThirdPartyAdapter):
        name = "empty"
        capabilities = models.Capabilities(frozenset())

    with pytest.raises(errors.ConfigError, match="no supported tasks"):
        conformance.check_conformance(Empty())


def test_plugin_loading_isolates_failures(registry, monkeypatch):
    class BrokenEntry:
        name = "broken"

        def load(self):
            raise RuntimeError("bad plugin")

    class GoodEntry:
        name = "good"

        def load(self):
            return lambda: [ThirdPartyAdapter()]

    monkeypatch.setattr(
        scmle.models.registry.metadata,
        "entry_points",
        lambda group: [BrokenEntry(), GoodEntry()],
    )
    assert registry.load_plugins() == ["broken"]
    assert "third_party" in registry.adapters


def test_svm_and_knn_pass_conformance_and_respect_size_limits(registry):
    for name in ("svm", "knn"):
        conformance.check_conformance(registry.get(name))
    compatible, rejected = registry.candidates(
        tasks.TaskType.BINARY, row_count=100_000
    )
    assert (
        "svm" not in {a.name for a in compatible}
        and "too large" in rejected["svm"]
    )
    assert "too large" in rejected["knn"]
    small = registry.candidates(tasks.TaskType.BINARY, row_count=1_000)[0]
    assert {"svm", "knn"} <= {a.name for a in small}


def test_svm_classifier_is_calibrated_and_regressor_is_plain(registry):
    estimator = registry.get("svm").build(
        tasks.TaskType.REGRESSION, 0, {"C": 2.0}
    )
    assert type(estimator).__name__ == "SVR" and estimator.C == 2.0
    classifier = registry.get("svm").build(tasks.TaskType.BINARY, 3, {"C": 0.5})
    assert type(classifier).__name__ == "CalibratedClassifierCV"
    assert (
        classifier.estimator.random_state == 3 and classifier.estimator.C == 0.5
    )
    assert classifier.ensemble is False


def test_neural_family_registers_through_the_same_plugin_contract(registry):
    adapter = registry.get("neural_network")
    conformance.check_conformance(adapter)
    assert adapter.family == "neural" and adapter.capabilities.early_stopping
    compatible = registry.candidates(tasks.TaskType.BINARY, row_count=1_000)[0]
    assert "neural_network" in {a.name for a in compatible}
    tuned = adapter.build(
        tasks.TaskType.BINARY, 0, {"hidden_layer_sizes": (16,), "alpha": 0.01}
    )
    assert tuned.hidden_layer_sizes == (16,) and tuned.alpha == 0.01
    assert registry.candidates(tasks.TaskType.BINARY, row_count=500_000)[1][
        "neural_network"
    ]


def test_a_third_party_deep_learning_adapter_needs_no_core_edit(registry):
    from sklearn import neural_network

    class TinyNet(models.ModelAdapter):
        name = "tiny_net"
        family = "neural"
        capabilities = models.Capabilities(
            frozenset({tasks.TaskType.BINARY}), gpu=True
        )

        def build(self, task, seed, params=None):
            return neural_network.MLPClassifier(
                (4,), max_iter=200, random_state=seed
            )

    registry.register(TinyNet())
    conformance.check_conformance(registry.get("tiny_net"))
    assert registry.get("tiny_net").capabilities.gpu is True


def test_a_default_orchestrator_discovers_conforming_plugins_only(
    monkeypatch, binary_frame
):
    from scmle.core import orchestrator

    class NoProba:
        def fit(self, features, target):
            return self

        def predict(self, features):
            return [0] * len(features)

        def get_params(self, deep=True):
            return {}

        def set_params(self, **params):
            return self

    class NonConforming(ThirdPartyAdapter):
        name = "non_conforming"

        def build(self, task, seed, params=None):
            return NoProba()

    class Entry:
        def __init__(self, name, adapters):
            self.name, self.adapters = name, adapters

        def load(self):
            return lambda: self.adapters

    monkeypatch.setattr(
        scmle.models.registry.metadata,
        "entry_points",
        lambda group: [
            Entry("good", [ThirdPartyAdapter()]),
            Entry("bad", [NonConforming()]),
        ],
    )
    spec = tasks.ProjectSpec(data=tasks.DataSpec(uri="<in-memory>", target="y"))
    engine = orchestrator.Orchestrator(spec, binary_frame)
    assert "third_party" in engine.registry.adapters
    assert "non_conforming" not in engine.registry.adapters
    assert "predict_proba" in engine.registry.plugin_failures["bad"]
