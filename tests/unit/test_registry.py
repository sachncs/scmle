import pytest
from sklearn.dummy import DummyClassifier

from mml.contracts.errors import ConfigError
from mml.contracts.models import Capabilities, ModelSpec
from mml.contracts.tasks import TaskType
from mml.models import registry
from mml.models.conformance import check_conformance
from mml.models.registry import candidates, get_model, register_model


def test_builtin_candidates_per_task():
    clf, _ = candidates(TaskType.BINARY)
    reg, rej = candidates(TaskType.REGRESSION)
    assert "logistic_regression" in {m.name for m in clf}
    assert "logistic_regression" not in {m.name for m in reg}
    assert "does not support" in rej["logistic_regression"]
    assert [m.capabilities.cost for m in clf] == sorted(m.capabilities.cost for m in clf)


@pytest.mark.parametrize(
    "name",
    [
        "dummy",
        "logistic_regression",
        "ridge",
        "elastic_net",
        "decision_tree",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
    ],
)
def test_builtins_pass_conformance(name):
    check_conformance(get_model(name))


def test_third_party_registration_without_core_edits():
    spec = ModelSpec(
        "my_dummy",
        "baseline",
        Capabilities(frozenset({TaskType.BINARY})),
        lambda task, seed: DummyClassifier(strategy="prior"),
    )
    register_model(spec)
    try:
        check_conformance(spec)
        assert "my_dummy" in {m.name for m in candidates(TaskType.BINARY)[0]}
        with pytest.raises(ConfigError):
            register_model(spec)
    finally:
        registry._REGISTRY.pop("my_dummy")


def test_missing_dependency_and_size_exclusion():
    spec = ModelSpec(
        "needs_x",
        "x",
        Capabilities(frozenset({TaskType.BINARY}), max_rows=10),
        lambda t, s: DummyClassifier(),
        requires=("definitely_not_installed_pkg",),
    )
    register_model(spec)
    try:
        _, rej = candidates(TaskType.BINARY)
        assert "missing dependency" in rej["needs_x"]
    finally:
        registry._REGISTRY.pop("needs_x")


def test_conformance_catches_bad_adapter():
    class NoProba:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return [0] * len(X)

        def get_params(self, deep=True):
            return {}

        def set_params(self, **p):
            return self

    bad = ModelSpec("bad", "x", Capabilities(frozenset({TaskType.BINARY})), lambda t, s: NoProba())
    with pytest.raises(ConfigError, match="predict_proba"):
        check_conformance(bad)
