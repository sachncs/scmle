"""Built-in scikit-learn model adapters (the only ones with a hard dependency)."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mml.contracts.models import Capabilities, ModelSpec
from mml.contracts.tasks import TaskType
from mml.models.registry import register_model

_B, _M, _R = TaskType.BINARY, TaskType.MULTICLASS, TaskType.REGRESSION
CLF = frozenset({_B, _M})
REG = frozenset({_R})
BOTH = CLF | REG


def _pair(clf, reg):
    def factory(task: TaskType, seed: int):
        cls = clf if task.is_classification else reg
        return cls(random_state=seed) if "random_state" in cls().get_params() else cls()

    return factory


def _only(cls, **kw):
    def factory(task: TaskType, seed: int):
        params = cls().get_params()
        if "random_state" in params:
            kw["random_state"] = seed
        return cls(**kw)

    return factory


def _dummy(task: TaskType, seed: int):
    if task.is_classification:
        return DummyClassifier(strategy="prior")
    return DummyRegressor(strategy="mean")


for _spec in [
    ModelSpec(
        "dummy", "baseline", Capabilities(BOTH, sparse_input=True, handles_missing=True), _dummy
    ),
    ModelSpec(
        "logistic_regression",
        "linear",
        Capabilities(CLF, sparse_input=True, needs_scaling=True),
        _only(LogisticRegression, max_iter=1000),
    ),
    ModelSpec(
        "ridge", "linear", Capabilities(REG, sparse_input=True, needs_scaling=True), _only(Ridge)
    ),
    ModelSpec(
        "elastic_net",
        "linear",
        Capabilities(REG, sparse_input=True, needs_scaling=True, cost=2),
        _only(ElasticNet, alpha=0.1, max_iter=5000),
    ),
    ModelSpec(
        "decision_tree",
        "tree",
        Capabilities(BOTH, handles_missing=True),
        _pair(DecisionTreeClassifier, DecisionTreeRegressor),
    ),
    ModelSpec(
        "random_forest",
        "tree",
        Capabilities(BOTH, cost=3),
        _pair(RandomForestClassifier, RandomForestRegressor),
    ),
    ModelSpec(
        "extra_trees",
        "tree",
        Capabilities(BOTH, cost=3),
        _pair(ExtraTreesClassifier, ExtraTreesRegressor),
    ),
    ModelSpec(
        "hist_gradient_boosting",
        "boosting",
        Capabilities(
            BOTH, handles_missing=True, native_categorical=True, early_stopping=True, cost=3
        ),
        _pair(HistGradientBoostingClassifier, HistGradientBoostingRegressor),
    ),
]:
    register_model(_spec)
