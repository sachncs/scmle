"""Tests for fold-safe preprocessing pipelines."""

import numpy as np
import pandas as pd
import pytest
from sklearn import base

from scml.contracts import data as data_contracts
from scml.contracts import tasks
from scml.data.curation import preprocessing
from scml.models import registry as registry_module

PLAN = data_contracts.FeaturePlan(("num", "flag"), ("cat",))
BINARY = tasks.TaskType.BINARY


@pytest.fixture(scope="module")
def registry():
    return registry_module.ModelRegistry.with_builtins()


def make_frame(rows=200, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "num": rng.normal(size=rows),
            "flag": rng.random(rows) < 0.5,
            "cat": rng.choice(["a", "b", "c"], rows),
        }
    )
    frame.loc[::7, "num"] = np.nan
    frame.loc[::11, "cat"] = None
    target = pd.Series((frame["num"].fillna(0) > 0).astype(int))
    return frame, target


@pytest.mark.parametrize(
    "name",
    ["logistic_regression", "random_forest", "hist_gradient_boosting", "dummy"],
)
def test_handles_missing_bool_and_categoricals(registry, name):
    features, target = make_frame()
    pipe = preprocessing.build_pipeline(PLAN, registry.get(name), BINARY, 0)
    pipe.fit(features, target)
    assert pipe.predict_proba(features).shape == (200, 2)


def test_unknown_category_and_extra_columns_at_inference(registry):
    features, target = make_frame()
    pipe = preprocessing.build_pipeline(
        PLAN, registry.get("logistic_regression"), BINARY, 0
    ).fit(features, target)
    unseen = features.head(5).assign(cat="never_seen", junk="x")
    assert pipe.predict(unseen).shape == (5,)


def test_statistics_come_from_training_rows_only(registry):
    features, target = make_frame()
    pipe = preprocessing.build_pipeline(
        PLAN, registry.get("logistic_regression"), BINARY, 0
    )
    pipe.fit(features.iloc[:100], target.iloc[:100])
    scaler = pipe.named_steps["prep"].named_transformers_["numeric"]
    scaler = scaler.named_steps["scale"]
    expected = features["num"].iloc[:100].median()
    imputer = pipe.named_steps["prep"].named_transformers_["numeric"]
    assert imputer.named_steps["impute"].statistics_[0] == pytest.approx(
        expected
    )
    assert scaler.n_samples_seen_ == 100


def test_scaling_follows_adapter_capability(registry):
    linear = preprocessing.build_pipeline(
        PLAN, registry.get("logistic_regression"), BINARY, 0
    )
    forest = preprocessing.build_pipeline(
        PLAN, registry.get("random_forest"), BINARY, 0
    )
    assert "scale" in linear.named_steps["prep"].transformers[0][1].named_steps
    assert (
        "scale" not in forest.named_steps["prep"].transformers[0][1].named_steps
    )


def test_clone_is_unfitted_and_refit_is_deterministic(registry):
    features, target = make_frame()
    fitted = preprocessing.build_pipeline(
        PLAN, registry.get("random_forest"), BINARY, 3
    ).fit(features, target)
    twin = base.clone(fitted)
    assert not hasattr(twin.named_steps["model"], "estimators_")
    np.testing.assert_allclose(
        twin.fit(features, target).predict_proba(features),
        fitted.predict_proba(features),
    )


def test_plan_without_numeric_columns(registry):
    plan = data_contracts.FeaturePlan((), ("cat",))
    features, target = make_frame()
    pipe = preprocessing.build_pipeline(plan, registry.get("dummy"), BINARY, 0)
    assert pipe.fit(features, target).predict(features).shape == (200,)
