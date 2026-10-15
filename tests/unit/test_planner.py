import numpy as np
import pandas as pd
import pytest

from mml.contracts.errors import ConfigError
from mml.contracts.tasks import DataSpec, ProjectSpec, ResourceSpec, TaskType
from mml.data.profiling import profile_dataset
from mml.data.readers import build_manifest
from mml.intelligence.planner import make_plan


def plan_for(df, task=TaskType.BINARY, budget=1800, allowed=None, **kw):
    spec = ProjectSpec(
        data=DataSpec(uri="m", target="y"), resources=ResourceSpec(time_budget_seconds=budget), **kw
    )
    prof = profile_dataset(df, spec, build_manifest(df, "y"), task)
    return make_plan(spec, prof, task, allowed)


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"x": rng.normal(size=300), "y": rng.integers(0, 2, 300)})


def test_classification_includes_baseline_linear_tree(df):
    p = plan_for(df)
    assert {"dummy", "logistic_regression", "random_forest"} <= set(p.candidate_models)
    assert p.metric == "roc_auc" and p.validation_strategy == "stratified"
    assert "Candidates:" in p.summary()


def test_plan_is_deterministic(df):
    assert plan_for(df).plan_id == plan_for(df.copy()).plan_id


def test_small_budget_prefers_fast_models(df):
    p = plan_for(df, budget=60)
    assert "random_forest" not in p.candidate_models
    assert any(d.technique == "random_forest" and d.action == "reject" for d in p.decisions)


def test_imbalance_defers_techniques(df):
    imb = pd.concat([df[df.y == 0], df[df.y == 1].head(8)])
    p = plan_for(imb)
    assert {"class_weighting", "resampling_inside_cv"} <= {
        d.technique for d in p.decisions if d.action == "defer"
    }


def test_approved_plan_subset_and_invalid(df):
    p = plan_for(df, allowed=("dummy", "logistic_regression"))
    assert p.candidate_models == ("dummy", "logistic_regression")
    with pytest.raises(ConfigError):
        plan_for(df, allowed=("ridge",))
