"""Tests for the planner and its rules."""

import numpy as np
import pandas as pd
import pytest

from mml.contracts import errors
from mml.contracts import experiments
from mml.contracts import tasks
from mml.data import profiling
from mml.data import readers
from mml.intelligence import planner as planner_module
from mml.models import registry as registry_module

BINARY = tasks.TaskType.BINARY


@pytest.fixture(scope="module")
def planner():
    return planner_module.Planner(registry_module.ModelRegistry.with_builtins())


def make_plan(planner, frame, budget=1800, approved=None, task=BINARY, **extra):
    spec = tasks.ProjectSpec(
        data=tasks.DataSpec(uri="m", target="y"),
        resources=tasks.ResourceSpec(time_budget_seconds=budget),
        **extra,
    )
    manifest = readers.build_manifest(frame, "y")
    profile = profiling.DatasetProfiler(spec).profile(frame, manifest, task)
    return planner.plan(
        planner_module.PlanningContext(spec, profile, task, approved)
    )


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"x": rng.normal(size=300), "y": rng.integers(0, 2, 300)}
    )


def test_classification_includes_baseline_linear_and_trees(planner, frame):
    plan = make_plan(planner, frame)
    assert {"dummy", "logistic_regression", "random_forest"} <= set(
        plan.candidate_models
    )
    assert plan.metric == "roc_auc" and plan.validation_strategy == "stratified"
    assert "Candidates:" in plan.summary()


def test_plan_is_deterministic(planner, frame):
    assert (
        make_plan(planner, frame).plan_id
        == make_plan(planner, frame.copy()).plan_id
    )


def test_small_budget_prefers_fast_models(planner, frame):
    plan = make_plan(planner, frame, budget=60)
    assert "random_forest" not in plan.candidate_models
    assert "hist_gradient_boosting" in plan.candidate_models
    assert any(
        d.technique == "random_forest"
        and d.action is experiments.DecisionAction.REJECT
        for d in plan.decisions
    )


def test_severe_imbalance_defers_techniques(planner, frame):
    skewed = pd.concat([frame[frame.y == 0], frame[frame.y == 1].head(8)])
    plan = make_plan(planner, skewed)
    deferred = {
        d.technique
        for d in plan.decisions
        if d.action is experiments.DecisionAction.DEFER
    }
    assert {"class_weighting", "resampling_inside_cv"} <= deferred


def test_approved_subset_and_invalid_approval(planner, frame):
    plan = make_plan(planner, frame, approved=("dummy", "logistic_regression"))
    assert plan.candidate_models == ("dummy", "logistic_regression")
    with pytest.raises(errors.ConfigError, match="approved"):
        make_plan(planner, frame, approved=("ridge",))


def test_custom_rule_chain_is_honoured(frame):
    class KeepOnlyDummy(planner_module.PlanningRule):
        def apply(self, context, candidates, decisions):
            return [a for a in candidates if a.name == "dummy"]

    custom = planner_module.Planner(
        registry_module.ModelRegistry.with_builtins(), rules=(KeepOnlyDummy(),)
    )
    assert make_plan(custom, frame).candidate_models == ("dummy",)


def test_no_candidates_is_an_error(frame):
    class DropAll(planner_module.PlanningRule):
        def apply(self, context, candidates, decisions):
            return []

    empty = planner_module.Planner(
        registry_module.ModelRegistry.with_builtins(), rules=(DropAll(),)
    )
    with pytest.raises(errors.ConfigError, match="no candidate"):
        make_plan(empty, frame)
