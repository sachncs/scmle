"""Declared model capabilities change what the pipeline and planner do."""

import numpy
import pandas
import pytest

from scmle.contracts import data as data_contracts
from scmle.contracts import tasks
from scmle.data.curation import preprocessing
from scmle.intelligence import planner
from scmle.models import registry as registry_module
from scmle.sampling import strategies

BINARY = tasks.TaskType.BINARY


@pytest.fixture
def registry():
    return registry_module.ModelRegistry.with_builtins()


def high_cardinality_frame(rows=900, levels=40, seed=0):
    rng = numpy.random.default_rng(seed)
    cat = rng.choice([f"c{i}" for i in range(levels)], rows)
    effect = {f"c{i}": rng.normal() for i in range(levels)}
    x = rng.normal(size=rows)
    signal = numpy.array([effect[c] for c in cat]) + x
    target = (signal + rng.normal(scale=0.5, size=rows) > 0).astype(int)
    return pandas.DataFrame({"cat": cat, "x": x}), pandas.Series(
        target, name="y"
    )


PLAN = data_contracts.FeaturePlan(
    numeric=("x",), categorical=("cat",), high_cardinality=("cat",)
)


def step_names(pipeline_):
    prep = pipeline_.named_steps["prep"]
    return {name: transformer for name, transformer, cols in prep.transformers}


def test_models_that_handle_missing_values_skip_imputation(registry):
    plain = PLAN.__class__(numeric=("x",), categorical=("cat",))
    native = preprocessing.build_pipeline(
        plain, registry.get("hist_gradient_boosting"), BINARY, 0
    )
    linear = preprocessing.build_pipeline(
        plain, registry.get("logistic_regression"), BINARY, 0
    )
    assert "impute" not in step_names(native)["numeric"].named_steps
    assert "impute" in step_names(linear)["numeric"].named_steps


def test_native_categorical_models_get_integer_codes_and_column_marks(
    registry,
):
    features, target = high_cardinality_frame()
    pipeline_ = preprocessing.build_pipeline(
        PLAN, registry.get("hist_gradient_boosting"), BINARY, 0
    )
    assert "high_cardinality" in step_names(pipeline_)
    model = pipeline_.named_steps["model"]
    assert model.categorical_features == [0]
    pipeline_.fit(features, target)
    assert pipeline_.predict_proba(features).shape == (len(features), 2)


def test_non_native_models_keep_one_hot_for_high_cardinality(registry):
    pipeline_ = preprocessing.build_pipeline(
        PLAN, registry.get("logistic_regression"), BINARY, 0
    )
    assert "high_cardinality" not in step_names(pipeline_)
    assert "categorical" in step_names(pipeline_)


def test_sparse_capable_models_get_sparse_one_hot_on_high_cardinality(
    registry,
):
    features, target = high_cardinality_frame()
    pipeline_ = preprocessing.build_pipeline(
        PLAN, registry.get("logistic_regression"), BINARY, 0
    )
    pipeline_.fit(features, target)
    encoded = pipeline_.named_steps["prep"].transform(features)
    assert hasattr(encoded, "tocsr")  # scipy sparse
    dense = preprocessing.build_pipeline(
        PLAN, registry.get("random_forest"), BINARY, 0
    )
    dense.fit(features, target)
    assert not hasattr(dense.named_steps["prep"].transform(features), "tocsr")


def test_target_encoding_strategy_is_cross_fitted_and_conditional(registry):
    features, target = high_cardinality_frame()
    strategy = strategies.get_strategy("target_encoding")
    adapter = registry.get("logistic_regression")
    pipeline_ = strategy.build(PLAN, adapter, BINARY, 0)
    pipeline_.fit(features, target)
    encoder = step_names(pipeline_)["high_cardinality"].named_steps["encode"]
    assert type(encoder).__name__ == "TargetEncoder"
    assert pipeline_.predict_proba(features).shape == (len(features), 2)
    profile = data_contracts.DatasetProfile(
        manifest=None,
        column_statistics={},
        quality_findings=(),
        risk_findings=(),
        target_summary={},
        feature_plan=data_contracts.FeaturePlan(("x",), ("cat",)),
    )
    reason = strategy.applicability(profile, BINARY, adapter)
    assert reason == "no high-cardinality categorical feature"


def planning_context(spec, plan):
    profile = data_contracts.DatasetProfile(
        manifest=data_contracts.DatasetManifest(
            "ds", "s", "h", 900, 3, "y", "<in-memory>"
        ),
        column_statistics={},
        quality_findings=(),
        risk_findings=(),
        target_summary={"imbalance_ratio": 1.0},
        feature_plan=plan,
    )
    return planner.PlanningContext(spec, profile, BINARY)


def make_spec(**resources):
    return tasks.ProjectSpec(
        data=tasks.DataSpec(uri="<in-memory>", target="y"),
        resources=tasks.ResourceSpec(**resources),
    )


def test_planner_prefers_native_categorical_models_for_high_cardinality(
    registry,
):
    plan = planner.Planner(registry).plan(
        planning_context(make_spec(time_budget_seconds=900), PLAN)
    )
    by_name = {d.technique: d for d in plan.decisions}
    assert "native_categorical_boosting" in by_name
    assert by_name["native_categorical_boosting"].action == "include"
    assert "regularized_target_encoding" in by_name
    models_ = list(plan.candidate_models)
    assert models_.index("hist_gradient_boosting") < models_.index(
        "random_forest"
    )


def test_no_high_cardinality_decisions_without_such_features(registry):
    plain = data_contracts.FeaturePlan(numeric=("x",), categorical=("cat",))
    plan = planner.Planner(registry).plan(
        planning_context(make_spec(time_budget_seconds=900), plain)
    )
    techniques = {d.technique for d in plan.decisions}
    assert "native_categorical_boosting" not in techniques


def test_short_budget_reduces_search_complexity_and_is_recorded(registry):
    spec = make_spec(time_budget_seconds=120)
    assert spec.fast_budget
    search = spec.effective_search()
    assert search.top_models == 1 and search.configs_per_model == 3
    assert spec.effective_strategies().top_models == 1
    plan = planner.Planner(registry).plan(planning_context(spec, PLAN))
    by_name = {d.technique: d for d in plan.decisions}
    assert by_name["reduce_search_complexity"].action == "include"
    relaxed = make_spec(time_budget_seconds=900)
    assert relaxed.effective_search() == relaxed.search
