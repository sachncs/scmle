"""Pre-training capability boundary tests."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from scml.contracts import capabilities
from scml.contracts import tasks
from scml.core import context as context_module
from scml.core import orchestrator
from scml.core import stages
from scml.models import registry as registry_module
from scml.models.sklearn import classical


def binary_frame() -> pd.DataFrame:
    """Builds a deterministic tabular fixture."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "feature": rng.normal(size=120),
            "y": rng.integers(0, 2, size=120),
        }
    )


def make_context(
    model_registry: registry_module.ModelRegistry | None = None,
    spec: tasks.ProjectSpec | None = None,
) -> context_module.RunContext:
    """Builds a profiled context without entering model fitting."""
    project_spec = spec or tasks.ProjectSpec(
        data=tasks.DataSpec(uri="<in-memory>", target="y"),
        search=tasks.SearchSpec(top_models=0),
        strategies=tasks.StrategySpec(top_models=0),
        ensemble=tasks.EnsembleSpec(size=0),
    )
    models = model_registry or registry_module.ModelRegistry(
        (
            classical.DummyAdapter(),
            classical.LogisticRegressionAdapter(),
        )
    )
    project = orchestrator.Orchestrator(
        project_spec, binary_frame(), registry=models
    )
    context = project.new_context()
    stages.IngestStage().execute(context)
    stages.ProfileStage().execute(context)
    return context


def test_tabular_beta_plan_has_an_explicit_library_capability_path():
    """The existing tabular plan succeeds through declared capabilities."""
    context = make_context()

    plan = stages.build_plan(context)

    assert set(plan.candidate_models) == {"dummy", "logistic_regression"}
    declarations = context.capability_registry.list_capabilities()
    assert declarations
    assert {declaration.key.modality for declaration in declarations} == {
        capabilities.Modality.TABULAR
    }
    assert {declaration.key.inference for declaration in declarations} == {
        capabilities.Inference.LIBRARY
    }


def test_deep_learning_paths_declare_their_optional_extra():
    """Deep-learning capability declarations identify their installation gate."""
    registry = context_module.tabular_beta_capability_registry()

    for task in (
        tasks.TaskType.BINARY,
        tasks.TaskType.MULTICLASS,
        tasks.TaskType.REGRESSION,
    ):
        key = capabilities.CapabilityKey(
            modality=capabilities.Modality.TABULAR,
            task=task,
            model_family="deep_learning",
            evaluation=capabilities.Evaluation.STANDARD,
            uncertainty=capabilities.Uncertainty.NONE,
            fairness=capabilities.Fairness.NONE,
            explainability=capabilities.Explainability.NONE,
            inference=capabilities.Inference.LIBRARY,
        )
        assert registry.require(key).required_extras == ("deeplearning",)


def test_unimplemented_contract_dimensions_are_not_promoted():
    """Missing modality, uncertainty, and explanation paths fail closed."""
    registry = context_module.tabular_beta_capability_registry()
    key = capabilities.CapabilityKey(
        modality=capabilities.Modality.TABULAR,
        task=tasks.TaskType.BINARY,
        model_family="linear",
        evaluation=capabilities.Evaluation.STANDARD,
        uncertainty=capabilities.Uncertainty.NONE,
        fairness=capabilities.Fairness.NONE,
        explainability=capabilities.Explainability.NONE,
        inference=capabilities.Inference.LIBRARY,
    )

    assert (
        registry.require(key).status is capabilities.CapabilityStatus.SUPPORTED
    )
    for unsupported in (
        dataclasses.replace(key, modality=capabilities.Modality.IMAGE),
        dataclasses.replace(
            key, uncertainty=capabilities.Uncertainty.CONFORMAL
        ),
        dataclasses.replace(
            key, explainability=capabilities.Explainability.FULL
        ),
    ):
        with pytest.raises(capabilities.UnsupportedCapabilityError):
            registry.require(unsupported)


def test_empty_override_fails_before_model_fitting():
    """An empty injected registry cannot silently fall back to defaults."""
    context = make_context()
    context.capability_registry = capabilities.CapabilityRegistry()

    with pytest.raises(
        capabilities.UnsupportedCapabilityError,
        match="unsupported capability before training",
    ):
        stages.build_plan(context)


def test_declared_non_tabular_modality_fails_closed_before_model_fitting():
    """A modality declaration cannot silently inherit tabular support."""
    spec = tasks.ProjectSpec(
        data=tasks.DataSpec(uri="<in-memory>", target="y", modality="image"),
        search=tasks.SearchSpec(top_models=0),
        strategies=tasks.StrategySpec(top_models=0),
        ensemble=tasks.EnsembleSpec(size=0),
    )
    context = make_context(spec=spec)

    with pytest.raises(
        capabilities.UnsupportedCapabilityError,
        match="unsupported capability",
    ) as failure:
        stages.build_plan(context)

    assert failure.value.details["key"]["modality"] == "image"
    assert context.manifest is not None
    assert context.manifest.modality == "image"


def test_file_backed_non_tabular_modality_fails_before_tabular_reader(
    tmp_path,
):
    """Unsupported file-backed data returns a structured boundary error."""
    source = tmp_path / "image-list.txt"
    source.write_text("image-a.jpg\n")
    spec = tasks.ProjectSpec(
        data=tasks.DataSpec(uri=str(source), target="label", modality="image"),
        search=tasks.SearchSpec(top_models=0),
        strategies=tasks.StrategySpec(top_models=0),
        ensemble=tasks.EnsembleSpec(size=0),
    )
    project = orchestrator.Orchestrator(spec, None)

    with pytest.raises(
        capabilities.UnsupportedCapabilityError,
        match="file-backed modality has no registered ingestion adapter",
    ) as failure:
        stages.IngestStage().execute(project.new_context())

    assert failure.value.details == {
        "status": "unsupported",
        "reason": "file-backed modality has no ingestion adapter",
        "modality": "image",
        "uri": str(source),
        "supported_modalities": ["tabular"],
    }


def test_unlisted_model_family_fails_before_model_fitting():
    """A planner candidate needs its own explicit family declaration."""

    class UnlistedAdapter(classical.DummyAdapter):
        """Test adapter representing an undeclared model family."""

        name = "unlisted_model"
        family = "unlisted"

    model_registry = registry_module.ModelRegistry((UnlistedAdapter(),))
    context = make_context(model_registry=model_registry)

    with pytest.raises(capabilities.UnsupportedCapabilityError):
        stages.build_plan(context)


def test_group_thresholds_reject_non_binary_tasks_before_fitting():
    """The registry boundary rejects a known unsupported fairness path."""
    spec = tasks.ProjectSpec(
        data=tasks.DataSpec(
            uri="<in-memory>",
            target="y",
            sensitive_columns=["group"],
        ),
        task="regression",
        fairness=tasks.FairnessSpec(
            mitigation="group_thresholds", column="group"
        ),
        search=tasks.SearchSpec(top_models=0),
        strategies=tasks.StrategySpec(top_models=0),
        ensemble=tasks.EnsembleSpec(size=0),
    )
    frame = binary_frame().assign(group="a", y=lambda value: value.feature)
    project = orchestrator.Orchestrator(
        spec,
        frame,
        registry=registry_module.ModelRegistry((classical.DummyAdapter(),)),
    )
    context = project.new_context()
    stages.IngestStage().execute(context)
    stages.ProfileStage().execute(context)

    with pytest.raises(
        capabilities.UnsupportedCapabilityError,
        match="group-threshold mitigation",
    ):
        stages.build_plan(context)
