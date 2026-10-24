"""Public opt-in integration tests for experimental tabular model paths."""

import dataclasses
from importlib import util

import numpy as np
import pandas as pd
import pytest

import scmle
from scmle.contracts import capabilities
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.core import context as context_module
from scmle.core import orchestrator
from scmle.core import stages
from scmle.models import registry as registry_module
from scmle.models.deeplearning import estimators
from scmle.models.deeplearning import jepa
from scmle.models.deeplearning import networks
from scmle.models.sklearn import classical


def test_unknown_experimental_model_fails_closed():
    """A typo in the public opt-in list cannot silently change the plan."""
    registry = registry_module.ModelRegistry((classical.DummyAdapter(),))

    with pytest.raises(
        errors.ConfigError,
        match="unknown experimental model",
    ):
        registry.candidates(
            tasks.TaskType.BINARY,
            row_count=100,
            experimental_models=("torch_jepa",),
        )


def test_unavailable_explicit_experimental_model_fails_closed(monkeypatch):
    """An explicit model cannot silently fall back when its extra is absent."""
    representation = networks.TorchJepaAdapter()
    monkeypatch.setattr(representation, "is_available", lambda: False)
    registry = registry_module.ModelRegistry(
        (classical.DummyAdapter(), representation)
    )

    with pytest.raises(
        errors.ConfigError,
        match="explicit experimental model 'torch_jepa'",
    ):
        registry.candidates(
            tasks.TaskType.BINARY,
            row_count=100,
            experimental_models=("torch_jepa",),
        )


def test_transformer_and_jepa_are_explicit_opt_in_candidates(monkeypatch):
    """Both experimental adapters enter planning only after opt-in."""
    transformer = networks.TorchTransformerAdapter()
    representation = networks.TorchJepaAdapter()
    monkeypatch.setattr(transformer, "is_available", lambda: True)
    monkeypatch.setattr(representation, "is_available", lambda: True)
    registry = registry_module.ModelRegistry(
        (classical.DummyAdapter(), transformer, representation)
    )
    spec = tasks.ProjectSpec(
        data=tasks.DataSpec(uri="<in-memory>", target="y"),
        models=tasks.ModelSpec(
            experimental_models=[transformer.name, representation.name]
        ),
        search=tasks.SearchSpec(top_models=0),
        strategies=tasks.StrategySpec(top_models=0),
        ensemble=tasks.EnsembleSpec(size=0),
    )
    frame = pd.DataFrame({"feature": np.arange(120.0), "y": np.arange(120) % 2})
    project = orchestrator.Orchestrator(spec, frame, registry=registry)
    context = project.new_context()
    context.capability_registry.bind_runtime_environment(
        capabilities.CapabilityEnvironment(
            available_extras=frozenset({"core", "deeplearning"}),
            cpu_cores=1,
            memory_gb=8.0,
            disk_gb=1.0,
            gpu_available=False,
            network_available=False,
        )
    )
    stages.IngestStage().execute(context)
    stages.ProfileStage().execute(context)

    plan = stages.build_plan(context)

    assert {transformer.name, representation.name} <= set(plan.candidate_models)
    decisions = {
        decision.technique: decision.reason
        for decision in plan.decisions
        if decision.technique in {transformer.name, representation.name}
        and decision.action is not None
    }
    assert "experimental-only" in decisions[transformer.name]
    assert "experimental-only" in decisions[representation.name]
    context.close()


def test_jepa_has_a_complete_experimental_tabular_capability():
    """The public JEPA path has a task-level capability declaration."""
    registry = context_module.tabular_beta_capability_registry()

    key = capabilities.CapabilityKey(
        modality=capabilities.Modality.TABULAR,
        task=tasks.TaskType.BINARY,
        model_family=capabilities.ModelFamily.JEPA.value,
        evaluation=capabilities.Evaluation.STANDARD,
        uncertainty=capabilities.Uncertainty.NONE,
        fairness=capabilities.Fairness.NONE,
        explainability=capabilities.Explainability.NONE,
        inference=capabilities.Inference.LIBRARY,
    )

    declaration = registry.find(key)

    assert declaration is not None
    assert declaration.status is capabilities.CapabilityStatus.EXPERIMENTAL
    assert declaration.required_extras == ("deeplearning",)
    assert "multimodal JEPA" in " ".join(declaration.limitations)


def test_jepa_ignores_supervised_labels(monkeypatch):
    """A downstream target is never forwarded into JEPA representation fit."""
    received_targets = []

    def capture_fit(instance, features, target, sample_weight=None):
        del features, sample_weight
        received_targets.append(np.asarray(target))
        return instance

    monkeypatch.setattr(estimators.NetworkEstimator, "fit", capture_fit)
    features = np.ones((8, 2))
    labels = np.arange(8)

    jepa.JepaEstimator().fit(features, labels)

    np.testing.assert_array_equal(received_targets[0], np.zeros(8))


def test_jepa_adapter_builds_a_serializable_pipeline_when_torch_is_available(
    monkeypatch,
):
    """The adapter exposes representation and prediction stages together."""
    adapter = networks.TorchJepaAdapter()
    monkeypatch.setattr(adapter, "is_available", lambda: True)

    candidate = adapter.build(
        tasks.TaskType.BINARY,
        seed=11,
        params={"hidden_size": 32, "embedding_size": 16, "epochs": 3},
    )

    assert [step[0] for step in candidate.steps] == [
        "representation",
        "model",
    ]
    assert candidate.get_params()["representation__embedding_size"] == 16
    assert candidate.get_params()["model__random_state"] == 11


@pytest.mark.skipif(
    util.find_spec("torch") is None,
    reason="the optional deeplearning extra is not installed",
)
@pytest.mark.parametrize(
    "model_name",
    ["torch_ft_transformer", "torch_jepa"],
)
def test_explicit_deep_model_project_packages_and_predicts(
    binary_frame, tmp_path, model_name
):
    """An opt-in tabular deep model uses normal package/predict APIs."""
    project = scmle.MLProject(
        binary_frame,
        "y",
        time_budget=600,
        output_dir=tmp_path,
        enforcement="cooperative",
        search=tasks.SearchSpec(top_models=0),
        strategies=tasks.StrategySpec(top_models=0),
        ensemble=tasks.EnsembleSpec(size=0),
        models=tasks.ModelSpec(experimental_models=[model_name]),
    )
    plan = dataclasses.replace(
        project.plan(),
        candidate_models=(model_name,),
    )

    result = project.run(plan)
    features = binary_frame.drop(columns="y").head(12)
    saved = result.save(tmp_path / f"{model_name}-bundle")
    loaded = scmle.load_model(saved, trusted=True)

    assert result.selected_model == model_name
    assert len(result.predict(features)) == 12
    np.testing.assert_array_equal(
        loaded.predict(features),
        result.predict(features),
    )
