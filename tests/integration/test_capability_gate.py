"""Pre-training capability boundary tests."""

import dataclasses

import numpy
import pandas
import pytest

from scmle.contracts import capabilities
from scmle.contracts import tasks
from scmle.core import orchestrator
from scmle.core import stages
import scmle.core.context
from scmle.models.deeplearning import networks
import scmle.models.registry
from scmle.models.sklearn import classical


def binary_frame() -> pandas.DataFrame:
    """Builds a deterministic tabular fixture."""
    rng = numpy.random.default_rng(7)
    return pandas.DataFrame(
        {
            "feature": rng.normal(size=120),
            "y": rng.integers(0, 2, size=120),
        }
    )


def make_context(
    model_registry: scmle.models.registry.Catalog | None = None,
    spec: tasks.Config | None = None,
) -> scmle.core.context.Context:
    """Builds a profiled context without entering model fitting."""
    project_spec = spec or tasks.Config(
        data=tasks.Dataset(uri="<in-memory>", target="y"),
        search=tasks.Search(top_models=0),
        strategies=tasks.Strategies(top_models=0),
        ensemble=tasks.Stacking(size=0),
    )
    models = model_registry or scmle.models.registry.Catalog(
        (
            classical.Dummy(),
            classical.LogisticRegression(),
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
    family = context.capability_registry.find_model_family(
        capabilities.ModelFamily.JEPA.value
    )
    assert family is not None
    assert family.status is capabilities.Status.EXPERIMENTAL
    assert family.modalities == (capabilities.Modality.TABULAR,)


def test_deep_learning_paths_declare_their_optional_extra():
    """Deep-learning capability declarations identify their installation gate."""
    registry = scmle.core.context.tabular_beta_capability_registry()

    for task in (
        tasks.Task.BINARY,
        tasks.Task.MULTICLASS,
        tasks.Task.REGRESSION,
    ):
        key = capabilities.Coordinate(
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


def test_ft_transformer_uses_an_experimental_tabular_boundary():
    """FT-Transformer must not inherit supported deep-learning status."""
    adapter = networks.FtTransformer()
    assert adapter.family == "deep_learning"
    assert adapter.architecture == "ft_transformer"
    assert adapter.capability_family == "transformer"

    registry = scmle.core.context.tabular_beta_capability_registry()
    declaration = registry.find_model_family("transformer")
    assert declaration is not None
    assert declaration.status is capabilities.Status.EXPERIMENTAL
    assert declaration.modalities == (capabilities.Modality.TABULAR,)

    context = make_context()
    key = stages.capability_key(
        context, tasks.Task.BINARY, adapter.capability_family
    )
    diagnostics = registry.diagnose(key, row_count=120)
    assert {item.code for item in diagnostics} == {
        "status_not_allowed",
    }


def test_experimental_transformer_requires_explicit_model_opt_in():
    """An experimental FT-Transformer can be selected only by declaration."""
    pytest.importorskip(
        "torch",
        reason="the optional deeplearning extra is not installed",
    )
    adapter = networks.FtTransformer()
    registry = scmle.models.registry.Catalog((classical.Dummy(), adapter))
    default_context = make_context(model_registry=registry)
    default_plan = stages.build_plan(default_context)
    assert adapter.name not in default_plan.candidate_models

    spec = default_context.spec.model_copy(
        update={"models": tasks.Candidates(experimental_models=[adapter.name])}
    )
    opted_in_context = make_context(model_registry=registry, spec=spec)
    opted_in_plan = stages.build_plan(opted_in_context)
    assert adapter.name in opted_in_plan.candidate_models


def test_capability_family_is_used_before_training():
    """A coarse deep-learning family cannot bypass its boundary identity."""

    class ExperimentalTransformerAdapter(classical.Dummy):
        """Test adapter with the FT-Transformer boundary identity."""

        name = "experimental_transformer"
        family = "deep_learning"
        capability_family = "transformer"

    context = make_context(
        model_registry=scmle.models.registry.Catalog(
            (ExperimentalTransformerAdapter(),)
        )
    )

    with pytest.raises(capabilities.UnsupportedCapabilityError):
        stages.build_plan(context)


def test_extension_only_foundation_adapter_cannot_break_auto_planning(
    monkeypatch,
):
    """An installed extension-only model must not enter the beta plan."""
    registry = scmle.models.registry.Catalog.with_builtins()
    adapter = registry.get("tabpfn")
    monkeypatch.setattr(adapter, "is_available", lambda: True)

    plan = stages.build_plan(make_context(model_registry=registry))

    assert "tabpfn" not in plan.candidate_models


def test_unimplemented_contract_dimensions_are_not_promoted():
    """Missing modality, uncertainty, and explanation paths fail closed."""
    registry = scmle.core.context.tabular_beta_capability_registry()
    key = capabilities.Coordinate(
        modality=capabilities.Modality.TABULAR,
        task=tasks.Task.BINARY,
        model_family="linear",
        evaluation=capabilities.Evaluation.STANDARD,
        uncertainty=capabilities.Uncertainty.NONE,
        fairness=capabilities.Fairness.NONE,
        explainability=capabilities.Explainability.NONE,
        inference=capabilities.Inference.LIBRARY,
    )

    assert registry.require(key).status is capabilities.Status.SUPPORTED
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
    context.capability_registry = capabilities.Registry()

    with pytest.raises(
        capabilities.UnsupportedCapabilityError,
        match="unsupported capability before training",
    ):
        stages.build_plan(context)


def test_declared_non_tabular_modality_fails_closed_before_model_fitting():
    """A modality declaration cannot silently inherit tabular support."""
    spec = tasks.Config(
        data=tasks.Dataset(uri="<in-memory>", target="y", modality="image"),
        search=tasks.Search(top_models=0),
        strategies=tasks.Strategies(top_models=0),
        ensemble=tasks.Stacking(size=0),
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
    spec = tasks.Config(
        data=tasks.Dataset(uri=str(source), target="label", modality="image"),
        search=tasks.Search(top_models=0),
        strategies=tasks.Strategies(top_models=0),
        ensemble=tasks.Stacking(size=0),
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

    class UnlistedAdapter(classical.Dummy):
        """Test adapter representing an undeclared model family."""

        name = "unlisted_model"
        family = "unlisted"

    model_registry = scmle.models.registry.Catalog((UnlistedAdapter(),))
    context = make_context(model_registry=model_registry)

    with pytest.raises(capabilities.UnsupportedCapabilityError):
        stages.build_plan(context)


def test_group_thresholds_reject_non_binary_tasks_before_fitting():
    """The registry boundary rejects a known unsupported fairness path."""
    spec = tasks.Config(
        data=tasks.Dataset(
            uri="<in-memory>",
            target="y",
            sensitive_columns=["group"],
        ),
        task="regression",
        fairness=tasks.Parity(mitigation="group_thresholds", column="group"),
        search=tasks.Search(top_models=0),
        strategies=tasks.Strategies(top_models=0),
        ensemble=tasks.Stacking(size=0),
    )
    frame = binary_frame().assign(group="a", y=lambda value: value.feature)
    project = orchestrator.Orchestrator(
        spec,
        frame,
        registry=scmle.models.registry.Catalog((classical.Dummy(),)),
    )
    context = project.new_context()
    stages.IngestStage().execute(context)
    stages.ProfileStage().execute(context)

    with pytest.raises(
        capabilities.UnsupportedCapabilityError,
        match="group-threshold mitigation",
    ):
        stages.build_plan(context)
