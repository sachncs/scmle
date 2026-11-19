"""Pre-training capability boundary tests."""

import dataclasses
import importlib.util

import numpy
import pandas
import pytest

from scmle.contracts import capabilities
from scmle.contracts import tasks
from scmle.models.deeplearning import networks
import scmle.models.registry
from scmle.models.sklearn import classical
from scmle.pipeline import engine
from scmle.pipeline import orchestrator
from scmle.pipeline import roster
from scmle.pipeline import stages
from scmle.pipeline.stages import ingestion


def binary_frame() -> pandas.DataFrame:
    """Builds a deterministic tabular fixture."""
    rng = numpy.random.default_rng(7)
    return pandas.DataFrame(
        {
            "feature": rng.normal(size=120),
            "y": rng.integers(0, 2, size=120),
        }
    )


def profiled(
    model_registry: scmle.models.registry.Catalog | None = None,
    spec: tasks.Config | None = None,
) -> engine.Board:
    """Builds a profiled run's board without entering model fitting."""
    project_spec = spec or tasks.Config(
        data=tasks.Dataset(uri="<in-memory>", target="y"),
        search=tasks.Search(top_models=0),
        strategies=tasks.Strategies(top_models=0),
        ensemble=tasks.Stacking(size=0),
    )
    models = model_registry or scmle.models.registry.Catalog(
        (
            classical.Dummy(),
            classical.LOGISTIC_REGRESSION,
        )
    )
    project = orchestrator.Orchestrator(
        project_spec, binary_frame(), registry=models
    )
    board = project.start()
    stages.IngestStage().execute(board)
    stages.ProfileStage().execute(board)
    return board


def test_tabular_beta_plan_has_an_explicit_library_capability_path():
    """The existing tabular plan succeeds through declared capabilities."""
    board = profiled()

    plan = stages.devise(board)

    assert set(plan.candidates) == {"dummy", "logistic_regression"}
    declarations = board.services.capability_registry.catalogue()
    assert declarations
    assert {declaration.key.modality for declaration in declarations} == {
        capabilities.Modality.TABULAR
    }
    assert {declaration.key.inference for declaration in declarations} == {
        capabilities.Inference.LIBRARY
    }
    family = board.services.capability_registry.find(
        capabilities.ModelFamily.JEPA.value
    )
    assert family is not None
    assert family.status is capabilities.Status.EXPERIMENTAL
    assert family.modalities == (capabilities.Modality.TABULAR,)


def test_deep_learning_paths_declare_their_optional_extra():
    """Deep-learning capability declarations identify their installation gate."""
    registry = roster.enrol()

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
    adapter = networks.FT_TRANSFORMER
    assert adapter.family == "deep_learning"
    assert adapter.architecture == "ft_transformer"
    assert adapter.capability_family == "transformer"

    registry = roster.enrol()
    declaration = registry.find("transformer")
    assert declaration is not None
    assert declaration.status is capabilities.Status.EXPERIMENTAL
    assert declaration.modalities == (capabilities.Modality.TABULAR,)

    board = profiled()
    key = stages.requisition(
        board, tasks.Task.BINARY, adapter.capability_family
    )
    diagnostics = registry.diagnose(key, row_count=120)
    assert {item.code for item in diagnostics} == {
        "status_not_allowed",
    }


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="the optional deeplearning extra is not installed",
)
def test_experimental_transformer_requires_explicit_model_opt_in():
    """An experimental FT-Transformer can be selected only by declaration.

    Also enforces website/src/pages/docs/modeling.astro:78.
    """
    adapter = networks.FT_TRANSFORMER
    registry = scmle.models.registry.Catalog((classical.Dummy(), adapter))
    default_board = profiled(model_registry=registry)
    default_plan = stages.devise(default_board)
    assert adapter.name not in default_plan.candidates

    spec = default_board.services.spec.model_copy(
        update={"models": tasks.Candidates(experimental_models=[adapter.name])}
    )
    opted_in_board = profiled(model_registry=registry, spec=spec)
    opted_in_plan = stages.devise(opted_in_board)
    assert adapter.name in opted_in_plan.candidates


def test_capability_family_is_used_before_training():
    """A coarse deep-learning family cannot bypass its boundary identity."""

    class ExperimentalTransformerAdapter(classical.Dummy):
        """Test adapter with the FT-Transformer boundary identity."""

        name = "experimental_transformer"
        family = "deep_learning"
        capability_family = "transformer"

    board = profiled(
        model_registry=scmle.models.registry.Catalog(
            (ExperimentalTransformerAdapter(),)
        )
    )

    with pytest.raises(capabilities.UnsupportedError):
        stages.devise(board)


def test_extension_only_foundation_adapter_cannot_break_auto_planning(
    monkeypatch,
):
    """An installed extension-only model must not enter the beta plan.

    Also enforces website/src/pages/docs/capabilities.astro:10.
    """
    registry = scmle.models.registry.Catalog.stock()
    adapter = registry.get("tabpfn")
    monkeypatch.setattr(adapter, "is_available", lambda: True)

    plan = stages.devise(profiled(model_registry=registry))

    assert "tabpfn" not in plan.candidates


def test_unimplemented_contract_dimensions_are_not_promoted():
    """Missing modality, uncertainty, and explanation paths fail closed.

    Also enforces website/src/pages/docs/capabilities.astro:82.
    Also enforces website/src/pages/docs/evidence.astro:32.
    """
    registry = roster.enrol()
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
        with pytest.raises(capabilities.UnsupportedError):
            registry.require(unsupported)


def test_empty_override_fails_before_model_fitting():
    """An empty injected registry cannot silently fall back to defaults."""
    board = profiled()
    board.services = dataclasses.replace(
        board.services, capability_registry=capabilities.Registry()
    )

    with pytest.raises(
        capabilities.UnsupportedError,
        match="unsupported capability before training",
    ):
        stages.devise(board)


def test_declared_non_tabular_modality_fails_closed_before_model_fitting():
    """A modality declaration cannot silently inherit tabular support.

    Also enforces website/src/pages/docs/api.astro:33.
    Also enforces website/src/pages/docs/data.astro:25.
    """
    spec = tasks.Config(
        data=tasks.Dataset(uri="<in-memory>", target="y", modality="image"),
        search=tasks.Search(top_models=0),
        strategies=tasks.Strategies(top_models=0),
        ensemble=tasks.Stacking(size=0),
    )
    board = profiled(spec=spec)

    with pytest.raises(
        capabilities.UnsupportedError,
        match="unsupported capability",
    ) as failure:
        stages.devise(board)

    assert failure.value.details["key"]["modality"] == "image"
    assert board.get(ingestion.Intake).manifest.modality == "image"


def test_file_backed_non_tabular_modality_fails_before_tabular_reader(
    tmp_path,
):
    """Unsupported file-backed data returns a structured boundary error.

    Also enforces website/src/pages/docs/data.astro:25.
    """
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
        capabilities.UnsupportedError,
        match="file-backed modality has no registered ingestion adapter",
    ) as failure:
        stages.IngestStage().execute(project.start())

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
    board = profiled(model_registry=model_registry)

    with pytest.raises(capabilities.UnsupportedError):
        stages.devise(board)


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
    board = project.start()
    stages.IngestStage().execute(board)
    stages.ProfileStage().execute(board)

    with pytest.raises(
        capabilities.UnsupportedError,
        match="group-threshold mitigation",
    ):
        stages.devise(board)
