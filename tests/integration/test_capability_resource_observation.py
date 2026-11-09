"""Focused tests for runtime facts at the capability boundary."""

import json
import pathlib

import numpy
import pandas
import pytest

from scmle.contracts import capabilities
from scmle.contracts import tasks
from scmle.core import orchestrator
from scmle.core import resource_observation
from scmle.core import stages
import scmle.models.registry
from scmle.models.sklearn import classical


def binary_frame() -> pandas.DataFrame:
    """Build a small deterministic frame for the planning integration."""
    rng = numpy.random.default_rng(23)
    return pandas.DataFrame(
        {
            "feature": rng.normal(size=80),
            "y": rng.integers(0, 2, size=80),
        }
    )


def capability(resources: capabilities.ResourceAssumptions):
    """Return the baseline binary capability used by the test gate."""
    return capabilities.Capability(
        key=capabilities.CapabilityKey(
            modality=capabilities.Modality.TABULAR,
            task=tasks.TaskType.BINARY,
            model_family="baseline",
            evaluation=capabilities.Evaluation.STANDARD,
            uncertainty=capabilities.Uncertainty.NONE,
            fairness=capabilities.Fairness.NONE,
            explainability=capabilities.Explainability.NONE,
            inference=capabilities.Inference.LIBRARY,
        ),
        status=capabilities.CapabilityStatus.SUPPORTED,
        required_extras=("core",),
        resources=resources,
        failure_modes=("resource observation unavailable",),
        evidence_links=(
            "tests/integration/test_capability_resource_observation.py",
        ),
    )


def observation(
    *,
    cpu_cores: int | None,
    memory_gb: float | None,
    disk_gb: float | None,
) -> resource_observation.RuntimeResourceObservation:
    """Build a fixed observation for tests without probing the host."""
    environment = capabilities.CapabilityEnvironment(
        available_extras=frozenset({"core"}),
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        disk_gb=disk_gb,
    )
    return resource_observation.RuntimeResourceObservation(
        environment,
        captured_at="2026-01-01T00:00:00+00:00",
        disk_path="/tmp",
        extra_observations={"core": True},
        sources={
            "cpu_cores": "test",
            "memory_gb": "test",
            "disk_gb": "test",
            "extra:core": "test",
        },
    )


def make_spec() -> tasks.ProjectSpec:
    """Return a planning-only project specification."""
    return tasks.ProjectSpec(
        data=tasks.DataSpec(uri="<in-memory>", target="y"),
        search=tasks.SearchSpec(top_models=0),
        strategies=tasks.StrategySpec(top_models=0),
        ensemble=tasks.EnsembleSpec(size=0),
    )


def test_observation_captures_bounded_facts_and_unknowns(monkeypatch, tmp_path):
    """The probe serializes CPU, memory, disk, extras, and null unknowns."""
    monkeypatch.setattr(
        resource_observation,
        "observe_cpu_cores",
        lambda: (4, "test.cpu"),
    )
    monkeypatch.setattr(
        resource_observation,
        "observe_memory_gb",
        lambda: (8.0, "test.memory"),
    )
    monkeypatch.setattr(
        resource_observation,
        "observe_disk_gb",
        lambda path: (12.0, f"test.disk:{path}"),
    )
    monkeypatch.setattr(
        resource_observation,
        "observe_extras",
        lambda: (
            frozenset({"core", "deeplearning"}),
            {"core": True, "deeplearning": True},
            {"extra:core": "test", "extra:deeplearning": "test"},
        ),
    )

    record = resource_observation.observe_runtime_resources(tmp_path).to_dict()

    assert record["environment"] == {
        "available_extras": ["core", "deeplearning"],
        "cpu_cores": 4,
        "memory_gb": 8.0,
        "disk_gb": 12.0,
        "gpu_available": None,
        "network_available": None,
    }
    assert record["optional_extras"]["deeplearning"]["available"] is True
    assert record["observations"]["gpu_available"]["value"] is None


def test_bound_observation_is_used_by_the_capability_gate(monkeypatch):
    """A context-bound observation is applied by the existing gate call."""
    record = capability(
        capabilities.ResourceAssumptions(
            cpu_cores=2,
            memory_gb=4.0,
            disk_gb=1.0,
        )
    )
    registry = capabilities.CapabilityRegistry((record,))
    monkeypatch.setattr(
        orchestrator.resource_observation,
        "observe_runtime_resources",
        lambda path=None: observation(
            cpu_cores=2,
            memory_gb=4.0,
            disk_gb=1.0,
        ),
    )
    project = orchestrator.Orchestrator(
        make_spec(),
        binary_frame(),
        registry=scmle.models.registry.ModelRegistry(
            (classical.DummyAdapter(),)
        ),
        capability_registry=registry,
    )
    context = project.new_context()
    stages.IngestStage().execute(context)
    stages.ProfileStage().execute(context)

    plan = stages.build_plan(context)

    assert plan.candidate_models == ("dummy",)


def test_unknown_declared_facts_fail_closed_and_are_persisted(
    monkeypatch, tmp_path
):
    """The planning gate rejects unknown facts after saving its evidence."""
    record = capability(
        capabilities.ResourceAssumptions(
            cpu_cores=2,
            memory_gb=4.0,
            disk_gb=1.0,
        )
    )
    registry = capabilities.CapabilityRegistry((record,))
    monkeypatch.setattr(
        orchestrator.resource_observation,
        "observe_runtime_resources",
        lambda path=None: observation(
            cpu_cores=None,
            memory_gb=None,
            disk_gb=None,
        ),
    )
    project = orchestrator.Orchestrator(
        make_spec(),
        binary_frame(),
        registry=scmle.models.registry.ModelRegistry(
            (classical.DummyAdapter(),)
        ),
        capability_registry=registry,
    )

    with pytest.raises(capabilities.UnsupportedCapabilityError) as failure:
        project.run(tmp_path)

    diagnostics = failure.value.details["diagnostics"]
    assert {item["code"] for item in diagnostics} == {
        "unknown_cpu_cores",
        "unknown_memory_gb",
        "unknown_disk_gb",
    }
    evidence_path = (
        pathlib.Path(failure.value.details["run_dir"])
        / "evidence"
        / "capability_resources.json"
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["environment"]["cpu_cores"] is None
    assert evidence["environment"]["available_extras"] == ["core"]
