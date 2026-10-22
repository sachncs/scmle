"""Tests for the pre-training capability registry contract."""

import dataclasses
import json

import pytest

from scml.contracts import capabilities
from scml.contracts import errors
from scml.contracts import tasks


def make_key(**overrides):
    values = {
        "modality": capabilities.Modality.TABULAR,
        "task": tasks.TaskType.BINARY,
        "model_family": "tree",
        "evaluation": capabilities.Evaluation.CALIBRATED,
        "uncertainty": capabilities.Uncertainty.CALIBRATED_PROBABILITY,
        "fairness": capabilities.Fairness.GROUP_REPORT,
        "explainability": capabilities.Explainability.FULL,
        "inference": capabilities.Inference.LIBRARY,
    }
    values.update(overrides)
    return capabilities.CapabilityKey(**values)


def make_capability(
    status=capabilities.CapabilityStatus.SUPPORTED,
    key=None,
):
    return capabilities.Capability(
        key=key or make_key(),
        status=status,
        required_extras=("core",),
        resources=capabilities.ResourceAssumptions(
            cpu_cores=2,
            memory_gb=4.0,
            disk_gb=1.0,
            notes=("uses bounded in-memory batches",),
        ),
        failure_modes=("missing sensitive columns", "calibration failure"),
        evidence_links=("tests/unit/test_capability_registry.py",),
        limitations=("library inference only",),
    )


def test_key_covers_every_support_boundary_dimension():
    key = make_key()

    assert key.to_dict() == {
        "modality": "tabular",
        "task": "binary_classification",
        "model_family": "tree",
        "evaluation": "calibrated",
        "uncertainty": "calibrated_probability",
        "fairness": "group_report",
        "explainability": "full",
        "inference": "library",
    }


def test_capability_declaration_serializes_resources_failures_and_evidence():
    record = make_capability()

    assert record.to_dict() == {
        "key": record.key.to_dict(),
        "status": "supported",
        "required_extras": ["core"],
        "resources": {
            "cpu_cores": 2,
            "memory_gb": 4.0,
            "disk_gb": 1.0,
            "gpu_required": False,
            "network_required": False,
            "max_rows": None,
            "notes": ["uses bounded in-memory batches"],
        },
        "failure_modes": ["missing sensitive columns", "calibration failure"],
        "evidence_links": ["tests/unit/test_capability_registry.py"],
        "limitations": ["library inference only"],
    }


def test_supported_capability_is_required_by_default():
    record = make_capability()
    registry = capabilities.CapabilityRegistry((record,))

    assert registry.find(record.key) is record
    assert registry.require(record.key) is record
    assert registry.list_capabilities() == (record,)


def test_non_supported_status_requires_explicit_opt_in():
    record = make_capability(capabilities.CapabilityStatus.EXPERIMENTAL)
    registry = capabilities.CapabilityRegistry((record,))

    with pytest.raises(
        capabilities.UnsupportedCapabilityError,
        match="registered as experimental",
    ):
        registry.require(record.key)
    assert (
        registry.require(
            record.key,
            {capabilities.CapabilityStatus.EXPERIMENTAL},
        )
        is record
    )


def test_unknown_combination_fails_before_training_with_structured_diagnostic():
    registered = make_capability()
    requested = make_key(
        modality=capabilities.Modality.IMAGE,
        model_family="vision",
        inference=capabilities.Inference.BATCH,
    )
    registry = capabilities.CapabilityRegistry((registered,))

    with pytest.raises(capabilities.UnsupportedCapabilityError) as failure:
        registry.require(requested)

    assert failure.value.category == "capability"
    assert failure.value.details["status"] == "unsupported"
    assert failure.value.details["reason"].startswith("no declaration")
    assert failure.value.details["key"] == requested.to_dict()
    assert failure.value.details["registered_keys"] == [
        registered.key.to_dict()
    ]


def test_duplicate_keys_and_incomplete_declarations_are_rejected():
    record = make_capability()
    registry = capabilities.CapabilityRegistry((record,))

    with pytest.raises(errors.ConfigError, match="already registered"):
        registry.register(record)
    with pytest.raises(ValueError, match="evidence_links"):
        capabilities.Capability(
            key=record.key,
            status=capabilities.CapabilityStatus.SUPPORTED,
            required_extras=("core",),
            resources=record.resources,
            failure_modes=record.failure_modes,
        )


def test_contract_rejects_noncanonical_dimensions_and_resources():
    with pytest.raises(TypeError, match="modality"):
        make_key(modality="tabular")
    with pytest.raises(ValueError, match="model_family"):
        make_key(model_family=" Tree ")
    with pytest.raises(ValueError, match="memory_gb"):
        capabilities.ResourceAssumptions(memory_gb=0)


def test_registry_serialization_is_canonical_across_registration_order():
    """Registry evidence does not depend on insertion order."""
    first = capabilities.CapabilityRegistry.with_declared_model_families(
        (make_capability(),)
    )
    second = capabilities.CapabilityRegistry(
        (make_capability(),),
        tuple(reversed(capabilities.DEFAULT_MODEL_FAMILY_DECLARATIONS)),
    )

    assert first.canonical_json() == second.to_json()
    assert json.loads(first.canonical_json()) == first.to_dict()
    assert first.to_dict()["schema_version"] == 1


def test_transformer_and_jepa_family_statuses_are_explicit():
    """Family metadata records experimental boundaries without promotion."""
    registry = capabilities.CapabilityRegistry.with_declared_model_families()

    families = {
        item.model_family: item for item in registry.list_model_families()
    }
    assert {
        capabilities.ModelFamily.TRANSFORMER.value,
        capabilities.ModelFamily.TEXT_TRANSFORMER.value,
        capabilities.ModelFamily.VISION_TRANSFORMER.value,
        capabilities.ModelFamily.AUDIO_TRANSFORMER.value,
        capabilities.ModelFamily.VIDEO_TRANSFORMER.value,
        capabilities.ModelFamily.MULTIMODAL_TRANSFORMER.value,
        capabilities.ModelFamily.JEPA.value,
    } <= families.keys()
    assert families[capabilities.ModelFamily.JEPA.value].status is (
        capabilities.CapabilityStatus.EXPERIMENTAL
    )
    assert families[capabilities.ModelFamily.JEPA.value].modalities == (
        capabilities.Modality.TABULAR,
    )
    assert families[capabilities.ModelFamily.JEPA.value].required_extras == (
        "deeplearning",
    )

    diagnostic = registry.diagnose(make_key(model_family="jepa"))
    assert diagnostic[0].code == "missing_declaration"
    assert diagnostic[0].observed == {"model_family_status": "experimental"}


def test_environment_checks_collect_missing_extras_and_resources():
    """A declared capability fails closed with all observed environment gaps."""
    record = make_capability()
    registry = capabilities.CapabilityRegistry((record,))
    environment = capabilities.CapabilityEnvironment(
        available_extras=frozenset(),
        cpu_cores=1,
        memory_gb=1.0,
        disk_gb=0.0,
    )

    diagnostics = registry.diagnose(record.key, environment=environment)

    assert {item.code for item in diagnostics} == {
        "missing_extra",
        "insufficient_cpu_cores",
        "insufficient_memory_gb",
        "insufficient_disk_gb",
    }
    with pytest.raises(capabilities.UnsupportedCapabilityError) as failure:
        registry.require(record.key, environment=environment)
    assert [item["code"] for item in failure.value.details["diagnostics"]] == [
        item.code for item in diagnostics
    ]


def test_required_resources_reject_unknown_runtime_facts():
    """Required GPU and network facts cannot be inferred optimistically."""
    record = make_capability()
    gpu_key = make_key(model_family="vision_transformer")
    gpu_record = dataclasses.replace(
        record,
        key=gpu_key,
        resources=capabilities.ResourceAssumptions(
            memory_gb=4.0,
            gpu_required=True,
            network_required=True,
        ),
    )
    registry = capabilities.CapabilityRegistry((gpu_record,))
    environment = capabilities.CapabilityEnvironment(
        available_extras=frozenset({"core"}),
        cpu_cores=2,
        memory_gb=4.0,
        disk_gb=1.0,
    )

    diagnostics = registry.diagnose(
        gpu_key,
        environment=environment,
    )

    assert {item.code for item in diagnostics} == {
        "unknown_gpu_available",
        "unknown_network_available",
    }


def test_row_limits_are_checked_before_training():
    """A bounded capability requires an observed row count."""
    record = dataclasses.replace(
        make_capability(),
        resources=capabilities.ResourceAssumptions(
            memory_gb=4.0,
            max_rows=10,
        ),
    )
    registry = capabilities.CapabilityRegistry((record,))

    assert registry.diagnose(record.key) == (
        capabilities.CapabilityDiagnostic(
            code="unknown_row_count",
            message="row count is required for this capability limit",
            expected=10,
        ),
    )
    with pytest.raises(capabilities.UnsupportedCapabilityError):
        registry.require(record.key, row_count=11)
