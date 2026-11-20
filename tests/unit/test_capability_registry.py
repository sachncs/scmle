"""Tests for the pre-training capability registry contract."""

import dataclasses
import json

import pytest

from scmle.contracts import capabilities
from scmle.contracts import errors
from scmle.contracts import tasks


def make_key(**overrides):
    values = {
        "modality": capabilities.Modality.TABULAR,
        "task": tasks.Task.BINARY,
        "family": "tree",
        "evaluation": capabilities.Evaluation.CALIBRATED,
        "uncertainty": capabilities.Uncertainty.CALIBRATED_PROBABILITY,
        "fairness": capabilities.Fairness.GROUP_REPORT,
        "explainability": capabilities.Explainability.FULL,
        "inference": capabilities.Inference.LIBRARY,
    }
    values.update(overrides)
    return capabilities.Coordinate(**values)


def make_capability(
    status=capabilities.Status.SUPPORTED,
    key=None,
):
    return capabilities.Capability(
        key=key or make_key(),
        status=status,
        extras=("core",),
        resources=capabilities.Resources(
            cores=2,
            memory=4.0,
            disk=1.0,
            notes=("uses bounded in-memory batches",),
        ),
        modes=("missing sensitive columns", "calibration failure"),
        links=("tests/unit/test_capability_registry.py",),
        limitations=("library inference only",),
    )


def test_key_covers_every_support_boundary_dimension():
    key = make_key()

    assert key.to_dict() == {
        "modality": "tabular",
        "task": "binary_classification",
        "family": "tree",
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
        "extras": ["core"],
        "resources": {
            "cpu": 2,
            "memory": 4.0,
            "disk": 1.0,
            "gpu": False,
            "network": False,
            "rows": None,
            "notes": ["uses bounded in-memory batches"],
        },
        "failures": ["missing sensitive columns", "calibration failure"],
        "links": ["tests/unit/test_capability_registry.py"],
        "limitations": ["library inference only"],
    }


def test_supported_capability_is_required_by_default():
    record = make_capability()
    registry = capabilities.Registry((record,))

    assert registry.records.get(record.key) is record
    assert registry.require(record.key) is record
    assert registry.catalogue() == (record,)


def test_non_supported_status_requires_explicit_opt_in():
    record = make_capability(capabilities.Status.EXPERIMENTAL)
    registry = capabilities.Registry((record,))

    with pytest.raises(
        capabilities.UnsupportedError,
        match="registered as experimental",
    ):
        registry.require(record.key)
    assert (
        registry.require(
            record.key,
            {capabilities.Status.EXPERIMENTAL},
        )
        is record
    )


def test_unknown_combination_fails_before_training_with_structured_diagnostic():
    registered = make_capability()
    requested = make_key(
        modality=capabilities.Modality.IMAGE,
        family="vision",
        inference=capabilities.Inference.BATCH,
    )
    registry = capabilities.Registry((registered,))

    with pytest.raises(capabilities.UnsupportedError) as failure:
        registry.require(requested)

    assert failure.value.category == "capability"
    assert failure.value.details["status"] == "unsupported"
    assert failure.value.details["reason"].startswith("no declaration")
    assert failure.value.details["key"] == requested.to_dict()
    assert failure.value.details["registered_keys"] == [
        registered.key.to_dict()
    ]


def test_duplicate_keys_and_incomplete_declarations_are_rejected():
    # Enforces website/src/pages/docs/capabilities.astro:133.
    record = make_capability()
    registry = capabilities.Registry((record,))

    with pytest.raises(errors.ConfigError, match="already registered"):
        registry.register(record)
    with pytest.raises(ValueError, match="evidence_links"):
        capabilities.Capability(
            key=record.key,
            status=capabilities.Status.SUPPORTED,
            extras=("core",),
            resources=record.resources,
            modes=record.modes,
        )


def test_contract_rejects_noncanonical_dimensions_and_resources():
    with pytest.raises(TypeError, match="modality"):
        make_key(modality="tabular")
    with pytest.raises(ValueError, match="model_family"):
        make_key(family=" Tree ")
    with pytest.raises(ValueError, match="memory_gb"):
        capabilities.Resources(memory=0)


def test_registry_serialization_is_canonical_across_registration_order():
    """Registry evidence does not depend on insertion order."""
    first = capabilities.Registry.seed((make_capability(),))
    second = capabilities.Registry(
        (make_capability(),),
        tuple(reversed(capabilities.DEFAULT_MODEL_FAMILY_DECLARATIONS)),
    )

    assert first.canonize() == second.to_json()
    assert json.loads(first.canonize()) == first.to_dict()
    assert first.to_dict()["schema_version"] == 2


def test_transformer_and_jepa_family_statuses_are_explicit():
    """Family metadata records implemented boundaries without promotion.

    Also enforces website/src/pages/docs/modeling.astro:144.
    Also enforces website/src/pages/docs/modeling.astro:13.
    Also enforces website/src/pages/docs/modeling.astro:101.
    """
    registry = capabilities.Registry.seed()

    families = {item.family: item for item in registry.inventory()}
    assert {
        capabilities.Family.TRANSFORMER.value,
        capabilities.Family.TEXT_TRANSFORMER.value,
        capabilities.Family.VISION_TRANSFORMER.value,
        capabilities.Family.AUDIO_TRANSFORMER.value,
        capabilities.Family.VIDEO_TRANSFORMER.value,
        capabilities.Family.MULTIMODAL_TRANSFORMER.value,
        capabilities.Family.JEPA.value,
    } <= families.keys()
    assert families[capabilities.Family.JEPA.value].status is (
        capabilities.Status.EXPERIMENTAL
    )
    assert families[capabilities.Family.JEPA.value].modalities == (
        capabilities.Modality.TABULAR,
    )
    assert families[capabilities.Family.JEPA.value].extras == ("deeplearning",)
    assert families[capabilities.Family.TRANSFORMER.value].status is (
        capabilities.Status.EXPERIMENTAL
    )
    assert families[capabilities.Family.TRANSFORMER.value].modalities == (
        capabilities.Modality.TABULAR,
    )
    assert families[capabilities.Family.TRANSFORMER.value].accommodates(
        capabilities.Modality.TABULAR
    )
    assert not families[capabilities.Family.TRANSFORMER.value].accommodates(
        capabilities.Modality.IMAGE
    )
    for family_name in (
        capabilities.Family.TEXT_TRANSFORMER.value,
        capabilities.Family.VISION_TRANSFORMER.value,
        capabilities.Family.AUDIO_TRANSFORMER.value,
        capabilities.Family.VIDEO_TRANSFORMER.value,
        capabilities.Family.MULTIMODAL_TRANSFORMER.value,
    ):
        assert families[family_name].status is (
            capabilities.Status.EXTENSION_ONLY
        )

    diagnostic = registry.diagnose(make_key(family="jepa"))
    assert diagnostic[0].code == "missing_declaration"
    assert diagnostic[0].observed == {"model_family_status": "experimental"}


@pytest.mark.parametrize(
    ("model_family", "modality"),
    [
        (capabilities.Family.JEPA.value, capabilities.Modality.IMAGE),
        (
            capabilities.Family.TRANSFORMER.value,
            capabilities.Modality.TEXT,
        ),
        (
            capabilities.Family.MULTIMODAL_TRANSFORMER.value,
            capabilities.Modality.GRAPH,
        ),
    ],
)
def test_unsupported_family_modality_reports_structured_diagnostic(
    model_family, modality
):
    """Known families reject undeclared modalities before training."""
    registry = capabilities.Registry.seed()
    key = make_key(family=model_family, modality=modality)

    diagnostics = registry.diagnose(key)

    modality_diagnostic = next(
        item
        for item in diagnostics
        if item.code == "model_family_modality_unsupported"
    )
    assert modality_diagnostic.observed == modality.value
    assert modality.value not in modality_diagnostic.expected
    with pytest.raises(capabilities.UnsupportedError) as failure:
        registry.require(key)
    assert {item["code"] for item in failure.value.details["diagnostics"]} >= {
        "missing_declaration",
        "model_family_modality_unsupported",
    }


def test_experimental_family_requires_explicit_status_opt_in():
    """A catalog entry never becomes training support by being known.

    Also enforces website/src/pages/docs/modeling.astro:13.
    """
    registry = capabilities.Registry.seed()
    key = make_key(
        family=capabilities.Family.TRANSFORMER.value,
        modality=capabilities.Modality.TABULAR,
    )

    diagnostics = registry.diagnose(key)

    assert {item.code for item in diagnostics} == {
        "missing_declaration",
        "model_family_status_not_allowed",
    }
    with pytest.raises(capabilities.UnsupportedError):
        registry.require(key)


def test_environment_checks_collect_missing_extras_and_resources():
    """A declared capability fails closed with all observed environment gaps."""
    record = make_capability()
    registry = capabilities.Registry((record,))
    environment = capabilities.Environment(
        extras=frozenset(),
        cores=1,
        memory=1.0,
        disk=0.0,
    )

    diagnostics = registry.diagnose(record.key, environment=environment)

    assert {item.code for item in diagnostics} == {
        "missing_extra",
        "insufficient_cpu_cores",
        "insufficient_memory_gb",
        "insufficient_disk_gb",
    }
    with pytest.raises(capabilities.UnsupportedError) as failure:
        registry.require(record.key, environment=environment)
    assert [item["code"] for item in failure.value.details["diagnostics"]] == [
        item.code for item in diagnostics
    ]


def test_required_resources_reject_unknown_runtime_facts():
    """Required GPU and network facts cannot be inferred optimistically."""
    record = make_capability()
    gpu_key = make_key(family="vision_transformer")
    gpu_record = dataclasses.replace(
        record,
        key=gpu_key,
        resources=capabilities.Resources(
            memory=4.0,
            required=True,
            network=True,
        ),
    )
    registry = capabilities.Registry((gpu_record,))
    environment = capabilities.Environment(
        extras=frozenset({"core"}),
        cores=2,
        memory=4.0,
        disk=1.0,
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
        resources=capabilities.Resources(
            memory=4.0,
            rows=10,
        ),
    )
    registry = capabilities.Registry((record,))

    assert registry.diagnose(record.key) == (
        capabilities.Reason(
            code="unknown_row_count",
            message="row count is required for this capability limit",
            expected=10,
        ),
    )
    with pytest.raises(capabilities.UnsupportedError):
        registry.require(record.key, row_count=11)
