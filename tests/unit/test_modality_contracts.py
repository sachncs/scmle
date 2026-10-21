"""Tests for modality lifecycle support contracts."""

import dataclasses
import json
from typing import cast

import pytest

from scml.contracts import capabilities
from scml.contracts import data
from scml.contracts import errors
from scml.contracts import modalities
from scml.data.modalities import curation


def make_evidence(
    stage: modalities.LifecycleStage,
    links: tuple[str, ...] = ("tests/lifecycle/test_stage.py",),
) -> modalities.StageEvidence:
    return modalities.StageEvidence(
        stage=stage,
        evidence_links=links,
        notes=(f"verified {stage.value}",),
    )


def make_declaration(
    status: modalities.ModalityStatus = modalities.ModalityStatus.SUPPORTED,
    stage_evidence: tuple[modalities.StageEvidence, ...] | None = None,
    modality: modalities.Modality = modalities.Modality.TABULAR,
) -> modalities.ModalityDeclaration:
    return modalities.ModalityDeclaration(
        modality=modality,
        status=status,
        stage_evidence=(
            tuple(
                make_evidence(stage)
                for stage in modalities.REQUIRED_LIFECYCLE_STAGES
            )
            if stage_evidence is None
            else stage_evidence
        ),
        required_extras=("core",),
        resources=capabilities.ResourceAssumptions(
            cpu_cores=2,
            memory_gb=4.0,
            disk_gb=1.0,
            notes=("bounded local execution",),
        ),
        failure_modes=("malformed input", "resource exhaustion"),
        limitations=("library-first support",),
    )


def test_enums_define_all_required_modalities_and_lifecycle_stages():
    assert [item.value for item in modalities.Modality] == [
        "tabular",
        "text",
        "document",
        "image",
        "audio",
        "video",
        "graph",
        "geospatial",
        "raster",
    ]
    assert len(modalities.REQUIRED_LIFECYCLE_STAGES) == 15
    assert modalities.LifecycleStage.RECOVERY_AND_RESOURCE_TESTS.value == (
        "recovery_and_resource_tests"
    )


def test_complete_declaration_serializes_evidence_in_required_stage_order():
    declaration = make_declaration()

    assert declaration.complete
    assert declaration.missing_stages == ()
    payload = declaration.to_dict()
    stage_records = cast(list[dict[str, object]], payload["stage_evidence"])
    resources = cast(dict[str, object], payload["resources"])

    assert payload["status"] == "supported"
    assert payload["complete"] is True
    assert [item["stage"] for item in stage_records] == [
        stage.value for stage in modalities.REQUIRED_LIFECYCLE_STAGES
    ]
    assert resources["memory_gb"] == 4.0
    assert stage_records[0]["evidence"] == {}


def test_stage_evidence_requires_json_native_structured_output():
    """Stage payloads remain machine-readable and reject non-JSON values."""
    record = modalities.StageEvidence(
        stage=modalities.LifecycleStage.INGESTION,
        evidence={"sample_count": 2, "groups": ["a", "b"]},
    )

    assert record.to_dict()["evidence"] == {
        "groups": ["a", "b"],
        "sample_count": 2,
    }
    with pytest.raises(ValueError, match="JSON-native"):
        modalities.StageEvidence(
            stage=modalities.LifecycleStage.INGESTION,
            evidence={"invalid": object()},
        )


def test_modality_dataset_card_keeps_curation_and_governance_gaps_explicit():
    """A card cannot claim provenance or privacy that the adapter lacks."""
    report = {
        "dataset_hash": "a" * 64,
        "sample_count": 3,
        "release_ready": True,
        "quality_findings": [],
        "risk_findings": [],
    }

    card = curation.ModalityDatasetCard.from_report(
        "image", report, ("bounded representation",)
    )

    payload = card.to_dict()
    assert payload["curation"] == report
    assert payload["privacy"]["status"] == "not_assessed"
    assert payload["provenance"]["status"] == (
        "adapter_source_metadata_not_declared"
    )

    with pytest.raises(ValueError, match="does not match dataset_hash"):
        curation.ModalityDatasetCard(
            modality="image",
            dataset_hash="different",
            sample_count=3,
            curation=report,
            intended_use="test",
            limitations=("bounded",),
            provenance={"status": "declared"},
            privacy={"status": "assessed"},
        )


def test_modality_manifest_round_trip_binds_governance_to_content():
    """A modality manifest preserves source and privacy evidence exactly."""
    manifest = data.ModalityManifest(
        dataset_id="image-dataset",
        snapshot_id="snapshot-1",
        modality="image",
        task="binary_classification",
        sample_count=3,
        content_hash="b" * 64,
        source=data.SourceMetadata(
            owner="vision-team",
            acquisition_method="licensed object export",
            license="internal-use",
        ),
        lineage=data.LineageMetadata(
            source_snapshot="source-1",
            transformations=("resize-v1",),
        ),
        privacy=data.PrivacyMetadata(
            classification="confidential",
            contains_pii=True,
            consent_basis="contract",
            retention_period="30d",
            deletion_policy="delete-on-request",
        ),
    )

    restored = data.ModalityManifest.from_dict(manifest.to_dict())

    assert restored == manifest
    assert restored.privacy.contains_pii is True


def test_declaration_is_immutable():
    declaration = make_declaration()

    with pytest.raises(dataclasses.FrozenInstanceError):
        declaration.status = (  # type: ignore[misc]
            modalities.ModalityStatus.EXPERIMENTAL
        )


def test_supported_declaration_requires_every_stage_with_evidence():
    evidence = tuple(
        make_evidence(stage)
        for stage in modalities.REQUIRED_LIFECYCLE_STAGES
        if stage is not modalities.LifecycleStage.MONITORING
    )

    with pytest.raises(
        ValueError,
        match="supported modalities require evidenced lifecycle stages",
    ):
        make_declaration(stage_evidence=evidence)


def test_empty_stage_evidence_is_incomplete_and_fails_closed():
    evidence = tuple(
        make_evidence(stage)
        if stage is not modalities.LifecycleStage.INFERENCE
        else make_evidence(stage, links=())
        for stage in modalities.REQUIRED_LIFECYCLE_STAGES
    )
    declaration = make_declaration(
        status=modalities.ModalityStatus.EXPERIMENTAL,
        stage_evidence=evidence,
    )
    registry = modalities.ModalityRegistry((declaration,))

    assert declaration.missing_stages == (modalities.LifecycleStage.INFERENCE,)
    with pytest.raises(
        modalities.UnsupportedModalityError,
        match="declaration is incomplete",
    ) as failure:
        registry.require(
            modalities.Modality.TABULAR,
            {modalities.ModalityStatus.EXPERIMENTAL},
        )
    assert failure.value.details["missing_stages"] == ["inference"]


def test_registry_fails_closed_for_undeclared_modality():
    registry = modalities.ModalityRegistry((make_declaration(),))

    with pytest.raises(
        modalities.UnsupportedModalityError,
        match="no declaration exists",
    ) as failure:
        registry.require(modalities.Modality.IMAGE)

    assert failure.value.category == "modality"
    assert failure.value.details["status"] == "unsupported"
    assert failure.value.details["modality"] == "image"
    assert failure.value.details["registered_modalities"] == ["tabular"]


def test_status_requires_explicit_opt_in_after_completeness_is_proven():
    declaration = make_declaration(
        status=modalities.ModalityStatus.EXPERIMENTAL,
    )
    registry = modalities.ModalityRegistry((declaration,))

    with pytest.raises(
        modalities.UnsupportedModalityError,
        match="registered as experimental",
    ):
        registry.require(modalities.Modality.TABULAR)
    assert (
        registry.require(
            modalities.Modality.TABULAR,
            {modalities.ModalityStatus.EXPERIMENTAL},
        )
        is declaration
    )


def test_registry_serialization_is_deterministic_and_duplicate_safe():
    tabular = make_declaration()
    image = make_declaration(
        modality=modalities.Modality.IMAGE,
        status=modalities.ModalityStatus.EXTENSION_ONLY,
    )
    first = modalities.ModalityRegistry((image, tabular))
    second = modalities.ModalityRegistry((tabular, image))

    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == first.to_dict()
    assert [item.modality for item in first.list_declarations()] == [
        modalities.Modality.IMAGE,
        modalities.Modality.TABULAR,
    ]

    with pytest.raises(errors.ConfigError, match="already registered"):
        first.register(tabular)


def test_declaration_rejects_duplicate_stages_and_missing_failure_modes():
    duplicate = make_evidence(modalities.LifecycleStage.INGESTION)
    with pytest.raises(ValueError, match="duplicate stages"):
        make_declaration(stage_evidence=(duplicate, duplicate))

    with pytest.raises(ValueError, match="failure_modes"):
        modalities.ModalityDeclaration(
            modality=modalities.Modality.TEXT,
            status=modalities.ModalityStatus.EXPERIMENTAL,
            stage_evidence=(),
        )
