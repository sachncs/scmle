"""Tests for modality lifecycle support contracts."""

import dataclasses
import json
from typing import cast

import pytest

from scmle.contracts import capabilities
from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.data.modalities import curation


def make_evidence(
    stage: modalities.Phase,
    links: tuple[str, ...] = ("tests/lifecycle/test_stage.py",),
) -> modalities.Proof:
    return modalities.Proof(
        stage=stage,
        evidence_links=links,
        notes=(f"verified {stage.value}",),
    )


def make_declaration(
    status: modalities.Status = modalities.Status.SUPPORTED,
    stage_evidence: tuple[modalities.Proof, ...] | None = None,
    modality: modalities.ModalityIdentity = modalities.Modality.TABULAR,
) -> modalities.Declaration:
    return modalities.Declaration(
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
        resources=capabilities.Resources(
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
    assert modalities.Phase.RECOVERY_AND_RESOURCE_TESTS.value == (
        "recovery_and_resource_tests"
    )


def test_extension_modality_has_stable_namespaced_identity():
    """Caller-defined families serialize without colliding with built-ins.

    Also enforces website/src/pages/docs/extensions.astro:31.
    """
    identity = modalities.Bespoke("acme/point-cloud")

    assert identity.value == "extension:acme/point-cloud"
    assert modalities.restore(identity.value) == identity
    with pytest.raises(ValueError, match="lowercase namespaced"):
        modalities.Bespoke("Acme point cloud")
    with pytest.raises(ValueError, match="conflicts"):
        modalities.Bespoke("image")


def test_extension_declarations_are_never_promoted_to_supported():
    """Complete extension evidence still requires an explicit extension gate."""
    identity = modalities.Bespoke("acme/point-cloud")
    declaration = make_declaration(
        modality=identity,
        status=modalities.Status.EXTENSION_ONLY,
    )
    registry = modalities.Registry((declaration,))

    assert (
        registry.require(identity, {modalities.Status.EXTENSION_ONLY})
        is declaration
    )
    assert declaration.to_dict()["modality"] == "extension:acme/point-cloud"
    with pytest.raises(ValueError, match="extension-only status"):
        make_declaration(
            modality=identity,
            status=modalities.Status.SUPPORTED,
        )


def test_complete_declaration_serializes_evidence_in_required_stage_order():
    declaration = make_declaration()

    assert declaration.complete
    assert declaration.missing_stages == ()
    payload = declaration.to_dict()
    stage_records = cast("list[dict[str, object]]", payload["evidence"])
    resources = cast("dict[str, object]", payload["resources"])

    assert payload["status"] == "supported"
    assert payload["complete"] is True
    assert [item["stage"] for item in stage_records] == [
        stage.value for stage in modalities.REQUIRED_LIFECYCLE_STAGES
    ]
    assert resources["memory"] == 4.0
    assert stage_records[0]["evidence"] == {}


def test_stage_evidence_requires_json_native_structured_output():
    """Stage payloads remain machine-readable and reject non-JSON values."""
    record = modalities.Proof(
        stage=modalities.Phase.INGESTION,
        evidence={"sample_count": 2, "groups": ["a", "b"]},
    )

    assert record.to_dict()["evidence"] == {
        "groups": ["a", "b"],
        "sample_count": 2,
    }
    with pytest.raises(ValueError, match="JSON-native"):
        modalities.Proof(
            stage=modalities.Phase.INGESTION,
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

    card = curation.Datasheet.from_report(
        "image", report, ("bounded representation",)
    )

    payload = card.to_dict()
    assert payload["curation"] == report
    assert payload["privacy"]["status"] == "not_assessed"
    assert payload["provenance"]["status"] == (
        "adapter_source_metadata_not_declared"
    )

    with pytest.raises(ValueError, match="does not match dataset_hash"):
        curation.Datasheet(
            modality="image",
            dataset_hash="different",
            sample_count=3,
            curation=report,
            intended_use="test",
            limitations=("bounded",),
            provenance={"status": "declared"},
            privacy={"status": "assessed"},
        )


def test_modality_curation_card_can_describe_an_empty_snapshot():
    """Empty raw inputs are evidence, not an invitation to invent samples."""
    report = {
        "dataset_hash": "c" * 64,
        "sample_count": 0,
        "release_ready": False,
        "quality_findings": [],
        "risk_findings": [],
    }

    card = curation.Datasheet.from_report(
        "extension:acme/empty-source", report, ("empty snapshot",)
    )

    assert card.sample_count == 0
    assert card.manifest is not None
    assert card.manifest.samples == 0


def test_modality_manifest_round_trip_binds_governance_to_content():
    """A modality manifest preserves source and privacy evidence exactly."""
    manifest = data.Passport(
        identifier="image-dataset",
        snapshot="snapshot-1",
        modality="image",
        task="binary_classification",
        samples=3,
        digests={"content": "b" * 64},
        source=data.Origin(
            owner="vision-team",
            acquisition="licensed object export",
            license="internal-use",
        ),
        lineage=data.Ancestry(
            snapshot="source-1",
            transformations=("resize-v1",),
        ),
        privacy=data.Confidentiality(
            classification="confidential",
            identifying=True,
            consent="contract",
            retention="30d",
            deletion="delete-on-request",
        ),
    )

    restored = data.Passport.from_dict(manifest.to_dict())

    assert restored == manifest
    assert restored.privacy.identifying is True


@pytest.mark.parametrize("samples", [True, 1.5])
def test_modality_manifest_rejects_non_integer_sample_counts(samples):
    """Lifecycle counts must remain exact integers in persisted evidence."""
    with pytest.raises(TypeError, match="samples must be an integer"):
        data.Passport(
            identifier="image-dataset",
            snapshot="snapshot-1",
            modality="image",
            task="binary_classification",
            samples=samples,
            digests={"content": "b" * 64},
        )


def test_declaration_is_immutable():
    declaration = make_declaration()

    with pytest.raises(dataclasses.FrozenInstanceError):
        declaration.status = (  # type: ignore[misc]
            modalities.Status.EXPERIMENTAL
        )


def test_supported_declaration_requires_every_stage_with_evidence():
    evidence = tuple(
        make_evidence(stage)
        for stage in modalities.REQUIRED_LIFECYCLE_STAGES
        if stage is not modalities.Phase.MONITORING
    )

    with pytest.raises(
        ValueError,
        match="supported modalities require evidenced lifecycle stages",
    ):
        make_declaration(stage_evidence=evidence)


def test_empty_stage_evidence_is_incomplete_and_fails_closed():
    evidence = tuple(
        make_evidence(stage)
        if stage is not modalities.Phase.INFERENCE
        else make_evidence(stage, links=())
        for stage in modalities.REQUIRED_LIFECYCLE_STAGES
    )
    declaration = make_declaration(
        status=modalities.Status.EXPERIMENTAL,
        stage_evidence=evidence,
    )
    registry = modalities.Registry((declaration,))

    assert declaration.missing_stages == (modalities.Phase.INFERENCE,)
    with pytest.raises(
        modalities.UnsupportedError,
        match="declaration is incomplete",
    ) as failure:
        registry.require(
            modalities.Modality.TABULAR,
            {modalities.Status.EXPERIMENTAL},
        )
    assert failure.value.details["missing_stages"] == ["inference"]


def test_registry_fails_closed_for_undeclared_modality():
    registry = modalities.Registry((make_declaration(),))

    with pytest.raises(
        modalities.UnsupportedError,
        match="no declaration exists",
    ) as failure:
        registry.require(modalities.Modality.IMAGE)

    assert failure.value.category == "modality"
    assert failure.value.details["status"] == "unsupported"
    assert failure.value.details["modality"] == "image"
    assert failure.value.details["registered_modalities"] == ["tabular"]


def test_status_requires_explicit_opt_in_after_completeness_is_proven():
    declaration = make_declaration(
        status=modalities.Status.EXPERIMENTAL,
    )
    registry = modalities.Registry((declaration,))

    with pytest.raises(
        modalities.UnsupportedError,
        match="registered as experimental",
    ):
        registry.require(modalities.Modality.TABULAR)
    assert (
        registry.require(
            modalities.Modality.TABULAR,
            {modalities.Status.EXPERIMENTAL},
        )
        is declaration
    )


def test_registry_serialization_is_deterministic_and_duplicate_safe():
    tabular = make_declaration()
    image = make_declaration(
        modality=modalities.Modality.IMAGE,
        status=modalities.Status.EXTENSION_ONLY,
    )
    first = modalities.Registry((image, tabular))
    second = modalities.Registry((tabular, image))

    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == first.to_dict()
    assert [item.modality for item in first.catalog()] == [
        modalities.Modality.IMAGE,
        modalities.Modality.TABULAR,
    ]

    with pytest.raises(errors.ConfigError, match="already registered"):
        first.register(tabular)


def test_declaration_rejects_duplicate_stages_and_missing_failure_modes():
    duplicate = make_evidence(modalities.Phase.INGESTION)
    with pytest.raises(ValueError, match="duplicate stages"):
        make_declaration(stage_evidence=(duplicate, duplicate))

    with pytest.raises(ValueError, match="failure_modes"):
        modalities.Declaration(
            modality=modalities.Modality.TEXT,
            status=modalities.Status.EXPERIMENTAL,
            stage_evidence=(),
        )
