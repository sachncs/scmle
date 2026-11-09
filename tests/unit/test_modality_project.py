"""Tests for the public non-tabular lifecycle project surface."""

import hashlib
import json
import pathlib
from pathlib import Path
from typing import cast

import pytest

import scmle.api.project
from scmle.contracts import modalities
from scmle.data.modalities import curation
from scmle.data.modalities import document
from scmle.models import modality_lifecycle
import scmle.models.modality_lifecycle


def text_adapter() -> modality_lifecycle.TextPreparer:
    """Build a small labelled text corpus for the public API test."""
    samples = tuple(
        document.Sample(
            document_id=f"text-{index}",
            content=("alpha signal " if index % 2 else "beta signal ")
            * (index + 2),
            label="alpha" if index % 2 else "beta",
        )
        for index in range(12)
    )
    return modality_lifecycle.TextPreparer(document.Dataset(samples))


class ExtensionTextAdapter(modality_lifecycle.TextPreparer):
    """Test caller-owned adapter for a dataset family outside core enums."""

    modality = modalities.ExtensionModality("acme/point-cloud")


def test_modality_project_runs_and_persists_all_stages(tmp_path):
    """The public project surface preserves the complete lifecycle contract."""
    project = scmle.api.project.ModalityProject(
        text_adapter(), "test-signing-key", output_dir=tmp_path / "run"
    )

    report = project.run()

    assert report.modality == "text"
    assert len(report.stage_evidence) == 15
    assert project.declaration().status.value == "experimental"
    assert (tmp_path / "run" / "model" / "manifest.json").is_file()
    assert (tmp_path / "run" / "modality_lifecycle.json").is_file()
    predictor = scmle.models.modality_lifecycle.Predictor.from_package(
        project.adapter,
        report.package.path,
        "test-signing-key",
    )
    assert predictor.predict(project.adapter.samples[:2]).shape == (2,)


def test_modality_project_rejects_invalid_configuration(tmp_path):
    """Configuration errors are explicit before a lifecycle can start."""
    with pytest.raises(TypeError, match="Preparer"):
        scmle.api.project.ModalityProject(
            cast("modality_lifecycle.Preparer", object()),
            "key",
            output_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="signing_key"):
        scmle.api.project.ModalityProject(
            text_adapter(), "", output_dir=tmp_path
        )

    project = scmle.api.project.ModalityProject(
        text_adapter(), "key", output_dir=tmp_path / "run"
    )
    with pytest.raises(RuntimeError, match="run must complete"):
        project.declaration()


def test_modality_project_can_cancel_before_publication(tmp_path):
    """Cancellation prevents a modality package and report from appearing."""
    project = scmle.api.project.ModalityProject(
        text_adapter(), "key", output_dir=tmp_path / "cancelled"
    )

    project.cancel()

    with pytest.raises(
        modality_lifecycle.LifecycleCancelledError,
        match="cancelled at a stage boundary",
    ):
        project.run()
    assert project.cancellation.is_cancelled() is True
    assert not (tmp_path / "cancelled" / "model").exists()
    assert not (tmp_path / "cancelled" / "modality_lifecycle.json").exists()


def test_modality_project_can_require_independent_explanation_ground_truth(
    tmp_path,
):
    """Production-configured runs fail before publishing an unverified package."""
    project = scmle.api.project.ModalityProject(
        text_adapter(),
        "key",
        output_dir=tmp_path / "run",
        require_independent_ground_truth=True,
    )

    with pytest.raises(
        ValueError,
        match="independent explanation ground-truth suite is required",
    ):
        project.run()

    assert not (tmp_path / "run" / "modality_lifecycle.json").exists()
    assert not (tmp_path / "run" / "model" / "manifest.json").exists()


def test_modality_project_rejects_file_output(tmp_path):
    """A file cannot be used as the lifecycle output directory."""
    output = Path(tmp_path) / "output"
    output.write_text("occupied", encoding="utf-8")
    project = scmle.api.project.ModalityProject(
        text_adapter(), "key", output_dir=output
    )

    with pytest.raises(ValueError, match="output_dir"):
        project.run()


def test_modality_project_preserves_extension_identity_and_status(tmp_path):
    """Arbitrary dataset families use the lifecycle without support inflation."""
    project = scmle.api.project.ModalityProject(
        ExtensionTextAdapter(text_adapter().dataset),
        "test-signing-key",
        output_dir=tmp_path / "extension-run",
    )

    report = project.run()
    declaration = project.declaration()

    assert report.modality == "extension:acme/point-cloud"
    assert report.documentation["status"] == "extension-only"
    assert declaration.modality == ExtensionTextAdapter.modality
    assert declaration.status is modalities.Status.EXTENSION_ONLY
    assert declaration.complete
    assert pathlib.PurePath(report.package.path).name == "model"


def test_modality_curation_project_accepts_unlabeled_data_without_modelling(
    tmp_path,
):
    """Curation remains available when a classification baseline is invalid."""
    samples = tuple(
        document.Sample(
            document_id=f"unlabelled-{index}",
            content=f"raw record {index}",
        )
        for index in range(3)
    )
    adapter = modality_lifecycle.TextPreparer(document.Dataset(samples))

    result = scmle.api.project.Curator(
        adapter,
        "test-signing-key",
        output_dir=tmp_path / "curation",
    ).run()

    assert result.modality == "text"
    assert result.sample_count == 3
    assert result.curation["sample_count"] == 3
    assert result.dataset_card["privacy"]["status"] == "not_assessed"
    assert result.dataset_card["manifest"]["task"] == "not_declared"
    assert Path(result.path).is_file()

    restored = scmle.models.modality_lifecycle.Curation.read_verified(
        result.path, "test-signing-key"
    )
    assert restored == result
    assert json.loads(Path(result.path).read_text(encoding="utf-8")) == (
        result.to_dict()
    )
    tampered = json.loads(Path(result.path).read_text(encoding="utf-8"))
    tampered["dataset_card"]["intended_use"] = "tampered"
    Path(result.path).write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=r"signature does not verify|content_digest does not match evidence",
    ):
        scmle.models.modality_lifecycle.Curation.read_verified(
            result.path, "test-signing-key"
        )
    with pytest.raises(
        ValueError,
        match=r"signature does not verify|content_digest does not match evidence",
    ):
        scmle.api.project.Curator(
            adapter,
            "test-signing-key",
            output_dir=tmp_path / "curation",
        ).run()


def test_modality_curation_project_reuses_identical_artifact(tmp_path):
    """A verified identical publication is idempotent and not replaced."""
    project = scmle.api.project.Curator(
        text_adapter(),
        "test-signing-key",
        output_dir=tmp_path / "curation",
    )

    first = project.run()
    original_bytes = Path(first.path).read_bytes()

    second = project.run()

    assert second == first
    assert second.to_dict() == first.to_dict()
    assert Path(first.path).read_bytes() == original_bytes


def test_modality_curation_project_rejects_identity_conflict_without_overwrite(
    tmp_path,
):
    """A valid artifact for another dataset remains untouched and fails."""
    output_dir = tmp_path / "curation"
    first = scmle.api.project.Curator(
        text_adapter(), "test-signing-key", output_dir=output_dir
    ).run()
    original_bytes = Path(first.path).read_bytes()
    samples = list(text_adapter().dataset.samples)
    samples[0] = document.Sample(
        document_id=samples[0].document_id,
        content="different source content",
        label=samples[0].label,
    )
    changed_adapter = modality_lifecycle.TextPreparer(
        document.Dataset(tuple(samples))
    )

    with pytest.raises(
        ValueError,
        match="dataset_hash does not match current adapter",
    ):
        scmle.api.project.Curator(
            changed_adapter, "test-signing-key", output_dir=output_dir
        ).run()

    assert Path(first.path).read_bytes() == original_bytes


def test_modality_curation_project_rejects_unsigned_artifact_tampering(
    tmp_path,
):
    """A verified signature cannot authorize unknown record content."""
    project = scmle.api.project.Curator(
        text_adapter(),
        "test-signing-key",
        output_dir=tmp_path / "curation",
    )
    result = project.run()
    artifact = Path(result.path)
    record = json.loads(artifact.read_text(encoding="utf-8"))
    record["unexpected"] = "tampered"
    artifact.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(
        ValueError, match="canonical content does not match current adapter"
    ):
        project.run()


def test_extension_curation_adapter_accepts_arbitrary_records_without_model_hooks(
    tmp_path,
):
    """Curation-only extensions do not implement unused modelling methods."""
    extension = modalities.ExtensionModality("acme/point-cloud")
    records = ({"x": 1, "y": 2}, {"x": 3, "y": 4})
    dataset_hash = hashlib.sha256(b"point-cloud-snapshot-1").hexdigest()
    report = curation.Curation(
        modality=extension.value,
        dataset_hash=dataset_hash,
        sample_count=len(records),
        profile={"record_type": "point", "dimensions": 2},
    )
    adapter = scmle.models.modality_lifecycle.ExtensionCurator(
        extension=extension,
        sample_values=records,
        dataset_hash_value=dataset_hash,
        curation_callback=lambda: report,
        limitations=("The caller owns point-cloud decoding and validation.",),
    )

    assert not isinstance(adapter, modality_lifecycle.Preparer)
    result = scmle.api.project.Curator(
        adapter,
        "test-signing-key",
        output_dir=tmp_path / "point-cloud-curation",
    ).run()

    assert result.modality == "extension:acme/point-cloud"
    assert result.sample_count == 2
    assert result.dataset_card["manifest"]["task"] == "not_declared"


def test_extension_curation_adapter_rejects_detached_report():
    """A curation callback cannot publish evidence for another snapshot."""
    extension = modalities.ExtensionModality("acme/point-cloud")
    report = curation.Curation(
        modality=extension.value,
        dataset_hash="different-snapshot",
        sample_count=1,
        profile={},
    )
    adapter = scmle.models.modality_lifecycle.ExtensionCurator(
        extension=extension,
        sample_values=({"x": 1},),
        dataset_hash_value=hashlib.sha256(b"declared-snapshot").hexdigest(),
        curation_callback=lambda: report,
        limitations=("The caller owns validation.",),
    )

    with pytest.raises(ValueError, match="hash does not match"):
        adapter.curate()


def test_extension_curation_adapter_accepts_empty_raw_snapshot(tmp_path):
    """Empty snapshots remain valid curation evidence without modelling."""
    extension = modalities.ExtensionModality("acme/empty-stream")
    dataset_hash = hashlib.sha256(b"empty-snapshot").hexdigest()
    adapter = scmle.models.modality_lifecycle.ExtensionCurator(
        extension=extension,
        sample_values=(),
        dataset_hash_value=dataset_hash,
        curation_callback=lambda: curation.Curation(
            modality=extension.value,
            dataset_hash=dataset_hash,
            sample_count=0,
            profile={"record_type": "event", "empty": True},
        ),
        limitations=("The caller owns source availability checks.",),
    )

    result = scmle.api.project.Curator(
        adapter,
        "test-signing-key",
        output_dir=tmp_path / "empty-curation",
    ).run()

    assert result.sample_count == 0
    assert result.curation["profile"]["empty"] is True
