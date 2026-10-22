"""Tests for the public non-tabular lifecycle project surface."""

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

import scml
from scml.contracts import modalities
from scml.data.modalities import curation
from scml.data.modalities import document
from scml.models import modality_lifecycle


def text_adapter() -> modality_lifecycle.TextAdapter:
    """Build a small labelled text corpus for the public API test."""
    samples = tuple(
        document.DocumentSample(
            document_id=f"text-{index}",
            content=("alpha signal " if index % 2 else "beta signal ")
            * (index + 2),
            label="alpha" if index % 2 else "beta",
        )
        for index in range(12)
    )
    return modality_lifecycle.TextAdapter(document.DocumentDataset(samples))


class ExtensionTextAdapter(modality_lifecycle.TextAdapter):
    """Test caller-owned adapter for a dataset family outside core enums."""

    modality = modalities.ExtensionModality("acme/point-cloud")


def test_modality_project_runs_and_persists_all_stages(tmp_path):
    """The public project surface preserves the complete lifecycle contract."""
    project = scml.ModalityProject(
        text_adapter(), "test-signing-key", output_dir=tmp_path / "run"
    )

    report = project.run()

    assert report.modality == "text"
    assert len(report.stage_evidence) == 15
    assert project.declaration().status.value == "experimental"
    assert (tmp_path / "run" / "model" / "manifest.json").is_file()
    assert (tmp_path / "run" / "modality_lifecycle.json").is_file()
    predictor = scml.ModalityPredictor.from_package(
        project.adapter,
        report.package.path,
        "test-signing-key",
    )
    assert predictor.predict(project.adapter.samples[:2]).shape == (2,)


def test_modality_project_rejects_invalid_configuration(tmp_path):
    """Configuration errors are explicit before a lifecycle can start."""
    with pytest.raises(TypeError, match="ModalityAdapter"):
        scml.ModalityProject(
            cast(modality_lifecycle.ModalityAdapter, object()),
            "key",
            output_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="signing_key"):
        scml.ModalityProject(text_adapter(), "", output_dir=tmp_path)

    project = scml.ModalityProject(
        text_adapter(), "key", output_dir=tmp_path / "run"
    )
    with pytest.raises(RuntimeError, match="run must complete"):
        project.declaration()


def test_modality_project_can_cancel_before_publication(tmp_path):
    """Cancellation prevents a modality package and report from appearing."""
    project = scml.ModalityProject(
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
    project = scml.ModalityProject(
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
    project = scml.ModalityProject(text_adapter(), "key", output_dir=output)

    with pytest.raises(ValueError, match="output_dir"):
        project.run()


def test_modality_project_preserves_extension_identity_and_status(tmp_path):
    """Arbitrary dataset families use the lifecycle without support inflation."""
    project = scml.ModalityProject(
        ExtensionTextAdapter(text_adapter().dataset),
        "test-signing-key",
        output_dir=tmp_path / "extension-run",
    )

    report = project.run()
    declaration = project.declaration()

    assert report.modality == "extension:acme/point-cloud"
    assert report.documentation["status"] == "extension-only"
    assert declaration.modality == ExtensionTextAdapter.modality
    assert declaration.status is modalities.ModalityStatus.EXTENSION_ONLY
    assert declaration.complete
    assert report.package.path.endswith("/model")


def test_modality_curation_project_accepts_unlabeled_data_without_modelling(
    tmp_path,
):
    """Curation remains available when a classification baseline is invalid."""
    samples = tuple(
        document.DocumentSample(
            document_id=f"unlabelled-{index}",
            content=f"raw record {index}",
        )
        for index in range(3)
    )
    adapter = modality_lifecycle.TextAdapter(document.DocumentDataset(samples))

    result = scml.ModalityCurationProject(
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

    restored = scml.ModalityCurationResult.read_verified(
        result.path, "test-signing-key"
    )
    assert restored == result
    assert json.loads(Path(result.path).read_text(encoding="utf-8")) == (
        result.to_dict()
    )

    tampered = json.loads(Path(result.path).read_text(encoding="utf-8"))
    tampered["dataset_card"]["intended_use"] = "tampered"
    Path(result.path).write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="signature does not verify"):
        scml.ModalityCurationResult.read_verified(
            result.path, "test-signing-key"
        )


def test_extension_curation_adapter_accepts_arbitrary_records_without_model_hooks(
    tmp_path,
):
    """Curation-only extensions do not implement unused modelling methods."""
    extension = modalities.ExtensionModality("acme/point-cloud")
    records = ({"x": 1, "y": 2}, {"x": 3, "y": 4})
    dataset_hash = hashlib.sha256(b"point-cloud-snapshot-1").hexdigest()
    report = curation.ModalityCurationReport(
        modality=extension.value,
        dataset_hash=dataset_hash,
        sample_count=len(records),
        profile={"record_type": "point", "dimensions": 2},
    )
    adapter = scml.ExtensionCurationAdapter(
        extension=extension,
        sample_values=records,
        dataset_hash_value=dataset_hash,
        curation_callback=lambda: report,
        limitations=("The caller owns point-cloud decoding and validation.",),
    )

    assert not isinstance(adapter, modality_lifecycle.ModalityAdapter)
    result = scml.ModalityCurationProject(
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
    report = curation.ModalityCurationReport(
        modality=extension.value,
        dataset_hash="different-snapshot",
        sample_count=1,
        profile={},
    )
    adapter = scml.ExtensionCurationAdapter(
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
    adapter = scml.ExtensionCurationAdapter(
        extension=extension,
        sample_values=(),
        dataset_hash_value=dataset_hash,
        curation_callback=lambda: curation.ModalityCurationReport(
            modality=extension.value,
            dataset_hash=dataset_hash,
            sample_count=0,
            profile={"record_type": "event", "empty": True},
        ),
        limitations=("The caller owns source availability checks.",),
    )

    result = scml.ModalityCurationProject(
        adapter,
        "test-signing-key",
        output_dir=tmp_path / "empty-curation",
    ).run()

    assert result.sample_count == 0
    assert result.curation["profile"]["empty"] is True
