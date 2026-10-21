"""Tests for the public non-tabular lifecycle project surface."""

from pathlib import Path
from typing import cast

import pytest

import scml
from scml.contracts import modalities
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
    assert Path(result.path).is_file()
