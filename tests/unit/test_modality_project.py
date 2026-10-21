"""Tests for the public non-tabular lifecycle project surface."""

from pathlib import Path
from typing import cast

import pytest

import scml
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
