"""Tests for runtime dependency license evidence collection."""

from email.message import Message
from pathlib import Path
from types import SimpleNamespace

from scripts import check_dependency_licenses


def distribution_with_license_file(
    tmp_path: Path, content: str, file_name: str = "LICENSE"
) -> SimpleNamespace:
    """Builds the smallest distribution-shaped object used by the checker."""
    license_path = tmp_path / file_name
    license_path.write_text(content, encoding="utf-8")
    return SimpleNamespace(
        metadata=Message(),
        files=[Path(file_name)],
        locate_file=lambda path: tmp_path / path,
    )


def test_license_evidence_reads_standard_license_file(tmp_path: Path) -> None:
    """Recognizes a license when wheel metadata omits its license fields."""
    distribution = distribution_with_license_file(
        tmp_path,
        "MIT License\n\nPermission is hereby granted, free of charge.",
    )

    assert check_dependency_licenses.license_evidence(distribution) == (
        "MIT",
        "license-file",
    )


def test_license_evidence_rejects_unrecognized_license_file(
    tmp_path: Path,
) -> None:
    """Leaves an unrecognized license as missing evidence."""
    distribution = distribution_with_license_file(
        tmp_path, "Copyright only; terms are supplied elsewhere."
    )

    assert check_dependency_licenses.license_evidence(distribution) == (
        "UNKNOWN",
        "missing-metadata",
    )
