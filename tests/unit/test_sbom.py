"""Tests for deterministic software inventory generation."""

import pytest

from scmle.governance import sbom


def test_sbom_has_deterministic_component_identity():
    packages = {"Pandas": "2.2.0", "scikit_learn": "1.6.0"}
    first = sbom.build(
        packages,
        run_id="run-1",
        software_version="0.1.0b1",
        observed_at="2026-10-20T00:00:00+00:00",
    )
    second = sbom.build(
        dict(reversed(list(packages.items()))),
        run_id="run-1",
        software_version="0.1.0b1",
        observed_at="2026-10-20T00:00:00+00:00",
    )

    assert first == second
    assert [item["name"] for item in first["components"]] == [
        "pandas",
        "scikit-learn",
    ]
    assert first["components"][1]["purl"] == "pkg:pypi/scikit-learn@1.6.0"


def test_sbom_rejects_empty_inventory_and_invalid_names():
    with pytest.raises(ValueError, match="must not be empty"):
        sbom.build({}, run_id="run-1", software_version="0.1.0b1")
    with pytest.raises(ValueError, match="invalid distribution name"):
        sbom.build(
            {"bad name": "1.0"}, run_id="run-1", software_version="0.1.0b1"
        )
