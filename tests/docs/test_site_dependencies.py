"""Dependency statements of the site that no other test enforced."""

import pathlib
import tomllib

import pytest

from scripts import check_dependency_licenses

PROJECT = pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_extras_are_audited_and_a_copyleft_licence_fails():
    """Extras join the audit only when enabled; copyleft fails the audit.

    Also enforces website/src/pages/docs/dependencies.astro:41.
    """
    project = tomllib.loads(PROJECT.read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]["boosting"]
    names = {item.split(">")[0].split("=")[0].lower() for item in extras}
    base = check_dependency_licenses.dependency_names(PROJECT, False)
    enabled = check_dependency_licenses.dependency_names(PROJECT, True)
    assert not names & base
    assert names <= enabled
    for licence in ("GPL-3.0", "LGPL-2.1", "AGPL-3.0", "SSPL", "BUSL-1.1"):
        with pytest.raises(SystemExit, match="forbidden license"):
            check_dependency_licenses.validate_inventory(
                [{"name": "extra", "license": licence}], []
            )
    check_dependency_licenses.validate_inventory(
        [{"name": "extra", "license": "MIT"}], []
    )
