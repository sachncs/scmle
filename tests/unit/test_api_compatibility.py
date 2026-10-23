"""Tests for the versioned public API compatibility gate."""

import json
from pathlib import Path

import pytest

from scripts import check_api_compatibility

SNAPSHOT = Path(__file__).parents[1] / "fixtures" / "public_api.json"


def test_current_public_api_matches_versioned_snapshot():
    """The published top-level API must match its checked-in contract."""
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    assert (
        check_api_compatibility.differences(
            expected, check_api_compatibility.current_snapshot()
        )
        == []
    )


def test_compatibility_gate_reports_removed_and_changed_exports():
    """A removed or changed signature must fail with a useful diagnosis."""
    expected = check_api_compatibility.current_snapshot()
    changed = json.loads(json.dumps(expected))
    changed["exports"]["predict"]["signature_sha256"] = "changed"
    changed["exports"].pop("load_model")

    differences = check_api_compatibility.differences(expected, changed)

    assert "export removed: load_model" in differences
    assert "export contract changed: predict" in differences


@pytest.mark.parametrize("value", [{"b", "a"}, frozenset({"b", "a"})])
def test_set_defaults_are_canonicalized(value):
    """Set-like defaults cannot make the compatibility record nondeterministic."""
    assert check_api_compatibility.stable_value(value) == ["a", "b"]


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        ("pathlib._local.Path", "pathlib.Path"),
        ("pathlib._windows.Path | None", "pathlib.Path | None"),
    ],
)
def test_platform_private_pathlib_annotations_are_canonicalized(
    annotation, expected
):
    """Private pathlib module spellings must not change the public contract."""
    assert check_api_compatibility.stable_annotation(annotation) == expected
