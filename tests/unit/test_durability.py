"""Tests for filesystem durability boundaries."""

import pathlib

import pytest

from scmle import durability


def test_sync_directory_flushes_a_real_directory(tmp_path: pathlib.Path):
    """A directory handle can flush an atomic replacement on Unix."""
    durability.sync_directory(tmp_path)


def test_write_atomically_replaces_complete_content(tmp_path: pathlib.Path):
    """Atomic writes expose only the complete replacement to readers."""
    destination = tmp_path / "evidence.json"
    destination.write_bytes(b"old")

    durability.write_atomically(destination, b"new evidence")

    assert destination.read_bytes() == b"new evidence"
    assert tuple(tmp_path.glob(".*.tmp")) == ()


def test_write_atomically_preserves_previous_content_on_write_failure(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """A failed replacement does not destroy an already committed file."""
    destination = tmp_path / "evidence.json"
    destination.write_bytes(b"previous evidence")

    def fail_replace(source: str, target: pathlib.Path) -> None:
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(durability.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replacement failure"):
        durability.write_atomically(destination, b"new evidence")

    assert destination.read_bytes() == b"previous evidence"
    assert tuple(tmp_path.glob(".*.tmp")) == ()
