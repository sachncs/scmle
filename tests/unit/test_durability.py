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


class FlakyReplace:
    """`os.replace` that fails with PermissionError a set number of times."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0
        self.real = durability.os.replace

    def __call__(self, source: str, target: pathlib.Path) -> None:
        self.calls += 1
        if self.calls <= self.failures:
            raise PermissionError(13, "Access is denied")
        self.real(source, target)


def test_windows_retries_a_transiently_denied_replacement(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Windows denies a replace while a reader still has the file open."""
    destination = tmp_path / "state.json"
    destination.write_bytes(b"old")
    flaky = FlakyReplace(failures=3)
    monkeypatch.setattr(durability, "WINDOWS", True)
    monkeypatch.setattr(durability.os, "replace", flaky)
    monkeypatch.setattr(durability.time, "sleep", lambda seconds: None)

    durability.write_atomically(destination, b"new")

    assert destination.read_bytes() == b"new"
    assert flaky.calls == 4
    assert tuple(tmp_path.glob(".*.tmp")) == ()


def test_windows_gives_up_after_a_bounded_number_of_attempts(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """A permanent denial still surfaces, and keeps the previous content."""
    destination = tmp_path / "state.json"
    destination.write_bytes(b"old")
    flaky = FlakyReplace(failures=10**6)
    monkeypatch.setattr(durability, "WINDOWS", True)
    monkeypatch.setattr(durability.os, "replace", flaky)
    monkeypatch.setattr(durability.time, "sleep", lambda seconds: None)

    with pytest.raises(PermissionError):
        durability.write_atomically(destination, b"new")

    assert flaky.calls == durability.REPLACE_ATTEMPTS
    assert destination.read_bytes() == b"old"
    assert tuple(tmp_path.glob(".*.tmp")) == ()


def test_other_platforms_do_not_retry_a_denied_replacement(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Off Windows a PermissionError is a real error, not a race."""
    destination = tmp_path / "state.json"
    flaky = FlakyReplace(failures=1)
    monkeypatch.setattr(durability, "WINDOWS", False)
    monkeypatch.setattr(durability.os, "replace", flaky)

    with pytest.raises(PermissionError):
        durability.write_atomically(destination, b"new")

    assert flaky.calls == 1


def test_replacement_waits_for_a_reader_holding_the_path_lock(
    tmp_path: pathlib.Path,
):
    """A writer takes the same per-path lock readers hold while reading."""
    import threading

    destination = tmp_path / "state.json"
    destination.write_bytes(b"old")
    finished = threading.Event()

    def write() -> None:
        durability.write_atomically(destination, b"new")
        finished.set()

    with durability.path_lock(destination):
        writer = threading.Thread(target=write)
        writer.start()
        assert not finished.wait(timeout=0.3)
        assert destination.read_bytes() == b"old"
    writer.join(timeout=5)

    assert finished.is_set()
    assert destination.read_bytes() == b"new"
