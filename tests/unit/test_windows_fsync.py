"""`fsync` needs a writable descriptor on Windows (it fails with EBADF).

POSIX lets a read-only descriptor be flushed, so the bug only showed on the
Windows CI jobs. On POSIX a strict `fsync` reproduces the Windows rule; on
Windows the real `os.fsync` is already strict, so the same tests run as is.
"""

import errno
import os
import pathlib
import re
import stat
import sys

import pytest

from scmle import durability

POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32", reason="Windows handles are not file descriptors"
)


def is_directory(descriptor: int) -> bool:
    """Whether the descriptor refers to a directory."""
    return stat.S_ISDIR(os.fstat(descriptor).st_mode)


@pytest.fixture
def strict_fsync(monkeypatch):
    """`os.fsync` that rejects read-only files, as Windows does."""
    if sys.platform == "win32":
        return
    import fcntl

    real = os.fsync

    def fsync(descriptor: int) -> None:
        access = fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE
        if access == os.O_RDONLY and not is_directory(descriptor):
            raise OSError(errno.EBADF, "Bad file descriptor")
        real(descriptor)

    monkeypatch.setattr(os, "fsync", fsync)


@POSIX_ONLY
def test_strict_fsync_really_rejects_read_only_files(tmp_path, strict_fsync):
    target = tmp_path / "file.bin"
    target.write_bytes(b"data")

    with target.open("rb") as stream, pytest.raises(OSError, match="Bad file"):
        os.fsync(stream.fileno())


def test_sync_file_flushes_an_existing_file_without_changing_it(
    tmp_path, strict_fsync
):
    target = tmp_path / "file.bin"
    target.write_bytes(b"payload")

    durability.sync_file(target)

    assert target.read_bytes() == b"payload"


def test_sync_file_reports_a_missing_file(tmp_path, strict_fsync):
    with pytest.raises(FileNotFoundError):
        durability.sync_file(tmp_path / "absent.bin")


def test_no_module_flushes_a_file_opened_read_only():
    """Guard against reintroducing `open("rb")` followed by `os.fsync`."""
    pattern = re.compile(
        r"open\(\s*[\"']rb[\"']\s*\)\s+as\s+\w+\s*:\s*\n\s*os\.fsync"
    )
    root = pathlib.Path(durability.__file__).parent
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []
