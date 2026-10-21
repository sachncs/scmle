"""Filesystem durability primitives for content-addressed state writes."""

import contextlib
import os
import pathlib
import tempfile


def sync_directory(directory: str | pathlib.Path) -> None:
    """Flush a directory entry after an atomic replacement.

    ``fsync`` on the temporary file protects its contents. A second ``fsync``
    on the containing directory protects the rename itself from being lost
    during a sudden power failure. Windows does not expose a portable,
    supported directory-handle equivalent, so file flushing remains the
    available durability guarantee there.

    Args:
        directory: Directory whose metadata must be flushed.
    """
    if os.name == "nt":
        return
    flags = os.O_RDONLY
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(pathlib.Path(directory), flags | directory_flag)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_atomically(path: str | pathlib.Path, content: bytes) -> None:
    """Write bytes with flush, fsync, atomic replacement, and cleanup.

    The temporary file is created beside the destination so the replacement
    remains atomic on the supported filesystems. The containing directory is
    flushed after replacement so the directory entry is durable as well.

    Args:
        path: Destination file path.
        content: Complete file contents to persist.
    """
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
        sync_directory(target.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
