"""Filesystem durability primitives for content-addressed state writes."""

import os
import pathlib


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
