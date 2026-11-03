"""Filesystem durability primitives for content-addressed state writes."""

import contextlib
import os
import pathlib
import tempfile
import threading

PATH_LOCKS: dict[str, threading.RLock] = {}
PATH_LOCKS_GUARD = threading.Lock()


def path_lock(path: str | pathlib.Path) -> threading.RLock:
    """Returns the process-wide re-entrant lock that guards one file.

    Every writer of a shared journal or index takes the lock for the same
    path, whichever object it goes through, so a read-modify-write or an
    append cannot interleave with another thread's. The lock is per process;
    it does not coordinate separate processes.

    Args:
        path: The guarded file or directory (symlinks are resolved).

    Returns:
        The lock for that path; the same object on every call.
    """
    key = os.path.realpath(path)
    with PATH_LOCKS_GUARD:
        lock = PATH_LOCKS.get(key)
        if lock is None:
            lock = PATH_LOCKS[key] = threading.RLock()
        return lock


def append_text(path: str | pathlib.Path, text: str) -> None:
    """Appends text as one unit, never interleaved with another thread's.

    Args:
        path: File to append to (parent directories are created).
        text: Complete text to add, including its trailing newline.
    """
    target = pathlib.Path(path)
    with path_lock(target):
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(text)


def sync_file(path: str | pathlib.Path) -> None:
    """Flush an existing file's contents to stable storage.

    The file is opened for update because Windows rejects ``fsync`` on a
    read-only descriptor (``EBADF``) while POSIX accepts it.

    Args:
        path: File to flush; it must exist.
    """
    with open(path, "r+b") as handle:
        os.fsync(handle.fileno())


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
