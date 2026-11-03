"""Filesystem durability primitives for content-addressed state writes."""

import contextlib
import os
import pathlib
import sys
import tempfile
import threading
import time

WINDOWS = sys.platform == "win32"
# Windows refuses to replace a file another handle still has open, usually
# for a few milliseconds (a reader, an indexer, a scanner).
REPLACE_ATTEMPTS = 8
REPLACE_DELAY_SECONDS = 0.02

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


def replace_file(
    source: str | pathlib.Path, target: str | pathlib.Path
) -> None:
    """Atomically move ``source`` over ``target``, tolerating Windows sharing.

    The replacement holds the target's path lock, so in-process readers that
    take the same lock never overlap it. On Windows a replace can still be
    denied by a handle held elsewhere (another process); it is retried a
    bounded number of times with a short growing delay. Other platforms, and
    other errors, surface immediately.

    Args:
        source: Fully written temporary file.
        target: Destination path.

    Raises:
        PermissionError: If the replacement is still denied after the
            attempts, or at once on a platform other than Windows.
    """
    with path_lock(target):
        attempt = 1
        while True:
            try:
                os.replace(source, target)
                return
            except PermissionError:
                if not WINDOWS or attempt >= REPLACE_ATTEMPTS:
                    raise
                time.sleep(REPLACE_DELAY_SECONDS * attempt)
                attempt += 1


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
        replace_file(temporary_name, target)
        sync_directory(target.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
