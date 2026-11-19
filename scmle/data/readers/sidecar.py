"""The lock and acknowledgement files beside a durable ingestion state file.

Both durable ledgers, the incremental event `store.Ledger` and the streaming
`journal.Journal`, keep their state in one JSON file and serialize writers
with an advisory lock on a sibling file. They share that layout here.
"""

from collections.abc import Generator
import contextlib
import dataclasses
import os
import pathlib
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class Sidecar:
    """A durable state file and the sidecar files kept next to it.

    Attributes:
        path: JSON file holding the durable state.
    """

    path: pathlib.Path

    @property
    def lockfile(self) -> pathlib.Path:
        """The sidecar file used to serialize ledger writers."""
        return self.path.with_name(f"{self.path.name}.lock")

    @property
    def pending(self) -> pathlib.Path:
        """The durable sidecar file holding an undelivered acknowledgement."""
        return self.path.with_name(f"{self.path.name}.ack")

    @contextlib.contextmanager
    def lock(self) -> Generator[None, None, None]:
        """Serializes ledger writers with the platform's advisory file lock.

        Yields:
            None, while the exclusive lock on the sidecar file is held.
        """
        self.lockfile.parent.mkdir(parents=True, exist_ok=True)
        windows_lock: Any = None
        with self.lockfile.open("a+b") as handle:
            if os.name == "nt":
                import msvcrt  # noqa: PLC0415 -- Windows-only module, imported only on Windows

                windows_lock = msvcrt
                handle.seek(0)
                windows_lock.locking(handle.fileno(), windows_lock.LK_LOCK, 1)
            else:
                import fcntl  # noqa: PLC0415 -- POSIX-only module, imported only off Windows

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    handle.seek(0)
                    windows_lock.locking(
                        handle.fileno(), windows_lock.LK_UNLCK, 1
                    )
                else:
                    import fcntl  # noqa: PLC0415 -- POSIX-only module, imported only off Windows

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
