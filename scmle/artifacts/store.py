"""Run directory management, the local backend for run evidence."""

import datetime
import json
import os
import pathlib
import tempfile
from typing import Any
from typing import Self

from scmle import durability


class RunStore:
    """A run directory with helpers that write files inside it.

    `create` allocates a new uniquely named directory and `open` wraps an
    existing one.

    Attributes:
        run_id: Identifier of the run.
        directory: The run directory.
    """

    def __init__(self, run_id: str, directory: pathlib.Path) -> None:
        """Initializes the store over an existing directory.

        Args:
            run_id: Identifier of the run.
            directory: The run directory.
        """
        self.run_id = run_id
        self.directory = directory

    @classmethod
    def create(cls, root: str | pathlib.Path) -> Self:
        """Allocates `run_YYYYMMDD_NNN` under `root` without collisions.

        Args:
            root: Parent directory (created if needed).

        Returns:
            A store for the new directory.
        """
        parent = pathlib.Path(root)
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d")
        number = 1
        while True:
            run_id = f"run_{stamp}_{number:03d}"
            try:
                (parent / run_id).mkdir()
                return cls(run_id, parent / run_id)
            except FileExistsError:
                number += 1

    @classmethod
    def open(cls, directory: str | pathlib.Path) -> Self:
        """Opens an existing run directory (for resume).

        Args:
            directory: A directory previously created by `create`.

        Returns:
            A store over it.

        Raises:
            FileNotFoundError: If the path is not an existing directory.
        """
        path = pathlib.Path(directory)
        if not path.is_dir():
            raise FileNotFoundError(f"run directory not found: {path}")
        return cls(path.name, path)

    def path(self, relative: str) -> pathlib.Path:
        """Returns a path inside the run directory.

        Args:
            relative: Path relative to the run directory.

        Returns:
            The run directory joined with `relative` (absolute only if the
            directory is; the file need not exist).
        """
        return self.directory / relative

    def write_json(self, relative: str, payload: Any) -> None:
        """Writes `payload` as indented JSON, creating parent directories.

        Values `json` cannot encode are rendered with `str`. The write goes
        through `write_text`, so it is atomic.

        Args:
            relative: File path relative to the run directory.
            payload: The JSON-serializable value to write.
        """
        self.write_text(relative, json.dumps(payload, indent=2, default=str))

    def write_text(self, relative: str, text: str) -> None:
        """Atomically writes plain text, creating parent directories.

        The text goes to a temporary file beside the destination, is flushed
        and fsynced, and then replaces the destination with `os.replace`; the
        directory is synced afterwards (not on Windows). Any failure removes
        the temporary file and propagates. The temporary file sits in the same
        directory so the replace stays atomic on the supported operating
        systems and filesystems.

        Args:
            relative: File path relative to the run directory.
            text: The text to write, encoded as UTF-8 with LF line endings.
        """
        target = self.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", dir=target.parent
        )
        temporary = pathlib.Path(temporary_name)
        try:
            with os.fdopen(
                descriptor, "w", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            durability.sync_directory(target.parent)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
