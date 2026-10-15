"""Run directory management (the evidence plane's local backend)."""

import datetime
import json
import pathlib
from typing import Any
from typing import Self


class RunStore:
    """A uniquely named run directory with typed write helpers."""

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
            FileNotFoundError: If the directory does not exist.
        """
        path = pathlib.Path(directory)
        if not path.is_dir():
            raise FileNotFoundError(f"run directory not found: {path}")
        return cls(path.name, path)

    def path(self, relative: str) -> pathlib.Path:
        """Returns an absolute path inside the run directory."""
        return self.directory / relative

    def make_directory(self, relative: str) -> pathlib.Path:
        """Creates (if needed) and returns a subdirectory."""
        target = self.path(relative)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def write_json(self, relative: str, payload: Any) -> None:
        """Writes `payload` as indented JSON, creating parent directories.

        Unknown value types are rendered as strings.
        """
        target = self.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, default=str))

    def write_text(self, relative: str, text: str) -> None:
        """Writes plain text."""
        self.path(relative).write_text(text)
