"""The training history and the preserved datasets it points to."""

import hashlib
import json
import pathlib
from typing import Any

import pandas

from scmle import durability
from scmle.contracts import errors

HISTORY_FILE = "training_history.jsonl"
LABELS_FILE = "retained_outcomes.jsonl"
DATASET_DIRECTORY = "datasets"


def serialize(frame: pandas.DataFrame) -> bytes:
    """The bytes a dataset version is computed from.

    Args:
        frame: The dataset.

    Returns:
        The UTF-8 CSV of the frame with columns sorted by name, no index
        and newline line endings.
    """
    ordered = frame.reindex(sorted(frame.columns), axis=1)
    return ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")


def stamp(frame: pandas.DataFrame) -> str:
    """``sha256:<hex>`` of the canonical CSV of a dataset.

    Args:
        frame: The dataset.

    Returns:
        The version string.
    """
    return "sha256:" + hashlib.sha256(serialize(frame)).hexdigest()


def load(directory: str | pathlib.Path, version: str) -> pandas.DataFrame:
    """Reads a preserved dataset back and checks its digest.

    Args:
        directory: The directory holding the preserved datasets.
        version: The dataset version, as returned by `stamp`. The
            file read is named after it, with the colon replaced by a hyphen.

    Returns:
        The stored dataset.

    Raises:
        ValueError: If the stored copy no longer has the digest it is named
            for.
    """
    path = pathlib.Path(directory) / DATASET_DIRECTORY
    frame = pandas.read_csv(
        path / (version.replace(":", "-") + ".csv"),
        float_precision="round_trip",
    )
    if stamp(frame) != version:
        raise errors.InputError(f"the stored copy of {version} has changed")
    return frame


class History:
    """An append-only, hash-chained record of what was trained and decided."""

    __module__ = "scmle.operations.continuous"

    def __init__(self, path: str | pathlib.Path) -> None:
        """Opens (or starts) the history at ``path``.

        Args:
            path: The history file.
        """
        self.path = pathlib.Path(path)

    def rows(self) -> list[dict[str, Any]]:
        """Every row, verified.

        Returns:
            The rows, oldest first; empty if the file does not exist.

        Raises:
            ValueError: If the chain is broken.
        """
        if not self.path.exists():
            return []
        found = []
        previous = "0" * 64
        for number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines()
        ):
            row = json.loads(line)
            content = {
                k: row[k] for k in ("seq", "at", "kind", "facts", "prev")
            }
            digest = hashlib.sha256(
                json.dumps(content, sort_keys=True).encode()
            ).hexdigest()
            if (
                row["seq"] != number
                or row["prev"] != previous
                or (row["hash"] != digest)
            ):
                raise errors.InputError(
                    f"training history broken at row {number}"
                )
            previous = row["hash"]
            found.append(row)
        return found

    def record(self, kind: str, at: float, **facts: Any) -> dict[str, Any]:
        """Appends one row and returns it.

        Args:
            kind: The kind of event.
            at: Time of the event, in epoch seconds; stored rounded to three
                decimals.
            **facts: Facts recorded with the event; values that JSON cannot
                encode are stored as their string form.

        Returns:
            The row, with its sequence number, previous hash and own hash.
        """
        rows = self.rows()
        content = {
            "seq": len(rows),
            "at": round(at, 3),
            "kind": kind,
            "facts": json.loads(json.dumps(facts, default=str)),
            "prev": rows[-1]["hash"] if rows else "0" * 64,
        }
        row = {
            **content,
            "hash": hashlib.sha256(
                json.dumps(content, sort_keys=True).encode()
            ).hexdigest(),
        }
        durability.append_text(
            self.path, json.dumps(row, sort_keys=True) + "\n"
        )
        return row

    def select(self, kind: str) -> list[dict[str, Any]]:
        """Rows of one kind, oldest first.

        Args:
            kind: The kind of event.

        Returns:
            The verified rows of that kind.
        """
        return [row for row in self.rows() if row["kind"] == kind]
