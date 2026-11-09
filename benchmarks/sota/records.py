"""The one result format every system writes, so results compare like for like.

Each line of a results file is a JSON object with the fields of `RunRecord`.
A run that fails or times out is still a record (`status != "ok"`); nothing is
dropped, and the statistics decide how failures are scored.
"""

import dataclasses
import json
import pathlib
import platform
from typing import Any

STATUSES = ("ok", "failed", "timeout")


@dataclasses.dataclass
class RunRecord:
    """Outcome of one system on one task split at one budget.

    Attributes:
        system: System name, for example `scmle` or `autogluon-best`.
        budget_s: Wall-clock budget handed to the system, in seconds.
        task_id: OpenML task id from the pinned task table.
        fold: Fold index of the task's split.
        repeat: Repeat index of the task's split.
        metric: Name of the task's evaluation metric.
        score: Test score in the metric's natural units, or None on failure.
        seconds: Wall-clock time the run actually used.
        status: `ok`, `failed` or `timeout`.
        error: Failure text, empty on success.
        versions: Versions of the system and its main dependencies.
        machine: Hardware and operating-system identity.
    """

    system: str
    budget_s: int
    task_id: int
    fold: int
    repeat: int
    metric: str
    score: float | None
    seconds: float
    status: str = "ok"
    error: str = ""
    versions: dict[str, str] = dataclasses.field(default_factory=dict)
    machine: str = dataclasses.field(default_factory=platform.platform)

    def __post_init__(self) -> None:
        """Rejects an unknown status so a typo cannot hide a failure.

        Raises:
            ValueError: If `status` is not one of `STATUSES`, or an `ok` record
                has no score.
        """
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")
        if self.status == "ok" and self.score is None:
            raise ValueError("an ok record needs a score")


def append(path: pathlib.Path, record: RunRecord) -> None:
    """Appends one record to a JSON-lines results file.

    Args:
        path: The results file; created if missing.
        record: The record to write as one line.
    """
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dataclasses.asdict(record)) + "\n")


def read(path: pathlib.Path) -> list[RunRecord]:
    """Reads every record of a results file.

    Args:
        path: The JSON-lines results file.

    Returns:
        The records in file order; blank lines are skipped.
    """
    records: list[RunRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                fields: dict[str, Any] = json.loads(line)
                records.append(RunRecord(**fields))
    return records


def key(record: RunRecord) -> tuple[str, int, int, int, int]:
    """Identifies one run: system, budget, task, fold and repeat.

    Args:
        record: The record to identify.

    Returns:
        The tuple (system, budget_s, task_id, fold, repeat).
    """
    return (
        record.system,
        record.budget_s,
        record.task_id,
        record.fold,
        record.repeat,
    )


def completed(path: pathlib.Path) -> set[tuple[str, int, int, int, int]]:
    """Keys of the runs that finished `ok`, so a restart skips them.

    A failed or timed-out run is not complete and will be tried again.

    Args:
        path: The JSON-lines results file; it may not exist yet.

    Returns:
        The `key` of every `ok` record; empty when the file is missing.
    """
    if not path.exists():
        return set()
    return {key(r) for r in read(path) if r.status == "ok"}
