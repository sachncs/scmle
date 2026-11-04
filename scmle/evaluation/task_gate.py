"""Which task types have passed a measured gate (canon P1, F12 criterion 11).

`task_gate_status.json` lists every `TaskType` as `passed`, `failed` or
`no_referent`. A type that has not passed may still run, but its outputs are
capped at the `unverified` tier. The file is written by hand from the
certificates in `benchmarks/qualification/uncertainty_certificates.json`; a
test keeps it complete.
"""

import json
from pathlib import Path
from typing import Any

from scmle.contracts import tasks

STATUS_PATH = Path(__file__).with_name("task_gate_status.json")
PASSED = "passed"
STATUSES = (PASSED, "failed", "no_referent")
UNVERIFIED = "unverified"


def load() -> dict[str, Any]:
    """Reads the status file."""
    return json.loads(STATUS_PATH.read_text(encoding="utf-8"))["tasks"]


def status(task: tasks.TaskType) -> str:
    """The gate status of a task type (`no_referent` when unlisted)."""
    row = load().get(task.value)
    return row["status"] if row else "no_referent"


def tier_ceiling(task: tasks.TaskType, tier: str) -> str:
    """Caps a tier at `unverified` unless the task type passed its gate."""
    return tier if status(task) == PASSED else UNVERIFIED
