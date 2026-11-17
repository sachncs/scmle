"""Controlled local fault injectors for release-rehearsal evidence.

The executor uses real subprocesses and the repository's public durability,
lineage, ingestion, attestation, and governance contracts. It is intentionally
local and bounded: it does not pretend to reproduce a cloud broker outage,
physical disk exhaustion, or a kernel-wide OOM event.

The package splits the scenarios by what they break: `workspace` (the
evidence workspace and its artifacts), `interruptions` (process death, stale
checkpoints, worker loss, timeouts, retries), `resources` (disk, memory,
files, dependencies), `provenance` (signatures, events, lineage, approvals,
snapshots) and `executor` (`Executor`, which dispatches them).
"""

from scmle.rehearsal.local_faults.executor import Executor
from scmle.rehearsal.local_faults.interruptions import unreachable
from scmle.rehearsal.local_faults.resources import cap
from scmle.rehearsal.local_faults.resources import DISK_BUDGET_BYTES
from scmle.rehearsal.local_faults.resources import MEMORY_BUDGET_BYTES
from scmle.rehearsal.local_faults.resources import OBSERVED_DEPENDENCY
from scmle.rehearsal.local_faults.resources import ration
from scmle.rehearsal.local_faults.resources import REQUIRED_DEPENDENCY

__all__ = [
    "DISK_BUDGET_BYTES",
    "MEMORY_BUDGET_BYTES",
    "OBSERVED_DEPENDENCY",
    "REQUIRED_DEPENDENCY",
    "Executor",
    "cap",
    "ration",
    "unreachable",
]
