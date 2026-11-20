"""Local scenarios for the fault catalogue.

Each scenario starts the real local service (the generated ``serve.py`` in a
subprocess, as ``scmle.rehearsal.demo`` does), the local object-store and
metrics emulators, a real ledger or a real workflow executor, breaks it the
way the fault says and measures recovery with a monotonic clock. Everything
the scenarios start is owned by a `Fleet`, which stops it in ``close`` and
counts anything still alive afterwards.

These are local processes and emulators. The control plane in ``release``
runs revisions as processes. None of this is a cloud, a container
orchestrator, a network partition or a machine crash; the catalogue
(`scmle.operations.recovery`) says so per fault.

The package splits the scenarios by what breaks: `harness` (the fleet, the
service and its probes), `crashes` (killed processes, jobs and ledgers),
`service` (refused bundles, keys and schemas, and a rolling restart),
`outages` (the object store and the metrics sink), `release` (a bad release
on a process control plane) and `catalogue` (`SCENARIOS` and `rehearse`).
"""

from scmle.rehearsal.faults.catalogue import rehearse
from scmle.rehearsal.faults.catalogue import SCENARIOS
from scmle.rehearsal.faults.crashes import JOB_CHILD
from scmle.rehearsal.faults.crashes import KILLED_RETURN_CODE
from scmle.rehearsal.faults.crashes import LEDGER_WRITER
from scmle.rehearsal.faults.crashes import process
from scmle.rehearsal.faults.crashes import resilience
from scmle.rehearsal.faults.crashes import resume
from scmle.rehearsal.faults.crashes import STATE_WRITER
from scmle.rehearsal.faults.crashes import stress
from scmle.rehearsal.faults.harness import boot
from scmle.rehearsal.faults.harness import corrupt
from scmle.rehearsal.faults.harness import digest
from scmle.rehearsal.faults.harness import ENVIRONMENT
from scmle.rehearsal.faults.harness import export
from scmle.rehearsal.faults.harness import failing
from scmle.rehearsal.faults.harness import Fleet
from scmle.rehearsal.faults.harness import FULL_TRAFFIC_PERCENT
from scmle.rehearsal.faults.harness import GOLDEN
from scmle.rehearsal.faults.harness import Instance
from scmle.rehearsal.faults.harness import interrupt
from scmle.rehearsal.faults.harness import INTERRUPTED_STEP_RUNS
from scmle.rehearsal.faults.harness import invert
from scmle.rehearsal.faults.harness import LABEL_NOISE_RATE
from scmle.rehearsal.faults.harness import METHOD_CLOCK
from scmle.rehearsal.faults.harness import MODEL
from scmle.rehearsal.faults.harness import predict
from scmle.rehearsal.faults.harness import provision
from scmle.rehearsal.faults.harness import READY_SECONDS
from scmle.rehearsal.faults.harness import records
from scmle.rehearsal.faults.harness import ROTATED_KEY
from scmle.rehearsal.faults.harness import specify
from scmle.rehearsal.faults.harness import SPOOLED_SAMPLES
from scmle.rehearsal.faults.harness import WRONG_KEY
from scmle.rehearsal.faults.outages import blackout
from scmle.rehearsal.faults.outages import disconnect
from scmle.rehearsal.faults.release import degradation
from scmle.rehearsal.faults.release import Processes
from scmle.rehearsal.faults.release import register
from scmle.rehearsal.faults.release import release
from scmle.rehearsal.faults.service import bundle
from scmle.rehearsal.faults.service import Client
from scmle.rehearsal.faults.service import DRAIN_REQUESTS
from scmle.rehearsal.faults.service import DRAIN_SECONDS
from scmle.rehearsal.faults.service import incompatible_schema
from scmle.rehearsal.faults.service import invalid_credentials
from scmle.rehearsal.faults.service import rekey
from scmle.rehearsal.faults.service import restart

__all__ = [
    "DRAIN_REQUESTS",
    "DRAIN_SECONDS",
    "ENVIRONMENT",
    "FULL_TRAFFIC_PERCENT",
    "GOLDEN",
    "INTERRUPTED_STEP_RUNS",
    "JOB_CHILD",
    "KILLED_RETURN_CODE",
    "LABEL_NOISE_RATE",
    "LEDGER_WRITER",
    "METHOD_CLOCK",
    "MODEL",
    "READY_SECONDS",
    "ROTATED_KEY",
    "SCENARIOS",
    "SPOOLED_SAMPLES",
    "STATE_WRITER",
    "WRONG_KEY",
    "Client",
    "Fleet",
    "Instance",
    "Processes",
    "blackout",
    "boot",
    "bundle",
    "corrupt",
    "degradation",
    "digest",
    "disconnect",
    "export",
    "failing",
    "incompatible_schema",
    "interrupt",
    "invalid_credentials",
    "invert",
    "predict",
    "process",
    "provision",
    "records",
    "register",
    "rehearse",
    "rekey",
    "release",
    "resilience",
    "restart",
    "resume",
    "specify",
    "stress",
]
