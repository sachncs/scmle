"""The local executor: the same definition, run with durable state.

Layout under ``workflows/<name>/runs/<run id>/`` in a `Depot`:
``run.json`` (state, trigger, requester, cost), ``steps/<step>.json`` (state,
attempts, outputs, idempotency key, cost), ``approvals/<step>.json`` and
``lock`` (a lease). Everything is written atomically, so a crashed executor
leaves a consistent record and a new one continues from it.

* **Durable run id.** The run id is derived from the workflow name and the
  caller's `trigger_key` (a schedule slot, a request id), so invoking the
  same trigger twice is the same run.
* **Idempotency keys.** A step's key hashes the run id, the step, its
  parameters and the digests of its upstream outputs. A step that already
  succeeded with the same key is not run again; the key is also handed to the
  handler so that external effects can be deduplicated.
* **Partial failure.** A step that fails (after its retries) stops its
  dependents, which are recorded as `skipped`; independent steps still finish.
  `resume` runs only what has not succeeded.
* **Concurrency.** `max_concurrent_runs` is enforced with run leases;
  `max_parallel` and per-group limits bound the steps in flight.
* **Timeouts** are cooperative here: the handler calls `context.check()`;
  a hosting engine enforces its own.
* **Cost.** Handlers report cost with `context.spend`; a step over its budget
  fails, and a run over its budget stops the remaining steps
  (`budget_stopped`).
* **Approval.** An approval step waits (`waiting_approval`) until a signed
  decision from a listed approver, who must not be the requester, is recorded;
  a rejection or expiry ends the run without the approved actions.
* **Leases.** A run holds a lease with an owner and a fencing token, renewed
  by a heartbeat while steps run. A second executor gets `BusyError`
  while it is valid and may take over after it expires (the first one is
  presumed dead) with one compare-and-swap that raises the token, so two
  executors racing for it cannot both win. Renewal and release check owner
  and token, and every run record is written only after that check
  (`EvictionError` otherwise), so a slow first executor stops instead of
  overwriting the new holder's state (CON-1). The check and the write are
  two store calls, so handlers must still be idempotent.

* **One machine, one archive.** A run's state moves on the shared state
  machine (`scmle.runtime.state.Machine`) along `status.TABLE`, and its
  records are kept through `records.Bucket`, the object-store `Archive` of
  the pipeline engine; neither changes the keys or fields above.

The package splits the executor by concern: `status` (run and step
states and the run's transition table), `failures` (errors), `context`
(what a handler sees), `records` (run ids and the run's `Bucket`), `leases`
(leases and fenced writes), `steps` (one step and its retries), `approvals`,
`loop` (the run loop) and `runs` (`Executor`).
"""

from scmle.workflow.executor.context import Context
from scmle.workflow.executor.context import Handler
from scmle.workflow.executor.context import Outcome
from scmle.workflow.executor.failures import ApprovalError
from scmle.workflow.executor.failures import BudgetError
from scmle.workflow.executor.failures import BusyError
from scmle.workflow.executor.failures import CapacityError
from scmle.workflow.executor.failures import DeadlineError
from scmle.workflow.executor.failures import EvictionError
from scmle.workflow.executor.failures import PreemptedError
from scmle.workflow.executor.leases import CHANGED
from scmle.workflow.executor.leases import LEASE_ATTEMPTS
from scmle.workflow.executor.leases import LEASE_SECONDS
from scmle.workflow.executor.leases import LOGGER
from scmle.workflow.executor.records import distill
from scmle.workflow.executor.records import identify
from scmle.workflow.executor.records import parse
from scmle.workflow.executor.runs import Executor
from scmle.workflow.executor.status import FINISHED
from scmle.workflow.executor.status import Phase
from scmle.workflow.executor.status import Progress
from scmle.workflow.executor.steps import WRITES

__all__ = [
    "CHANGED",
    "FINISHED",
    "LEASE_ATTEMPTS",
    "LEASE_SECONDS",
    "LOGGER",
    "WRITES",
    "ApprovalError",
    "BudgetError",
    "BusyError",
    "CapacityError",
    "Context",
    "DeadlineError",
    "EvictionError",
    "Executor",
    "Handler",
    "Outcome",
    "Phase",
    "PreemptedError",
    "Progress",
    "distill",
    "identify",
    "parse",
]
