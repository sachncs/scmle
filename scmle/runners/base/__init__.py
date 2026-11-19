"""The job spec, the run manifest and the `Runner` template.

One `Job` describes a training run independent of where it runs. A
`Runner` adapter knows how to start, poll and cancel a job in one place
(this process, a container job, a managed service). The shared `Runner.run`
does everything that must not differ between places:

* **Durable run id.** The spec is stored at ``runs/<id>/spec.json``; running
  an id again with a different spec is refused, and running a finished id
  returns its recorded outcome without starting anything.
* **Manifest.** ``runs/<id>/manifest.json`` is rewritten atomically at every
  transition: attempts, states, costs, reasons, and at the end the digests of
  the checkpoints and the result.
* **Controlled retries.** An attempt that ends interrupted (preemption,
  spot reclaim) or failed with a retryable cause is retried under the spec's
  `Retries`, up to its attempt limit; the retry resumes from the run's
  checkpoints. Other failures, timeouts and budget stops are final.
* **Cost budget.** Cost is the declared rate times resources times elapsed
  time, accumulated across attempts. When it passes the budget the job is
  cancelled and the run ends `budget_stopped`. It is an estimate from the
  declared rates, not billing data.
* **Separate roles.** A job may not run as the serving identity.
"""

from scmle.runners.base.job import Clock
from scmle.runners.base.job import ConflictError
from scmle.runners.base.job import CostBudget
from scmle.runners.base.job import CostRate
from scmle.runners.base.job import Exit
from scmle.runners.base.job import FINAL
from scmle.runners.base.job import Job
from scmle.runners.base.job import MAX_RUN_ID_LENGTH
from scmle.runners.base.job import Meter
from scmle.runners.base.job import Progress
from scmle.runners.base.job import Resources
from scmle.runners.base.job import RUN_ID_PATTERN
from scmle.runners.base.job import SECRET_WORDS
from scmle.runners.base.job import stamp
from scmle.runners.base.job import State
from scmle.runners.base.job import Status
from scmle.runners.base.job import TERMINAL
from scmle.runners.base.job import Ticket
from scmle.runners.base.ledger import Attempt
from scmle.runners.base.ledger import MANIFEST_FORMAT
from scmle.runners.base.ledger import RunManifest
from scmle.runners.base.runner import Runner

__all__ = [
    "FINAL",
    "MANIFEST_FORMAT",
    "MAX_RUN_ID_LENGTH",
    "RUN_ID_PATTERN",
    "SECRET_WORDS",
    "TERMINAL",
    "Attempt",
    "Clock",
    "ConflictError",
    "CostBudget",
    "CostRate",
    "Exit",
    "Job",
    "Meter",
    "Progress",
    "Resources",
    "RunManifest",
    "Runner",
    "State",
    "Status",
    "Ticket",
    "stamp",
]
