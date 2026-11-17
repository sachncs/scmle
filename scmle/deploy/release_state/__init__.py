"""Registry state bound to what the runtime says is running.

The registry (`registry.py`) knows versions and aliases; it does not know what
a cloud is serving. This module adds, beside it, the release state of each
version and a record of what each environment runs, and it moves them only
after the platform confirms.

States, all distinct:

``candidate``
    Registered, nobody has judged it.
``challenger``
    Under comparison with the production model.
``approved``
    Signed bundle and an independent, signed approval for this exact bundle;
    may be released, is not released.
``production``
    The runtime confirmed it serves all traffic in the environment.
``retired``
    Superseded, rolled back or withdrawn. A retired version returns to
    ``production`` only through a rollback that the runtime confirms.

Rules this module enforces:

* ``promote`` asks a `ControlPlane` to deploy and to move traffic, step by
  step, and reads the platform back after each request. The registry state,
  the champion alias and the `promoted` ledger event change only after the
  last read shows the revision, its image digest, readiness and the full
  share of traffic. If any read disagrees, nothing in the registry changes,
  a `rejected` event is recorded and the previous revision is restored.
* ``rollback`` restores the previous known-good revision and changes the
  registry only after the platform confirms it.
* ``live`` answers "what does the registry say is deployed" only
  when an observation agrees; otherwise it raises `RuntimeDisagrees`, so an
  alias can never be read as "deployed" against the runtime.
* ``reconcile`` lists every disagreement between the registry, the ledger
  and an observation.

Nothing here talks to a cloud; the `ControlPlane` does. The state file is a
separate file next to ``registry.json`` and does not change its format.

`status` holds the release states and the records a call returns,
`records` the state file and event log, `approval` the hand transitions and
approvals, `rollout` the confirmed traffic moves, `promotion` and `rollback`
the two releases the runtime confirms, and `reconcile` the
`ReleaseRegistry` that joins them.
"""

from scmle.deploy.release_state.reconcile import ReleaseRegistry
from scmle.deploy.release_state.status import ALLOWED
from scmle.deploy.release_state.status import Finding
from scmle.deploy.release_state.status import FULL_ROLLOUT
from scmle.deploy.release_state.status import halt
from scmle.deploy.release_state.status import Health
from scmle.deploy.release_state.status import MANUAL
from scmle.deploy.release_state.status import Outcome
from scmle.deploy.release_state.status import predecessor
from scmle.deploy.release_state.status import PRODUCTION_ENVIRONMENT
from scmle.deploy.release_state.status import Reconciliation
from scmle.deploy.release_state.status import ReleaseState
from scmle.deploy.release_state.status import RuntimeDisagrees
from scmle.deploy.release_state.status import STATE_FILE
from scmle.deploy.release_state.status import STATE_FORMAT

__all__ = [
    "ALLOWED",
    "FULL_ROLLOUT",
    "MANUAL",
    "PRODUCTION_ENVIRONMENT",
    "STATE_FILE",
    "STATE_FORMAT",
    "Finding",
    "Health",
    "Outcome",
    "Reconciliation",
    "ReleaseRegistry",
    "ReleaseState",
    "RuntimeDisagrees",
    "halt",
    "predecessor",
]
