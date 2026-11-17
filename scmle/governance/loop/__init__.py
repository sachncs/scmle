"""The closed loop: drift, retrain, gated promotion and asking a person.

    loop = Loop.open(deploy_dir, charter, policy)
    routing = loop.observe(batch, predictions, decisions)
    loop.mature(losses)                    # outcomes arrive, with delay
    result = loop.step()                   # alarms, proposals, escalations
    pending = loop.pending_escalations()
    loop.reply(pending[0].id, attestation)  # a person's signed reply

What the loop does by itself, and how each rule behaves:

* L1. A betting alarm on the matured, propensity-weighted mean loss
  (`Tripwire`): false alarm probability at most ``alarm`` over all time
  for the first alarm of an episode. On an alarm with a verified label
  mechanism it records an `alarm` row. If ``retrain`` is set it
  then records a `retrain` request (and calls the `retrain` hook, if given,
  which must train and stage a challenger and return its name); otherwise it
  escalates for investigation and retrains nothing. When the naive
  (unweighted) alarm crosses on an unverified mechanism it halts and
  escalates; it never retrains.
* L2. Each challenger is registered in the ledger before it is scored, scored
  by an e-process on paired losses, and decided in registration order by
  e-LOND. A crossing becomes a `promotion_proposal`; the signed approval of
  the registry is still needed to move the champion alias.
* L3. After a promotion, a reverse e-process (`surveil`) on the new
  champion against the previous one can create a `rollback_proposal` and
  tightens `tau`.
* E1. Cases below the calibrated threshold, and any case beyond the model's
  reach, are escalated. If the selective risk of answered cases is evidenced
  above ``r*`` the loop halts: every case is escalated until a recalibration
  passes. Escalations beyond ``escalation`` get the safe default action.
* E2. System-level triggers write an `escalation` row with the question, the
  evidence and the proposed default before anything waits.

What it cannot do: see `scmle.governance.never_list`. The decision on the
champion alias, the sealed sets, the charter and the other listed actions
needs a person's attestation; `Loop.reply` records it. Key custody for that
signature is a deployment requirement the software cannot enforce.

State. The ledger holds what happened; `loop_state.json` holds the monitors'
running state and the batches awaiting maturity. State is saved after every
public call and carries the ledger head: if they disagree at `open`, the loop
starts halted rather than guess.

Layout. `terms` holds the policy, the records and the constants; `memory`
the running state and the state file; `desk` escalations, halts and
replies; `risk` the weighted estimates and calibration; `contest` the
challengers and the rollback watch; `intake` routing and matured outcomes;
`alarms` what a crossed monitor leads to; `guard` the never-list actions;
and `cycle` the `Loop` that joins them.
"""

from scmle.governance.loop.cycle import Loop
from scmle.governance.loop.terms import amend
from scmle.governance.loop.terms import Consequence
from scmle.governance.loop.terms import ESCALATING_ROLLBACKS
from scmle.governance.loop.terms import Escalation
from scmle.governance.loop.terms import HALT_LEDGER
from scmle.governance.loop.terms import HALT_SELECTIVE
from scmle.governance.loop.terms import HALT_STATE
from scmle.governance.loop.terms import HALT_UNVERIFIED
from scmle.governance.loop.terms import HALTED
from scmle.governance.loop.terms import Hooks
from scmle.governance.loop.terms import INSUFFICIENT_LABELS
from scmle.governance.loop.terms import jsonify
from scmle.governance.loop.terms import LABEL_MECHANISM_FILE
from scmle.governance.loop.terms import LOOSER
from scmle.governance.loop.terms import NUMERIC_ORIENTATION
from scmle.governance.loop.terms import Observation
from scmle.governance.loop.terms import OK
from scmle.governance.loop.terms import Outcome
from scmle.governance.loop.terms import Preregistration
from scmle.governance.loop.terms import RefusalError
from scmle.governance.loop.terms import RISK_COVERAGE_FILE
from scmle.governance.loop.terms import STATE_FILE
from scmle.governance.loop.terms import STATE_VERSION
from scmle.governance.loop.terms import TIGHTER

__all__ = [
    "ESCALATING_ROLLBACKS",
    "HALTED",
    "HALT_LEDGER",
    "HALT_SELECTIVE",
    "HALT_STATE",
    "HALT_UNVERIFIED",
    "INSUFFICIENT_LABELS",
    "LABEL_MECHANISM_FILE",
    "LOOSER",
    "NUMERIC_ORIENTATION",
    "OK",
    "RISK_COVERAGE_FILE",
    "STATE_FILE",
    "STATE_VERSION",
    "TIGHTER",
    "Consequence",
    "Escalation",
    "Hooks",
    "Loop",
    "Observation",
    "Outcome",
    "Preregistration",
    "RefusalError",
    "amend",
    "jsonify",
]
