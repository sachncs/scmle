"""The deployment ledger: a hash-chained record of the closed loop.

The hypothesis ledger (`scmle.assurance.ledger.ledger`) closes at its
confirmation, so a production loop cannot append to it. This ledger has its
own chain, with the same row layout and hash (`ledger.digest`), and the same
limits: it shows that nothing was edited, removed from the middle or
reordered, not that the newest rows were never removed (compare `head` with a
copy kept elsewhere), and not that nobody rewrote file and copies together.

Row kinds (``open`` first and only once):

``open``
    Charter hash, the loop policy and its hash, the fingerprints of keys the
    system itself holds (an attestation signed with one of them is refused).
``observe_summary``, ``alarm``, ``retrain``
    What the loop saw and did, free-form but chained.
``challenger_result``
    A challenger's registration (``phase = "registered"``, before it is
    scored) and its decision (``phase = "decided"``) with the e-value, the
    e-LOND level and the wealth. Replay re-runs e-LOND over the decided rows
    and compares level, decision and wealth exactly.
``promotion_proposal``, ``rollback_proposal``
    Proposals only; the registry's signed approval still moves the alias.
``escalation``
    The full pending question, evidence references and the system's proposed
    default, written before anything waits (``phase = "open"``); a timeout
    closes it with the safe default (``phase = "timeout"``).
``attestation``
    A human reply, bound to one escalation and to the hash of its question,
    signed with a key the system does not hold, with an expiry.
``halt``
    Entering (``phase = "enter"``) or leaving (``phase = "release"``) the
    state in which every case is escalated.
``incident``
    An incident opened because of a halt: the reason, the batch and the
    status ``open``. It is a record that a person must look, not an incident
    workflow (`scmle.monitoring.incidents` wants an SLO evaluation, which a
    halt does not have).
``rollout_experiment``, ``rollout_assignment``, ``rollout_decision``
    The decision layer's rollout: an experiment is started with the
    level the `DecisionBudget` assigned and the budget after the assignment;
    each batch of units is logged with the arm and the probability that arm
    had (the propensity), and only for a started experiment; the decision
    (`hold`, `promote`, `hold_final`, `hold_invalid`, or a rollback) is
    chained with the e-process values. A promotion or rollback still needs the
    registry's signed approval.
``envelope``, ``envelope_used``, ``envelope_revoked``
    Authority a person signed in advance for an action that is not on the
    never list (`scmle.governance.envelope`): the actions with a use limit
    each, the scope, an expiry and the signing key's fingerprint. A use is
    chained before the action runs and is refused at ingest when the
    envelope is unknown, revoked, out of scope, does not cover the action or
    has spent its limit; revoking is allowed to anyone because it only
    removes authority.
``controller``
    The unattended controller's journal (`scmle.operations.controller`): the
    intent to apply an inbox file (by content digest) and that it was
    applied, a rejected file, and one row per tick. A digest is applied once.
``deployment``
    A release event confirmed or rejected by the runtime (`ReleaseRegistry`):
    the model, version, environment, revision and image digest, and whether
    the platform confirmed the promotion, the rollback or refused it.
``audit``
    A security event of the prediction service: a model activation, an
    access decision (allowed or denied) or an administrative action. Chained
    so an entry cannot be removed or edited afterwards; it carries who, what,
    the outcome and a request id, never a credential or a request body.

`kinds` holds the row kinds and shared checks, `grants` the signed replies
and envelopes, `rules` the state machine of each row kind and `chain` the
ledger itself.
"""

from scmle.governance.deployment_ledger.chain import DeploymentLedger
from scmle.governance.deployment_ledger.grants import Attestation
from scmle.governance.deployment_ledger.grants import Envelope
from scmle.governance.deployment_ledger.kinds import ALARM
from scmle.governance.deployment_ledger.kinds import APPLIED
from scmle.governance.deployment_ledger.kinds import ATTESTATION
from scmle.governance.deployment_ledger.kinds import ATTESTATION_USED
from scmle.governance.deployment_ledger.kinds import AUDIT
from scmle.governance.deployment_ledger.kinds import CHALLENGER_RESULT
from scmle.governance.deployment_ledger.kinds import CONTROLLER
from scmle.governance.deployment_ledger.kinds import CONTROLLER_INPUTS
from scmle.governance.deployment_ledger.kinds import CONTROLLER_PHASES
from scmle.governance.deployment_ledger.kinds import DECIDED
from scmle.governance.deployment_ledger.kinds import DEPLOYMENT
from scmle.governance.deployment_ledger.kinds import ENTER
from scmle.governance.deployment_ledger.kinds import ENVELOPE
from scmle.governance.deployment_ledger.kinds import ENVELOPE_ACTIONS
from scmle.governance.deployment_ledger.kinds import ENVELOPE_REVOKED
from scmle.governance.deployment_ledger.kinds import ENVELOPE_USED
from scmle.governance.deployment_ledger.kinds import ESCALATION
from scmle.governance.deployment_ledger.kinds import FILE_NAME
from scmle.governance.deployment_ledger.kinds import HALT
from scmle.governance.deployment_ledger.kinds import INCIDENT
from scmle.governance.deployment_ledger.kinds import INTENT
from scmle.governance.deployment_ledger.kinds import KINDS
from scmle.governance.deployment_ledger.kinds import LEDGER_VERSION
from scmle.governance.deployment_ledger.kinds import mapped
from scmle.governance.deployment_ledger.kinds import MisuseError
from scmle.governance.deployment_ledger.kinds import OBSERVE_SUMMARY
from scmle.governance.deployment_ledger.kinds import OPEN
from scmle.governance.deployment_ledger.kinds import parse_date
from scmle.governance.deployment_ledger.kinds import PHASE_OPEN
from scmle.governance.deployment_ledger.kinds import PHASE_TIMEOUT
from scmle.governance.deployment_ledger.kinds import PROMOTION_PROPOSAL
from scmle.governance.deployment_ledger.kinds import REGISTERED
from scmle.governance.deployment_ledger.kinds import REJECTED
from scmle.governance.deployment_ledger.kinds import RELEASE
from scmle.governance.deployment_ledger.kinds import Reply
from scmle.governance.deployment_ledger.kinds import RETRAIN
from scmle.governance.deployment_ledger.kinds import ROLLBACK_PROPOSAL
from scmle.governance.deployment_ledger.kinds import ROLLOUT_ASSIGNMENT
from scmle.governance.deployment_ledger.kinds import ROLLOUT_DECISION
from scmle.governance.deployment_ledger.kinds import ROLLOUT_EXPERIMENT
from scmle.governance.deployment_ledger.kinds import thumbprint
from scmle.governance.deployment_ledger.kinds import TICK
from scmle.monitoring import sequential_tests

__all__ = [
    "ALARM",
    "APPLIED",
    "ATTESTATION",
    "ATTESTATION_USED",
    "AUDIT",
    "CHALLENGER_RESULT",
    "CONTROLLER",
    "CONTROLLER_INPUTS",
    "CONTROLLER_PHASES",
    "DECIDED",
    "DEPLOYMENT",
    "ENTER",
    "ENVELOPE",
    "ENVELOPE_ACTIONS",
    "ENVELOPE_REVOKED",
    "ENVELOPE_USED",
    "ESCALATION",
    "FILE_NAME",
    "HALT",
    "INCIDENT",
    "INTENT",
    "KINDS",
    "LEDGER_VERSION",
    "OBSERVE_SUMMARY",
    "OPEN",
    "PHASE_OPEN",
    "PHASE_TIMEOUT",
    "PROMOTION_PROPOSAL",
    "REGISTERED",
    "REJECTED",
    "RELEASE",
    "RETRAIN",
    "ROLLBACK_PROPOSAL",
    "ROLLOUT_ASSIGNMENT",
    "ROLLOUT_DECISION",
    "ROLLOUT_EXPERIMENT",
    "TICK",
    "Attestation",
    "DeploymentLedger",
    "Envelope",
    "MisuseError",
    "Reply",
    "mapped",
    "parse_date",
    "sequential_tests",
    "thumbprint",
]
