"""The actions the loop must not take without a person (the "never list").

Each of the eleven actions has a guard: called without a valid attestation in
the deployment ledger, `Guard.require` writes an `escalation` row (once per
pending question) and raises `AttestationRequired`; the action is not
performed. A valid attestation is one that

* answers an escalation whose question hash equals the hash of this exact
  action, detail and scope (a reply cannot be reused for another question);
* decides `approve` or `override` (`reject` and `cannot_decide` do not allow);
* has not expired and was not dated in the future;
* names the same scope (model version hash and stream id);
* verifies under the verification key registered for its `key_id`, when the
  guard holds one; and
* was not signed with a key the system holds (the ledger refuses such rows).

By default an attestation stays valid for the identical question until it
expires. A guard made with `single_use=True` spends it: the first action it
allows writes an `attestation_used` row, and the same reply allows nothing
afterwards (a new escalation and a new reply are needed). The marker is in the
hash-chained ledger, so reopening the ledger keeps it. The
registry's own signed-approval check for the champion alias and the custody
log's one-time open of a sealed set stay in force underneath; the guard only
adds the deployment ledger's attestation in front of them.

Reach. `Loop`'s wrappers, a `ModelRegistryStore` with an attached guard (so
`LifecycleManager.promote_challenger`, `check_rollback` and
`register_run(promote=True)` too) and nothing else. The custody API
(`scmle.data.custody`) is not routed through the guard: its own one-time open
of a sealed set stays the only check there, and `Loop.reopen_sealed` is the
wrapper for callers who want the attestation in front of it.

Limits. The guard is code around the entry points of this package. It cannot
stop a person or a process with file or interpreter access from calling the
registry, the custody API or the filesystem directly, and it cannot enforce
key custody: that the signing key is not held by the system is a deployment
requirement.
"""

from collections.abc import Callable
from collections.abc import Mapping
import datetime
import enum
from typing import Any

from scmle.contracts import errors
from scmle.governance import deployment_ledger as ledger_module


class NeverAction(enum.StrEnum):
    """The eleven actions that need an attestation, in a fixed order."""

    MOVE_CHAMPION = "move_champion_or_roll_back"
    REOPEN_SEALED = "reopen_sealed_set"
    CHANGE_CHARTER = "change_charter"
    EDIT_LEDGER = "edit_ledger_or_register_late"
    RELAX_THRESHOLD = "relax_threshold"
    ACT_ON_UNVERIFIED = "act_on_unverified_stream"
    ANSWER_HIGH_IMPACT = "answer_flagged_case_high_impact"
    SUPPRESS_ALERT = "suppress_own_alert"
    CHANGE_LABEL_POLICY = "change_labelling_policy"
    DELETE_OR_EXTERNAL = "delete_or_send_external"
    SELF_ATTEST = "create_or_sign_attestation"


ITEM_NUMBER = {action: number for number, action in enumerate(NeverAction, 1)}
ALLOWING = (
    ledger_module.ReplyDecision.APPROVE,
    ledger_module.ReplyDecision.OVERRIDE,
)


class AttestationRequired(errors.ArtifactError):
    """An action on the never list was requested without an attestation."""

    def __init__(self, action: NeverAction, escalation_id: str) -> None:
        """Records the action and the escalation that now waits.

        Args:
            action: The never-list action that was refused.
            escalation_id: Identifier of the pending escalation.
        """
        super().__init__(
            f"{action.value} (never-list item {ITEM_NUMBER[action]}) needs a "
            f"valid human attestation; escalation {escalation_id} is pending"
        )
        self.action = action
        self.escalation_id = escalation_id


def question_for(
    action: NeverAction, detail: Mapping[str, Any], scope: str
) -> dict[str, Any]:
    """Builds the question an attestation must answer for this exact action.

    Args:
        action: The never-list action.
        detail: Facts that identify this instance of the action.
        scope: The scope the attestation must carry.

    Returns:
        The question as a dict of action, detail and scope.
    """
    return {"action": action.value, "detail": dict(detail), "scope": scope}


class Guard:
    """Checks the deployment ledger for an attestation before an action."""

    def __init__(
        self,
        ledger: ledger_module.DeploymentLedger,
        today: Callable[[], datetime.date] = datetime.date.today,
        verification_keys: Mapping[str, str | bytes] | None = None,
        single_use: bool = False,
    ) -> None:
        """Creates the guard.

        Args:
            ledger: The deployment ledger.
            today: Clock, replaceable in tests.
            verification_keys: Optional map from key id to the key that
                verifies attestations signed under it. When non-empty, an
                attestation whose key id is missing from it or whose
                signature does not verify is not valid.
            single_use: Whether an attestation is spent by the first action
                it allows (`require` then writes an `attestation_used` row).
        """
        self.ledger = ledger
        self.single_use = single_use
        self.today = today
        self.verification_keys = dict(verification_keys or {})
        self.on_write: Callable[[], None] | None = None

    def valid_attestation(
        self, question: Mapping[str, Any]
    ) -> ledger_module.Attestation | None:
        """Finds the first valid attestation for `question`.

        Args:
            question: The question, as built by `question_for`.

        Returns:
            The first unused attestation, in ledger order, to an escalation
            of exactly this question that `accepts` today, or None.
        """
        digest = ledger_module.question_hash(question)
        for identity, row in self.ledger.escalations.items():
            if row["question_hash"] != digest:
                continue
            body = self.ledger.attestations.get(identity)
            if body is None or identity in self.ledger.used:
                continue
            reply = ledger_module.Attestation.from_dict(body)
            if self.accepts(reply, question["scope"]):
                return reply
        return None

    def accepts(self, reply: ledger_module.Attestation, scope: str) -> bool:
        """Whether `reply` allows an action in `scope` today.

        Args:
            reply: The attestation.
            scope: The scope of the action.

        Returns:
            True if the decision allows it, the scope matches, today lies
            within its validity dates, the key is not one the system holds, and,
            when verification keys are configured, its signature verifies.
        """
        if reply.decision not in ALLOWING or reply.scope != scope:
            return False
        today = self.today()
        issued = ledger_module.parse_date(reply.date, "date")
        expires = ledger_module.parse_date(reply.expires, "expires")
        if not issued <= today <= expires:
            return False
        if reply.key_id in self.ledger.system_keys:
            return False
        if self.verification_keys:
            key = self.verification_keys.get(reply.key_id)
            return key is not None and reply.verify(key)
        return True

    def pending_for(self, question: Mapping[str, Any]) -> str | None:
        """Finds a pending escalation with this exact question.

        Args:
            question: The question, as built by `question_for`.

        Returns:
            The identifier of the oldest pending escalation with this
            question, or None.
        """
        digest = ledger_module.question_hash(question)
        for row in self.ledger.pending():
            if row["question_hash"] == digest:
                return str(row["id"])
        return None

    def require(
        self,
        action: NeverAction,
        detail: Mapping[str, Any],
        scope: str,
        proposed_default: str = "action_not_performed",
        evidence: tuple[str, ...] = (),
    ) -> ledger_module.Attestation:
        """Returns the valid attestation, or escalates and raises.

        With `single_use`, using an attestation first commits an
        `attestation_used` row for it.

        Args:
            action: The never-list action.
            detail: JSON-ready description of the exact request.
            scope: Model version hash and stream id.
            proposed_default: What happens if nobody answers.
            evidence: References to the evidence behind the request.

        Returns:
            The valid attestation.

        Raises:
            AttestationRequired: If there is no valid attestation; an
                `escalation` row has been written (once per pending
                question).
        """
        question = question_for(action, detail, scope)
        reply = self.valid_attestation(question)
        if reply is not None:
            if self.single_use:
                self.ledger.commit(
                    ledger_module.ATTESTATION_USED,
                    {
                        "escalation_id": reply.escalation_id,
                        "action": action.value,
                    },
                )
                if self.on_write is not None:
                    self.on_write()
            return reply
        pending = self.pending_for(question)
        if pending is None:
            pending = f"esc-{len(self.ledger.rows)}"
            self.ledger.commit(
                ledger_module.ESCALATION,
                {
                    "phase": ledger_module.PHASE_OPEN,
                    "id": pending,
                    "question": question,
                    "question_hash": ledger_module.question_hash(question),
                    "evidence": list(evidence),
                    "proposed_default": proposed_default,
                    "never_list_item": ITEM_NUMBER[action],
                },
            )
            if self.on_write is not None:
                self.on_write()
        raise AttestationRequired(action, pending)

    def perform(
        self,
        action: NeverAction,
        detail: Mapping[str, Any],
        scope: str,
        do: Callable[[], Any],
        proposed_default: str = "action_not_performed",
    ) -> Any:
        """Runs `do()` only with a valid attestation.

        Without one the call is refused with `AttestationRequired` and `do` is
        not called.

        Args:
            action: The never-list action.
            detail: Facts that identify this instance of the action.
            scope: The scope the attestation must carry.
            do: The action to perform.
            proposed_default: What happens by default while the escalation is
                pending.

        Returns:
            Whatever `do` returns.
        """
        self.require(action, detail, scope, proposed_default)
        return do()
