"""Questions for a person: escalations, halts, replies and timeouts."""

from collections.abc import Callable
from collections.abc import Mapping
from typing import Any

import scmle.assurance.ledger.ledger
from scmle.contracts import qualification
from scmle.governance import deployment_ledger
from scmle.governance import never_list
from scmle.governance.loop import memory
from scmle.governance.loop import terms


class Desk(memory.Memory):
    """Opening, answering and timing out the questions a person must decide."""

    def survey(self) -> list[terms.Escalation]:
        """Lists the escalations with no reply and no timeout, oldest first.

        Returns:
            The unanswered, un-timed-out escalations, oldest first.
        """
        return [
            terms.Escalation(
                id=str(row["id"]),
                question=row["question"],
                question_hash=str(row["question_hash"]),
                evidence=tuple(row["evidence"]),
                proposed_default=str(row["proposed_default"]),
            )
            for row in self.ledger.pending()
        ]

    def escalate(
        self,
        action: never_list.Forbidden | str,
        detail: Mapping[str, Any],
        evidence: tuple[str, ...] = (),
        proposed_default: str | None = None,
    ) -> terms.Escalation | None:
        """Writes an escalation row for a system-level question.

        Args:
            action: The never-list action or other question name.
            detail: Facts that identify the question.
            evidence: Names of the evidence files the person should read.
            proposed_default: What happens if nobody answers; the policy's
                safe default if None.

        Returns:
            The new escalation, or None if an identical question is already
            pending.
        """
        name = (
            action.value if isinstance(action, never_list.Forbidden) else action
        )
        question = {
            "action": name,
            "detail": terms.jsonify(dict(detail)),
            "scope": self.policy.scope,
        }
        if self.guard.outstanding(question) is not None:
            return None
        identity = f"esc-{len(self.ledger.rows)}"
        self.ledger.commit(
            deployment_ledger.ESCALATION,
            {
                "phase": deployment_ledger.PHASE_OPEN,
                "id": identity,
                "question": question,
                "question_hash": scmle.assurance.ledger.ledger.digest(question),
                "evidence": list(evidence),
                "proposed_default": proposed_default or self.policy.fallback,
            },
        )
        self.escalation[identity] = self.batch
        return next(e for e in self.survey() if e.id == identity)

    def suspend(self, reason: str, detail: Mapping[str, Any]) -> None:
        """Writes a halt row and an incident row, then calls the incident hook.

        Does nothing if the ledger is already halted. Otherwise it commits a
        halt row, records `reason` as the halt reason, writes the default
        `incident` row and calls the ``open_incident`` hook, if one was given.

        Args:
            reason: Why the loop halts.
            detail: Facts recorded with the halt.
        """
        if self.ledger.halted:
            return
        self.ledger.commit(
            deployment_ledger.HALT,
            {
                "phase": deployment_ledger.ENTER,
                "reason": reason,
                "batch": self.batch,
                "detail": terms.jsonify(dict(detail)),
            },
        )
        self.reason = reason
        self.record(reason)
        if self.hooks.open_incident is not None:
            self.hooks.open_incident(reason)

    def record(self, reason: str) -> None:
        """Writes an `incident` ledger row with status ``open``.

        It records that a person must look; it is not the SLO incident
        workflow of `scmle.monitoring.incidents`, which needs an SLO
        evaluation that a halt does not have.

        Args:
            reason: Why the incident is opened.
        """
        self.ledger.commit(
            deployment_ledger.INCIDENT,
            {"status": "open", "reason": reason, "batch": self.batch},
        )

    def reply(
        self,
        escalation_id: str,
        attestation: deployment_ledger.Attestation,
        key: str | bytes | None = None,
    ) -> terms.Consequence:
        """Records a person's signed reply to a pending escalation.

        On success the attestation is committed as a ledger row, which closes
        the escalation.

        Args:
            escalation_id: The escalation answered.
            attestation: The signed reply.
            key: Key that verifies the signature; defaults to the one
                registered for the reply's `key_id` at `open`.

        Returns:
            The outcome: tier `attested` if the person approved or overrode,
            else `unverified` with the safe default applying; also the
            decision itself.

        Raises:
            RefusalError: If the escalation is not pending, the reply is
                bound to another escalation, question or scope, was signed
                with a key the system holds (or is verified by one), no
                verification key is known for its key id, it does not verify,
                or it is expired or dated in the future. A malformed date
                raises the ledger's `MisuseError` instead.
        """
        pending = {e.id: e for e in self.survey()}
        target = pending.get(escalation_id)
        if target is None:
            raise terms.RefusalError(
                f"{escalation_id!r} is not a pending escalation"
            )
        if attestation.escalation != escalation_id:
            raise terms.RefusalError("the reply answers a different escalation")
        if attestation.digests["question"] != target.question_hash:
            raise terms.RefusalError(
                "the reply is bound to a different question"
            )
        if attestation.scope != target.question["scope"]:
            raise terms.RefusalError("the reply names a different scope")
        if attestation.key_id in self.ledger.keys:
            raise terms.RefusalError(
                "the reply is signed with a key the system holds"
            )
        verifier = key if key is not None else self.keys.get(attestation.key_id)
        if verifier is None:
            raise terms.RefusalError(
                "no verification key for the reply's key id"
            )
        if deployment_ledger.thumbprint(verifier) in self.ledger.keys:
            raise terms.RefusalError(
                "the verification key is one the system holds"
            )
        if not attestation.verify(verifier):
            raise terms.RefusalError("the reply's signature does not verify")
        today = self.today()
        issued = deployment_ledger.parse(attestation.date, "date")
        expires = deployment_ledger.parse(attestation.expires, "expires")
        if not issued <= today <= expires:
            raise terms.RefusalError(
                "the reply is expired or dated in the future"
            )
        self.ledger.commit(deployment_ledger.ATTESTATION, attestation.to_dict())
        self.save()
        allowed = attestation.decision in never_list.ALLOWING
        return terms.Consequence(
            qualification.Tier.ATTESTED
            if allowed
            else qualification.Tier.UNVERIFIED,
            attestation.decision,
            not allowed,
        )

    def perform(
        self,
        action: never_list.Forbidden,
        detail: Mapping[str, Any],
        do: Callable[[], Any],
        proposed_default: str = "action_not_performed",
    ) -> Any:
        """Runs `do()` only with a valid attestation for this exact action.

        Without one the guard raises `AttestationRequired` and `do` is not
        called; an `escalation` row is written unless an identical question
        is already pending. The loop state is saved either way.

        Args:
            action: The never-list action.
            detail: Facts that identify this instance of the action.
            do: The action to perform.
            proposed_default: What happens by default while the escalation is
                pending.

        Returns:
            Whatever `do` returns.
        """
        try:
            return self.guard.perform(
                action, detail, self.policy.scope, do, proposed_default
            )
        finally:
            self.save()

    def expire(self) -> terms.Escalation | None:
        """Closes escalations unanswered for ``timeout`` batches.

        Each is closed with a timeout row that applies its own proposed
        default (the safe default unless the escalation proposed another).
        An escalation the loop has no opening batch for counts from the
        current batch.

        Returns:
            Always None; the closed escalations are recorded in the ledger.
        """
        for row in list(self.ledger.pending()):
            identity = str(row["id"])
            opened = self.escalation.setdefault(identity, self.batch)
            if self.batch - opened >= self.policy.timeout:
                self.ledger.commit(
                    deployment_ledger.ESCALATION,
                    {
                        "phase": deployment_ledger.PHASE_TIMEOUT,
                        "id": identity,
                        "applied": row["proposed_default"],
                        "tier": qualification.Tier.UNVERIFIED.value,
                    },
                )
        return None
