"""Retraining: the policy applied to a window, and approved retrains."""

from collections.abc import Sequence

from scmle.contracts import errors
from scmle.governance import deployment_ledger
from scmle.governance import never_list
from scmle.operations import signals
from scmle.operations.continuous import feeding
from scmle.operations.continuous import policies
from scmle.operations.continuous import records


class Retraining(feeding.Feeding):
    """Applies the retraining policy and starts challengers."""

    def review(
        self,
        rows: Sequence[signals.Signal],
        age_days: float | None = None,
    ) -> policies.RetrainDecision:
        """Applies the policy to a window of signals.

        A proposal always goes through `Loop.escalate`. In ``trigger`` mode a
        challenger is also started through `train` at once (which
        starts none without a retrain hook or while one is already queued).
        A review that makes no proposal is written to the history as
        ``retrain_review`` and changes no state; a proposal is written as
        ``retrain_decision`` and moves the alarm count and the cooldown
        reference to the present.

        Args:
            rows: The window's signals.
            age_days: Age of the model in days, if known.

        Returns:
            The decision: action ``none`` (with the reason for holding back
            if there was a primary reason), or the policy's mode with the
            pending escalation's id and, in trigger mode, the challenger's
            name if one was started.
        """
        alarms = sum(
            1
            for r in self.loop.ledger.rows
            if r["kind"] == deployment_ledger.ALARM
        )
        fresh_alarm = alarms > self.seen_alarms
        batch = self.loop.batch
        since = (
            None
            if self.last_proposal_batch is None
            else batch - self.last_proposal_batch
        )
        primary, support, held = policies.decide(
            self.policy, rows, fresh_alarm, age_days, since
        )
        gates = self.loop.policy.retrain
        reasons = tuple(primary + support)
        if not primary or held is not None:
            decision = policies.RetrainDecision(
                "none", reasons, held if primary else None, None, None, gates
            )
            self.history.record(
                "retrain_review",
                self.clock(),
                batch=batch,
                **decision.to_dict(),
            )
            return decision
        self.seen_alarms = alarms
        pending = self.loop.escalate(
            policies.RETRAIN_QUESTION,
            {"reasons": list(reasons), "mode": self.policy.mode},
            evidence=("retained_outcomes.jsonl", records.HISTORY_FILE),
            proposed_default=policies.DEFAULT_IF_UNANSWERED,
        )
        identity = pending.id if pending is not None else self.find()
        challenger = None
        if self.policy.mode == policies.TRIGGER:
            challenger = self.train(reasons)
        decision = policies.RetrainDecision(
            self.policy.mode, reasons, None, identity, challenger, gates
        )
        self.last_proposal_batch = batch
        self.history.record(
            "retrain_decision", self.clock(), batch=batch, **decision.to_dict()
        )
        return decision

    def find(self) -> str | None:
        """Id of the pending retrain question, if one is open.

        Returns:
            The identifier of the open retrain escalation, or None.
        """
        for item in self.loop.pending_escalations():
            if item.question.get("action") == policies.RETRAIN_QUESTION:
                return item.id
        return None

    def train(self, reasons: Sequence[str]) -> str | None:
        """Trains a challenger with the hook and registers it with the loop.

        Args:
            reasons: Why the challenger is trained, recorded in the ledger.

        Returns:
            The challenger's name, or None if there is no hook or the loop
            already has a challenger waiting.
        """
        if self.retrain is None or self.loop.queue:
            return None
        name = self.retrain()
        if name is None:
            return None
        self.loop.ledger.commit(
            deployment_ledger.RETRAIN,
            {
                "reason": "operations retraining policy",
                "mode": self.policy.mode,
                "batch": self.loop.batch,
                "challenger": name,
                "reasons": list(reasons),
            },
        )
        if name not in self.loop.ledger.registered:
            self.loop.enroll(name)
        return name

    def run_approved_retrain(self, escalation_id: str) -> str | None:
        """Trains a challenger once a person has approved the recommendation.

        Args:
            escalation_id: The retrain escalation that was answered.

        Returns:
            The challenger's name, or None if none was started.

        Raises:
            ValueError: If the ledger holds no attestation for this
                escalation whose decision approves or overrides. Only the
                ledger row is looked at; its signature and expiry are not
                checked here.
        """
        for row in self.loop.ledger.rows:
            if row["kind"] != deployment_ledger.ATTESTATION:
                continue
            reply = deployment_ledger.Attestation.from_dict(row["body"])
            if (
                reply.escalation == escalation_id
                and reply.decision in never_list.ALLOWING
            ):
                return self.train(("approved recommendation",))
        raise errors.InputError(f"{escalation_id} has no approving reply")
