"""Challengers: registration, paired evidence, promotion and rollback."""

from fractions import Fraction
import math
from typing import Any

import numpy

from scmle.assurance.ledger import online_fdr
from scmle.governance import deployment_ledger
from scmle.governance.loop import risk
from scmle.governance.loop import terms
from scmle.monitoring import sequential_tests


class Contest(risk.Risk):
    """Challengers and the champion's rollback watch, decided by e-processes."""

    def enroll(self, name: str) -> int:
        """Registers a challenger in the ledger before it is scored.

        Args:
            name: Name of the challenger.

        Returns:
            The number of challengers now queued, including this one.
        """
        self.ledger.commit(
            deployment_ledger.CHALLENGER_RESULT,
            {
                "name": name,
                "phase": deployment_ledger.REGISTERED,
                "batch": self.batch,
            },
        )
        policy = self.policy
        self.active[name] = {
            "process": sequential_tests.Comparison(
                policy.promotion["delta"],
                policy.alpha,
                policy.scale,
            ),
            "batches": 0,
        }
        self.queue.append(name)
        self.save()
        return len(self.queue)

    def surveil(self, previous: str) -> None:
        """Starts the reverse e-process after a promotion.

        Args:
            previous: Name under which the previous champion's losses are
                passed to `mature`.
        """
        policy = self.policy
        self.watch = {
            "previous": previous,
            "process": sequential_tests.Comparison(
                policy.rollback["delta"], policy.rollback["alpha"], policy.scale
            ),
        }
        self.save()

    def challenge(
        self,
        record: dict[str, Any],
        ids: list[str],
        losses: numpy.ndarray,
        observed: numpy.ndarray,
        model_cases: numpy.ndarray,
    ) -> None:
        """Feeds the e-processes with Horvitz-Thompson paired differences.

        Only challengers registered with the loop, and the previous champion
        under a rollback watch, are fed. A challenger is skipped for this
        batch if it lacks a loss for any labelled model case or the estimate
        has no mean. The batch mean is clamped to the policy's scale, and an
        active challenger's matured-batch count rises by one.

        Args:
            record: The stored batch with its challenger losses.
            ids: Case identifiers of the batch.
            losses: The served model's loss per case.
            observed: Whether each case's label is observed.
            model_cases: Cases that went through the model.
        """
        needed = observed & model_cases
        for name, other in record["challenger_losses"].items():
            holder: Any
            if name in self.active:
                holder = self.active[name]
            elif self.watch is not None and name == self.watch["previous"]:
                holder = self.watch
            else:
                continue
            paired = numpy.array([i in other for i in ids])
            if not paired[needed].all():
                continue
            second = numpy.array(
                [other.get(i, math.nan) for i in ids], dtype=float
            )
            gap = numpy.where(needed, losses - second, 0.0)
            estimate = self.centred(record, gap, observed, model_cases)
            if estimate.mean is None:
                continue
            scale = self.policy.scale
            holder["process"].ingest(min(max(estimate.mean, -scale), scale))
            if "batches" in holder:
                holder["batches"] += 1

    def refit(self, batch: int) -> None:
        """Records a retrain request and registers the hook's challenger.

        If a challenger is already queued, the hook is not called and the
        request is recorded as skipped. Otherwise the ``retrain`` hook, if
        any, is called. A returned name not yet registered in the ledger is
        registered as a challenger.

        Args:
            batch: Index of the batch whose alarm triggered the request.
        """
        name = None
        skipped = None
        if self.queue:
            skipped = "a challenger is already pending"
        elif self.hooks.retrain is not None:
            name = self.hooks.retrain()
        body = {"reason": "L1 alarm", "batch": batch, "challenger": name}
        if skipped is not None:
            body["skipped"] = skipped
        self.ledger.commit(deployment_ledger.RETRAIN, body)
        if name is not None and name not in self.ledger.registered:
            self.enroll(name)

    def propose(self) -> terms.Escalation | None:
        """Turns a crossed reverse process into a rollback proposal.

        If the watch has crossed, commits the proposal, raises a finite
        `tau` by ``step`` (capped at 1; an infinite `tau` is kept) and
        ends the watch.

        Returns:
            The escalation opened for a second consecutive rollback proposal,
            or None if there is none, no proposal was made or an identical
            question is already pending.
        """
        watch = self.watch
        if watch is None or not watch["process"].crossed:
            return None
        self.ledger.commit(
            deployment_ledger.ROLLBACK_PROPOSAL,
            {
                "previous": watch["previous"],
                "e_value": watch["process"].e_value,
                "batch": self.batch - 1,
                "tau_before": None if math.isinf(self.tau) else self.tau,
            },
        )
        if not math.isinf(self.tau):
            self.tau = min(1.0, self.tau + self.policy.step)
        self.watch = None
        self.rollbacks += 1
        if self.rollbacks >= terms.ESCALATING_ROLLBACKS:
            return self.escalate(
                "two_consecutive_rollbacks", {"batch": self.batch - 1}
            )
        return None

    def drain(self) -> terms.Escalation | None:
        """Decides challengers in registration order with e-LOND.

        Decides the first queued challenger once its evidence reaches the
        current e-LOND level or it has used ``budget`` matured
        batches, then the next, and stops at the first that is neither. A
        rejection writes a promotion proposal.

        Returns:
            The newest escalation opened for a challenger that used its budget
            without enough evidence, or None.
        """
        newest: terms.Escalation | None = None
        sequence = self.ledger.elond()
        while self.queue:
            name = self.queue[0]
            item = self.active[name]
            level = sequence.rule.level(sequence.state)
            value = min(item["process"].e_value, 1e300)
            reached = Fraction(value) * level >= 1
            exhausted = item["batches"] >= self.policy.budget
            if not (reached or exhausted):
                break
            step = sequence.rule.step(
                sequence.state,
                online_fdr.Evidence(e=value),
            )
            self.ledger.commit(
                deployment_ledger.CHALLENGER_RESULT,
                {
                    "name": name,
                    "phase": deployment_ledger.DECIDED,
                    "e_value": value,
                    "level": str(step.level),
                    "rejected": step.rejected,
                    "wealth_after": str(step.wealth_after),
                    "matured_batches": item["batches"],
                    "mechanism": self.mechanism.value,
                },
            )
            self.queue.pop(0)
            del self.active[name]
            if step.rejected:
                self.rollbacks = 0
                self.ledger.commit(
                    deployment_ledger.PROMOTION_PROPOSAL,
                    {
                        "name": name,
                        "e_value": value,
                        "level": str(step.level),
                        "wealth_after": str(step.wealth_after),
                        "matured_batches": item["batches"],
                        "needs": "signed registry approval and attestation",
                    },
                )
            elif not reached:
                newest = (
                    self.escalate(
                        "challenger_evidence_insufficient",
                        {"name": name, "matured_batches": item["batches"]},
                    )
                    or newest
                )
        return newest
