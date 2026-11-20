"""Alarm decisions: halts, investigations and replaced monitors."""

from scmle.governance import deployment_ledger
from scmle.governance.loop import contest
from scmle.governance.loop import terms
from scmle.monitoring import sequential_tests


class Alarms(contest.Contest):
    """What a crossed monitor leads to."""

    def alert(self, batch: int, verified: bool) -> terms.Escalation | None:
        """Acts on the loss alarms (L1, naive) and the selective-risk alarm.

        A naive alarm under an unverified mechanism halts and escalates; an
        L1 alarm writes an alarm row, then an investigation escalation or a
        retrain request, by ``retrain``; a selective-risk alarm
        halts and escalates. Each crossed alarm is replaced by a fresh one.

        Args:
            batch: Index of the newest observed batch.
            verified: Whether the label mechanism permits action.

        Returns:
            The newest escalation opened, or None if none was opened.
        """
        newest: terms.Escalation | None = None
        if self.naive_flag and not verified:
            self.naive_flag = False
            self.ledger.commit(
                deployment_ledger.ALARM,
                {
                    "rule": "L1_naive",
                    "reason": terms.HALT_UNVERIFIED,
                    "batch": batch,
                },
            )
            self.suspend(terms.HALT_UNVERIFIED, {"batch": batch})
            newest = (
                self.escalate(
                    "l1_alarm_unverified_label_mechanism",
                    {"batch": batch, "mechanism": self.mechanism.value},
                )
                or newest
            )
        if verified and bool(self.alarm.test.crossed):
            self.ledger.commit(
                deployment_ledger.ALARM,
                {
                    "rule": "L1",
                    "reason": "mean loss above reference plus tolerance",
                    "batch": batch,
                    "e_value": float(self.alarm.test.e_value),
                    "mechanism": self.mechanism.value,
                },
            )
            if not self.policy.retrain:
                newest = (
                    self.escalate(
                        "l1_alarm_investigate",
                        {"batch": batch, "reason": "alarm is not certified"},
                    )
                    or newest
                )
            else:
                self.refit(batch)
            self.alarm = sequential_tests.Tripwire(
                self.policy.reference,
                self.policy.tolerance,
                self.policy.alarm,
                self.policy.scale,
            )
        if verified and bool(self.selective.test.crossed):
            self.ledger.commit(
                deployment_ledger.ALARM,
                {
                    "rule": "selective_risk",
                    "reason": terms.HALT_SELECTIVE,
                    "batch": batch,
                },
            )
            self.suspend(terms.HALT_SELECTIVE, {"batch": batch})
            newest = (
                self.escalate("halt_selective_risk", {"batch": batch}) or newest
            )
            self.selective = sequential_tests.Tripwire(
                self.policy.target, 0.0, self.policy.delta, self.policy.scale
            )
        return newest

    def screen(self, batch: int) -> terms.Escalation | None:
        """Escalates a crossed coverage or novelty test and replaces it.

        Args:
            batch: Index of the newest observed batch.

        Returns:
            The newest escalation opened, or None if none was opened.
        """
        newest: terms.Escalation | None = None
        if self.test is not None and bool(self.test.crossed):
            newest = (
                self.escalate("coverage_below_nominal", {"batch": batch})
                or newest
            )
            self.test = sequential_tests.Mixture(
                self.policy.coverage or 0.9,
                sequential_tests.LESS,
                self.policy.alarm,
            )
        if bool(self.novelty.crossed):
            newest = (
                self.escalate(
                    "out_of_support_rate_above_bound", {"batch": batch}
                )
                or newest
            )
            self.novelty = sequential_tests.Mixture(
                min(max(self.policy.novelty, 1e-6), 1 - 1e-6),
                sequential_tests.GREATER,
                self.policy.alarm,
            )
        return newest
