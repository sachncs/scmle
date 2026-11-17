"""Risk estimates, threshold calibration and the feedback test."""

import json
from typing import Any

import numpy
import pandas

from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.governance import deployment_ledger
from scmle.governance import escalation
from scmle.governance import never_list
from scmle.governance.loop import desk
from scmle.governance.loop import terms
from scmle.monitoring import labels


class Risk(desk.Desk):
    """Weighted risk estimates and the tests that set or question `tau`."""

    def estimate(
        self,
        record: dict[str, Any],
        losses: numpy.ndarray,
        observed: numpy.ndarray,
        mask: numpy.ndarray | None = None,
    ) -> labels.Estimate:
        """Estimates the Horvitz-Thompson mean loss of a batch or a subset.

        Args:
            record: The stored batch, with its propensities and audit flags.
            losses: The loss of each case.
            observed: Whether each case's label is observed.
            mask: Boolean per case, True for those to include; all if None.

        Returns:
            The estimate of the mean loss with its label mechanism. Its mean
            is None if the mechanism is `unverified`.
        """
        keep = numpy.ones(len(observed), dtype=bool) if mask is None else mask
        propensity = record["propensity"]
        audit = record["audit"]
        return labels.weigh(
            losses[keep],
            observed[keep],
            None
            if propensity is None
            else numpy.asarray(propensity, float)[keep],
            None if audit is None else numpy.asarray(audit, bool)[keep],
            self.policy.epsilon,
            self.policy.loss,
        )

    def centred(
        self,
        record: dict[str, Any],
        values: numpy.ndarray,
        observed: numpy.ndarray,
        mask: numpy.ndarray,
    ) -> labels.Estimate:
        """Like `estimate`, for values in ``[-loss, loss]``.

        Args:
            record: The stored batch, with its propensities and audit flags.
            values: The signed value of each case.
            observed: Whether each case's label is observed.
            mask: Boolean per case, True for those to include.

        Returns:
            The estimate of the mean value with its label mechanism. Its mean
            is None if the mechanism is `unverified`.
        """
        propensity = record["propensity"]
        audit = record["audit"]
        return labels.weigh(
            values[mask],
            observed[mask],
            None
            if propensity is None
            else numpy.asarray(propensity, float)[mask],
            None if audit is None else numpy.asarray(audit, bool)[mask],
            self.policy.epsilon,
            self.policy.loss,
            signed=True,
        )

    def calibrate(
        self,
        score: Any,
        loss: Any,
        weights: Any = None,
        confidence: numpy.ndarray | None = None,  # noqa: ARG002 -- public parameter, documented as not used
    ) -> escalation.Threshold:
        """Selects `tau` on matured calibration cases and applies it.

        The first calibration, and any raise of `tau`, is free. Lowering `tau`
        after the first calibration relaxes the rule and needs an attestation
        (the relax-threshold action; without one the guard raises
        `AttestationRequired` and `tau` is unchanged), except as the
        recalibration that ends a halt, which must itself pass the
        fixed-sequence test (`tau` finite). While the ledger stays halted,
        `tau` is not changed. Every call also clears the in-memory halt that
        a failed ledger verification or a state mismatch sets.

        Args:
            score: The escalation score of each calibration case.
            loss: The loss of each case.
            weights: Optional Horvitz-Thompson weights ``1 / propensity``.
            confidence: Not used.

        Returns:
            The result of the threshold selection. It is written to
            `risk_coverage.json` first, then to the ledger: as the halt
            release when it ends a halt, otherwise as an observation summary.
        """
        result = escalation.select(
            score, loss, self.policy.target, self.policy.delta, weights
        )
        curve = escalation.risk_coverage(score, loss, weights)
        (self.directory / terms.RISK_COVERAGE_FILE).write_text(
            json.dumps(
                terms.jsonify(
                    {"curve": curve.to_dict(), "threshold": result.to_dict()}
                )
            ),
            encoding="utf-8",
            newline="\n",
        )
        released = False
        if self.ledger.halted and not result.saturated:
            self.ledger.commit(
                deployment_ledger.HALT,
                {
                    "phase": deployment_ledger.RELEASE,
                    "recalibration": terms.jsonify(result.to_dict()),
                    "batch": self.batch,
                },
            )
            self.halt_reason = None
            released = True
        elif result.tau < self.tau and self.calibrated:
            self.guard.require(
                never_list.Forbidden.RELAX_THRESHOLD,
                {"tau_from": self.tau, "tau_to": result.tau},
                self.policy.scope,
            )
        if not released:
            self.ledger.commit(
                deployment_ledger.OBSERVE_SUMMARY,
                {
                    "calibration": terms.jsonify(result.to_dict()),
                    "batch": self.batch,
                },
            )
        if not self.ledger.halted or released:
            self.tau = result.tau
        self.calibrated = True
        self.halted_by_state = (
            self.halted_by_state and released is False and False
        )
        self.save_state()
        return result

    def performativity(
        self,
        bypass_features: pandas.DataFrame,
        model_features: pandas.DataFrame,
    ) -> qualification.Check:
        """Tests whether the model's decisions shift the next-period features.

        Runs `labels.circularity` on the next-period features of the
        randomized bypass slice against those of model-decided cases, records
        the result in the ledger and in `label_mechanism.json`, and escalates
        if a feedback loop is detected. The test is not run by `step`: the
        next-period features are the caller's to supply.

        Args:
            bypass_features: Next-period features of the bypass slice.
            model_features: Next-period features of model-decided cases.

        Returns:
            The bypass test's result, as a check with its statistic and
            p-value.

        Raises:
            ValueError: If the policy declares no bypass slice.
        """
        if self.policy.bypass <= 0.0:
            raise errors.InputError(
                "no bypass slice is declared (policy.bypass is 0): "
                "feedback cannot be tested"
            )
        check = labels.circularity(
            bypass_features, model_features, seed=len(self.ledger.rows)
        )
        verdict = labels.assessment(self.policy.bypass, check)
        self.feedback = {
            "verdict": verdict,
            "p_value": check.p_value,
            "statistic": check.statistic,
            "n_bypass": len(bypass_features),
            "n_model": len(model_features),
            "batch": self.batch,
        }
        self.ledger.commit(
            deployment_ledger.OBSERVE_SUMMARY,
            {"feedback_test": terms.jsonify(self.feedback)},
        )
        if verdict == "loop_detected":
            self.escalate(
                "feedback_loop_detected",
                {"p_value": check.p_value, "batch": self.batch},
                proposed_default=self.policy.fallback,
            )
        self.write_label_mechanism()
        self.save_state()
        return check

    def write_label_mechanism(self) -> None:
        """Writes `label_mechanism.json`."""
        record = {
            "mechanism": self.mechanism.value,
            "actions_permitted": labels.actionable(self.mechanism),
            "epsilon": self.policy.epsilon,
            "bypass_rate": self.policy.bypass,
            "feedback": (
                labels.UNVERIFIED
                if self.policy.bypass <= 0.0
                else (
                    "bypass slice declared"
                    if self.feedback is None
                    else self.feedback["verdict"]
                )
            ),
            "feedback_test": self.feedback,
            "feedback_note": (
                labels.FEEDBACK_UNVERIFIED if self.policy.bypass <= 0.0 else ""
            ),
            "outcome_window": self.policy.maturity,
        }
        (self.directory / terms.LABEL_MECHANISM_FILE).write_text(
            json.dumps(record, sort_keys=True), encoding="utf-8", newline="\n"
        )
