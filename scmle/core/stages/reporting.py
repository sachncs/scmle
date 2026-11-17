"""The coverage pledge and the decision report."""

from typing import Any

import numpy
import pandas

from scmle.assurance import coverage_run
from scmle.assurance.charter import sealing
import scmle.assurance.ledger.ledger
from scmle.contracts import tasks
import scmle.core.context
from scmle.core.stages import relabelling
from scmle.core.stages import scoring
from scmle.decision import reports
from scmle.evaluation import coverage


class Reporting(relabelling.Relabelling):
    """The coverage claim pledged and the decision files written.

    Part of `FinalEvaluationStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def pledge(self, context: scmle.core.context.Context) -> None:
        """Fixes and hashes the coverage claim before the final rows open.

        Written to `coverage_claim.json`, announced by a `coverage.registered`
        event and, in a charter run, appended to the custody log. A task that
        stores no interval or set has no claim, and nothing is written.

        Args:
            context: The run context that stores the claim and its hash.
        """
        split = scmle.core.context.require(context.split, "split")
        charter_hash = None
        if context.spec.charter_path:
            charter_hash = sealing.read_verified(
                str(context.spec.charter_path)
            ).charter_hash
        claim = coverage_run.claim_document(
            task=scmle.core.context.require(context.task, "task").value,
            levels=scoring.COVERAGE_LEVELS,
            tolerance=coverage.DEFAULT_TOLERANCE,
            split_fingerprint=split.manifest.fingerprint,
            charter_hash=charter_hash,
            slice_columns=scoring.slicers(context),
        )
        if claim is None:
            return
        sha = scmle.assurance.ledger.ledger.digest(claim)
        context.write_json(coverage_run.CLAIM_FILE, claim)
        context.coverage_claim_sha256 = sha
        vault = context.custody_vault
        if vault is not None:
            vault.log.append(
                "final_test",
                coverage_run.REGISTRATION_EVENT,
                "scmle.coverage",
                {"sha256": sha},
            )
        context.emit(
            "coverage.registered",
            "coverage claim fixed before the final test rows are released",
            (coverage_run.CLAIM_FILE,),
            {"sha256": sha},
        )

    def notarize(self, context: scmle.core.context.Context) -> None:
        """Appends the hashes of the four decision files to the custody log.

        A file that does not exist is logged with a null hash. Does nothing
        without a custody vault or a stored run.

        Args:
            context: The run context, holding the vault and the run directory.
        """
        vault = context.custody_vault
        if vault is None or context.store is None:
            return
        vault.log.append(
            "final_test",
            reports.REGISTRATION_EVENT,
            "scmle.decision",
            {"files": reports.hashsums(context.store.directory)},
        )

    def decide(
        self,
        context: scmle.core.context.Context,
        pipeline: Any,
        features: pandas.DataFrame,
        target: pandas.Series,
        decision: dict[str, Any] | None,
        net_benefit: dict[str, Any] | None,
    ) -> None:
        """Writes the policy value, fairness satisfaction and rollout design.

        Runs only when the charter has a decision layer and cost model and the
        task is binary, on the locked rows that were just opened. It writes
        `policy_value.json` and `decision_report.json`; the threshold and
        net-benefit files if those are given; and `rollout_design.json` and
        `rollout_decisions.jsonl` if the charter yields a rollout design. It
        then logs the hashes of the decision files and emits an event.

        Args:
            context: The run context that stores the reports.
            pipeline: The fitted final model.
            features: The released rows.
            target: The labels of the released rows.
            decision: The chartered threshold description, or None.
            net_benefit: The net-benefit report, or None.
        """
        charter = scoring.decisional(context)
        if charter is None or charter.decision is None:
            return
        if (
            scmle.core.context.require(context.task, "task")
            is not tasks.Task.BINARY
        ):
            return
        layer = charter.decision
        data = context.spec.data
        positive = pipeline.classes_[1]
        outcome = (target == positive).to_numpy(dtype=float)
        actions = (
            numpy.asarray(pipeline.predict(features)) == positive
        ).astype(float)
        logged = (
            features[data.logged_action_column].to_numpy(dtype=float)
            if data.logged_action_column
            and data.logged_action_column in features
            else None
        )
        propensity = (
            features[data.propensity_column].to_numpy(dtype=float)
            if data.propensity_column and data.propensity_column in features
            else None
        )
        value = reports.policy_value(
            layer, actions, outcome, logged, propensity
        )
        satisfaction = reports.fairness(
            layer,
            features,
            outcome,
            actions,
            float(charter.error_budget.confirmation.alpha_confirm),
        )
        design_record = None
        built = reports.rollout_design(
            charter, float(numpy.mean(outcome)) if len(outcome) else 0.0
        )
        report = reports.decision_report(
            charter,
            decision,
            net_benefit,
            value,
            satisfaction,
            None if built is None else built[0],
        )
        context.write_json(reports.POLICY_FILE, value)
        written = ["decision_report.json", reports.POLICY_FILE]
        if decision is not None:
            context.write_json(reports.THRESHOLD_FILE, decision)
            written.append(reports.THRESHOLD_FILE)
        if net_benefit is not None:
            context.write_json(reports.NET_BENEFIT_FILE, net_benefit)
            written.append(reports.NET_BENEFIT_FILE)
        if built is not None:
            design_record, log = built
            context.write_json("rollout_design.json", design_record)
            if context.store is not None:
                log.write(context.store.directory / "rollout_decisions.jsonl")
            written += ["rollout_design.json", "rollout_decisions.jsonl"]
        context.write_json("decision_report.json", report)
        self.notarize(context)
        context.emit(
            "decision.reported",
            "policy value, fairness satisfaction and rollout design written",
            tuple(written),
            {
                "policy_value": value.get("status"),
                "fairness": [f["status"] for f in satisfaction],
            },
        )
