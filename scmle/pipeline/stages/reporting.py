"""The coverage pledge and the decision report."""

from typing import Any

import numpy
import pandas

from scmle.assurance import coverage_run
from scmle.assurance.charter import sealing
import scmle.assurance.ledger.ledger
from scmle.contracts import records
from scmle.contracts import tasks
from scmle.decision import reports
from scmle.evaluation import coverage
from scmle.pipeline import engine
from scmle.pipeline.stages import examination
from scmle.pipeline.stages import exploration
from scmle.pipeline.stages import profiling
from scmle.pipeline.stages import relabelling
from scmle.pipeline.stages import scoring


class Reporting(relabelling.Relabelling):
    """The coverage claim pledged and the decision files written.

    Part of `Assessment`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def pledge(self, board: engine.Board) -> str | None:
        """Fixes and hashes the coverage claim before the final rows open.

        Written to `coverage_claim.json`, announced by a `coverage.registered`
        event and, in a charter run, appended to the custody log. A task that
        stores no interval or set has no claim, and nothing is written.

        Args:
            board: The run's board whose journal stores the claim.

        Returns:
            The claim's hash (`examination.Decision.claim`), or None when the
            task makes no claim.
        """
        partition = board.get(profiling.Partition)
        split = partition.split
        charter_hash = None
        if board.services.spec.charter_path:
            charter_hash = sealing.read(
                str(board.services.spec.charter_path)
            ).digest
        claim = coverage_run.claim(
            task=partition.task.value,
            levels=scoring.COVERAGE_LEVELS,
            tolerance=coverage.DEFAULT_TOLERANCE,
            split_fingerprint=split.manifest.fingerprint,
            charter_hash=charter_hash,
            slice_columns=scoring.slicers(board),
        )
        if claim is None:
            return None
        sha = scmle.assurance.ledger.ledger.digest(claim)
        board.journal.write(coverage_run.CLAIM_FILE, claim)
        custody = board.find(exploration.Custody)
        if custody is not None:
            custody.vault.log.append(
                "final_test",
                coverage_run.REGISTRATION_EVENT,
                "scmle.coverage",
                {"sha256": sha},
            )
        board.journal.emit(
            "coverage.registered",
            "coverage claim fixed before the final test rows are released",
            (coverage_run.CLAIM_FILE,),
            {"sha256": sha},
        )
        return sha

    def notarize(self, board: engine.Board) -> None:
        """Appends the hashes of the four decision files to the custody log.

        A file that does not exist is logged with a null hash. Does nothing
        without a custody vault or a stored run.

        Args:
            board: The run's board, holding the vault and the run directory.
        """
        custody = board.find(exploration.Custody)
        if custody is None or board.services.store is None:
            return
        custody.vault.log.append(
            "final_test",
            reports.REGISTRATION_EVENT,
            "scmle.decision",
            {"files": reports.hashsums(board.services.store.directory)},
        )

    def decide(
        self,
        board: engine.Board,
        pipeline: Any,
        features: pandas.DataFrame,
        target: pandas.Series,
        decision: examination.Decision,
    ) -> None:
        """Writes the policy value, fairness satisfaction and rollout design.

        Runs only when the charter has a decision layer and cost model and the
        task is binary, on the locked rows that were just opened. It writes
        `policy_value.json` and `decision_report.json`; the threshold and
        net-benefit files if those are given; and `rollout_design.json` and
        `rollout_decisions.jsonl` if the charter yields a rollout design. It
        then logs the hashes of the decision files and emits an event.

        Args:
            board: The run's board that stores the reports.
            pipeline: The fitted final model.
            features: The released rows.
            target: The labels of the released rows.
            decision: The chartered threshold and its net-benefit report.
        """
        charter = scoring.decisional(board)
        if charter is None or charter.decision is None:
            return
        if board.get(profiling.Partition).task is not tasks.Task.BINARY:
            return
        layer = charter.decision
        data = board.services.spec.data
        positive = pipeline.classes_[1]
        outcome = (target == positive).to_numpy(dtype=float)
        actions = (
            numpy.asarray(pipeline.predict(features)) == positive
        ).astype(float)
        logged = self.column(features, data.logged_action_column)
        propensity = self.column(features, data.propensity_column)
        value = reports.value(layer, actions, outcome, logged, propensity)
        satisfaction = reports.fairness(
            layer,
            features,
            outcome,
            actions,
            float(charter.budget.confirmation.alpha_confirm),
        )
        built = reports.plan(
            charter, float(numpy.mean(outcome)) if len(outcome) else 0.0
        )
        report = reports.report(
            charter,
            decision.threshold,
            decision.benefit,
            value,
            satisfaction,
            None if built is None else built[0],
        )
        written = self.deposit(
            board, value, decision.threshold, decision.benefit, built
        )
        board.journal.write("decision_report.json", report)
        self.notarize(board)
        board.journal.emit(
            "decision.reported",
            "policy value, fairness satisfaction and rollout design written",
            tuple(written),
            {
                "policy_value": value.get("status"),
                "fairness": [f["status"] for f in satisfaction],
            },
        )

    @staticmethod
    def column(features: pandas.DataFrame, name: str | None) -> Any:
        """A declared column of the released rows as floats, if present.

        Args:
            features: The released rows.
            name: The declared column, or None.

        Returns:
            The column's values, or None when it is not declared or absent.
        """
        if name and name in features:
            return features[name].to_numpy(dtype=float)
        return None

    def deposit(
        self,
        board: engine.Board,
        value: records.Layout,
        decision: records.Layout | None,
        net_benefit: records.Layout | None,
        built: Any,
    ) -> list[str]:
        """Writes the policy value, threshold, net-benefit and rollout files.

        Args:
            board: The run's board that stores the files.
            value: The policy value.
            decision: The chartered threshold description, or None.
            net_benefit: The net-benefit report, or None.
            built: The rollout design and its decision log, or None.

        Returns:
            The decision files, with `decision_report.json` (written later)
            first.
        """
        board.journal.write(reports.POLICY_FILE, value)
        written = ["decision_report.json", reports.POLICY_FILE]
        if decision is not None:
            board.journal.write(reports.THRESHOLD_FILE, decision)
            written.append(reports.THRESHOLD_FILE)
        if net_benefit is not None:
            board.journal.write(reports.NET_BENEFIT_FILE, net_benefit)
            written.append(reports.NET_BENEFIT_FILE)
        if built is not None:
            design, log = built
            board.journal.write("rollout_design.json", design)
            if board.services.store is not None:
                log.write(
                    board.services.store.directory / "rollout_decisions.jsonl"
                )
            written += ["rollout_design.json", "rollout_decisions.jsonl"]
        return written
