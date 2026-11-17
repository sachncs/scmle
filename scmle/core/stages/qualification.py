"""Robustness qualification of a charter run's exploration."""

from typing import Any

from scmle.assurance.charter import contracts
from scmle.assurance.ledger import confirm
from scmle.assurance.ledger import explore
from scmle.assurance.ledger import governed
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.core import ingest_stage
import scmle.core.context
from scmle.core.stages import screening
from scmle.data.admissibility import qualify
import scmle.data.framing.claim
from scmle.data.readers import tabular
from scmle.data.robustness import html_report
from scmle.data.robustness import prefilter
import scmle.data.robustness.qualify
import scmle.data.robustness.report
import scmle.data.robustness.stability


class Qualification(screening.Screening):
    """Robustness and stability qualification of the exploration.

    Part of `ProfileStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def audit(
        self,
        context: scmle.core.context.Context,
        run: governed.Governance,
        found: qualify.Qualification,
    ) -> None:
        """Requires the ledger to hold exactly the declared synthesized family.

        Does nothing when the run synthesized no features. A mismatch between
        the declared family size and the registered hypotheses blocks the run
        (the multiplicity check raises `BlockedError`).

        Args:
            context: The run context, holding the ingest result.
            run: The governed run whose ledger registered the hypotheses.
            found: The exploration qualification, whose quarantined columns
                are exempt.
        """
        result = context.ingest
        if result is None:
            return
        columns = tuple(str(c) for c in result.features.columns)
        registered = {
            h.spec.column
            for h in run.ledger.hypotheses.values()
            if h.spec.column
        }
        skipped = {pair[0] for pair in run.plan.skipped}
        ingest_stage.plurality(
            result.declared,
            columns,
            registered,
            set(found.quarantined) | skipped,
        )

    def publish(
        self,
        context: scmle.core.context.Context,
        run: governed.Governance,
        runner: confirm.Docket | None,
    ) -> None:
        """Writes the report files of the registered claims.

        If `C` could not test the baseline claim (no confirmation set, or a
        failure), its document is written as not evaluated rather than left
        out. The emitted event lists the status of every registered claim.

        Args:
            context: The run context that stores the documents.
            run: The governed run, holding the confirmation's documents.
            runner: The claim runner, or None if no claim was registered, in
                which case nothing is written.
        """
        if runner is None:
            return
        documents = dict(run.confirmation.documents)
        status = {
            claim_id: record["status"]
            for claim_id, record in (run.confirmation.registered or {}).items()
        }
        if scmle.data.framing.claim.DOCUMENT not in documents and any(
            i.kind == scmle.data.framing.claim.KIND for i in runner.registered
        ):
            documents[scmle.data.framing.claim.DOCUMENT] = {
                "schema_version": 1,
                "status": confirm.NOT_EVALUATED,
                "issued": False,
                "reasons": [
                    "the confirmation set was not available to test the claim"
                ],
            }
        for name, document in documents.items():
            context.write_json(name, document)
        context.emit(
            "claims.registered",
            "registered claims were evaluated on the one opening of C",
            tuple(documents),
            {"status": status},
        )

    def stress(
        self,
        context: scmle.core.context.Context,
        task: tasks.Task,
        charter: contracts.AnalysisCharter,
        view: Any,
        survivors: tuple[str, ...],
    ) -> scmle.data.robustness.qualify.Robustness:
        """Assesses the stability of the survivors and shift from a reference.

        Reads the exploration view and the data spec's reference sample, if
        any; never `C` and never `F`. It runs between exploration and the
        opening of `C`, so its inputs are exploration rows only.

        Args:
            context: The run context, holding the profile and specification.
            task: The resolved task.
            charter: The analysis charter, supplying the stability rule, seed
                and multiverse choices.
            view: The exploration view (features and target).
            survivors: Columns that survived exploration.

        Returns:
            The stability and shift assessment of the survivors.
        """
        data = context.spec.data
        profile = scmle.core.context.require(context.profile, "profile")
        plan = profile.feature_plan
        candidates = [*plan.numeric, *plan.categorical]
        codes = explore.vectorize(view.target, task.is_classification)
        kind = explore.target_kind(view.target, task.is_classification)
        reference = (
            tabular.read_table(data.reference_uri)
            if data.reference_uri
            else None
        )
        rules = charter.rules
        return scmle.data.robustness.qualify.run(
            view.features,
            codes,
            survivors=list(survivors),
            multiclass=kind == explore.MULTICLASS,
            choices=charter.multiverse,
            seed=charter.seed,
            permutations=charter.error_budget.permutations_b,
            strata=governed.strata_labels(
                view.features, data.group_column, data.time_column
            ),
            reference=reference,
            candidates=candidates,
            sensitive=tuple(data.sensitive_columns),
            temporal=bool(data.time_column),
            rule=(
                scmle.data.robustness.stability.DEFAULT_RULE
                if rules is None
                else scmle.data.robustness.stability.Thresholds(
                    rules.s_min, rules.band, rules.q_min
                )
            ),
        )

    def assess_robustness(
        self,
        context: scmle.core.context.Context,
        stage: prefilter.Prefilter,
    ) -> None:
        """Writes the stability and shift reports and the EDA page.

        The files are `stability_report.json`, `shift_report.json` and
        `eda_report.html`.

        The assessment itself ran as the second stage, before `C` was
        opened, on the exploration set only. Without a reference sample the
        shift status is `unverifiable`, never `pass`. Fragile survivors were
        withdrawn from the confirmation only if the charter's rules ask for
        it and the stability rule is certified for its thresholds; the
        report says which.

        Args:
            context: The run context that stores the reports.
            stage: The stability stage, holding the assessment it ran.

        Raises:
            BlockedError: If the stability stage holds no assessment; the
                governed exploration always runs it, so this guards the type.
        """
        found = stage.found
        if found is None:  # pragma: no cover - run_governed always calls it
            raise errors.BlockedError("the second stage did not run")
        stability_document = scmle.data.robustness.report.stability_report(
            found, stage.note if stage.requested else None
        )
        shift_document = scmle.data.robustness.report.shift_report(found)
        scmle.data.robustness.report.conform(stability_document)
        scmle.data.robustness.report.validate_shift(shift_document)
        context.write_json("stability_report.json", stability_document)
        context.write_json("shift_report.json", shift_document)
        context.write_text(
            "eda_report.html",
            html_report.render(stability_document, shift_document),
        )
        context.emit(
            "robustness.completed",
            "stability and shift were assessed on the exploration set",
            ("stability_report.json", "shift_report.json", "eda_report.html"),
            {
                "stability": stability_document["status"],
                "stability_counts": stability_document["counts"],
                "fragile": list(found.fragile),
                "stability_gate": found.stability_gate.value,
                "stability_filter": stage.note,
                "shift": shift_document["status"],
                "shift_reason": shift_document["reason"],
                "forward": found.forward.status,
            },
        )
