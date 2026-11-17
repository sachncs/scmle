"""Causal identification of a charter run's primary claim."""

from typing import Any

from scmle.assurance.charter import contracts
from scmle.assurance.ledger import confirm
from scmle.assurance.ledger import governed
from scmle.core import causal_claim
import scmle.core.context
from scmle.core.stages import qualification
from scmle.data.admissibility import qualify
from scmle.data.framing import draft
import scmle.data.identification.run


class Identification(qualification.Qualification):
    """The causal claim grounded, stated and identified.

    Part of `ProfileStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def ground(
        self,
        context: scmle.core.context.Context,
        charter: contracts.AnalysisCharter,
        view: Any,
    ) -> None:
        """Checks that the charter's words name real columns.

        Reads the column names of the exploration view and the target
        column, never row values. Writes `framing_link.json`. The result is
        descriptive: a name that matches no column is `unresolved`, and
        nothing in the run is gated on it.

        Args:
            context: The run context that stores the report.
            charter: The analysis charter whose target, moment and unit are
                matched.
            view: The exploration view, supplying the column names.
        """
        data = context.spec.data
        columns = [*view.features.columns, data.target]
        document = draft.linkage(charter, columns)
        context.write_json("framing_link.json", document)
        context.emit(
            "framing.completed",
            "the charter's target, moment and unit were matched to columns",
            ("framing_link.json",),
            {k: v["status"] for k, v in document["fields"].items()},
        )

    def hypothesise(
        self,
        context: scmle.core.context.Context,
        charter: contracts.AnalysisCharter,
        view: Any,
        graph: Any,
        found: qualify.Qualification,
    ) -> confirm.Docket | None:
        """Registers the primary causal hypothesis for the one opening.

        The strategy is chosen from the charter and the declared DAG alone.
        The estimate, the falsification battery, the replication, the
        sensitivity assessment and the tier are computed on the frame the
        confirmation releases, in the same opening that tests the bivariate
        family; nothing opens `C` for them.

        Args:
            context: The run context, supplying the target name.
            charter: The analysis charter, with its causal block and budget.
            view: The exploration view (features and target).
            graph: The declared DAG, or None.
            found: The exploration qualification, whose DAG checks are
                copied into the report.

        Returns:
            The claim runner, or None for a predictive charter and for a
            charter no strategy identifies (no estimate can exist).
        """
        block = charter.estimand.causal
        if block is None:
            return None
        frame = view.features.copy()
        frame[context.spec.data.target] = view.target
        return causal_claim.rig(
            block,
            frame,
            context.spec.data.target,
            charter.error_budget,
            dag=graph,
            seed=charter.seed,
            dag_report=found.dag,
            charter_hash=charter.charter_hash,
        )

    def identify(
        self,
        context: scmle.core.context.Context,
        charter: contracts.AnalysisCharter,
        view: Any,
        run: governed.Governance,
        graph: Any,
        found: qualify.Qualification,
    ) -> None:
        """Reports the identification of a causal charter.

        Runs only when the charter's estimand carries a causal block. When the
        confirmation evaluated the registered claim, its five report files
        were written with the other registered claims, bound to the ledger
        head, and only the event is emitted here. Otherwise (no strategy
        identifies the charter, no confirmation set exists, or the claim could
        not be evaluated) the strategy is chosen from the charter, the design
        is diagnosed on `X`, the returned documents are written and the claim
        is `unverified` or `not_identified` with no estimate.

        Args:
            context: The run context that stores the documents.
            charter: The analysis charter.
            view: The exploration view (features and target).
            run: The governed run, holding the confirmation's documents.
            graph: The declared DAG, or None.
            found: The exploration qualification, whose DAG checks are used.
        """
        block = charter.estimand.causal
        if block is None:
            return
        documents = run.confirmation.documents
        if causal_claim.CLAIM_FILE in documents:
            claims = documents[causal_claim.CLAIM_FILE]
            context.emit(
                "identification.completed",
                f"causal claim status {claims['status']}, tier {claims['tier']}"
                " (estimated on the single opening of C)",
                tuple(scmle.data.identification.run.DOCUMENTS),
                {
                    "status": claims["status"],
                    "tier": claims["tier"],
                    "estimated_on_c": True,
                },
            )
            return
        frame = view.features.copy()
        frame[context.spec.data.target] = view.target
        outcome = scmle.data.identification.run.design_only(
            block,
            frame,
            charter.error_budget,
            dag=graph,
            seed=charter.seed,
            dag_checks=found.dag,
        )
        for name, document in outcome.documents.items():
            context.write_json(name, document)
        claims = outcome.documents["causal_claims.json"]
        context.emit(
            "identification.completed",
            f"causal claim status {claims['status']}, tier {claims['tier']}",
            tuple(outcome.documents),
            {
                "status": claims["status"],
                "tier": claims["tier"],
                "estimated_on_c": False,
            },
        )
