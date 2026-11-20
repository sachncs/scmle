"""The packaging stage, which assembles the `RunResult`."""

import dataclasses
from typing import ClassVar

from scmle.artifacts import report
from scmle.artifacts import results
import scmle.artifacts.bundle
from scmle.contracts import lineage
from scmle.contracts import records
from scmle.contracts import signing
from scmle.contracts import tasks
import scmle.data.reference
from scmle.governance import attestation
from scmle.governance import ledger
from scmle.governance import model_card
from scmle.inference import novelty
from scmle.pipeline import engine
from scmle.pipeline import products
from scmle.pipeline.stages import assessment
from scmle.pipeline.stages import examination
from scmle.pipeline.stages import ingestion
from scmle.pipeline.stages import labelling
from scmle.pipeline.stages import planning
from scmle.pipeline.stages import profiling
from scmle.pipeline.stages import selecting
from scmle.pipeline.stages import tracing
import scmle.runtime.state

PARITY_ROWS = 200


@dataclasses.dataclass(frozen=True)
class Package(products.Derived):
    """The signed bundle and the run's result; the run is then complete.

    Attributes:
        bundle: The packaged model.
        result: The run's result.
        graph: The graph of the run's evidence.
        evaluation: The evaluation record at its current layout.
    """

    bundle: scmle.artifacts.bundle.Bundle | None
    result: results.Run | None
    graph: lineage.Graph
    evaluation: records.Layout | None

    def summary(self) -> object:
        """The digest of the evidence graph, which names every artifact.

        Returns:
            The graph digest.
        """
        return self.graph.digest()


class Packaging(tracing.Tracing):
    """Writes the model bundle, leaderboard and report."""

    recorded: ClassVar[str] = "PackageStage"
    requires: ClassVar[tuple[type, ...]] = (assessment.Assessment,)
    makes: ClassVar[type[products.Product] | None] = Package
    state: ClassVar[scmle.runtime.state.Progress] = (
        scmle.runtime.state.Progress.PACKAGING
    )

    def describe(
        self,
        board: engine.Board,  # noqa: ARG002 -- Stage.describe
    ) -> str:
        """Names the stage.

        Args:
            board: The run's board; not consulted.

        Returns:
            The text "write bundle and reports".
        """
        return "write bundle and reports"

    def execute(self, board: engine.Board) -> Package:
        """Packages artifacts and assembles the `RunResult`.

        Builds the model bundle with its conformal, quantile and coverage
        evidence, saves it, and writes the model card, attestation, evidence
        graph, leaderboard and HTML report. The stages build the evaluation
        record in its version-1 layout; it is moved to the current layout
        (`results.evaluation`) just before `evaluation.json` is written, and
        the result carries that layout. The plan is the product
        `planning.Blueprint` the board holds.

        Args:
            board: The run's board; read only (`Package.apply` sets it).

        Returns:
            The bundle, result, evidence graph and evaluation record.
        """
        store = board.services.workspace
        partition = board.get(profiling.Partition)
        split = partition.split
        selected = products.require(
            board.get(selecting.Choice).selected, "selected"
        )
        evaluation = products.require(
            board.get(assessment.Exam).evaluation, "evaluation"
        )
        profile = partition.profile
        bundle = self.assemble(board)
        evaluation.update(self.probe(board, selected).entries())
        bundle.conformal = self.conformalise(board, selected, bundle)
        bundle.quantile = self.bracket(board)
        self.settle(board, bundle)
        if partition.task is tasks.Task.FORECAST:
            evaluation["forecast_backtest"] = self.forecast(board, selected)
            board.journal.write(
                "diagnostics/forecast_backtest.json",
                evaluation["forecast_backtest"],
            )
        uncertainty = self.gauge(board, bundle)
        uncertainty.equip(bundle)
        evaluation.update(uncertainty.entries())
        bundle.novelty = novelty.Strangeness.fit(
            split.features,
            bundle.schema.numeric,
            bundle.schema.categorical,
            board.services.spec.validation.seed,
        )
        bundle.save(store.path("model"))
        self.certify(board, bundle, evaluation)
        self.release(board, store, evaluation)
        evaluation = results.evaluation(evaluation)
        store.write("evaluation.json", evaluation)
        evidence_graph = self.trace(board, store)
        graph_record = evidence_graph.to_dict()
        graph_record["graph_digest"] = evidence_graph.digest()
        store.write("evidence_graph.json", graph_record)
        result = results.Run(
            identifier=store.run,
            directory=store.directory,
            state=scmle.runtime.state.Progress.COMPLETED.value,
            plan=products.require(board.get(planning.Blueprint).plan, "plan"),
            profile=profile,
            trials=board.journal.trials,
            model=selected.model,
            evaluation=evaluation,
            bundle=bundle,
            diagnostics=board.get(examination.Readings).diagnostics,
        )
        board.journal.track(artifacts=self.lineage(store))
        ranking = result.leaderboard()
        ranking.to_parquet(store.path("leaderboard.parquet"), index=False)
        store.write_text(
            "report.html",
            report.Renderer().render(result, ranking),
        )
        return Package(bundle, result, evidence_graph, evaluation)

    def assemble(self, board: engine.Board) -> scmle.artifacts.bundle.Bundle:
        """Builds the model bundle of the final pipeline.

        The labels the model's outputs decode to are the kind's
        (`kinds.Kind.decode`).

        Args:
            board: The run's board, holding the model to serve, the
                split, the selection and the evaluation, and the intake.

        Returns:
            The bundle, before its conformal, quantile and novelty evidence.
        """
        partition = board.get(profiling.Partition)
        split = partition.split
        selected = products.require(
            board.get(selecting.Choice).selected, "selected"
        )
        exam = board.get(assessment.Exam)
        evaluation = products.require(exam.evaluation, "evaluation")
        profile = partition.profile
        intake = board.get(ingestion.Intake)
        readings = board.get(examination.Readings)
        return scmle.artifacts.bundle.Bundle.create(
            exam.shipped if exam.shipped is not None else exam.pipeline,
            profile.plan,
            split.features,
            board.services.spec.data.target,
            partition.task.value,
            partition.metric.name,
            selected.model,
            model_params=selected.params,
            strategy=selected.strategy,
            decision_threshold=(evaluation["decision"] or {}).get("threshold"),
            graph_fingerprint=board.services.spec.feature_graph.fingerprint()
            if board.services.spec.feature_graph.nodes
            else "",
            group_column=board.services.spec.data.group_column,
            time_column=board.services.spec.data.time_column,
            history_source=partition.frame,
            class_labels=intake.kind.decode(
                board.get(labelling.Resolution).labels
            ),
            seed=board.services.spec.validation.seed,
            configuration=board.services.spec.config_hash(),
            dataset=intake.manifest.digests["data"],
            split=split.manifest.fingerprint,
            reference=scmle.data.reference.Baseline.build(
                split.features,
                profile.plan,
                products.require(readings.scores, "reference_scores"),
                str(
                    products.require(
                        readings.scores,
                        "reference_scores",
                    ).name
                ),
                evaluation["test_metrics"],
            ),
        )

    def certify(
        self,
        board: engine.Board,
        bundle: scmle.artifacts.bundle.Bundle,
        evaluation: records.Layout,
    ) -> None:
        """Writes the model card, attestation and access records of a bundle.

        Args:
            board: The run's board.
            bundle: The saved bundle.
            evaluation: The evaluation record; gains the card, reliability
                plan, SBOM, provenance, attestation, explanation evidence and
                round-trip check.
        """
        store = board.services.workspace
        partition = board.get(profiling.Partition)
        split = partition.split
        selected = products.require(
            board.get(selecting.Choice).selected, "selected"
        )
        profile = partition.profile
        card = model_card.Card.assemble(
            board.get(ingestion.Intake).manifest,
            profile,
            bundle,
            evaluation,
            scmle.artifacts.bundle.shasum(
                store.path(f"model/{scmle.artifacts.bundle.MANIFEST_FILE}")
            ),
        )
        evaluation["model_card"] = card.to_dict()
        store.write("model_card.json", card.to_dict())
        evaluation["reliability_plan"] = self.persist(board, store)
        evaluation["sbom"] = self.inventory(store, bundle)
        evaluation["provenance"] = self.notarise(board, store, bundle)
        release_attestation = attestation.Attestation.from_records(
            evaluation["provenance"],
            evaluation["sbom"],
            signing_key=signing.keying(None),
        ).to_dict()
        evaluation["release_attestation"] = release_attestation
        store.write("release_attestation.json", release_attestation)
        evaluation["explanation_evidence"] = self.explain(board, store)
        evaluation["artifact_verification"] = bundle.roundtrip(
            store.path("model"), split.features.head(PARITY_ROWS)
        )
        store.write(
            "diagnostics/data_access.json",
            ledger.compose(
                split,
                board.journal.trials,
                int(split.test.opened),
                evaluation["decision"] is not None,
            ),
        )
        store.write(
            "diagnostics/decision_log.json",
            {
                "plan": [
                    d.to_dict()
                    for d in products.require(
                        board.get(planning.Blueprint).plan, "plan"
                    ).decisions
                ],
                "outcomes": [d.to_dict() for d in board.journal.outcomes],
                "selection": {
                    "model": selected.model,
                    "strategy": selected.strategy,
                    "params": selected.params,
                    "policy": board.services.policy.description,
                },
            },
        )
