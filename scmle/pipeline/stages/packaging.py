"""The packaging stage, which assembles the `RunResult`."""

from typing import ClassVar

from scmle.artifacts import report
from scmle.artifacts import results
import scmle.artifacts.bundle
from scmle.contracts import signing
from scmle.contracts import tasks
import scmle.data.reference
from scmle.governance import attestation
from scmle.governance import ledger
from scmle.governance import model_card
from scmle.inference import novelty
import scmle.pipeline.context
from scmle.pipeline.stages import assessment
from scmle.pipeline.stages import tracing
import scmle.runtime.state

PARITY_ROWS = 200


class PackageStage(tracing.Tracing):
    """Writes the model bundle, leaderboard and report."""

    requires: ClassVar[tuple[type, ...]] = (assessment.FinalEvaluationStage,)
    state: ClassVar[scmle.runtime.state.Progress] = (
        scmle.runtime.state.Progress.PACKAGING
    )

    def describe(
        self,
        context: scmle.pipeline.context.Context,  # noqa: ARG002 -- Stage.describe
    ) -> str:
        """Names the stage.

        Args:
            context: The run context; not consulted.

        Returns:
            The text "write bundle and reports".
        """
        return "write bundle and reports"

    def execute(self, context: scmle.pipeline.context.Context) -> None:
        """Packages artifacts and assembles the `RunResult`.

        Builds the model bundle with its conformal, quantile and coverage
        evidence, saves it, and writes the model card, attestation, evidence
        graph, leaderboard and HTML report.

        Args:
            context: The run context; receives the bundle and the result.
        """
        store = context.run_store
        split = scmle.pipeline.context.require(context.split, "split")
        selected = scmle.pipeline.context.require(context.selected, "selected")
        evaluation = scmle.pipeline.context.require(
            context.evaluation, "evaluation"
        )
        profile = scmle.pipeline.context.require(context.profile, "profile")
        bundle = scmle.artifacts.bundle.Bundle.create(
            context.final_pipeline,
            profile.feature_plan,
            split.dev_features,
            context.spec.data.target,
            scmle.pipeline.context.require(context.task, "task").value,
            scmle.pipeline.context.require(context.metric, "metric").name,
            selected.model_name,
            model_params=selected.params,
            strategy=selected.strategy,
            decision_threshold=(evaluation["decision"] or {}).get("threshold"),
            graph_fingerprint=context.spec.feature_graph.fingerprint()
            if context.spec.feature_graph.nodes
            else "",
            group_column=context.spec.data.group_column,
            time_column=context.spec.data.time_column,
            history_source=context.frame,
            class_labels=context.ordinal_labels,
            seed=context.spec.validation.seed,
            configuration=context.spec.config_hash(),
            dataset=context.manifest.digests["data"]
            if context.manifest
            else "",
            split=split.manifest.fingerprint,
            reference=scmle.data.reference.Baseline.build(
                split.dev_features,
                profile.feature_plan,
                scmle.pipeline.context.require(
                    context.reference_scores, "reference_scores"
                ),
                str(
                    scmle.pipeline.context.require(
                        context.reference_scores, "reference_scores"
                    ).name
                ),
                evaluation["test_metrics"],
            ),
        )
        evaluation["learning_curve"] = self.learning_curve(context, selected)
        evaluation["nested_cv"] = self.nested_validation(context, selected)
        evaluation["bias_variance"] = self.bias_variance(context, selected)
        evaluation["sensitivity"] = self.sensitivity(context, selected)
        bundle.conformal = self.conformalise(context, selected, bundle)
        bundle.quantile = self.bracket(context)
        self.settle(context, bundle)
        if (
            scmle.pipeline.context.require(context.task, "task")
            is tasks.Task.FORECAST
        ):
            evaluation["forecast_backtest"] = self.forecast_backtest(
                context, selected
            )
            context.write_json(
                "diagnostics/forecast_backtest.json",
                evaluation["forecast_backtest"],
            )
        self.gauge(context, bundle, evaluation)
        bundle.novelty = novelty.Strangeness.fit(
            split.dev_features,
            bundle.schema.numeric,
            bundle.schema.categorical,
            context.spec.validation.seed,
        )
        bundle.save(store.path("model"))
        card = model_card.Card.from_run(
            scmle.pipeline.context.require(context.manifest, "manifest"),
            profile,
            bundle,
            evaluation,
            scmle.artifacts.bundle.shasum(
                store.path(f"model/{scmle.artifacts.bundle.MANIFEST_FILE}")
            ),
        )
        evaluation["model_card"] = card.to_dict()
        store.write_json("model_card.json", card.to_dict())
        evaluation["reliability_plan"] = self.persist(context, store)
        evaluation["sbom"] = self.inventory(store, bundle)
        evaluation["provenance"] = self.notarise(context, store, bundle)
        release_attestation = attestation.Attestation.from_records(
            evaluation["provenance"],
            evaluation["sbom"],
            signing_key=signing.keying(None),
        ).to_dict()
        evaluation["release_attestation"] = release_attestation
        store.write_json("release_attestation.json", release_attestation)
        evaluation["explanation_evidence"] = self.explanation_evidence(
            context, store
        )
        evaluation["artifact_verification"] = bundle.roundtrip(
            store.path("model"), split.dev_features.head(PARITY_ROWS)
        )
        store.write_json(
            "diagnostics/data_access.json",
            ledger.compose(
                split,
                context.trials,
                int(split.locked_test.opened),
                evaluation["decision"] is not None,
            ),
        )
        store.write_json(
            "diagnostics/decision_log.json",
            {
                "plan": [
                    d.to_dict()
                    for d in scmle.pipeline.context.require(
                        context.plan, "plan"
                    ).decisions
                ],
                "outcomes": [d.to_dict() for d in context.outcomes],
                "selection": {
                    "model": selected.model_name,
                    "strategy": selected.strategy,
                    "params": selected.params,
                    "policy": context.policy.description,
                },
            },
        )
        self.release_evaluation(context, store, evaluation)
        store.write_json("evaluation.json", evaluation)
        evidence_graph = self.trace(context, store)
        graph_record = evidence_graph.to_dict()
        graph_record["graph_digest"] = evidence_graph.digest()
        store.write_json("evidence_graph.json", graph_record)
        result = results.Run(
            run_id=store.run_id,
            run_dir=store.directory,
            state=scmle.runtime.state.Progress.COMPLETED.value,
            plan=scmle.pipeline.context.require(context.plan, "plan"),
            profile=profile,
            trials=context.trials,
            selected_model=selected.model_name,
            evaluation=evaluation,
            bundle=bundle,
            diagnostics=context.diagnostics,
        )
        context.track(artifacts=self.lineage(store))
        board = result.leaderboard()
        board.to_parquet(store.path("leaderboard.parquet"), index=False)
        store.write_text(
            "report.html",
            report.Renderer().render(result, board),
        )
        context.bundle, context.result = bundle, result
