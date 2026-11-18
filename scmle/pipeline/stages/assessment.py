"""The final evaluation stage on the locked test set."""

import dataclasses
import time
from typing import Any
from typing import ClassVar

import pandas

from scmle.assurance.custody import journal
from scmle.contracts import experiments
from scmle.contracts import records
from scmle.contracts import tasks
from scmle.pipeline import products
import scmle.pipeline.context
from scmle.pipeline.stages import cost
from scmle.pipeline.stages import scoring
from scmle.pipeline.stages import selecting
from scmle.pipeline.stages import summary
from scmle.preparation import strategies
import scmle.runtime.state


@dataclasses.dataclass(frozen=True)
class Exam(products.Derived):
    """The final model and its one test-set evaluation; never written.

    It holds the opened test rows. A run whose test set was opened is not
    resumed (`journal.Journal.admit`), so it is never needed again.

    Attributes:
        evaluation: The evaluation record.
        final_pipeline: The model refit on the development data.
        test_rows: The opened test rows and labels.
    """

    evaluation: records.Layout | None
    final_pipeline: Any
    test_rows: tuple[pandas.DataFrame, pandas.Series] | None

    def summary(self) -> object:
        """The evaluation record and whether the test rows were opened.

        Returns:
            The evaluation and the opening.
        """
        return {"evaluation": self.evaluation, "opened": bool(self.test_rows)}


class FinalEvaluationStage(summary.Summary):
    """Refits the frozen candidate and opens the locked test set once."""

    requires: ClassVar[tuple[type, ...]] = (selecting.SelectStage,)
    makes: ClassVar[type[products.Product] | None] = Exam
    state: ClassVar[scmle.runtime.state.Progress] = (
        scmle.runtime.state.Progress.TESTING
    )

    def describe(self, context: scmle.pipeline.context.Context) -> str:
        """Names the selected model.

        Args:
            context: The run context, holding the selected trial.

        Returns:
            The text "final one-time test evaluation of" and the model name.
        """
        selected = scmle.pipeline.context.require(context.selected, "selected")
        return f"final one-time test evaluation of {selected.model_name}"

    def execute(self, context: scmle.pipeline.context.Context) -> None:
        """Refits on all development data and evaluates on the test set.

        Reuses the refit made during selection if there is one. Applies
        pseudo-labeling, streamed training, the decision threshold and the
        fairness mitigation, registers the coverage claim, then opens the
        locked rows once, after the selection is frozen. Records the test
        metrics, diagnostics and the evaluation document in the context.
        The locked rows are first wrapped so that opening them is recorded in
        the run's custody journal (`journal.Custodian`).

        Args:
            context: The run context; receives the final model, the test rows
                and the evaluation.
        """
        context.check_cancelled()
        split = scmle.pipeline.context.require(context.split, "split")
        if not isinstance(split.locked_test, journal.Custodian):
            split.locked_test = journal.Custodian(
                split.locked_test,
                journal.Journal.locate(context.run_store.directory),
            )
        selected = scmle.pipeline.context.require(context.selected, "selected")
        task = scmle.pipeline.context.require(context.task, "task")
        pipeline = self.refit(context, selected, task)
        pipeline = self.pseudolabel(context, selected, pipeline)
        pipeline = self.relearn(context, pipeline)
        pipeline, decision = scoring.choose_decision_threshold(
            context, pipeline
        )
        unmitigated = pipeline
        pipeline = self.mitigate(context, selected, pipeline, decision)
        self.pledge(context)
        locked = split.locked_test
        test_features, test_target = self.unlock(context, decision)
        context.test_rows = (test_features, test_target)
        self.corroborate(
            context, unmitigated, pipeline, test_features, test_target
        )
        context.emit(
            "test.opened",
            "locked test set released once, after selection was frozen",
            data={
                "rows": len(test_features),
                "fingerprint": split.manifest.fingerprint,
            },
        )
        context.final_pipeline = pipeline
        usage = cost.meter(context, pipeline, split.dev_features)
        test_metrics = self.measure(
            context, pipeline, test_features, test_target
        )
        self.examine(context, pipeline, test_features, test_target)
        net_benefit = scoring.appraise(
            context, pipeline, test_features, test_target, decision
        )
        self.decide(
            context, pipeline, test_features, test_target, decision, net_benefit
        )
        context.evaluation = self.document(
            context,
            pipeline,
            (test_features, test_target),
            test_metrics,
            {"decision": decision, "net_benefit": net_benefit},
        ) | {
            "resources": usage.to_dict(),
            "resource_limits": context.regulator.report(),
            "skipped_for_time": context.skipped,
            "test_set_opened": locked.opened,
            "diagnostics": sorted(context.diagnostics),
            "elapsed_seconds": time.monotonic() - context.started_at,
            "reproducibility": (
                "same seed/data/environment replays; threaded backends may "
                "differ"
            ),
        }

    def refit(
        self,
        context: scmle.pipeline.context.Context,
        selected: experiments.Trial,
        task: tasks.Task,
    ) -> Any:
        """The selected pipeline fitted on all development data.

        Args:
            context: The run context.
            selected: The selected trial.
            task: The resolved task.

        Returns:
            The refit made during selection if there is one, else a new fit.
        """
        pipeline = context.refits.get(selected.trial_id)
        if pipeline is None:
            split = scmle.pipeline.context.require(context.split, "split")
            pipeline = strategies.lookup(selected.strategy).build(
                scmle.pipeline.context.require(
                    context.profile, "profile"
                ).feature_plan,
                context.registry.get(selected.model_name),
                task,
                context.spec.validation.seed,
                selected.params,
            )
            pipeline.fit(split.dev_features, split.dev_target)
        return pipeline

    def unlock(
        self,
        context: scmle.pipeline.context.Context,
        decision: records.Layout | None,
    ) -> tuple[pandas.DataFrame, pandas.Series]:
        """Opens the locked rows once, bound to the frozen selection.

        Args:
            context: The run context, holding the split and the selection.
            decision: The decision threshold the selection is frozen with.

        Returns:
            The test features and labels.
        """
        split = scmle.pipeline.context.require(context.split, "split")
        selected = scmle.pipeline.context.require(context.selected, "selected")
        locked = split.locked_test
        binding = scoring.pin(selected, decision)
        locked.bind(binding)
        features, target = locked.open(locked.mint(), binding)
        if (
            context.spec.data.weight_column == tasks.FAIRNESS_WEIGHT
            and tasks.FAIRNESS_WEIGHT not in features
        ):
            # Reweighing weights development rows only; locked rows carry a
            # neutral weight so the pipeline finds the column it expects.
            features = features.assign(**{tasks.FAIRNESS_WEIGHT: 1.0})
        return features, target
