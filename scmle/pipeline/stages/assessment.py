"""The final evaluation stage on the locked test set."""

import dataclasses
from typing import Any
from typing import ClassVar

import numpy
import pandas

from scmle.assurance.custody import journal
from scmle.contracts import experiments
from scmle.contracts import records
from scmle.contracts import tasks
from scmle.models import thresholds
from scmle.pipeline import products
import scmle.pipeline.context
from scmle.pipeline.stages import cost
from scmle.pipeline.stages import scoring
from scmle.pipeline.stages import selecting
from scmle.pipeline.stages import summary
from scmle.preparation import pseudo_labeling
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
        shipped: The model the bundle serves: the same choice refit on
            every row, or `final_pipeline` when it was not refit.
        test_rows: The opened test rows and labels.
    """

    evaluation: records.Layout | None
    final_pipeline: Any
    shipped: Any
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
        metrics, diagnostics and the evaluation document in the context; the
        bundle's model is the choice refit on every row (`ship`).
        The locked rows are first wrapped so that opening them is recorded in
        the run's custody journal (`journal.Custodian`).

        Args:
            context: The run context; receives the final model, the test rows
                and the evaluation.
        """
        context.services.check()
        split = scmle.pipeline.context.require(context.split, "split")
        if not isinstance(split.locked_test, journal.Custodian):
            split.locked_test = journal.Custodian(
                split.locked_test,
                journal.Journal.locate(context.services.workspace.directory),
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
        context.journal.emit(
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
        context.shipped, shipped = self.ship(context, pipeline)
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
            "resource_limits": context.services.governor.report(),
            "skipped_for_time": context.journal.skipped,
            "test_set_opened": locked.opened,
            "shipped": shipped,
            "diagnostics": sorted(context.diagnostics),
            "elapsed_seconds": context.services.elapsed(),
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
                context.services.registry.get(selected.model_name),
                task,
                context.services.spec.validation.seed,
                selected.params,
            )
            pipeline.fit(split.dev_features, split.dev_target)
        return pipeline

    def ship(
        self,
        context: scmle.pipeline.context.Context,
        evaluated: Any,
    ) -> tuple[Any, records.Layout]:
        """The frozen choice refit on development and test rows, to serve.

        Selection and the test score are done, so the held-out rows have no
        further use as a judge; a model fitted on every row is expected to
        be at least as good as the one evaluated, as a practitioner refits
        on all data after choosing. The same configuration, pseudo-labels,
        decision threshold and group thresholds are kept; rows keep their
        ingested order. The reported test metrics stay those of
        `evaluated`, which never saw the test rows, so they are a
        conservative estimate for the served model. The refit is skipped,
        on record, when a charter governs the run (its confirmatory claim
        is about the evaluated model itself), when training was streamed,
        or when it would not end before the packaging reserve.

        Args:
            context: The run context, holding the split, the selection and
                the opened test rows.
            evaluated: The evaluated model, wrapped as it was scored.

        Returns:
            The model to serve and the record of the refit (`refit`, the
            `rows` it was fitted on and the `reason` it was skipped).
        """
        split = scmle.pipeline.context.require(context.split, "split")
        selected = scmle.pipeline.context.require(context.selected, "selected")
        rows = scmle.pipeline.context.require(context.test_rows, "test_rows")
        reason = self.hold(context, len(rows[0]))
        if reason:
            context.journal.forgo("refit on every row", reason)
            return evaluated, {
                "refit": False,
                "rows": len(split.dev_features),
                "reason": reason,
            }
        features, target = pool(split, rows)
        task = scmle.pipeline.context.require(context.task, "task")
        fitted = strategies.lookup(selected.strategy).build(
            scmle.pipeline.context.require(
                context.profile, "profile"
            ).feature_plan,
            context.services.registry.get(selected.model_name),
            task,
            context.services.spec.validation.seed,
            selected.params,
        )
        fitted.fit(features, target)
        unlabeled = context.unlabeled
        if (context.semi_supervised or {}).get(
            "used"
        ) and unlabeled is not None:
            fitted = pseudo_labeling.selftrain(
                fitted,
                features,
                target,
                unlabeled,
                context.services.spec.data.pseudo_label_confidence,
            )[0]
        context.journal.emit(
            "model.shipped",
            f"selected model refit on all {len(features)} rows to serve",
            data={"rows": len(features)},
        )
        return thresholds.rewrap(evaluated, fitted), {
            "refit": True,
            "rows": len(features),
            "reason": "",
        }

    def hold(
        self, context: scmle.pipeline.context.Context, tested: int
    ) -> str | None:
        """Why the selected model must not be refit on every row, or None.

        Args:
            context: The run context, holding the split, selection and
                governor.
            tested: Number of test rows the refit would add.

        Returns:
            The reason: a charter governs the run, training was streamed, or
            the refit would not end before the packaging reserve.
        """
        if context.services.spec.charter_path is not None:
            return "a charter governs the run"
        if (
            context.stream is not None
            and context.services.spec.data.training == "stream"
        ):
            return "training was streamed"
        split = scmle.pipeline.context.require(context.split, "split")
        selected = scmle.pipeline.context.require(context.selected, "selected")
        developed = max(len(split.dev_features), 1)
        return context.services.governor.overdue(
            cost.refitcost(context, selected)
            * (developed + tested)
            / developed,
            context.services.governor.reserve.parts.get("package", 0.0),
        )

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
            context.services.spec.data.weight_column == tasks.FAIRNESS_WEIGHT
            and tasks.FAIRNESS_WEIGHT not in features
        ):
            # Reweighing weights development rows only; locked rows carry a
            # neutral weight so the pipeline finds the column it expects.
            features = features.assign(**{tasks.FAIRNESS_WEIGHT: 1.0})
        return features, target


def pool(
    split: Any, rows: tuple[pandas.DataFrame, pandas.Series]
) -> tuple[pandas.DataFrame, pandas.Series]:
    """Development and test rows together, in their ingested order.

    Args:
        split: The split, holding the development rows and the manifest.
        rows: The opened test features and labels.

    Returns:
        Every row's features and labels, ordered by row id.
    """
    order = numpy.argsort(
        numpy.concatenate([split.manifest.train_ids, split.manifest.test_ids]),
        kind="stable",
    )
    parts = [split.dev_features, rows[0], split.dev_target, rows[1]]
    parts = [part.copy(deep=False) for part in parts]
    for part in parts:
        # `pandas.concat` compares the parts' `attrs`, which fails when a
        # user's frame keeps a Series there; the pooled rows need none.
        part.attrs = {}
    features = pandas.concat(parts[:2], ignore_index=True)
    target = pandas.concat(parts[2:], ignore_index=True)
    return (
        features.iloc[order].reset_index(drop=True),
        target.iloc[order].reset_index(drop=True),
    )
