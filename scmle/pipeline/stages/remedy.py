"""Fairness mitigation and the incremental refit."""

from typing import Any

import numpy

from scmle.contracts import experiments
from scmle.contracts import tasks
from scmle.data.readers import streaming
from scmle.evaluation import mitigation
from scmle.models import thresholds
import scmle.pipeline.context
from scmle.pipeline.stages import examination
from scmle.preparation import strategies


class Remedy(examination.Examination):
    """Group thresholds and the incremental refit.

    Part of `FinalEvaluationStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def mitigate(
        self,
        context: scmle.pipeline.context.Context,
        selected: experiments.Trial,
        pipeline: Any,
        decision: dict[str, Any] | None,
    ) -> Any:
        """Gives each group its own threshold when that is judged worthwhile.

        The judgement uses out-of-fold development predictions only. Under
        the group-thresholds mitigation the outcome (applied or not) is
        written to the decision log; with any other mitigation nothing is done.

        Args:
            context: The run context; receives the decision-log entry and the
                mitigation record.
            selected: The selected trial, used to rebuild the model for the
                out-of-fold predictions.
            pipeline: The fitted final model.
            decision: The chartered or default threshold description, or None;
                its threshold is the base cut of the groups (0.5 if None).

        Returns:
            A group-threshold classifier when applied, else `pipeline`.
        """
        fairness = context.services.spec.fairness
        task = scmle.pipeline.context.require(context.task, "task")
        if fairness.mitigation != "group_thresholds":
            return pipeline
        column = (
            fairness.column or context.services.spec.data.sensitive_columns[0]
        )
        label = f"fairness:{column}"
        if task is not tasks.Task.BINARY:
            context.journal.enter(
                label,
                experiments.Edict.REJECT,
                "group thresholds apply to binary classification only",
            )
            return pipeline
        split = scmle.pipeline.context.require(context.split, "split")
        probabilities, positions = self.outfold(context, selected)
        actual = (
            split.dev_target.iloc[positions] == pipeline.classes_[1]
        ).to_numpy()
        assigner = thresholds.Grouper.fit(split.dev_features[column])
        groups = assigner.assign(
            split.dev_features[column].iloc[positions]
        ).reset_index(drop=True)
        base_cut = decision["threshold"] if decision else 0.5
        result = mitigation.judge(
            probabilities,
            groups,
            actual,
            column,
            fairness.criterion,
            base_cut,
            fairness.trigger_gap,
            fairness.min_reduction,
            fairness.max_accuracy_cost,
        )
        context.journal.enter(
            label,
            experiments.Edict.INCLUDE
            if result.applied
            else experiments.Edict.REJECT,
            result.reason,
        )
        context.fairness_mitigation = result.to_dict()
        if not result.applied:
            return pipeline
        return thresholds.Segmenter(
            pipeline, column, assigner, result.thresholds, base_cut
        )

    def outfold(
        self,
        context: scmle.pipeline.context.Context,
        selected: experiments.Trial,
    ) -> tuple[Any, Any]:
        """Out-of-fold development probabilities of the selected candidate.

        Args:
            context: The run context, holding the split and feature plan.
            selected: The selected trial, rebuilt for each fold.

        Returns:
            The held-out probabilities and the positions they belong to.
        """
        split = scmle.pipeline.context.require(context.split, "split")
        task = scmle.pipeline.context.require(context.task, "task")
        strategy = strategies.lookup(selected.strategy)
        adapter = context.services.registry.get(selected.model_name)
        plan = scmle.pipeline.context.require(
            context.profile, "profile"
        ).feature_plan
        seed = context.services.spec.validation.seed
        return thresholds.heldout(
            lambda: strategy.build(plan, adapter, task, seed, selected.params),
            split.dev_features,
            split.dev_target,
            split.manifest.folds,
        )

    def relearn(
        self,
        context: scmle.pipeline.context.Context,
        pipeline: Any,
    ) -> Any:
        """Passes the whole development partition through an incremental model.

        The preprocessing stays as fitted on the development sample; the
        final estimator, which must support `partial_fit`, keeps learning from
        every development row, batch by batch, in one pass over the file.

        Args:
            context: The run context, holding the stream and settings.
            pipeline: The fitted final model.

        Returns:
            `pipeline`, updated in place when the data is streamed for training,
            otherwise unchanged.
        """
        if (
            context.stream is None
            or context.services.spec.data.training != "stream"
        ):
            return pipeline
        source = context.stream[0]
        data, seed = (
            context.services.spec.data,
            context.services.spec.validation.seed,
        )
        fraction = context.services.spec.validation.test_fraction
        preprocess, model = pipeline[:-1], pipeline.steps[-1][1]
        classes = getattr(pipeline, "classes_", None)
        offset, rows_seen = 0, 0
        for batch in source.batches():
            positions = numpy.arange(offset, offset + len(batch))
            offset += len(batch)
            side = streaming.allot(positions, fraction, seed)
            part = batch[side == streaming.DEVELOPMENT]
            if part.empty:
                continue
            features = part.drop(columns=[data.target])
            matrix = preprocess.transform(features)
            if classes is None:
                model.partial_fit(matrix, part[data.target])
            else:
                model.partial_fit(matrix, part[data.target], classes=classes)
            rows_seen += len(part)
        context.journal.emit(
            "training.streamed",
            f"final model updated on {rows_seen} streamed development rows",
            data={"rows": rows_seen},
        )
        return pipeline
