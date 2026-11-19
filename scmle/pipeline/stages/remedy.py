"""Fairness mitigation and the incremental refit."""

from typing import Any

import numpy

from scmle.contracts import experiments
from scmle.contracts import records
from scmle.contracts import tasks
from scmle.data.readers import streaming
from scmle.evaluation import mitigation
from scmle.models import thresholds
from scmle.pipeline import engine
from scmle.pipeline.stages import examination
from scmle.pipeline.stages import ingestion
from scmle.pipeline.stages import profiling
from scmle.preparation import strategies


class Remedy(examination.Examination):
    """Group thresholds and the incremental refit.

    Part of `FinalEvaluationStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def mitigate(
        self,
        board: engine.Board,
        selected: experiments.Trial,
        pipeline: Any,
        decision: records.Layout | None,
    ) -> tuple[Any, examination.Remedy]:
        """Gives each group its own threshold when that is judged worthwhile.

        The judgement uses out-of-fold development predictions only. Under
        the group-thresholds mitigation the outcome (applied or not) is
        written to the decision log; with any other mitigation nothing is done.

        Args:
            board: The run's board; receives the decision-log entry.
            selected: The selected trial, used to rebuild the model for the
                out-of-fold predictions.
            pipeline: The fitted final model.
            decision: The chartered or default threshold description, or None;
                its threshold is the base cut of the groups (0.5 if None).

        Returns:
            A group-threshold classifier when applied, else `pipeline`; and
            the judgement (`examination.Remedy`), empty when none was made.
        """
        fairness = board.services.spec.fairness
        partition = board.get(profiling.Partition)
        task = partition.task
        if fairness.mitigation != "group_thresholds":
            return pipeline, examination.Remedy()
        column = (
            fairness.column or board.services.spec.data.sensitive_columns[0]
        )
        label = f"fairness:{column}"
        if task is not tasks.Task.BINARY:
            board.journal.enter(
                label,
                experiments.Edict.REJECT,
                "group thresholds apply to binary classification only",
            )
            return pipeline, examination.Remedy()
        split = partition.split
        probabilities, positions = self.outfold(board, selected)
        actual = (
            split.target.iloc[positions] == pipeline.classes_[1]
        ).to_numpy()
        assigner = thresholds.Grouper.fit(split.features[column])
        groups = assigner.assign(
            split.features[column].iloc[positions]
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
        board.journal.enter(
            label,
            experiments.Edict.INCLUDE
            if result.applied
            else experiments.Edict.REJECT,
            result.reason,
        )
        remedy = examination.Remedy(result.to_dict())
        if not result.applied:
            return pipeline, remedy
        return thresholds.Segmenter(
            pipeline, column, assigner, result.thresholds, base_cut
        ), remedy

    def outfold(
        self,
        board: engine.Board,
        selected: experiments.Trial,
    ) -> tuple[Any, Any]:
        """Out-of-fold development probabilities of the selected candidate.

        Args:
            board: The run's board, holding the split and feature plan.
            selected: The selected trial, rebuilt for each fold.

        Returns:
            The held-out probabilities and the positions they belong to.
        """
        partition = board.get(profiling.Partition)
        split = partition.split
        task = partition.task
        strategy = strategies.lookup(selected.strategy)
        adapter = board.services.registry.get(selected.model_name)
        plan = partition.profile.feature_plan
        seed = board.services.spec.validation.seed
        return thresholds.heldout(
            lambda: strategy.build(
                plan,
                adapter,
                task,
                seed,
                selected.params,
                featurise=board.get(ingestion.Intake).kind.featurise,
            ),
            split.features,
            split.target,
            split.manifest.folds,
        )

    def relearn(
        self,
        board: engine.Board,
        pipeline: Any,
    ) -> Any:
        """Passes the whole development partition through an incremental model.

        The preprocessing stays as fitted on the development sample; the
        final estimator, which must support `partial_fit`, keeps learning from
        every development row, batch by batch, in one pass over the file.

        Args:
            board: The run's board, holding the stream and settings.
            pipeline: The fitted final model.

        Returns:
            `pipeline`, updated in place when the data is streamed for training,
            otherwise unchanged.
        """
        stream = board.get(ingestion.Intake).stream
        if stream is None or board.services.spec.data.training != "stream":
            return pipeline
        source = stream[0]
        data, seed = (
            board.services.spec.data,
            board.services.spec.validation.seed,
        )
        fraction = board.services.spec.validation.test_fraction
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
        board.journal.emit(
            "training.streamed",
            f"final model updated on {rows_seen} streamed development rows",
            data={"rows": rows_seen},
        )
        return pipeline
