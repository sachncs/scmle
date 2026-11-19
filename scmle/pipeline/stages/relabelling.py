"""Pseudo-labelling of the final fit."""

from typing import Any

from scmle.contracts import experiments
from scmle.contracts import records
import scmle.evaluation.metrics
from scmle.pipeline import engine
from scmle.pipeline.stages import cost
from scmle.pipeline.stages import examination
from scmle.pipeline.stages import labelling
from scmle.pipeline.stages import profiling
from scmle.pipeline.stages import remedy
from scmle.preparation import pseudo_labeling
from scmle.preparation import strategies


class Relabelling(remedy.Remedy):
    """Confident pseudo-labels added to the final fit when they help.

    Part of `FinalEvaluationStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def pseudolabel(
        self,
        board: engine.Board,
        selected: experiments.Trial,
        pipeline: Any,
    ) -> tuple[Any, examination.Pseudolabels]:
        """Adds confident pseudo-labels to the final fit when they help.

        The judgement uses only development folds. The unlabeled rows never
        touch the locked test rows. Does nothing unless the data specification
        asks for pseudo-labeling and unlabeled rows exist.

        Args:
            board: The run's board, holding the unlabeled rows and split.
            selected: The selected trial, used to build the template model.
            pipeline: The fitted final model.

        Returns:
            The pipeline to carry forward: the self-trained refit when
            cross-validation accepted it, otherwise `pipeline` unchanged
            (including when the judgement was abandoned for lack of time or
            confident rows); and the judgement's record
            (`examination.Pseudolabels`), empty when none was made.
        """
        data = board.services.spec.data
        unlabeled = board.get(labelling.Resolution).unlabeled
        if data.unlabeled != "pseudo_label" or unlabeled is None:
            return pipeline, examination.Pseudolabels()
        partition = board.get(profiling.Partition)
        split = partition.split
        task = partition.task
        metric = partition.metric
        template = strategies.lookup(selected.strategy).build(
            partition.profile.feature_plan,
            board.services.registry.get(selected.model_name),
            task,
            board.services.spec.validation.seed,
            selected.params,
        )
        outcome = pseudo_labeling.judge(
            template,
            split.dev_features,
            split.dev_target,
            split.manifest.folds,
            unlabeled,
            metric.scorer(
                scmle.evaluation.metrics.positive_label(split.dev_target, task)
            ),
            data.pseudo_label_confidence,
            lambda: bool(
                board.services.governor.overdue(
                    cost.foldcost(selected),
                    board.services.governor.reserve.parts.get("package", 0.0),
                )
            ),
        )
        record: records.Layout = {
            "unlabeled_rows": len(unlabeled),
            "confidence": data.pseudo_label_confidence,
            "outcome": None if outcome is None else outcome.to_dict(),
            "used": False,
        }
        if outcome is None or not outcome.accepted:
            return pipeline, examination.Pseudolabels(record)
        augmented, count = pseudo_labeling.selftrain(
            pipeline,
            split.dev_features,
            split.dev_target,
            unlabeled,
            data.pseudo_label_confidence,
        )
        return augmented, examination.Pseudolabels(
            record | {"used": count > 0, "pseudo_rows": count}
        )
