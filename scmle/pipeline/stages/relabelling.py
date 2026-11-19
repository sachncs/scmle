"""Pseudo-labelling of the final fit."""

from typing import Any

from scmle.contracts import experiments
from scmle.contracts import records
import scmle.evaluation.metrics
import scmle.pipeline.context
from scmle.pipeline.stages import cost
from scmle.pipeline.stages import examination
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
        context: scmle.pipeline.context.Context,
        selected: experiments.Trial,
        pipeline: Any,
    ) -> tuple[Any, examination.Pseudolabels]:
        """Adds confident pseudo-labels to the final fit when they help.

        The judgement uses only development folds. The unlabeled rows never
        touch the locked test rows. Does nothing unless the data specification
        asks for pseudo-labeling and unlabeled rows exist.

        Args:
            context: The run context, holding the unlabeled rows and split.
            selected: The selected trial, used to build the template model.
            pipeline: The fitted final model.

        Returns:
            The pipeline to carry forward: the self-trained refit when
            cross-validation accepted it, otherwise `pipeline` unchanged
            (including when the judgement was abandoned for lack of time or
            confident rows); and the judgement's record
            (`examination.Pseudolabels`), empty when none was made.
        """
        data = context.services.spec.data
        unlabeled = context.unlabeled
        if data.unlabeled != "pseudo_label" or unlabeled is None:
            return pipeline, examination.Pseudolabels()
        split = scmle.pipeline.context.require(context.split, "split")
        task = scmle.pipeline.context.require(context.task, "task")
        metric = scmle.pipeline.context.require(context.metric, "metric")
        template = strategies.lookup(selected.strategy).build(
            scmle.pipeline.context.require(
                context.profile, "profile"
            ).feature_plan,
            context.services.registry.get(selected.model_name),
            task,
            context.services.spec.validation.seed,
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
                context.services.governor.overdue(
                    cost.foldcost(selected),
                    context.services.governor.reserve.parts.get("package", 0.0),
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
