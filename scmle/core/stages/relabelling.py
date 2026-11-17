"""Pseudo-labelling of the final fit."""

from typing import Any

from scmle.contracts import experiments
import scmle.core.context
from scmle.core.stages import cost
from scmle.core.stages import remedy
import scmle.evaluation.metrics
from scmle.sampling import pseudo_labeling
from scmle.sampling import strategies


class Relabelling(remedy.Remedy):
    """Confident pseudo-labels added to the final fit when they help.

    Part of `FinalEvaluationStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def pseudolabel(
        self,
        context: scmle.core.context.Context,
        selected: experiments.Trial,
        pipeline: Any,
    ) -> Any:
        """Adds confident pseudo-labels to the final fit when they help.

        The judgement uses only development folds. The unlabeled rows never
        touch the locked test rows. Does nothing unless the data specification
        asks for pseudo-labeling and unlabeled rows exist; the outcome is
        recorded on the context.

        Args:
            context: The run context, holding the unlabeled rows and split.
            selected: The selected trial, used to build the template model.
            pipeline: The fitted final model.

        Returns:
            The pipeline to carry forward: the self-trained refit when
            cross-validation accepted it, otherwise `pipeline` unchanged
            (including when the judgement was abandoned for lack of time or
            confident rows).
        """
        data = context.spec.data
        unlabeled = context.unlabeled
        if data.unlabeled != "pseudo_label" or unlabeled is None:
            return pipeline
        split = scmle.core.context.require(context.split, "split")
        task = scmle.core.context.require(context.task, "task")
        metric = scmle.core.context.require(context.metric, "metric")
        template = strategies.lookup(selected.strategy).build(
            scmle.core.context.require(context.profile, "profile").feature_plan,
            context.registry.get(selected.model_name),
            task,
            context.spec.validation.seed,
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
                context.regulator.overdue(
                    cost.foldcost(selected),
                    context.regulator.reserve.parts.get("package", 0.0),
                )
            ),
        )
        record: dict[str, Any] = {
            "unlabeled_rows": len(unlabeled),
            "confidence": data.pseudo_label_confidence,
            "outcome": None if outcome is None else outcome.to_dict(),
            "used": False,
        }
        context.semi_supervised = record
        if outcome is None or not outcome.accepted:
            return pipeline
        augmented, count = pseudo_labeling.selftrain(
            pipeline,
            split.dev_features,
            split.dev_target,
            unlabeled,
            data.pseudo_label_confidence,
        )
        record["used"] = count > 0
        record["pseudo_rows"] = count
        return augmented
