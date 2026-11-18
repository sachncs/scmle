"""Conformal calibration of the bundle's intervals and sets."""

from typing import Any
from typing import cast

import numpy
from sklearn import base

import scmle.artifacts.bundle
from scmle.assurance.charter import contracts
from scmle.assurance.charter import sealing
from scmle.contracts import experiments
from scmle.contracts import tasks
import scmle.contracts.models
from scmle.evaluation import conformal
from scmle.evaluation import cv_plus
import scmle.pipeline.context
from scmle.pipeline.stages import cost
from scmle.pipeline.stages import documentation
from scmle.preparation import strategies
from scmle.runtime import randomness

MARGINLESS = (
    tasks.Task.SURVIVAL,
    tasks.Task.ORDINAL,
    tasks.Task.RANKING,
    tasks.Task.MULTILABEL,
    tasks.Task.MULTIOUTPUT,
    tasks.Task.FORECAST,
    tasks.Task.UPLIFT,
)
"""Tasks without a conformal margin: it needs point predictions or labels."""


class Calibration(documentation.Documentation):
    """Prediction intervals and sets calibrated from out-of-fold predictions.

    Part of `PackageStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def waive(
        self, context: scmle.pipeline.context.Context, reason: str
    ) -> None:
        """Records why the bundle will have no prediction intervals or sets.

        Args:
            context: The run context that receives the event.
            reason: Why calibration was skipped.
        """
        context.emit(
            "conformal.skipped",
            f"conformal calibration skipped: {reason}",
            data={"reason": reason},
        )

    def conformalise(
        self,
        context: scmle.pipeline.context.Context,
        selected: experiments.Trial,
        bundle: scmle.artifacts.bundle.Bundle,
    ) -> conformal.ConformalCalibrator | None:
        """Calibrates intervals or sets from out-of-fold predictions.

        The selected candidate is refit on each cross-validation fold's
        training rows and scored on its held-out rows, so every calibration
        score comes from a model that never saw that row. Under a charter whose
        unit of prediction is a new entity, one randomly chosen held-out score
        per entity is kept instead, and no fold models are kept.

        Args:
            context: The run context, holding the split, budget and charter.
            selected: The selected trial, used to build the fold models.
            bundle: The model bundle; the fold models are kept on it when the
                CV+ rule applies.

        Returns:
            The calibrator, or None when the task has no conformal margin, the
            time budget is spent or too few held-out rows exist.
        """
        task = scmle.pipeline.context.require(context.task, "task")
        if task in MARGINLESS:
            return None
        reason = cost.halt(context, selected, "conformal calibration")
        if reason:
            self.waive(context, reason)
            return None
        classes = list(bundle.classes or ())
        entity_column = self.grouping(context)
        held = self.holdout(context, selected, classes, entity_column)
        if held is None:
            return None
        held_labels, held_scores, held_entities, fold_models = held
        if entity_column is not None:
            chosen = conformal.representatives(
                held_entities,
                randomness.component_rng(
                    context.spec.validation.seed, "conformal.entity", 0
                ),
            )
            held_labels = [held_labels[i] for i in chosen]
            held_scores = [held_scores[i] for i in chosen]
            fold_models = []
            context.emit(
                "conformal.entity_wise",
                f"calibrated on one score for each of {len(chosen)} "
                "entities (unit_of_prediction: new_entity)",
                data={"entities": len(chosen)},
            )
        if len(held_labels) < conformal.MIN_CALIBRATION_ROWS:
            self.waive(
                context,
                f"only {len(held_labels)} "
                f"{'entities' if entity_column else 'held-out rows'}; need "
                f"{conformal.MIN_CALIBRATION_ROWS}",
            )
            return None
        if task is tasks.Task.REGRESSION:
            self.retain(
                context,
                bundle,
                fold_models,
                numpy.asarray(held_labels, dtype=float),
                numpy.asarray(held_scores, dtype=float),
            )
            return conformal.ConformalCalibrator.for_regression(
                held_labels, held_scores
            )
        return conformal.ConformalCalibrator.classify(
            held_labels, numpy.asarray(held_scores), classes
        )

    def holdout(
        self,
        context: scmle.pipeline.context.Context,
        selected: experiments.Trial,
        classes: list[Any],
        entity_column: str | None,
    ) -> tuple[list[Any], list[Any], list[Any], list[Any]] | None:
        """Scores each fold's held-out rows with a model refit without them.

        Args:
            context: The run context, holding the split, budget and charter.
            selected: The selected trial, rebuilt for each fold; the budget
                is checked before each fold.
            classes: The bundle's classes, in order (empty for regression).
            entity_column: The entity column when calibration is entity-wise,
                else None.

        Returns:
            The held-out labels, scores (predictions, or class probabilities
            in `classes` order), entities and, for regression, the fold
            models; None when the time budget ran out (recorded).
        """
        held_labels: list[Any] = []
        held_scores: list[Any] = []
        held_entities: list[Any] = []
        fold_models: list[Any] = []
        task = scmle.pipeline.context.require(context.task, "task")
        split = scmle.pipeline.context.require(context.split, "split")
        template = strategies.lookup(selected.strategy).build(
            scmle.pipeline.context.require(
                context.profile, "profile"
            ).feature_plan,
            context.registry.get(selected.model_name),
            task,
            context.spec.validation.seed,
            selected.params,
        )
        for fit_positions, evaluate_positions in split.manifest.folds:
            reason = cost.halt(context, selected, "conformal calibration")
            if reason:
                self.waive(context, reason)
                return None
            fit = numpy.asarray(fit_positions)
            evaluate = numpy.asarray(evaluate_positions)
            candidate = cast(
                "scmle.contracts.models.Estimator", base.clone(template)
            ).fit(split.dev_features.iloc[fit], split.dev_target.iloc[fit])
            rows = split.dev_features.iloc[evaluate]
            held_labels.extend(split.dev_target.iloc[evaluate])
            if entity_column is not None:
                held_entities.extend(rows[entity_column].astype(str))
            if task is tasks.Task.REGRESSION:
                held_scores.extend(candidate.predict(rows))
                fold_models.append(candidate)
                continue
            known = list(candidate.classes_)
            probabilities = candidate.predict_proba(rows)
            held_scores.extend(
                [
                    probabilities[i, known.index(c)] if c in known else 0.0
                    for c in classes
                ]
                for i in range(len(rows))
            )
        return held_labels, held_scores, held_entities, fold_models

    def grouping(self, context: scmle.pipeline.context.Context) -> str | None:
        """Returns the group column when the charter's unit is a new entity.

        Intervals for a new entity treat one score per entity as the
        exchangeable unit, so calibration keeps one random held-out row of
        each entity. Without a charter, or for any other unit, the result is
        None and row-wise calibration applies.

        Args:
            context: The run context, supplying the charter path and data spec.

        Returns:
            The group column name, or None when row-wise calibration applies.
        """
        path = context.spec.charter_path
        if not path:
            return None
        charter = sealing.read_verified(str(path)).charter
        if charter.purpose != contracts.NEW_ENTITY:
            return None
        return context.spec.data.group_column

    def retain(
        self,
        context: scmle.pipeline.context.Context,
        bundle: scmle.artifacts.bundle.Bundle,
        models: list[Any],
        observed: numpy.ndarray,
        predicted: numpy.ndarray,
    ) -> None:
        """Stores the fold models for CV+ intervals when the rule applies.

        CV+ needs each calibration row held out in exactly one fold; with
        `MIN_ROWS_FOR_SPLIT` or more rows the pooled margin is used instead,
        and a model set above the byte cap (`allowance`) is not kept. The
        source of the calibration scores is emitted either way, and the reason
        when no ensemble is kept.

        Args:
            context: The run context, holding the split.
            bundle: The model bundle that receives the ensemble.
            models: The fold models, one per cross-validation fold.
            observed: The held-out labels.
            predicted: The held-out predictions.
        """
        split = scmle.pipeline.context.require(context.split, "split")
        held = [fold[1] for fold in split.manifest.folds]
        plan = cv_plus.strategize(
            held,
            len(split.dev_features),
            len(observed),
            models,
            self.allowance(context),
        )
        context.emit(
            "conformal.scores",
            f"calibration scores are {plan.mode.replace('_', ' ')}",
            data={"source": plan.mode, "rows": len(observed)},
        )
        if plan.folds is None:
            context.emit(
                "cv_plus.skipped",
                f"CV+ ensemble not stored: {plan.reason}",
                data={"reason": plan.reason},
            )
            return
        bundle.cv_plus = cv_plus.CvPlusEnsemble(
            tuple(models), plan.folds, numpy.abs(observed - predicted)
        )

    def allowance(self, context: scmle.pipeline.context.Context) -> int:
        """Returns the byte cap for kept fold models (charter's, else default).

        Args:
            context: The run context, supplying the charter path.

        Returns:
            The byte limit from the charter's uncertainty block, or the default
            limit if the charter sets none.
        """
        path = context.spec.charter_path
        if path:
            block = sealing.read_verified(str(path)).charter.uncertainty
            if block is not None and block.limit is not None:
                return block.limit
        return cv_plus.MAX_BYTES
