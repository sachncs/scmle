"""Post-selection diagnostics and evidence on the released rows."""

from typing import Any
from typing import cast

import numpy
import pandas

from scmle.contracts import experiments
from scmle.contracts import tasks
import scmle.contracts.models
import scmle.core.context
from scmle.core.stages import interface
from scmle.data.readers import streaming
from scmle.evaluation import fit_health
from scmle.evaluation import mitigation
from scmle.evaluation import replay
from scmle.evaluation import uplift

DIAGNOSTIC_BUDGET_SHARE = 0.25


DIAGNOSTIC_MIN_SECONDS = 30.0


class Examination(interface.Stage):
    """Diagnostics, streamed scoring and evidence on the released rows.

    Part of `FinalEvaluationStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    @staticmethod
    def ration(context: scmle.core.context.Context) -> float:
        """Returns the time the post-selection diagnostics may use in total.

        A share (`DIAGNOSTIC_BUDGET_SHARE`) of the time budget, never less than
        `DIAGNOSTIC_MIN_SECONDS`, so wide datasets cannot turn explanation into
        the slowest step of the run; and never more than what is left before
        the run's end once the packaging reserve is kept.

        Args:
            context: The run context, supplying the time budget.

        Returns:
            The total seconds the diagnostics may use: a fixed share of the time
            budget (never less than the minimum), capped by the time left.
        """
        budget = context.spec.resources.time_budget_seconds
        share = DIAGNOSTIC_BUDGET_SHARE * budget
        allowance = max(DIAGNOSTIC_MIN_SECONDS, share)
        governor = context.regulator
        left = governor.left("finish") - governor.reserve.parts.get(
            "package", 0.0
        )
        return max(min(allowance, left), 0.0)

    def diagnose(
        self,
        context: scmle.core.context.Context,
        selected: experiments.Trial,
    ) -> dict[str, Any] | None:
        """Returns the fit-health record of the selected trial, if judgeable.

        Args:
            context: The run context, holding the trials and the metric.
            selected: The selected trial.

        Returns:
            The fit-health assessment as a dict, or None if the trial has no
            training scores or fold scores, or the run has no baseline trial
            to compare with.
        """
        health = fit_health.sanity(
            selected,
            context.trials,
            scmle.core.context.require(
                context.metric, "metric"
            ).greater_is_better,
        )
        return None if health is None else health.to_dict()

    def tally(
        self,
        context: scmle.core.context.Context,
        pipeline: Any,
    ) -> dict[str, Any] | None:
        """Scores every locked row of a streamed file, batch by batch.

        Args:
            context: The run context, holding the stream and catalog.
            pipeline: The fitted final model.

        Returns:
            The number of locked rows, the total number of rows and the metrics
            on the locked rows, or None if the data was not streamed.
        """
        if context.stream is None:
            return None
        source, totals = context.stream
        data, seed = context.spec.data, context.spec.validation.seed
        fraction = context.spec.validation.test_fraction
        task = scmle.core.context.require(context.task, "task")
        labels, guesses, scores = [], [], []
        offset = 0
        for batch in source.batches():
            positions = numpy.arange(offset, offset + len(batch))
            offset += len(batch)
            side = streaming.allot(positions, fraction, seed)
            part = batch[side == streaming.LOCKED]
            if part.empty:
                continue
            features = part.drop(columns=[data.target])
            labels.append(part[data.target].to_numpy())
            guesses.append(numpy.asarray(pipeline.predict(features)))
            if task.is_classification:
                scores.append(numpy.asarray(pipeline.predict_proba(features)))
        truth = pandas.Series(numpy.concatenate(labels))
        replay_class = (
            replay.ClassifierPredictions
            if task.is_classification
            else replay.Replayed
        )
        # Metrics only call predict and predict_proba on the stand-in.
        stand_in = cast(
            "scmle.contracts.models.Estimator",
            replay_class(
                numpy.concatenate(guesses),
                numpy.vstack(scores) if scores else None,
                getattr(pipeline, "classes_", None),
            ),
        )
        metrics = context.catalog.evaluate(
            stand_in, pandas.DataFrame(index=truth.index), truth, task
        )
        return {
            "rows": len(truth),
            "of_total": totals.rows,
            "metrics": metrics,
        }

    def uplift_report(
        self,
        context: scmle.core.context.Context,
        fitted: Any,
        features: pandas.DataFrame,
        outcome: pandas.Series,
    ) -> dict[str, Any] | None:
        """Returns treatment-effect evidence on the released rows, with caveat.

        Args:
            context: The run context, supplying the task, treatment column and
                propensity check.
            fitted: The fitted model whose predictions are the predicted effect.
            features: The released rows, including the treatment column.
            outcome: The observed outcome of those rows.

        Returns:
            The stated assumption, propensity check, average effect, mean
            predicted effect and a Qini curve sampled at up to 50 points, or
            None if the task is not uplift.
        """
        if (
            scmle.core.context.require(context.task, "task")
            is not tasks.Task.UPLIFT
        ):
            return None
        arm = features[context.spec.data.treatment_column].to_numpy(dtype=int)
        predicted = numpy.asarray(fitted.predict(features), dtype=float)
        rows, curve = uplift.qini_curve(outcome.to_numpy(float), arm, predicted)
        keep = numpy.unique(numpy.linspace(0, len(rows) - 1, 50).astype(int))
        return {
            "assumption": uplift.ASSUMPTION,
            "propensity": context.propensity,
            "average_effect": uplift.average_effect(
                outcome.to_numpy(float), arm
            ),
            "mean_predicted_effect": float(predicted.mean()),
            "qini_curve": {
                "rows": rows[keep].tolist(),
                "incremental": curve[keep].tolist(),
            },
        }

    def corroborate(
        self,
        context: scmle.core.context.Context,
        before: Any,
        after: Any,
        features: pandas.DataFrame,
        target: pandas.Series,
    ) -> None:
        """Adds the gap and accuracy on the released rows to the evidence.

        Stores the gap and accuracy before and after the mitigation, and the
        row count, under `locked_rows` in the mitigation record. Does nothing
        unless a group-threshold mitigation was applied.

        Args:
            context: The run context, holding the mitigation record.
            before: The model without the group thresholds.
            after: The model with the group thresholds.
            features: The released rows.
            target: The labels of the released rows.
        """
        record = context.fairness_mitigation
        if record is None or not record["applied"]:
            return
        column = record["column"]
        assigner = after.assigner
        groups = assigner.assign(features[column]).reset_index(drop=True)
        positive = after.classes_[1]
        actual = (target == positive).to_numpy()
        flags = {
            name: numpy.asarray(model.predict(features)) == positive
            for name, model in (("before", before), ("after", after))
        }
        criterion = record["criterion"]
        record["locked_rows"] = {
            "gap_before": mitigation.gap(
                groups, flags["before"], actual, criterion
            ),
            "gap_after": mitigation.gap(
                groups, flags["after"], actual, criterion
            ),
            "accuracy_before": float((flags["before"] == actual).mean()),
            "accuracy_after": float((flags["after"] == actual).mean()),
            "rows": len(features),
        }
