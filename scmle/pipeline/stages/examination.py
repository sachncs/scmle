"""Post-selection diagnostics and evidence on the released rows.

Also the parts of the final evaluation's product (`assessment.Exam`): the
pseudo-labels, the fairness remedy, the decision and the readings of the
test rows, each made by one of the final evaluation's mixins and returned
rather than set on the run context.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import cast
from typing import ClassVar
from typing import Self

import numpy
import pandas

from scmle.contracts import experiments
from scmle.contracts import records
from scmle.contracts import tasks
import scmle.contracts.models
from scmle.data.readers import streaming
from scmle.evaluation import fit_health
from scmle.evaluation import mitigation
from scmle.evaluation import replay
from scmle.evaluation import uplift
from scmle.pipeline import products
import scmle.pipeline.context
from scmle.pipeline.stages import interface

DIAGNOSTIC_BUDGET_SHARE = 0.25


DIAGNOSTIC_MIN_SECONDS = 30.0


@dataclasses.dataclass(frozen=True)
class Part(products.Derived):
    """A part of the final evaluation's product; never written.

    Its fields have single-word names; `names` maps each field that the run
    context still holds to the context attribute, so `apply` keeps filling
    the context fields that readers not yet converted read.

    Attributes:
        names: Context attribute of each field the context holds, by field.
    """

    names: ClassVar[Mapping[str, str]] = {}

    @classmethod
    def capture(cls, board: object) -> Self:
        """Reads the part from the run context's attributes.

        Args:
            board: The run context; it has an attribute per mapped field.

        Returns:
            The part; fields the context does not hold keep their defaults.
        """
        return cls(
            **{field: getattr(board, name) for field, name in cls.names.items()}
        )

    def apply(self, board: object) -> None:
        """Sets the mapped fields on the run context.

        Args:
            board: The run context.
        """
        for field, name in self.names.items():
            setattr(board, name, getattr(self, field))

    def summary(self) -> object:
        """Every field, as recorded.

        Returns:
            The fields by name.
        """
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
        }


@dataclasses.dataclass(frozen=True)
class Pseudolabels(Part):
    """The self-training judgement of the final fit (`Relabelling`).

    Attributes:
        record: Unlabeled rows, confidence, the judgement's outcome, whether
            the pseudo-labels were used and how many; None when the run
            asked for none or had no unlabeled rows.
    """

    names: ClassVar[Mapping[str, str]] = {"record": "semi_supervised"}

    record: records.Layout | None = None


@dataclasses.dataclass(frozen=True)
class Remedy(Part):
    """The group-threshold judgement of the final model (`remedy.Remedy`).

    Attributes:
        record: The mitigation's evidence, with the gap and accuracy on the
            released rows once corroborated (`locked_rows`); None without
            the group-thresholds mitigation or outside binary tasks.
    """

    names: ClassVar[Mapping[str, str]] = {"record": "fairness_mitigation"}

    record: records.Layout | None = None


@dataclasses.dataclass(frozen=True)
class Readings(Part):
    """What the final model reads off the released rows (`summary.Summary`).

    Attributes:
        scores: The model's scores or predicted classes on the released
            rows (`scoring.reference_scores`), the drift reference the
            bundle keeps; None before the rows are opened.
        diagnostics: The post-selection diagnostic reports by name.
    """

    names: ClassVar[Mapping[str, str]] = {
        "scores": "reference_scores",
        "diagnostics": "diagnostics",
    }

    scores: pandas.Series | None = None
    diagnostics: dict[str, dict[str, Any]] = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass(frozen=True)
class Decision(Part):
    """The decision rule the test set is opened for, and what it is worth.

    Attributes:
        threshold: The decision threshold chosen on development data
            (`scmle.decision.threshold.choose`), or None.
        benefit: The net benefit of the chartered rule on the released
            rows, or None.
        claim: Hash of the coverage claim pledged before the rows opened,
            or None when the task makes no claim.
    """

    names: ClassVar[Mapping[str, str]] = {"claim": "coverage_claim_sha256"}

    threshold: records.Layout | None = None
    benefit: records.Layout | None = None
    claim: str | None = None


class Examination(interface.Stage):
    """Diagnostics, streamed scoring and evidence on the released rows.

    Part of `FinalEvaluationStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    @staticmethod
    def ration(context: scmle.pipeline.context.Context) -> float:
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
        budget = context.services.spec.resources.time_budget_seconds
        share = DIAGNOSTIC_BUDGET_SHARE * budget
        allowance = max(DIAGNOSTIC_MIN_SECONDS, share)
        governor = context.services.governor
        left = governor.left("finish") - governor.reserve.parts.get(
            "package", 0.0
        )
        return max(min(allowance, left), 0.0)

    def diagnose(
        self,
        context: scmle.pipeline.context.Context,
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
            context.journal.trials,
            scmle.pipeline.context.require(
                context.metric, "metric"
            ).greater_is_better,
        )
        return None if health is None else health.to_dict()

    def tally(
        self,
        context: scmle.pipeline.context.Context,
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
        data, seed = (
            context.services.spec.data,
            context.services.spec.validation.seed,
        )
        fraction = context.services.spec.validation.test_fraction
        task = scmle.pipeline.context.require(context.task, "task")
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
        metrics = context.services.catalog.evaluate(
            stand_in, pandas.DataFrame(index=truth.index), truth, task
        )
        return {
            "rows": len(truth),
            "of_total": totals.rows,
            "metrics": metrics,
        }

    def uplift_report(
        self,
        context: scmle.pipeline.context.Context,
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
            scmle.pipeline.context.require(context.task, "task")
            is not tasks.Task.UPLIFT
        ):
            return None
        arm = features[context.services.spec.data.treatment_column].to_numpy(
            dtype=int
        )
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
        remedy: Remedy,
        before: Any,
        after: Any,
        features: pandas.DataFrame,
        target: pandas.Series,
    ) -> Remedy:
        """Adds the gap and accuracy on the released rows to the evidence.

        Stores the gap and accuracy before and after the mitigation, and the
        row count, under `locked_rows` in the mitigation record, unless no
        group-threshold mitigation was applied.

        Args:
            remedy: The mitigation judged on development rows.
            before: The model without the group thresholds.
            after: The model with the group thresholds.
            features: The released rows.
            target: The labels of the released rows.

        Returns:
            The remedy with the released rows' evidence, or `remedy` when
            nothing was applied.
        """
        record = remedy.record
        if record is None or not record["applied"]:
            return remedy
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
        return Remedy(
            record
            | {
                "locked_rows": {
                    "gap_before": mitigation.gap(
                        groups, flags["before"], actual, criterion
                    ),
                    "gap_after": mitigation.gap(
                        groups, flags["after"], actual, criterion
                    ),
                    "accuracy_before": float(
                        (flags["before"] == actual).mean()
                    ),
                    "accuracy_after": float((flags["after"] == actual).mean()),
                    "rows": len(features),
                }
            }
        )
