"""Post-selection diagnostics run while packaging."""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
import functools
from typing import Any
from typing import cast

import numpy
from sklearn import base

from scmle.contracts import experiments
from scmle.contracts import records
from scmle.contracts import tasks
import scmle.contracts.models
from scmle.evaluation import backtest
from scmle.evaluation import bias_variance
from scmle.evaluation import learning_curve
from scmle.evaluation import nested
from scmle.evaluation import sensitivity
import scmle.evaluation.metrics
from scmle.pipeline import engine
from scmle.pipeline.stages import assessment
from scmle.pipeline.stages import calibration
from scmle.pipeline.stages import cost
from scmle.pipeline.stages import ingestion
from scmle.pipeline.stages import profiling
from scmle.preparation import strategies


def recipe(
    board: engine.Board, selected: experiments.Trial
) -> Callable[[Mapping[str, Any] | None], Any]:
    """Builds the selected candidate from given parameters.

    The candidate is featurised by the data's kind (`kinds.Kind.featurise`).

    Args:
        board: The run's board, holding the partition and the intake.
        selected: The selected trial.

    Returns:
        The builder of the unfitted candidate from its parameters.
    """
    partition = board.get(profiling.Partition)
    return functools.partial(
        strategies.lookup(selected.strategy).build,
        partition.profile.feature_plan,
        board.services.registry.get(selected.model_name),
        partition.task,
        board.services.spec.validation.seed,
        featurise=board.get(ingestion.Intake).kind.featurise,
    )


@dataclasses.dataclass(frozen=True)
class Diagnostics:
    """The post-selection diagnostics of the selected model.

    Each report is None when it was not computed (budget spent, too few
    rows, or the task does not allow it).

    Attributes:
        curve: The learning curve on the first fold.
        nested: The nested cross-validation estimate.
        variance: The bias-variance decomposition.
        sensitivity: The hyperparameter sensitivity.
    """

    curve: records.Layout | None
    nested: records.Layout | None
    variance: records.Layout | None
    sensitivity: records.Layout | None

    def entries(self) -> records.Layout:
        """The evaluation record's diagnostic entries, in record order.

        Returns:
            The learning curve, nested CV, bias-variance and sensitivity.
        """
        return {
            "learning_curve": self.curve,
            "nested_cv": self.nested,
            "bias_variance": self.variance,
            "sensitivity": self.sensitivity,
        }


class Diagnosis(calibration.Calibration):
    """Post-selection diagnostics of the selected model.

    Part of `Packaging`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def probe(
        self,
        board: engine.Board,
        selected: experiments.Trial,
    ) -> Diagnostics:
        """Computes the diagnostics, in the order their fits are budgeted.

        Args:
            board: The run's board, holding the split, metric and budget.
            selected: The selected trial.

        Returns:
            The learning curve, nested CV, bias-variance and sensitivity.
        """
        return Diagnostics(
            self.chart(board, selected),
            self.revalidate(board, selected),
            self.decompose(board, selected),
            self.sensitivity(board, selected),
        )

    def chart(
        self,
        board: engine.Board,
        selected: experiments.Trial,
    ) -> dict[str, Any] | None:
        """Returns the selected candidate's learning curve on its first fold.

        Args:
            board: The run's board, holding the split, metric and budget.
            selected: The selected trial, used to build the model.

        Returns:
            The curve as a record, or None when the budget is spent, the
            fold has too few training rows or too few usable points result.
        """
        if cost.halt(board, selected, "learning curve"):
            return None
        partition = board.get(profiling.Partition)
        split = partition.split
        template = recipe(board, selected)(selected.params)
        metric = partition.metric
        fit_positions, evaluate_positions = split.manifest.folds[0]
        curve = learning_curve.compute(
            template,
            split.features,
            split.target,
            fit_positions,
            evaluate_positions,
            metric.scorer(
                scmle.evaluation.metrics.positive_label(
                    split.target,
                    partition.task,
                )
            ),
            metric.greater_is_better,
            lambda: bool(cost.halt(board, selected, "learning curve")),
        )
        return None if curve is None else curve.to_dict()

    def decompose(
        self,
        board: engine.Board,
        selected: experiments.Trial,
    ) -> dict[str, Any] | None:
        """Splits the selected model's error into bias and variance.

        Args:
            board: The run's board, holding the split and settings.
            selected: The selected trial.

        Returns:
            The decomposition; a record with a `skipped` reason when the task is
            not binary, multiclass or regression, or the budget was spent; None
            when no replicates are configured.
        """
        replicates = board.services.spec.selection.bias_variance_replicates
        partition = board.get(profiling.Partition)
        task = partition.task
        if not replicates:
            return None
        if task not in (
            tasks.Task.BINARY,
            tasks.Task.MULTICLASS,
            tasks.Task.REGRESSION,
        ):
            return {"skipped": f"not defined for {task.value}"}
        split = partition.split
        build = recipe(board, selected)
        found = bias_variance.decompose(
            lambda: build(selected.params),
            split.features,
            split.target,
            task.is_classification,
            replicates,
            board.services.spec.validation.seed,
            lambda: bool(cost.halt(board, selected, "bias-variance estimate")),
        )
        if found is None:
            return {"skipped": "time budget spent or too few predictions"}
        return found.to_dict()

    def sensitivity(
        self,
        board: engine.Board,
        selected: experiments.Trial,
    ) -> dict[str, Any] | None:
        """Measures how the selected model's score varies with its inputs.

        The sources are the training seed, the resample and the split, and
        added noise. Fewer rows than the analysis minimum
        (`sensitivity.MIN_ROWS`) raise `ValueError` from `sensitivity.analyze`.

        Args:
            board: The run's board, holding the split and settings.
            selected: The selected trial.

        Returns:
            The spread report with its dominant source; a record with a
            `skipped` reason for other tasks or when the budget was spent; None
            when no repeats are configured.
        """
        repeats = board.services.spec.selection.sensitivity_repeats
        partition = board.get(profiling.Partition)
        task = partition.task
        if not repeats:
            return None
        if task not in (
            tasks.Task.BINARY,
            tasks.Task.MULTICLASS,
            tasks.Task.REGRESSION,
        ):
            return {"skipped": f"not defined for {task.value}"}
        if cost.halt(board, selected, "sensitivity analysis"):
            return {"skipped": "time budget spent"}
        split = partition.split
        if len(split.target) < sensitivity.MIN_ROWS:
            return {"skipped": "too few rows"}
        build = recipe(board, selected)
        metric = partition.metric
        scorer = metric.scorer(
            scmle.evaluation.metrics.positive_label(split.target, task)
        )
        report = sensitivity.analyze(
            lambda: build(selected.params),
            split.features,
            split.target,
            scorer,
            repeats,
            board.services.spec.validation.seed,
        )
        return {**report.to_dict(), "dominant": report.dominant()}

    def revalidate(
        self,
        board: engine.Board,
        selected: experiments.Trial,
    ) -> dict[str, Any] | None:
        """Estimates the optimism of the winning score by nested validation.

        Args:
            board: The run's board, holding the split and settings.
            selected: The selected trial.

        Returns:
            The nested-score record, with the mean and optimism in the metric's
            natural direction; a record with a `skipped` reason when the
            validation strategy is not random or stratified, the budget was
            spent or there were too few rows; None when nested validation is
            off.
        """
        if not board.services.spec.selection.nested_cv:
            return None
        partition = board.get(profiling.Partition)
        split = partition.split
        if split.manifest.strategy not in {"random", "stratified"}:
            return {
                "skipped": f"not defined for {split.manifest.strategy} "
                "validation"
            }
        if cost.halt(board, selected, "nested validation"):
            return {"skipped": "time budget spent"}
        task = partition.task
        adapter = board.services.registry.get(selected.model_name)
        build = recipe(board, selected)
        metric = partition.metric
        positive = scmle.evaluation.metrics.positive_label(split.target, task)
        result = nested.nest(
            build,
            adapter.search_space,
            split.features,
            split.target,
            [(f[0], f[1]) for f in split.manifest.folds],
            metric.scorer(positive),
            task.is_classification,
            board.services.spec.validation.seed,
            should_stop=lambda: bool(
                cost.halt(board, selected, "nested validation")
            ),
        )
        if result is None:
            return {"skipped": "time budget spent or too few rows"}
        sign = 1.0 if metric.greater_is_better else -1.0
        record = result.to_dict()
        record["mean_natural"] = sign * result.mean
        record["optimism_natural"] = sign * result.optimism
        return record

    def forecast(
        self,
        board: engine.Board,
        selected: experiments.Trial,
    ) -> dict[str, Any] | None:
        """Per-step errors against baselines, and one row per rolling origin.

        The locked rows give the headline comparison. Each cross-validation
        fold (an origin) refits the selected candidate on its past and scores
        the future that followed.

        Args:
            board: The run's board, holding the split and test rows.
            selected: The selected trial.

        Returns:
            A record with the headline `steps` (None if the test rows were not
            opened), the per-origin `origins` and the `seasonal_period`, or
            None when the time budget is spent.
        """
        if cost.halt(board, selected, "forecast backtest"):
            return None
        partition = board.get(profiling.Partition)
        split = partition.split
        data = board.services.spec.data
        template = recipe(board, selected)(selected.params)
        fitted = []
        for fit_positions, evaluate_positions in split.manifest.folds:
            if cost.halt(board, selected, "forecast backtest"):
                return None
            fit = numpy.asarray(fit_positions)
            held = numpy.asarray(evaluate_positions)
            model = cast(
                "scmle.contracts.models.Estimator", base.clone(template)
            ).fit(split.features.iloc[fit], split.target.iloc[fit])
            fitted.append(
                (
                    model,
                    split.features.iloc[held],
                    split.target.iloc[held],
                )
            )
        headline = None
        exam = board.get(assessment.Exam)
        if exam.rows is not None:
            features, target = exam.rows
            headline = backtest.report(
                exam.pipeline,
                features,
                target,
                partition.seasonal,
            )
        return {
            "steps": headline,
            "origins": backtest.origins(fitted),
            "seasonal_period": data.seasonal_period,
        }
