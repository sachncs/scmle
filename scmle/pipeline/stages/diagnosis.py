"""Post-selection diagnostics run while packaging."""

from typing import Any
from typing import cast

import numpy
from sklearn import base

from scmle.contracts import experiments
from scmle.contracts import tasks
import scmle.contracts.models
from scmle.evaluation import backtest
from scmle.evaluation import bias_variance
from scmle.evaluation import learning_curve
from scmle.evaluation import nested
from scmle.evaluation import sensitivity
import scmle.evaluation.metrics
import scmle.pipeline.context
from scmle.pipeline.stages import calibration
from scmle.pipeline.stages import cost
from scmle.preparation import strategies


class Diagnosis(calibration.Calibration):
    """Post-selection diagnostics of the selected model.

    Part of `PackageStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def learning_curve(
        self,
        context: scmle.pipeline.context.Context,
        selected: experiments.Trial,
    ) -> dict[str, Any] | None:
        """Returns the selected candidate's learning curve on its first fold.

        Args:
            context: The run context, holding the split, metric and budget.
            selected: The selected trial, used to build the model.

        Returns:
            The curve as a record, or None when the budget is spent, the
            fold has too few training rows or too few usable points result.
        """
        if cost.halt(context, selected, "learning curve"):
            return None
        split = scmle.pipeline.context.require(context.split, "split")
        template = strategies.lookup(selected.strategy).build(
            scmle.pipeline.context.require(
                context.profile, "profile"
            ).feature_plan,
            context.services.registry.get(selected.model_name),
            scmle.pipeline.context.require(context.task, "task"),
            context.services.spec.validation.seed,
            selected.params,
        )
        metric = scmle.pipeline.context.require(context.metric, "metric")
        fit_positions, evaluate_positions = split.manifest.folds[0]
        curve = learning_curve.compute(
            template,
            split.dev_features,
            split.dev_target,
            fit_positions,
            evaluate_positions,
            metric.scorer(
                scmle.evaluation.metrics.positive_label(
                    split.dev_target,
                    scmle.pipeline.context.require(context.task, "task"),
                )
            ),
            metric.greater_is_better,
            lambda: bool(cost.halt(context, selected, "learning curve")),
        )
        return None if curve is None else curve.to_dict()

    def bias_variance(
        self,
        context: scmle.pipeline.context.Context,
        selected: experiments.Trial,
    ) -> dict[str, Any] | None:
        """Splits the selected model's error into bias and variance.

        Args:
            context: The run context, holding the split and settings.
            selected: The selected trial.

        Returns:
            The decomposition; a record with a `skipped` reason when the task is
            not binary, multiclass or regression, or the budget was spent; None
            when no replicates are configured.
        """
        replicates = context.services.spec.selection.bias_variance_replicates
        task = scmle.pipeline.context.require(context.task, "task")
        if not replicates:
            return None
        if task not in (
            tasks.Task.BINARY,
            tasks.Task.MULTICLASS,
            tasks.Task.REGRESSION,
        ):
            return {"skipped": f"not defined for {task.value}"}
        split = scmle.pipeline.context.require(context.split, "split")
        adapter = context.services.registry.get(selected.model_name)
        strategy = strategies.lookup(selected.strategy)
        plan = scmle.pipeline.context.require(
            context.profile, "profile"
        ).feature_plan
        seed = context.services.spec.validation.seed
        found = bias_variance.decompose(
            lambda: strategy.build(plan, adapter, task, seed, selected.params),
            split.dev_features,
            split.dev_target,
            task.is_classification,
            replicates,
            seed,
            lambda: bool(
                cost.halt(context, selected, "bias-variance estimate")
            ),
        )
        if found is None:
            return {"skipped": "time budget spent or too few predictions"}
        return found.to_dict()

    def sensitivity(
        self,
        context: scmle.pipeline.context.Context,
        selected: experiments.Trial,
    ) -> dict[str, Any] | None:
        """Measures how the selected model's score varies with its inputs.

        The sources are the training seed, the resample and the split, and
        added noise. Fewer rows than the analysis minimum
        (`sensitivity.MIN_ROWS`) raise `ValueError` from `sensitivity.analyze`.

        Args:
            context: The run context, holding the split and settings.
            selected: The selected trial.

        Returns:
            The spread report with its dominant source; a record with a
            `skipped` reason for other tasks or when the budget was spent; None
            when no repeats are configured.
        """
        repeats = context.services.spec.selection.sensitivity_repeats
        task = scmle.pipeline.context.require(context.task, "task")
        if not repeats:
            return None
        if task not in (
            tasks.Task.BINARY,
            tasks.Task.MULTICLASS,
            tasks.Task.REGRESSION,
        ):
            return {"skipped": f"not defined for {task.value}"}
        if cost.halt(context, selected, "sensitivity analysis"):
            return {"skipped": "time budget spent"}
        split = scmle.pipeline.context.require(context.split, "split")
        if len(split.dev_target) < sensitivity.MIN_ROWS:
            return {"skipped": "too few rows"}
        adapter = context.services.registry.get(selected.model_name)
        strategy = strategies.lookup(selected.strategy)
        plan = scmle.pipeline.context.require(
            context.profile, "profile"
        ).feature_plan
        seed = context.services.spec.validation.seed
        metric = scmle.pipeline.context.require(context.metric, "metric")
        scorer = metric.scorer(
            scmle.evaluation.metrics.positive_label(split.dev_target, task)
        )
        report = sensitivity.analyze(
            lambda: strategy.build(plan, adapter, task, seed, selected.params),
            split.dev_features,
            split.dev_target,
            scorer,
            repeats,
            seed,
        )
        return {**report.to_dict(), "dominant": report.dominant()}

    def nested_validation(
        self,
        context: scmle.pipeline.context.Context,
        selected: experiments.Trial,
    ) -> dict[str, Any] | None:
        """Estimates the optimism of the winning score by nested validation.

        Args:
            context: The run context, holding the split and settings.
            selected: The selected trial.

        Returns:
            The nested-score record, with the mean and optimism in the metric's
            natural direction; a record with a `skipped` reason when the
            validation strategy is not random or stratified, the budget was
            spent or there were too few rows; None when nested validation is
            off.
        """
        if not context.services.spec.selection.nested_cv:
            return None
        split = scmle.pipeline.context.require(context.split, "split")
        if split.manifest.strategy not in {"random", "stratified"}:
            return {
                "skipped": f"not defined for {split.manifest.strategy} "
                "validation"
            }
        if cost.halt(context, selected, "nested validation"):
            return {"skipped": "time budget spent"}
        task = scmle.pipeline.context.require(context.task, "task")
        adapter = context.services.registry.get(selected.model_name)
        strategy = strategies.lookup(selected.strategy)
        plan = scmle.pipeline.context.require(
            context.profile, "profile"
        ).feature_plan
        seed = context.services.spec.validation.seed
        metric = scmle.pipeline.context.require(context.metric, "metric")
        positive = scmle.evaluation.metrics.positive_label(
            split.dev_target, task
        )
        result = nested.nest(
            lambda params: strategy.build(plan, adapter, task, seed, params),
            adapter.search_space,
            split.dev_features,
            split.dev_target,
            [(f[0], f[1]) for f in split.manifest.folds],
            metric.scorer(positive),
            task.is_classification,
            seed,
            should_stop=lambda: bool(
                cost.halt(context, selected, "nested validation")
            ),
        )
        if result is None:
            return {"skipped": "time budget spent or too few rows"}
        sign = 1.0 if metric.greater_is_better else -1.0
        record = result.to_dict()
        record["mean_natural"] = sign * result.mean
        record["optimism_natural"] = sign * result.optimism
        return record

    def forecast_backtest(
        self,
        context: scmle.pipeline.context.Context,
        selected: experiments.Trial,
    ) -> dict[str, Any] | None:
        """Per-step errors against baselines, and one row per rolling origin.

        The locked rows give the headline comparison. Each cross-validation
        fold (an origin) refits the selected candidate on its past and scores
        the future that followed.

        Args:
            context: The run context, holding the split and test rows.
            selected: The selected trial.

        Returns:
            A record with the headline `steps` (None if the test rows were not
            opened), the per-origin `origins` and the `seasonal_period`, or
            None when the time budget is spent.
        """
        if cost.halt(context, selected, "forecast backtest"):
            return None
        split = scmle.pipeline.context.require(context.split, "split")
        data = context.services.spec.data
        template = strategies.lookup(selected.strategy).build(
            scmle.pipeline.context.require(
                context.profile, "profile"
            ).feature_plan,
            context.services.registry.get(selected.model_name),
            scmle.pipeline.context.require(context.task, "task"),
            context.services.spec.validation.seed,
            selected.params,
        )
        fitted = []
        for fit_positions, evaluate_positions in split.manifest.folds:
            if cost.halt(context, selected, "forecast backtest"):
                return None
            fit = numpy.asarray(fit_positions)
            held = numpy.asarray(evaluate_positions)
            model = cast(
                "scmle.contracts.models.Estimator", base.clone(template)
            ).fit(split.dev_features.iloc[fit], split.dev_target.iloc[fit])
            fitted.append(
                (
                    model,
                    split.dev_features.iloc[held],
                    split.dev_target.iloc[held],
                )
            )
        headline = None
        if context.test_rows is not None:
            features, target = context.test_rows
            headline = backtest.report(
                context.final_pipeline,
                features,
                target,
                context.seasonal_baseline,
            )
        return {
            "steps": headline,
            "origins": backtest.origins(fitted),
            "seasonal_period": data.seasonal_period,
        }
