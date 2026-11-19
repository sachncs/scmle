"""The test scores, diagnostics and evaluation record of the final model."""

from typing import Any

import numpy
import pandas

from scmle.contracts import experiments
from scmle.contracts import records
from scmle.contracts import tasks
from scmle.evaluation import confidence
from scmle.evaluation import criteria
from scmle.evaluation import diagnostics
from scmle.pipeline import suite
import scmle.pipeline.context
from scmle.pipeline.stages import reporting
from scmle.pipeline.stages import scoring

BASELINE = "dummy"


class Summary(reporting.Reporting):
    """The test scores, diagnostic reports and evaluation record.

    Part of `FinalEvaluationStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def measure(
        self,
        context: scmle.pipeline.context.Context,
        pipeline: Any,
        features: pandas.DataFrame,
        target: pandas.Series,
    ) -> records.Layout:
        """Scores the final pipeline on the test rows and tracks the scores.

        Args:
            context: The run context; receives the reference scores.
            pipeline: The final pipeline.
            features: The test features.
            target: The test labels.

        Returns:
            The test metrics.
        """
        selected = scmle.pipeline.context.require(context.selected, "selected")
        task = scmle.pipeline.context.require(context.task, "task")
        metric = scmle.pipeline.context.require(context.metric, "metric")
        test_metrics = context.services.catalog.evaluate(
            pipeline, features, target, task
        )
        context.journal.track(
            selected_strategy=selected.strategy,
            selected_model=selected.model_name,
            metric=metric.name,
            cv_score=selected.score_mean,
            test_metrics=test_metrics,
        )
        context.reference_scores = scoring.reference_scores(
            pipeline, features, task
        )
        return test_metrics

    def examine(
        self,
        context: scmle.pipeline.context.Context,
        pipeline: Any,
        features: pandas.DataFrame,
        target: pandas.Series,
    ) -> None:
        """Runs the diagnostic suite on the test rows and writes its reports.

        Args:
            context: The run context; receives the diagnostics.
            pipeline: The final pipeline.
            features: The test features.
            target: The test labels.
        """
        split = scmle.pipeline.context.require(context.split, "split")
        data = context.services.spec.data
        context.diagnostics = suite.run_suite(
            diagnostics.DiagnosticInput(
                pipeline,
                features,
                target,
                scmle.pipeline.context.require(context.task, "task"),
                scmle.pipeline.context.require(context.metric, "metric"),
                scmle.pipeline.context.require(
                    context.profile, "profile"
                ).feature_plan,
                context.services.spec.validation.seed,
                groups=(
                    features[data.group_column]
                    if (
                        split.manifest.strategy == "group"
                        or context.cluster_column
                    )
                    and data.group_column
                    else None
                ),
                temporal=split.manifest.strategy == "temporal",
                reference_features=split.dev_features,
                sensitive_columns=tuple(data.sensitive_columns),
            ),
            suite.lineup(),
            self.ration(context),
            context.services.governor.until("finish")
            - context.services.governor.reserve.parts.get("package", 0.0),
        )
        for name, report_body in context.diagnostics.items():
            context.journal.write_json(f"diagnostics/{name}.json", report_body)
            if report_body.get("skipped") is True:
                context.journal.forgo(
                    f"diagnostic {name}", report_body.get("reason", "")
                )
            elif report_body.get("truncated"):
                context.journal.forgo(
                    f"diagnostic {name} (cut short)",
                    str(report_body.get("reason", "")),
                )

    def document(
        self,
        context: scmle.pipeline.context.Context,
        pipeline: Any,
        rows: tuple[pandas.DataFrame, pandas.Series],
        test_metrics: records.Layout,
        verdict: records.Layout,
    ) -> records.Layout:
        """Builds the evaluation record, up to the decision it reports.

        Args:
            context: The run context.
            pipeline: The final pipeline.
            rows: The test features and labels.
            test_metrics: The test metrics.
            verdict: The decision threshold (`decision`) and its net benefit
                (`net_benefit`).

        Returns:
            The record's entries from `run_id` to `net_benefit`, in order.
        """
        split = scmle.pipeline.context.require(context.split, "split")
        selected = scmle.pipeline.context.require(context.selected, "selected")
        task = scmle.pipeline.context.require(context.task, "task")
        metric = scmle.pipeline.context.require(context.metric, "metric")
        features, target = rows
        baseline = next(
            (
                t
                for t in context.journal.trials
                if t.model_name == BASELINE
                and t.status is experiments.Termination.OK
            ),
            None,
        )
        return {
            "run_id": store.run_id if (store := context.services.store) else "",
            "task": task.value,
            "metric": metric.name,
            "metric_greater_is_better": metric.greater_is_better,
            "selection_policy": context.services.policy.description,
            "selected_model": selected.model_name,
            "selected_trial_id": selected.trial_id,
            "selected_params": selected.params,
            "selected_strategy": selected.strategy,
            "selected_cv_score": selected.score_mean,
            "selected_cv_std": selected.score_std,
            "fit_health": self.diagnose(context, selected),
            "cv_confidence": self.assure(context),
            "baseline_cv_score": baseline.score_mean if baseline else None,
            "test_metrics": test_metrics,
            "weighted_test_metrics": self.weigh(
                context, pipeline, features, target
            ),
            "semi_supervised": context.semi_supervised,
            "fairness_mitigation": context.fairness_mitigation,
            "execution_backend": context.services.backend.name,
            "pareto_front": context.pareto_front,
            "evaluation_prevalence": {
                "natural": True,
                "note": "resampling strategies change only training folds; "
                "validation and test rows keep their real class mix",
                "strategy": selected.strategy,
            },
            "success_criteria": [
                r.to_dict()
                for r in criteria.check(
                    context.services.spec.problem.success_criteria,
                    test_metrics,
                    context.services.catalog,
                )
            ],
            "streamed_locked": self.tally(context, pipeline),
            "uplift": self.uplift_report(context, pipeline, features, target),
            "n_dev": len(split.dev_features),
            "n_test": len(features),
            "split_fingerprint": split.manifest.fingerprint,
            "split_strategy": split.manifest.strategy,
            **verdict,
        }

    def assure(self, context: scmle.pipeline.context.Context) -> Any:
        """The confidence interval of the selected cross-validation score.

        Args:
            context: The run context, holding the split, trials and selection.

        Returns:
            The interval record of `confidence.cv_interval`.
        """
        split = scmle.pipeline.context.require(context.split, "split")
        selected = scmle.pipeline.context.require(context.selected, "selected")
        metric = scmle.pipeline.context.require(context.metric, "metric")
        return confidence.cv_interval(
            selected.fold_scores,
            float(numpy.mean([len(fold[0]) for fold in split.manifest.folds])),
            float(numpy.mean([len(fold[1]) for fold in split.manifest.folds])),
            split.manifest.strategy,
            [
                trial.fold_scores
                for trial in context.journal.trials
                if trial.status is experiments.Termination.OK
            ],
            metric.greater_is_better,
        )

    def weigh(
        self,
        context: scmle.pipeline.context.Context,
        pipeline: Any,
        features: pandas.DataFrame,
        target: pandas.Series,
    ) -> records.Layout | None:
        """Scores the test rows under the declared row weights.

        Args:
            context: The run context.
            pipeline: The final pipeline.
            features: The test features.
            target: The test labels.

        Returns:
            The weighted test metrics, or None without a weight column (the
            fairness reweighing column does not count).
        """
        column = context.services.spec.data.weight_column
        if not column or column == tasks.FAIRNESS_WEIGHT:
            return None
        return context.services.catalog.evaluate(
            pipeline,
            features,
            target,
            scmle.pipeline.context.require(context.task, "task"),
            features[column].to_numpy(dtype=float),
        )
