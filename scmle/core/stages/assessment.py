"""The final evaluation stage on the locked test set."""

import time
from typing import ClassVar

import numpy

from scmle.contracts import experiments
from scmle.contracts import tasks
import scmle.core.context
from scmle.core.stages import cost
from scmle.core.stages import reporting
from scmle.core.stages import scoring
from scmle.core.stages import selecting
from scmle.evaluation import confidence
from scmle.evaluation import criteria
from scmle.evaluation import diagnostics
from scmle.evaluation import suite
import scmle.runtime.state
from scmle.sampling import strategies

BASELINE = "dummy"


class FinalEvaluationStage(reporting.Reporting):
    """Refits the frozen candidate and opens the locked test set once."""

    requires: ClassVar[tuple[type, ...]] = (selecting.SelectStage,)
    state: ClassVar[scmle.runtime.state.Progress] = (
        scmle.runtime.state.Progress.TESTING
    )

    def describe(self, context: scmle.core.context.Context) -> str:
        """Names the selected model.

        Args:
            context: The run context, holding the selected trial.

        Returns:
            The text "final one-time test evaluation of" and the model name.
        """
        selected = scmle.core.context.require(context.selected, "selected")
        return f"final one-time test evaluation of {selected.model_name}"

    def execute(self, context: scmle.core.context.Context) -> None:
        """Refits on all development data and evaluates on the test set.

        Reuses the refit made during selection if there is one. Applies
        pseudo-labeling, streamed training, the decision threshold and the
        fairness mitigation, registers the coverage claim, then opens the
        locked rows once, after the selection is frozen. Records the test
        metrics, diagnostics and the evaluation document in the context.

        Args:
            context: The run context; receives the final model, the test rows
                and the evaluation.
        """
        context.check_cancelled()
        split = scmle.core.context.require(context.split, "split")
        selected = scmle.core.context.require(context.selected, "selected")
        task = scmle.core.context.require(context.task, "task")
        metric = scmle.core.context.require(context.metric, "metric")
        pipeline = context.refits.get(selected.trial_id)
        if pipeline is None:
            pipeline = strategies.lookup(selected.strategy).build(
                scmle.core.context.require(
                    context.profile, "profile"
                ).feature_plan,
                context.registry.get(selected.model_name),
                task,
                context.spec.validation.seed,
                selected.params,
            )
            pipeline.fit(split.dev_features, split.dev_target)
        pipeline = self.pseudolabel(context, selected, pipeline)
        pipeline = self.relearn(context, pipeline)
        pipeline, decision = scoring.choose_decision_threshold(
            context, pipeline
        )
        unmitigated = pipeline
        pipeline = self.mitigate(context, selected, pipeline, decision)
        self.pledge(context)
        locked = split.locked_test
        binding = scoring.pin(selected, decision)
        locked.bind(binding)
        test_features, test_target = locked.open(locked.mint(), binding)
        if (
            context.spec.data.weight_column == tasks.FAIRNESS_WEIGHT
            and tasks.FAIRNESS_WEIGHT not in test_features
        ):
            # Reweighing weights development rows only; locked rows carry a
            # neutral weight so the pipeline finds the column it expects.
            test_features = test_features.assign(**{tasks.FAIRNESS_WEIGHT: 1.0})
        context.test_rows = (test_features, test_target)
        self.corroborate(
            context, unmitigated, pipeline, test_features, test_target
        )
        context.emit(
            "test.opened",
            "locked test set released once, after selection was frozen",
            data={
                "rows": len(test_features),
                "fingerprint": split.manifest.fingerprint,
            },
        )
        baseline = next(
            (
                t
                for t in context.trials
                if t.model_name == BASELINE
                and t.status is experiments.Termination.OK
            ),
            None,
        )
        context.final_pipeline = pipeline
        usage = cost.meter(context, pipeline, split.dev_features)
        test_metrics = context.catalog.evaluate(
            pipeline, test_features, test_target, task
        )
        weight_column = context.spec.data.weight_column
        weighted_test_metrics = (
            context.catalog.evaluate(
                pipeline,
                test_features,
                test_target,
                task,
                test_features[weight_column].to_numpy(dtype=float),
            )
            if weight_column and weight_column != tasks.FAIRNESS_WEIGHT
            else None
        )
        context.track(
            selected_strategy=selected.strategy,
            selected_model=selected.model_name,
            metric=metric.name,
            cv_score=selected.score_mean,
            test_metrics=test_metrics,
        )
        context.reference_scores = scoring.reference_scores(
            pipeline, test_features, task
        )
        context.diagnostics = suite.run_suite(
            diagnostics.DiagnosticInput(
                pipeline,
                test_features,
                test_target,
                task,
                metric,
                scmle.core.context.require(
                    context.profile, "profile"
                ).feature_plan,
                context.spec.validation.seed,
                groups=(
                    test_features[context.spec.data.group_column]
                    if (
                        split.manifest.strategy == "group"
                        or context.cluster_column
                    )
                    and context.spec.data.group_column
                    else None
                ),
                temporal=split.manifest.strategy == "temporal",
                reference_features=split.dev_features,
                sensitive_columns=tuple(context.spec.data.sensitive_columns),
            ),
            suite.lineup(),
            self.ration(context),
            context.regulator.until("finish")
            - context.regulator.reserve.parts.get("package", 0.0),
        )
        for name, report_body in context.diagnostics.items():
            context.write_json(f"diagnostics/{name}.json", report_body)
            if report_body.get("skipped") is True:
                context.forgo(
                    f"diagnostic {name}", report_body.get("reason", "")
                )
            elif report_body.get("truncated"):
                context.forgo(
                    f"diagnostic {name} (cut short)",
                    str(report_body.get("reason", "")),
                )
        net_benefit = scoring.appraise(
            context, pipeline, test_features, test_target, decision
        )
        self.decide(
            context, pipeline, test_features, test_target, decision, net_benefit
        )
        context.evaluation = {
            "run_id": context.store.run_id if context.store else "",
            "task": task.value,
            "metric": metric.name,
            "metric_greater_is_better": metric.greater_is_better,
            "selection_policy": context.policy.description,
            "selected_model": selected.model_name,
            "selected_trial_id": selected.trial_id,
            "selected_params": selected.params,
            "selected_strategy": selected.strategy,
            "selected_cv_score": selected.score_mean,
            "selected_cv_std": selected.score_std,
            "fit_health": self.diagnose(context, selected),
            "cv_confidence": confidence.cv_interval(
                selected.fold_scores,
                float(
                    numpy.mean([len(fold[0]) for fold in split.manifest.folds])
                ),
                float(
                    numpy.mean([len(fold[1]) for fold in split.manifest.folds])
                ),
                split.manifest.strategy,
                [
                    trial.fold_scores
                    for trial in context.trials
                    if trial.status is experiments.Termination.OK
                ],
                metric.greater_is_better,
            ),
            "baseline_cv_score": baseline.score_mean if baseline else None,
            "test_metrics": test_metrics,
            "weighted_test_metrics": weighted_test_metrics,
            "semi_supervised": context.semi_supervised,
            "fairness_mitigation": context.fairness_mitigation,
            "execution_backend": context.backend.name,
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
                    context.spec.problem.success_criteria,
                    test_metrics,
                    context.catalog,
                )
            ],
            "streamed_locked": self.tally(context, pipeline),
            "uplift": self.uplift_report(
                context, pipeline, test_features, test_target
            ),
            "n_dev": len(split.dev_features),
            "n_test": len(test_features),
            "split_fingerprint": split.manifest.fingerprint,
            "split_strategy": split.manifest.strategy,
            "decision": decision,
            "net_benefit": net_benefit,
            "resources": usage.to_dict(),
            "resource_limits": context.regulator.report(),
            "skipped_for_time": context.skipped,
            "test_set_opened": locked.opened,
            "diagnostics": sorted(context.diagnostics),
            "elapsed_seconds": time.monotonic() - context.started_at,
            "reproducibility": (
                "same seed/data/environment replays; threaded backends may "
                "differ"
            ),
        }
