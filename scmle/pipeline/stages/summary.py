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
from scmle.pipeline import engine
from scmle.pipeline import products
from scmle.pipeline import suite
from scmle.pipeline.stages import examination
from scmle.pipeline.stages import exploration
from scmle.pipeline.stages import profiling
from scmle.pipeline.stages import reporting
from scmle.pipeline.stages import scoring
from scmle.pipeline.stages import selecting

BASELINE = "dummy"


class Summary(reporting.Reporting):
    """The test scores, diagnostic reports and evaluation record.

    Part of `Assessment`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def measure(
        self,
        board: engine.Board,
        pipeline: Any,
        features: pandas.DataFrame,
        target: pandas.Series,
    ) -> records.Layout:
        """Scores the final pipeline on the test rows and tracks the scores.

        Args:
            board: The run's board.
            pipeline: The final pipeline.
            features: The test features.
            target: The test labels.

        Returns:
            The test metrics.
        """
        selected = products.require(
            board.get(selecting.Choice).selected, "selected"
        )
        partition = board.get(profiling.Partition)
        task = partition.task
        metric = partition.metric
        test_metrics = board.services.catalog.evaluate(
            pipeline, features, target, task
        )
        board.journal.track(
            selected_strategy=selected.strategy,
            selected_model=selected.model_name,
            metric=metric.name,
            cv_score=selected.score_mean,
            test_metrics=test_metrics,
        )
        return test_metrics

    def examine(
        self,
        board: engine.Board,
        pipeline: Any,
        features: pandas.DataFrame,
        target: pandas.Series,
    ) -> examination.Readings:
        """Runs the diagnostic suite on the test rows and writes its reports.

        Args:
            board: The run's board.
            pipeline: The final pipeline.
            features: The test features.
            target: The test labels.

        Returns:
            The model's reference scores on the test rows and the
            diagnostic reports.
        """
        partition = board.get(profiling.Partition)
        split, task = partition.split, partition.task
        data = board.services.spec.data
        custody = board.find(exploration.Custody)
        clustered = custody is not None and custody.column is not None
        reports = suite.run(
            diagnostics.Input(
                pipeline,
                features,
                target,
                task,
                partition.metric,
                partition.profile.feature_plan,
                board.services.spec.validation.seed,
                groups=(
                    features[data.group_column]
                    if (split.manifest.strategy == "group" or clustered)
                    and data.group_column
                    else None
                ),
                temporal=split.manifest.strategy == "temporal",
                reference=split.features,
                sensitive=tuple(data.sensitive_columns),
            ),
            suite.lineup(),
            self.ration(board),
            board.services.governor.until("finish")
            - board.services.governor.reserve.parts.get("package", 0.0),
        )
        for name, report_body in reports.items():
            board.journal.write(f"diagnostics/{name}.json", report_body)
            if report_body.get("skipped") is True:
                board.journal.forgo(
                    f"diagnostic {name}", report_body.get("reason", "")
                )
            elif report_body.get("truncated"):
                board.journal.forgo(
                    f"diagnostic {name} (cut short)",
                    str(report_body.get("reason", "")),
                )
        return examination.Readings(
            scoring.anchor(pipeline, features, task), reports
        )

    def document(
        self,
        board: engine.Board,
        pipeline: Any,
        rows: tuple[pandas.DataFrame, pandas.Series],
        test_metrics: records.Layout,
        judged: tuple[examination.Pseudolabels, examination.Remedy],
        verdict: records.Layout,
    ) -> records.Layout:
        """Builds the evaluation record, up to the decision it reports.

        Args:
            board: The run's board.
            pipeline: The final pipeline.
            rows: The test features and labels.
            test_metrics: The test metrics.
            judged: The final fit's self-training judgement and the
                group-threshold remedy.
            verdict: The decision threshold (`decision`) and its net benefit
                (`net_benefit`).

        Returns:
            The record's entries from `run_id` to `net_benefit`, in order.
        """
        partition = board.get(profiling.Partition)
        split = partition.split
        choice = board.get(selecting.Choice)
        selected = products.require(choice.selected, "selected")
        task = partition.task
        metric = partition.metric
        features, target = rows
        pseudolabels, remedy = judged
        return {
            "run_id": store.run_id if (store := board.services.store) else "",
            "task": task.value,
            "metric": metric.name,
            "metric_greater_is_better": metric.greater_is_better,
            "selection_policy": board.services.policy.description,
            "selected_model": selected.model_name,
            "selected_trial_id": selected.trial_id,
            "selected_params": selected.params,
            "selected_strategy": selected.strategy,
            "selected_cv_score": selected.score_mean,
            "selected_cv_std": selected.score_std,
            "fit_health": self.diagnose(board, selected),
            "cv_confidence": self.assure(board),
            "baseline_cv_score": self.baseline(board),
            "test_metrics": test_metrics,
            "weighted_test_metrics": self.weigh(
                board, pipeline, features, target
            ),
            "semi_supervised": pseudolabels.record,
            "fairness_mitigation": remedy.record,
            "execution_backend": board.services.backend.name,
            "pareto_front": choice.front,
            "evaluation_prevalence": {
                "natural": True,
                "note": "resampling strategies change only training folds; "
                "validation and test rows keep their real class mix",
                "strategy": selected.strategy,
            },
            "success_criteria": [
                r.to_dict()
                for r in criteria.check(
                    board.services.spec.problem.success_criteria,
                    test_metrics,
                    board.services.catalog,
                )
            ],
            "streamed_locked": self.tally(board, pipeline),
            "uplift": self.report(board, pipeline, features, target),
            "n_dev": len(split.features),
            "n_test": len(features),
            "split_fingerprint": split.manifest.fingerprint,
            "split_strategy": split.manifest.strategy,
            **verdict,
        }

    def baseline(self, board: engine.Board) -> Any:
        """The cross-validation score of the `BASELINE` model.

        Args:
            board: The run's board, holding the trials.

        Returns:
            The mean score of the baseline's successful trial, or None
            when it has none.
        """
        found = next(
            (
                t
                for t in board.journal.trials
                if t.model_name == BASELINE
                and t.status is experiments.Termination.OK
            ),
            None,
        )
        return found.score_mean if found else None

    def assure(self, board: engine.Board) -> Any:
        """The confidence interval of the selected cross-validation score.

        Args:
            board: The run's board, holding the split, trials and selection.

        Returns:
            The interval record of `confidence.bound`.
        """
        partition = board.get(profiling.Partition)
        split = partition.split
        selected = products.require(
            board.get(selecting.Choice).selected, "selected"
        )
        metric = partition.metric
        return confidence.bound(
            selected.fold_scores,
            float(numpy.mean([len(fold[0]) for fold in split.manifest.folds])),
            float(numpy.mean([len(fold[1]) for fold in split.manifest.folds])),
            split.manifest.strategy,
            [
                trial.fold_scores
                for trial in board.journal.trials
                if trial.status is experiments.Termination.OK
            ],
            metric.greater_is_better,
        )

    def weigh(
        self,
        board: engine.Board,
        pipeline: Any,
        features: pandas.DataFrame,
        target: pandas.Series,
    ) -> records.Layout | None:
        """Scores the test rows under the declared row weights.

        Args:
            board: The run's board.
            pipeline: The final pipeline.
            features: The test features.
            target: The test labels.

        Returns:
            The weighted test metrics, or None without a weight column (the
            fairness reweighing column does not count).
        """
        column = board.services.spec.data.weight_column
        if not column or column == tasks.FAIRNESS_WEIGHT:
            return None
        return board.services.catalog.evaluate(
            pipeline,
            features,
            target,
            board.get(profiling.Partition).task,
            features[column].to_numpy(dtype=float),
        )
