"""Metric catalog with explicit orientation and a fixed scoring protocol.

Scores are always reported in natural units (log loss is positive);
`greater_is_better` tells rankers which direction wins.
"""

from collections.abc import Callable
import threading
from typing import Any

import pandas
from sklearn import metrics as sk_metrics

from scmle.contracts import errors
from scmle.contracts import models
from scmle.contracts import tasks
import scmle.contracts.metrics
from scmle.evaluation import concordance
from scmle.evaluation import forecast
from scmle.evaluation import multitarget
from scmle.evaluation import ordinal
from scmle.evaluation import ranking
from scmle.evaluation import survival
from scmle.evaluation import uplift

CLASSIFICATION = frozenset({tasks.TaskType.BINARY, tasks.TaskType.MULTICLASS})


def positive_label(target: pandas.Series, task: tasks.TaskType) -> Any:
    """Returns the positive class of a binary target (None otherwise).

    Follows scikit-learn's convention: the second class in sorted order
    among the non-missing labels, which is `classes_[1]` of every fitted
    classifier.

    Args:
        target: Labels.
        task: Resolved task type.

    Returns:
        The positive class for a binary task, otherwise None.

    Raises:
        ValueError: If a binary target has fewer than two distinct labels.
    """
    if task is not tasks.TaskType.BINARY:
        return None
    labels = sorted(target.dropna().unique())
    if len(labels) < 2:
        raise ValueError(
            "a binary target needs at least two distinct labels to name "
            f"the positive class; found {len(labels)}"
        )
    return labels[1]


DEFAULT_METRICS = (
    scmle.contracts.metrics.Metric(
        "roc_auc", "roc_auc", True, frozenset({tasks.TaskType.BINARY})
    ),
    scmle.contracts.metrics.Metric(
        "roc_auc_ovr",
        "roc_auc_ovr",
        True,
        frozenset({tasks.TaskType.MULTICLASS}),
    ),
    scmle.contracts.metrics.PositiveLabelMetric(
        "pr_auc",
        "average_precision",
        True,
        frozenset({tasks.TaskType.BINARY}),
        sk_metrics.average_precision_score,
        ("decision_function", "predict_proba"),
    ),
    scmle.contracts.metrics.Metric(
        "log_loss", "neg_log_loss", False, CLASSIFICATION
    ),
    scmle.contracts.metrics.PositiveLabelMetric(
        "f1",
        "f1",
        True,
        frozenset({tasks.TaskType.BINARY}),
        sk_metrics.f1_score,
        ("predict",),
    ),
    scmle.contracts.metrics.Metric(
        "f1_macro", "f1_macro", True, CLASSIFICATION
    ),
    scmle.contracts.metrics.Metric(
        "balanced_accuracy", "balanced_accuracy", True, CLASSIFICATION
    ),
    scmle.contracts.metrics.Metric(
        "accuracy", "accuracy", True, CLASSIFICATION
    ),
    scmle.contracts.metrics.Metric(
        "mae",
        "neg_mean_absolute_error",
        False,
        frozenset({tasks.TaskType.REGRESSION}),
    ),
    scmle.contracts.metrics.Metric(
        "rmse",
        "neg_root_mean_squared_error",
        False,
        frozenset({tasks.TaskType.REGRESSION}),
    ),
    scmle.contracts.metrics.Metric(
        "r2", "r2", True, frozenset({tasks.TaskType.REGRESSION})
    ),
    scmle.contracts.metrics.FunctionMetric(
        "qwk",
        "",
        True,
        frozenset({tasks.TaskType.ORDINAL}),
        ordinal.quadratic_weighted_kappa,
        "predict",
    ),
    scmle.contracts.metrics.FunctionMetric(
        "mace",
        "",
        False,
        frozenset({tasks.TaskType.ORDINAL}),
        ordinal.mean_absolute_class_error,
        "predict",
    ),
    scmle.contracts.metrics.Metric(
        "accuracy_levels", "accuracy", True, frozenset({tasks.TaskType.ORDINAL})
    ),
    multitarget.MultiTargetMetric(
        "f1_macro_labels",
        "",
        True,
        frozenset({tasks.TaskType.MULTILABEL}),
        "f1_macro_labels",
    ),
    multitarget.MultiTargetMetric(
        "f1_micro_labels",
        "",
        True,
        frozenset({tasks.TaskType.MULTILABEL}),
        "f1_micro_labels",
    ),
    multitarget.MultiTargetMetric(
        "hamming", "", False, frozenset({tasks.TaskType.MULTILABEL}), "hamming"
    ),
    multitarget.MultiTargetMetric(
        "subset_accuracy",
        "",
        True,
        frozenset({tasks.TaskType.MULTILABEL}),
        "subset_accuracy",
    ),
    multitarget.MultiTargetMetric(
        "r2_mean", "", True, frozenset({tasks.TaskType.MULTIOUTPUT}), "r2_mean"
    ),
    multitarget.MultiTargetMetric(
        "mae_scaled",
        "",
        False,
        frozenset({tasks.TaskType.MULTIOUTPUT}),
        "mae_scaled",
    ),
    forecast.ForecastMetric(
        "mase", "", False, frozenset({tasks.TaskType.FORECAST}), "mase"
    ),
    forecast.ForecastMetric(
        "forecast_mae", "", False, frozenset({tasks.TaskType.FORECAST}), "mae"
    ),
    forecast.ForecastMetric(
        "forecast_smape",
        "",
        False,
        frozenset({tasks.TaskType.FORECAST}),
        "smape",
    ),
    uplift.UpliftMetric(
        "qini", "", True, frozenset({tasks.TaskType.UPLIFT}), "qini"
    ),
    uplift.UpliftMetric(
        "uplift_at_30",
        "",
        True,
        frozenset({tasks.TaskType.UPLIFT}),
        "uplift_at_30",
    ),
    ranking.RankingMetric(
        "ndcg", "", True, frozenset({tasks.TaskType.RANKING}), "ndcg"
    ),
    ranking.RankingMetric(
        "map", "", True, frozenset({tasks.TaskType.RANKING}), "map"
    ),
    ranking.RankingMetric(
        "mrr", "", True, frozenset({tasks.TaskType.RANKING}), "mrr"
    ),
    ranking.RankingMetric(
        "precision_at_10",
        "",
        True,
        frozenset({tasks.TaskType.RANKING}),
        "precision_at_10",
    ),
    ranking.RankingMetric(
        "recall_at_10",
        "",
        True,
        frozenset({tasks.TaskType.RANKING}),
        "recall_at_10",
    ),
    concordance.ConcordanceMetric(
        "c_index", "", True, frozenset({tasks.TaskType.SURVIVAL})
    ),
    survival.SurvivalMetric(
        "ibs", "", False, frozenset({tasks.TaskType.SURVIVAL}), "ibs"
    ),
    survival.SurvivalMetric(
        "td_auc", "", True, frozenset({tasks.TaskType.SURVIVAL}), "td_auc"
    ),
)


class MetricCatalog:
    """Resolves metric names and applies the report protocol."""

    default_metric = {
        tasks.TaskType.BINARY: "roc_auc",
        tasks.TaskType.MULTICLASS: "log_loss",
        tasks.TaskType.REGRESSION: "rmse",
        tasks.TaskType.SURVIVAL: "c_index",
        tasks.TaskType.ORDINAL: "qwk",
        tasks.TaskType.RANKING: "ndcg",
        tasks.TaskType.MULTILABEL: "f1_macro_labels",
        tasks.TaskType.MULTIOUTPUT: "r2_mean",
        tasks.TaskType.FORECAST: "mase",
        tasks.TaskType.UPLIFT: "qini",
    }
    report_names = {
        tasks.TaskType.BINARY: (
            "roc_auc",
            "pr_auc",
            "log_loss",
            "f1",
            "balanced_accuracy",
            "accuracy",
        ),
        tasks.TaskType.MULTICLASS: (
            "log_loss",
            "f1_macro",
            "balanced_accuracy",
            "accuracy",
        ),
        tasks.TaskType.REGRESSION: ("mae", "rmse", "r2"),
        tasks.TaskType.SURVIVAL: ("c_index", "ibs", "td_auc"),
        tasks.TaskType.ORDINAL: ("qwk", "mace", "accuracy_levels"),
        tasks.TaskType.RANKING: (
            "ndcg",
            "map",
            "mrr",
            "precision_at_10",
            "recall_at_10",
        ),
        tasks.TaskType.MULTILABEL: (
            "f1_macro_labels",
            "f1_micro_labels",
            "hamming",
            "subset_accuracy",
        ),
        tasks.TaskType.MULTIOUTPUT: ("r2_mean", "mae_scaled"),
        tasks.TaskType.FORECAST: ("mase", "forecast_mae", "forecast_smape"),
        tasks.TaskType.UPLIFT: ("qini", "uplift_at_30"),
    }

    def __init__(
        self,
        metrics: tuple[scmle.contracts.metrics.Metric, ...] = DEFAULT_METRICS,
    ) -> None:
        """Initializes the catalog with a metric table and report names.

        The report names per task start as a copy of the class-level
        defaults, whatever `metrics` holds; a name missing from `metrics`
        is skipped by `evaluate`.

        Args:
            metrics: Available metrics, keyed by their names.
        """
        self.lock = threading.RLock()
        self.metrics = {metric.name: metric for metric in metrics}
        self.report_names = {
            task: tuple(names)
            for task, names in type(self).report_names.items()
        }

    def register(
        self, metric: scmle.contracts.metrics.Metric, replace: bool = False
    ) -> None:
        """Adds a user-defined metric, reported alongside the standard ones.

        The tables are replaced, never edited in place, so a reader that
        already holds one keeps a consistent view. `metrics` is published
        first so every reported name is already resolvable.

        Args:
            metric: The metric.
            replace: Allow overriding an existing metric of the same name.

        Raises:
            ConfigError: If the name is taken and `replace` is False.
        """
        with self.lock:
            if metric.name in self.metrics and not replace:
                raise errors.ConfigError(
                    f"metric '{metric.name}' already exists"
                )
            report_names = dict(self.report_names)
            for task in metric.tasks:
                names = report_names[task]
                if metric.name not in names:
                    report_names[task] = (*names, metric.name)
            self.metrics = {**self.metrics, metric.name: metric}
            self.report_names = report_names

    def unregister(self, name: str) -> None:
        """Removes a metric by name (no-op if it is not registered).

        Standard metrics are removed the same way as user-defined ones.

        The reported names shrink first, so no reader is told to score a
        metric that is already gone.

        Args:
            name: Name of the metric to remove.
        """
        with self.lock:
            self.report_names = {
                task: tuple(n for n in names if n != name)
                for task, names in self.report_names.items()
            }
            self.metrics = {
                key: value for key, value in self.metrics.items() if key != name
            }

    def resolve(
        self, name: str | None, task: tasks.TaskType
    ) -> scmle.contracts.metrics.Metric:
        """Finds the metric to optimize.

        Args:
            name: Requested name, or None for the task default.
            task: Resolved task type.

        Returns:
            The metric (`roc_auc` maps to its one-vs-rest variant for
            multiclass tasks).

        Raises:
            ConfigError: If the name is unknown or invalid for the task.
        """
        name = name or self.default_metric[task]
        if name == "roc_auc" and task is tasks.TaskType.MULTICLASS:
            name = "roc_auc_ovr"
        metric = self.metrics.get(name)
        if metric is None:
            raise errors.ConfigError(
                f"unknown metric '{name}'; available: {sorted(self.metrics)}"
            )
        if task not in metric.tasks:
            raise errors.ConfigError(
                f"metric '{name}' is not defined for {task.value}"
            )
        return metric

    def evaluate(
        self,
        estimator: models.Estimator,
        features: pandas.DataFrame,
        target: pandas.Series,
        task: tasks.TaskType,
        sample_weight: Any = None,
    ) -> dict[str, float]:
        """Applies every report metric for the task.

        Args:
            estimator: Fitted estimator.
            features: Evaluation features.
            target: Evaluation labels.
            task: Resolved task type.
            sample_weight: Optional per-row weights passed to every metric's
                scorer. A metric that ignores weights scores unweighted; one
                that cannot take them (a user function without a
                `sample_weight` parameter) scores NaN.

        Returns:
            Metric name to natural-unit score, in the report order of the
            task; NaN when scoring raises `ValueError` or `TypeError` (for
            example a class absent from the labels). A report name that was
            unregistered meanwhile is omitted.
        """
        results: dict[str, float] = {}
        classes = getattr(estimator, "classes_", None)
        positive = (
            classes[1]
            if task is tasks.TaskType.BINARY and classes is not None
            else None
        )
        names = self.report_names[task]
        table = self.metrics  # One snapshot, immune to concurrent changes.
        for name in names:
            if name not in table:  # Unregistered since `names` was read.
                continue
            try:
                results[name] = table[name].score(
                    estimator, features, target, positive, sample_weight
                )
            except (ValueError, TypeError):
                results[name] = float("nan")
        return results


CATALOG = MetricCatalog()


def register_metric(
    name: str,
    score_function: Callable[..., float],
    greater_is_better: bool,
    task_types: tuple[tasks.TaskType, ...] = tuple(tasks.STANDARD_TASKS),
    response_method: str = "predict",
    replace: bool = False,
) -> scmle.contracts.metrics.FunctionMetric:
    """Registers a user-defined metric with the default catalog.

    Args:
        name: Public metric name, usable as `MLProject(metric=name)`.
        score_function: Module-level function `(y_true, y_pred) -> float`;
            `y_pred` is what `response_method` supplies. Add a
            `sample_weight` parameter for it to be scored with weights.
        greater_is_better: Orientation of the score.
        task_types: Tasks the metric is defined for.
        response_method: `predict`, `predict_proba` or `decision_function`.
        replace: Allow overriding an existing metric of the same name.

    Returns:
        The registered metric. `MetricCatalog.register` raises `ConfigError`
        when the name is taken and `replace` is False.
    """
    metric = scmle.contracts.metrics.FunctionMetric(
        name,
        "",
        greater_is_better,
        frozenset(task_types),
        score_function,
        response_method,
    )
    CATALOG.register(metric, replace)
    return metric
