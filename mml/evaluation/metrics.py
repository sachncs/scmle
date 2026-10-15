"""Metric catalog with explicit orientation and a fixed scoring protocol.

Scores are always reported in natural units (log loss is positive);
`greater_is_better` tells rankers which direction wins.
"""

from collections.abc import Callable
import dataclasses
from typing import Any

import pandas as pd
from sklearn import metrics as sk_metrics

from mml.contracts import errors
from mml.contracts import models
from mml.contracts import tasks

BINARY = tasks.TaskType.BINARY
MULTICLASS = tasks.TaskType.MULTICLASS
REGRESSION = tasks.TaskType.REGRESSION
CLASSIFICATION = frozenset({BINARY, MULTICLASS})


def positive_label(target: pd.Series, task: tasks.TaskType) -> Any:
    """Returns the positive class of a binary target (None otherwise).

    Follows scikit-learn's convention: the second class in sorted order,
    which is `classes_[1]` of every fitted classifier.

    Args:
        target: Labels.
        task: Resolved task type.
    """
    if task is not BINARY:
        return None
    return sorted(target.dropna().unique())[1]


@dataclasses.dataclass(frozen=True, slots=True)
class Metric:
    """A scoring rule.

    Attributes:
        name: Public metric name.
        scorer_name: scikit-learn scorer (sign-adjusted so larger is better).
        greater_is_better: Whether larger natural values are better.
        tasks: Tasks the metric is defined for.
    """

    name: str
    scorer_name: str
    greater_is_better: bool
    tasks: frozenset[tasks.TaskType]

    def to_natural(self, scorer_value: float) -> float:
        """Converts a sign-adjusted scorer value to natural units."""
        return scorer_value if self.greater_is_better else -scorer_value

    def scorer(self, positive: Any = None) -> Any:
        """Returns the scikit-learn scorer.

        Args:
            positive: Positive class label (used by subclasses that need it).
        """
        return sk_metrics.get_scorer(self.scorer_name)

    def score(
        self,
        estimator: models.Estimator,
        features: pd.DataFrame,
        target: pd.Series,
        positive: Any = None,
    ) -> float:
        """Scores a fitted estimator in natural units."""
        scorer = self.scorer(positive)
        return self.to_natural(float(scorer(estimator, features, target)))


@dataclasses.dataclass(frozen=True, slots=True)
class PositiveLabelMetric(Metric):
    """A binary metric defined relative to a positive class.

    scikit-learn's stock scorers hard-code `pos_label=1`, which breaks on
    string or other non-{0, 1} labels; this builds the scorer with the real
    positive class instead.

    Attributes:
        score_function: Function `(y_true, y_pred_or_score, pos_label=...)`.
        response_methods: Estimator methods supplying the prediction.
    """

    score_function: Callable[..., float] = sk_metrics.f1_score
    response_methods: tuple[str, ...] = ("predict",)

    def scorer(self, positive: Any = None) -> Any:
        """Builds a scorer bound to `positive` (stock scorer if unknown)."""
        if positive is None:
            return super().scorer(None)
        return sk_metrics.make_scorer(
            self.score_function,
            response_method=self.response_methods,
            pos_label=positive,
        )


DEFAULT_METRICS = (
    Metric("roc_auc", "roc_auc", True, frozenset({BINARY})),
    Metric("roc_auc_ovr", "roc_auc_ovr", True, frozenset({MULTICLASS})),
    PositiveLabelMetric(
        "pr_auc",
        "average_precision",
        True,
        frozenset({BINARY}),
        sk_metrics.average_precision_score,
        ("decision_function", "predict_proba"),
    ),
    Metric("log_loss", "neg_log_loss", False, CLASSIFICATION),
    PositiveLabelMetric(
        "f1",
        "f1",
        True,
        frozenset({BINARY}),
        sk_metrics.f1_score,
        ("predict",),
    ),
    Metric("f1_macro", "f1_macro", True, CLASSIFICATION),
    Metric("balanced_accuracy", "balanced_accuracy", True, CLASSIFICATION),
    Metric("accuracy", "accuracy", True, CLASSIFICATION),
    Metric("mae", "neg_mean_absolute_error", False, frozenset({REGRESSION})),
    Metric(
        "rmse", "neg_root_mean_squared_error", False, frozenset({REGRESSION})
    ),
    Metric("r2", "r2", True, frozenset({REGRESSION})),
)


class MetricCatalog:
    """Resolves metric names and applies the report protocol."""

    default_metric = {
        BINARY: "roc_auc",
        MULTICLASS: "log_loss",
        REGRESSION: "rmse",
    }
    report_names = {
        BINARY: (
            "roc_auc",
            "pr_auc",
            "log_loss",
            "f1",
            "balanced_accuracy",
            "accuracy",
        ),
        MULTICLASS: ("log_loss", "f1_macro", "balanced_accuracy", "accuracy"),
        REGRESSION: ("mae", "rmse", "r2"),
    }

    def __init__(self, metrics: tuple[Metric, ...] = DEFAULT_METRICS) -> None:
        """Initializes the catalog.

        Args:
            metrics: Available metrics.
        """
        self.metrics = {metric.name: metric for metric in metrics}

    def resolve(self, name: str | None, task: tasks.TaskType) -> Metric:
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
        if name == "roc_auc" and task is MULTICLASS:
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
        features: pd.DataFrame,
        target: pd.Series,
        task: tasks.TaskType,
    ) -> dict[str, float]:
        """Applies every report metric for the task.

        Args:
            estimator: Fitted estimator.
            features: Evaluation features.
            target: Evaluation labels.
            task: Resolved task type.

        Returns:
            Metric name to natural-unit score; NaN when a metric cannot be
            computed (for example a class absent from the labels).
        """
        results: dict[str, float] = {}
        classes = getattr(estimator, "classes_", None)
        positive = (
            classes[1] if task is BINARY and classes is not None else None
        )
        for name in self.report_names[task]:
            try:
                results[name] = self.metrics[name].score(
                    estimator, features, target, positive
                )
            except ValueError:
                results[name] = float("nan")
        return results


CATALOG = MetricCatalog()
