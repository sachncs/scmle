"""Post-selection diagnostics and explainability.

Each `Diagnostic` subclass computes one JSON-serializable report from the
frozen, fitted candidate and the (already opened, single-use) test partition.
Diagnostics never influence selection; they explain it.
"""

import abc
import dataclasses
from typing import Any
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn import inspection
from sklearn import metrics as sk_metrics

from mml.contracts import data as data_contracts
from mml.contracts import tasks
from mml.evaluation import metrics as metric_module

CALIBRATION_BINS = 10
IMPORTANCE_REPEATS = 5
MAX_SEGMENT_LEVELS = 12
MAX_SEGMENT_FEATURES = 3
THRESHOLDS = tuple(round(0.1 * step, 1) for step in range(1, 10))
CLASSIFICATION = frozenset({tasks.TaskType.BINARY, tasks.TaskType.MULTICLASS})
BINARY_ONLY = frozenset({tasks.TaskType.BINARY})
REGRESSION_ONLY = frozenset({tasks.TaskType.REGRESSION})
EVERYTHING = frozenset(tasks.TaskType)


@dataclasses.dataclass(frozen=True)
class DiagnosticInput:
    """What diagnostics may look at.

    Attributes:
        estimator: Fitted candidate (preprocessing included).
        features: Evaluation features.
        target: Evaluation labels / values.
        task: Resolved task type.
        metric: Primary metric.
        feature_plan: Feature columns by kind.
        seed: Random seed.
    """

    estimator: Any
    features: pd.DataFrame
    target: pd.Series
    task: tasks.TaskType
    metric: metric_module.Metric
    feature_plan: data_contracts.FeaturePlan
    seed: int

    @property
    def positive_label(self) -> Any:
        """Positive class of a binary target, else None."""
        classes = getattr(self.estimator, "classes_", None)
        if self.task is tasks.TaskType.BINARY and classes is not None:
            return classes[1]
        return None


class Diagnostic(abc.ABC):
    """One diagnostic report.

    Attributes:
        name: File stem and key of the report.
        applies_to: Task types the diagnostic is defined for.
    """

    name: ClassVar[str]
    applies_to: ClassVar[frozenset[tasks.TaskType]] = EVERYTHING

    @abc.abstractmethod
    def compute(self, inputs: DiagnosticInput) -> dict[str, Any]:
        """Computes the report.

        Args:
            inputs: The fitted candidate and evaluation data.

        Returns:
            A JSON-serializable mapping.
        """


class ClassificationReport(Diagnostic):
    """Confusion matrix and per-class precision / recall / F1."""

    name = "classification_report"
    applies_to = CLASSIFICATION

    def compute(self, inputs: DiagnosticInput) -> dict[str, Any]:
        """Builds the confusion matrix and per-class scores."""
        predicted = inputs.estimator.predict(inputs.features)
        labels = list(inputs.estimator.classes_)
        matrix = sk_metrics.confusion_matrix(
            inputs.target, predicted, labels=labels
        )
        precision, recall, f1, support = (
            sk_metrics.precision_recall_fscore_support(
                inputs.target, predicted, labels=labels, zero_division=0
            )
        )
        return {
            "labels": [str(label) for label in labels],
            "confusion_matrix": matrix.tolist(),
            "per_class": {
                str(label): {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i, label in enumerate(labels)
            },
        }


class CalibrationReport(Diagnostic):
    """Reliability curve, Brier score and expected calibration error."""

    name = "calibration"
    applies_to = BINARY_ONLY

    def compute(self, inputs: DiagnosticInput) -> dict[str, Any]:
        """Bins predicted probabilities and compares them with outcomes."""
        probability = inputs.estimator.predict_proba(inputs.features)[:, 1]
        observed = (inputs.target == inputs.positive_label).astype(float)
        edges = np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)
        bins = np.clip(
            np.digitize(probability, edges[1:-1]), 0, CALIBRATION_BINS - 1
        )
        curve, error = [], 0.0
        for index in range(CALIBRATION_BINS):
            members = bins == index
            if not members.any():
                continue
            mean_predicted = float(probability[members].mean())
            frequency = float(observed[members].mean())
            error += members.mean() * abs(mean_predicted - frequency)
            curve.append(
                {
                    "bin": index,
                    "count": int(members.sum()),
                    "mean_predicted": mean_predicted,
                    "observed_frequency": frequency,
                }
            )
        return {
            "brier_score": float(np.mean((probability - observed) ** 2)),
            "expected_calibration_error": float(error),
            "curve": curve,
        }


class ThresholdReport(Diagnostic):
    """Precision / recall / F1 across decision thresholds."""

    name = "threshold_analysis"
    applies_to = BINARY_ONLY

    def compute(self, inputs: DiagnosticInput) -> dict[str, Any]:
        """Scores the positive class at a grid of thresholds."""
        probability = inputs.estimator.predict_proba(inputs.features)[:, 1]
        actual = (inputs.target == inputs.positive_label).to_numpy()
        rows = []
        for threshold in THRESHOLDS:
            flagged = probability >= threshold
            true_positive = int((flagged & actual).sum())
            precision = true_positive / max(int(flagged.sum()), 1)
            recall = true_positive / max(int(actual.sum()), 1)
            total = precision + recall
            rows.append(
                {
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1": 2 * precision * recall / total if total else 0.0,
                }
            )
        return {"thresholds": rows}


class ResidualReport(Diagnostic):
    """Bias, spread and error by segment for regression."""

    name = "residuals"
    applies_to = REGRESSION_ONLY

    def compute(self, inputs: DiagnosticInput) -> dict[str, Any]:
        """Summarizes residuals and the worst-served segments."""
        residual = inputs.target.to_numpy(dtype=float) - np.asarray(
            inputs.estimator.predict(inputs.features), dtype=float
        )
        absolute = np.abs(residual)
        report: dict[str, Any] = {
            "bias": float(residual.mean()),
            "residual_std": float(residual.std()),
            "mae": float(absolute.mean()),
            "abs_error_quantiles": {
                str(q): float(np.quantile(absolute, q))
                for q in (0.5, 0.9, 0.99)
            },
            "segments": {},
        }
        frame = inputs.features.assign(abs_error=absolute)
        for column in inputs.feature_plan.categorical[:MAX_SEGMENT_FEATURES]:
            if frame[column].nunique() > MAX_SEGMENT_LEVELS:
                continue
            grouped = frame.groupby(column, dropna=False)["abs_error"]
            report["segments"][column] = {
                str(level): {
                    "mae": float(group.mean()),
                    "count": len(group),
                }
                for level, group in grouped
            }
        return report


class PermutationImportance(Diagnostic):
    """Model-agnostic feature importance on the evaluation data."""

    name = "permutation_importance"

    def compute(self, inputs: DiagnosticInput) -> dict[str, Any]:
        """Measures the metric drop when each feature is shuffled."""
        columns = list(inputs.feature_plan.features)
        scorer = inputs.metric.scorer(inputs.positive_label)
        result = inspection.permutation_importance(
            inputs.estimator,
            inputs.features[columns],
            inputs.target,
            scoring=scorer,
            n_repeats=IMPORTANCE_REPEATS,
            random_state=inputs.seed,
        )
        ranked = np.argsort(-result.importances_mean)
        return {
            "metric": inputs.metric.name,
            "features": [
                {
                    "feature": columns[i],
                    "importance": float(result.importances_mean[i]),
                    "std": float(result.importances_std[i]),
                }
                for i in ranked
            ],
        }


def default_suite() -> tuple[Diagnostic, ...]:
    """Returns the standard diagnostics."""
    return (
        ClassificationReport(),
        CalibrationReport(),
        ThresholdReport(),
        ResidualReport(),
        PermutationImportance(),
    )


def run_suite(
    inputs: DiagnosticInput, suite: tuple[Diagnostic, ...] | None = None
) -> dict[str, dict[str, Any]]:
    """Runs every applicable diagnostic; one failure never hides the others.

    Args:
        inputs: Fitted candidate and evaluation data.
        suite: Diagnostics to run; defaults to `default_suite()`.

    Returns:
        Report per diagnostic name; failures appear as `{"error": ...}`.
    """
    reports: dict[str, dict[str, Any]] = {}
    for diagnostic in suite if suite is not None else default_suite():
        if inputs.task not in diagnostic.applies_to:
            continue
        try:
            reports[diagnostic.name] = diagnostic.compute(inputs)
        except Exception as error:  # Isolation: diagnostics are advisory.
            reports[diagnostic.name] = {
                "error": f"{type(error).__name__}: {error}"
            }
    return reports
