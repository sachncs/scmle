"""Post-selection diagnostics.

Each `Diagnostic` subclass computes one JSON-serializable report from the
frozen, fitted candidate and the (already opened, single-use) test partition.
Diagnostics never influence selection; they explain it. Model explanations
build on the same base class in `scmle.explainability`.
"""

import abc
import dataclasses
import time
from typing import Any
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn import base
from sklearn import metrics as sk_metrics

from scmle.contracts import data as data_contracts
from scmle.contracts import errors
from scmle.contracts import metrics as metric_contracts
from scmle.contracts import tasks
from scmle.evaluation import reliability

CALIBRATION_BINS = 10
MAX_SEGMENT_LEVELS = 12
MAX_SEGMENT_FEATURES = 3
SEGMENT_BANDS = 4
MAX_ERROR_ROWS = 20
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
        groups: Entity labels of the evaluation rows, when the data has
            repeated entities (resampling then draws whole groups).
        temporal: Whether rows are chronologically ordered and serially
            dependent (resampling then draws contiguous blocks).
        reference_features: Development rows the model was trained on, for
            diagnostics that compare them with the evaluation rows.
        sensitive_columns: Columns whose groups outcomes are compared across.
        deadline: Monotonic time by which a diagnostic should finish; slow
            diagnostics stop early and say so (None = no limit).
    """

    estimator: Any
    features: pd.DataFrame
    target: pd.Series
    task: tasks.TaskType
    metric: metric_contracts.Metric
    feature_plan: data_contracts.FeaturePlan
    seed: int
    groups: pd.Series | None = None
    temporal: bool = False
    reference_features: pd.DataFrame | None = None
    sensitive_columns: tuple[str, ...] = ()
    deadline: float | None = None

    def out_of_time(self) -> bool:
        """Whether the diagnostic's time allowance has run out.

        Returns:
            True if a deadline is set and the monotonic clock has passed it.
        """
        return self.deadline is not None and time.monotonic() > self.deadline

    @property
    def positive_label(self) -> Any:
        """The positive class of a binary target, else None.

        It is the second class of the fitted estimator.
        """
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
    """Confusion matrix and per-class precision / recall / F-measure."""

    name = "classification_report"
    applies_to = CLASSIFICATION

    def compute(self, inputs: DiagnosticInput) -> dict[str, Any]:
        """Builds the confusion matrix and per-class scores.

        Args:
            inputs: The fitted candidate and evaluation data.

        Returns:
            The class labels as text, the confusion matrix (rows are true
            classes, columns predicted, in label order) and, per class, the
            precision, recall, F-measure and support.
        """
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
        """Bins predicted probabilities and compares them with outcomes.

        Args:
            inputs: The fitted candidate and evaluation data.

        Returns:
            The Brier score, the expected calibration error (bin-weighted
            mean absolute gap between predicted and observed frequency) and
            the curve of the non-empty equal-width bins.
        """
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


class MulticlassCalibrationReport(Diagnostic):
    """Top-label reliability curve, Brier score and calibration error."""

    name = "calibration"
    applies_to = frozenset({tasks.TaskType.MULTICLASS})

    def compute(self, inputs: DiagnosticInput) -> dict[str, Any]:
        """Compares confidence in the predicted class with its accuracy.

        Args:
            inputs: The fitted candidate and evaluation data.

        Returns:
            The multiclass Brier score, the top-label expected calibration
            error and the top-label reliability curve.
        """
        probabilities = np.asarray(
            inputs.estimator.predict_proba(inputs.features)
        )
        classes = list(inputs.estimator.classes_)
        labels = np.asarray(inputs.target)
        error, curve = reliability.top_label_curve(
            probabilities, labels, classes, CALIBRATION_BINS
        )
        truth = (labels[:, None] == np.asarray(classes)[None, :]).astype(float)
        return {
            "brier_score": float(
                np.mean(np.sum((probabilities - truth) ** 2, axis=1))
            ),
            "expected_calibration_error": error,
            "curve": curve,
        }


class ThresholdReport(Diagnostic):
    """Precision / recall / F-measure across decision thresholds."""

    name = "threshold_analysis"
    applies_to = BINARY_ONLY

    def compute(self, inputs: DiagnosticInput) -> dict[str, Any]:
        """Scores the positive class at a grid of thresholds.

        Args:
            inputs: The fitted candidate and evaluation data.

        Returns:
            A mapping with key `thresholds`: one entry per value in
            `THRESHOLDS` with the precision, recall and F-measure there
            (a zero where nothing is flagged or the denominator is zero).
        """
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
        """Summarizes residuals and the worst-served segments.

        Args:
            inputs: The fitted candidate and evaluation data.

        Returns:
            The residual bias and standard deviation, the mean absolute
            error, quantiles (0.5, 0.9, 0.99) of the absolute error and,
            for each of the first `MAX_SEGMENT_FEATURES` categorical and
            numeric features, the mean absolute error and row count by
            level. Categorical features with more than `MAX_SEGMENT_LEVELS`
            levels are skipped; numeric features with more than that many
            distinct values are cut into quartile bands.
        """
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
        for column in inputs.feature_plan.numeric[:MAX_SEGMENT_FEATURES]:
            values = pd.to_numeric(frame[column], errors="coerce")
            keys = (
                values
                if values.nunique() <= MAX_SEGMENT_LEVELS
                else pd.qcut(values, SEGMENT_BANDS, duplicates="drop")
            )
            grouped = frame["abs_error"].groupby(keys, observed=True)
            report["segments"][column] = {
                str(level): {
                    "mae": float(group.mean()),
                    "count": len(group),
                }
                for level, group in grouped
            }
        return report


class ErrorAnalysis(Diagnostic):
    """Identifies the highest-risk held-out rows without exposing features."""

    name = "error_analysis"
    applies_to = CLASSIFICATION | REGRESSION_ONLY

    def compute(self, inputs: DiagnosticInput) -> dict[str, Any]:
        """Returns bounded row-level error evidence and its limitations.

        Args:
            inputs: The fitted candidate and evaluation data.

        Returns:
            The task, row count and up to `MAX_ERROR_ROWS` rows ranked
            worst first, with the ranking rule and stated limitations.
            Classification ranks misclassified rows first, then by low
            confidence, and also reports the misclassified count;
            regression ranks by absolute error and reports the mean
            absolute error.
        """
        predicted = np.asarray(inputs.estimator.predict(inputs.features))
        row_ids = [str(value) for value in inputs.features.index]
        if inputs.task in CLASSIFICATION:
            probabilities = np.asarray(
                inputs.estimator.predict_proba(inputs.features)
            )
            confidence = probabilities.max(axis=1)
            incorrect = predicted != inputs.target.to_numpy()
            positions = sorted(
                range(len(predicted)),
                key=lambda position: (
                    not bool(incorrect[position]),
                    float(confidence[position]),
                    position,
                ),
            )[:MAX_ERROR_ROWS]
            rows = [
                {
                    "row_id": row_ids[position],
                    "actual": str(inputs.target.iloc[position]),
                    "predicted": str(predicted[position]),
                    "confidence": float(confidence[position]),
                    "misclassified": bool(incorrect[position]),
                }
                for position in positions
            ]
            return {
                "task": inputs.task.value,
                "row_count": len(predicted),
                "misclassified_count": int(incorrect.sum()),
                "rows": rows,
                "ranked_by": "misclassification_then_low_confidence",
                "limitations": [
                    "Rows are identified by their evaluation index only; raw "
                    "feature values are not copied into the report.",
                    "Confidence is model probability, not correctness "
                    "certainty or causality.",
                ],
            }
        actual = inputs.target.to_numpy(dtype=float)
        numeric_prediction = predicted.astype(float)
        residual = actual - numeric_prediction
        absolute = np.abs(residual)
        positions = sorted(
            range(len(predicted)),
            key=lambda position: (-float(absolute[position]), position),
        )[:MAX_ERROR_ROWS]
        return {
            "task": inputs.task.value,
            "row_count": len(predicted),
            "mean_absolute_error": float(absolute.mean()),
            "rows": [
                {
                    "row_id": row_ids[position],
                    "actual": float(actual[position]),
                    "predicted": float(numeric_prediction[position]),
                    "residual": float(residual[position]),
                    "absolute_error": float(absolute[position]),
                }
                for position in positions
            ],
            "ranked_by": "absolute_error",
            "limitations": [
                "Rows are identified by their evaluation index only; raw "
                "feature values are not copied into the report.",
                "Residuals diagnose model behaviour and do not establish "
                "causal failure reasons.",
            ],
        }


class CachedPredictions(base.BaseEstimator):
    """Replays a fitted model's predictions by row label.

    Lets a bootstrap re-score thousands of resamples without re-predicting:
    the wrapped estimator runs once, and every later call is a lookup on the
    (unique) index of the rows it is handed.

    Attributes:
        predictions: Predicted labels / values by row label.
        probabilities: Class probabilities by row label (classifiers).
        classes_: Class labels of the wrapped model, if any.

    `probabilities` is None unless the wrapped model has both `classes_` and
    `predict_proba`.
    """

    def __init__(self, estimator: Any, features: pd.DataFrame) -> None:
        """Predicts every row once.

        Args:
            estimator: Fitted model.
            features: Rows with a unique index.
        """
        self.classes_ = getattr(estimator, "classes_", None)
        self.predictions = pd.Series(
            np.asarray(estimator.predict(features)), index=features.index
        )
        self.probabilities = (
            pd.DataFrame(
                estimator.predict_proba(features), index=features.index
            )
            if self.classes_ is not None and hasattr(estimator, "predict_proba")
            else None
        )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Looks up predictions for the given rows.

        Args:
            features: Rows whose index labels were seen at construction.

        Returns:
            The cached predictions, in the order of `features`.
        """
        return self.predictions.loc[features.index].to_numpy()

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Looks up probabilities for the given rows.

        Args:
            features: Rows whose index labels were seen at construction.

        Returns:
            The cached class probabilities, in the order of `features`.

        Raises:
            errors.ArtifactError: If no probabilities were cached, because the
                wrapped model lacks `classes_` or `predict_proba`.
        """
        if self.probabilities is None:
            raise errors.ArtifactError("the model has no probabilities")
        return self.probabilities.loc[features.index].to_numpy()


class ClassifierReplay(base.ClassifierMixin, CachedPredictions):
    """Cached predictions that scikit-learn treats as a classifier."""


class RegressorReplay(base.RegressorMixin, CachedPredictions):
    """Cached predictions that scikit-learn treats as a regressor."""


def resample_rows(
    rows: int,
    groups: pd.Series | None,
    temporal: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draws one bootstrap resample that respects the data's dependence.

    Args:
        rows: Number of evaluation rows.
        groups: Entity labels, if rows are clustered by entity.
        temporal: Whether to draw contiguous blocks; ignored when `groups`
            is given.
        rng: Source of randomness.

    Returns:
        Row positions of the resample: whole groups drawn with replacement
        (its length varies with the group sizes), or blocks of
        `max(5, int(sqrt(rows)))` positions truncated to `rows`, or `rows`
        positions drawn with replacement.
    """
    if groups is not None:
        members = (
            groups.reset_index(drop=True).groupby(groups.to_numpy()).indices
        )
        labels = list(members)
        drawn = rng.integers(0, len(labels), len(labels))
        return np.concatenate([members[labels[i]] for i in drawn])
    if temporal:
        length = max(5, int(np.sqrt(rows)))
        starts = rng.integers(
            0, max(rows - length + 1, 1), -(-rows // length)
        )
        blocks = np.concatenate(
            [np.arange(s, min(s + length, rows)) for s in starts]
        )
        return blocks[:rows]
    return rng.integers(0, rows, rows)


class MetricUncertainty(Diagnostic):
    """Bootstrap confidence interval for the primary test metric.

    Resampling follows the independent units of observation: whole groups
    for clustered data, contiguous blocks for temporal data, and rows
    otherwise, so intervals are not falsely narrow.
    """

    name = "metric_uncertainty"
    resamples = 200
    confidence = 95

    def compute(self, inputs: DiagnosticInput) -> dict[str, Any]:
        """Bootstraps the primary metric on the evaluation rows.

        Args:
            inputs: The fitted candidate and evaluation data.

        Returns:
            The metric name, its point estimate, the percentile interval
            (key `ci_` plus the confidence; None when every resample
            failed), the resampling unit, the number
            of successful resamples (resamples that fail with a
            `ValueError`, such as one missing a class, are skipped), the
            number of evaluation rows and the number of independent units
            (groups, else rows).
        """
        features = inputs.features[list(inputs.feature_plan.input_columns)]
        features = features.reset_index(drop=True)
        target = inputs.target.reset_index(drop=True)
        replay_class = (
            RegressorReplay
            if inputs.task is tasks.TaskType.REGRESSION
            else ClassifierReplay
        )
        replay = replay_class(inputs.estimator, features)
        metric, positive = inputs.metric, inputs.positive_label
        estimate = metric.score(replay, features, target, positive)
        groups = (
            inputs.groups.reset_index(drop=True)
            if inputs.groups is not None
            else None
        )
        rng = np.random.default_rng(inputs.seed)
        scores: list[float] = []
        unit = (
            "groups"
            if groups is not None
            else "blocks"
            if inputs.temporal
            else "rows"
        )
        draws = [
            resample_rows(len(target), groups, inputs.temporal, rng)
            for count in range(self.resamples)
        ]
        for rows in draws:
            try:
                scores.append(
                    metric.score(
                        replay,
                        features.iloc[rows],
                        target.iloc[rows],
                        positive,
                    )
                )
            except ValueError:  # A resample missing a class.
                continue
        interval = None
        if scores:
            low, high = np.percentile(
                scores,
                [
                    (100 - self.confidence) / 2,
                    100 - (100 - self.confidence) / 2,
                ],
            )
            interval = [float(low), float(high)]
        return {
            "metric": metric.name,
            "estimate": float(estimate),
            f"ci_{self.confidence}": interval,
            "resampling_unit": unit,
            "resamples": len(scores),
            "evaluation_rows": len(target),
            "independent_units": int(groups.nunique())
            if groups is not None
            else len(target),
        }


def default_suite() -> tuple[Diagnostic, ...]:
    """Returns the standard diagnostics.

    Returns:
        One instance of each diagnostic, in the order they are run.
    """
    return (
        ClassificationReport(),
        CalibrationReport(),
        MulticlassCalibrationReport(),
        ThresholdReport(),
        ResidualReport(),
        ErrorAnalysis(),
        MetricUncertainty(),
    )


def run_suite(
    inputs: DiagnosticInput,
    suite: tuple[Diagnostic, ...] | None = None,
    total_seconds: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Runs every applicable diagnostic; one failure never hides the others.

    Args:
        inputs: Fitted candidate and evaluation data.
        suite: Diagnostics to run; defaults to `default_suite()`.
        total_seconds: Time allowance for the whole suite, shared evenly by
            the diagnostics still to run (None = unlimited). It sets
            `DiagnosticInput.deadline`; only diagnostics that check
            `out_of_time` stop early, and none defined in this module do.

    Returns:
        Report per name for the diagnostics whose `applies_to` contains the
        task; a failure appears as `{"error": "<ErrorType>: <message>"}`.
    """
    reports: dict[str, dict[str, Any]] = {}
    chosen = [
        d
        for d in (suite if suite is not None else default_suite())
        if inputs.task in d.applies_to
    ]
    finish = None if total_seconds is None else time.monotonic() + total_seconds
    for position, diagnostic in enumerate(chosen):
        limited = inputs
        if finish is not None:
            share = (finish - time.monotonic()) / (len(chosen) - position)
            limited = dataclasses.replace(
                inputs, deadline=time.monotonic() + max(share, 0.0)
            )
        try:
            reports[diagnostic.name] = diagnostic.compute(limited)
        except Exception as error:  # Isolation: diagnostics are advisory.
            reports[diagnostic.name] = {
                "error": f"{type(error).__name__}: {error}"
            }
    return reports
