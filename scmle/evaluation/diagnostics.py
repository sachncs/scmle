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
from typing import cast
from typing import ClassVar

import numpy
import pandas
from sklearn import metrics

from scmle.contracts import data
from scmle.contracts import tasks
import scmle.contracts.metrics
from scmle.evaluation import reliability

CALIBRATION_BINS = 10
MAX_SEGMENT_LEVELS = 12
MAX_SEGMENT_FEATURES = 3
SEGMENT_BANDS = 4
MAX_ERROR_ROWS = 20
THRESHOLDS = tuple(round(0.1 * step, 1) for step in range(1, 10))
CLASSIFICATION = frozenset({tasks.Task.BINARY, tasks.Task.MULTICLASS})
BINARY_ONLY = frozenset({tasks.Task.BINARY})
REGRESSION_ONLY = frozenset({tasks.Task.REGRESSION})
EVERYTHING = frozenset(tasks.Task)


def clock() -> float:
    """Reads the clock every deadline is read from (tests replace it).

    Returns:
        The monotonic time in seconds.
    """
    return time.monotonic()


# Upper-envelope multiplier on the fitted cost models below. Derivation: the
# models were fitted by non-negative least squares on 31 runs (the 15
# TabArena tasks of benchmarks/evidence/quality.py, each with a gradient
# boosting and a linear model, two processes at a time, OMP_NUM_THREADS=1);
# the largest ratio of measured to fitted seconds among the diagnostics that
# run as one unit was 1.46 (adversarial validation; bootstrap 1.35,
# robustness 1.34, error analysis 1.30), rounded up to 1.5.
MARGIN = 1.5


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
        deadline: Monotonic time (on `clock`) by which the diagnostic must
            finish; diagnostics check it before each unit of work and
            reduce or skip what cannot fit, saying so (None = no limit).
        pace: Seconds one prediction over the evaluation rows takes, as
            timed by `suite.run_suite` (None = not timed).
    """

    estimator: Any
    features: pandas.DataFrame
    target: pandas.Series
    task: tasks.Task
    metric: scmle.contracts.metrics.Metric
    feature_plan: data.Blueprint
    seed: int
    groups: pandas.Series | None = None
    temporal: bool = False
    reference_features: pandas.DataFrame | None = None
    sensitive_columns: tuple[str, ...] = ()
    deadline: float | None = None
    pace: float | None = None

    def expired(self) -> bool:
        """Whether the diagnostic's time allowance has run out.

        Returns:
            True if a deadline is set and the monotonic clock has passed it.
        """
        return self.deadline is not None and clock() > self.deadline

    def affords(self, seconds: float) -> bool:
        """Whether work of the given length would end by the deadline.

        Args:
            seconds: Predicted length of the next unit of work.

        Returns:
            True if no deadline is set or the work would end by it.
        """
        return self.deadline is None or clock() + seconds <= self.deadline

    def left(self) -> float:
        """Seconds until the deadline (infinite without one)."""
        if self.deadline is None:
            return float("inf")
        return self.deadline - clock()

    @property
    def positive_label(self) -> Any:
        """The positive class of a binary target, else None.

        It is the second class of the fitted estimator.
        """
        classes = getattr(self.estimator, "classes_", None)
        if self.task is tasks.Task.BINARY and classes is not None:
            return classes[1]
        return None


def skipped(reason: str) -> dict[str, str | bool]:
    """Builds the report of a diagnostic that did not run (or not fully).

    Merge it over anything already measured to keep that in the report.

    Args:
        reason: Why it was skipped, naming the time needed and left.

    Returns:
        `available` False, `skipped` True and the reason, the shape release
        gates read as an unavailable diagnostic.
    """
    return {"available": False, "skipped": True, "reason": reason}


class Diagnostic(abc.ABC):
    """One diagnostic report.

    The cost model is `MARGIN * (fixed + passes * pace + rowcost * t)` for t
    evaluation rows: `passes` is the number of predictions over the
    evaluation rows the report needs, and the three class constants are
    fitted per diagnostic on the runs described at `MARGIN`.

    Attributes:
        name: File stem and key of the report.
        applies_to: Task types the diagnostic is defined for.
        fixed: Seconds that do not depend on the rows or the model.
        passes: Predictions over the evaluation rows.
        rowcost: Further seconds per evaluation row.
        units: Equal parts the full report divides into; the smallest
            useful report is one of them.
    """

    name: ClassVar[str]
    applies_to: ClassVar[frozenset[tasks.Task]] = EVERYTHING
    fixed: ClassVar[float] = 0.0
    passes: ClassVar[float] = 1.0
    rowcost: ClassVar[float] = 0.0
    units: ClassVar[int] = 1

    def estimate(self, inputs: DiagnosticInput) -> float:
        """Predicts the seconds of the full report.

        Args:
            inputs: The evaluation data, with `pace` when it was timed.

        Returns:
            The cost model's upper envelope, in seconds.
        """
        pace = inputs.pace or 0.0
        rows = len(inputs.features)
        return MARGIN * (self.fixed + self.passes * pace + self.rowcost * rows)

    def least(self, inputs: DiagnosticInput) -> float:
        """Predicts the seconds of the smallest useful report.

        Args:
            inputs: The evaluation data, with `pace` when it was timed.

        Returns:
            The full estimate over `units`: the full estimate for a report
            that runs as one unit.
        """
        return self.estimate(inputs) / self.units

    @abc.abstractmethod
    def compute(self, inputs: DiagnosticInput) -> dict[str, Any]:
        """Computes the report.

        Args:
            inputs: The fitted candidate and evaluation data.

        Returns:
            A JSON-serializable mapping.
        """


class Contingency(Diagnostic):
    """Confusion matrix and per-class precision / recall / F-measure."""

    name = "classification_report"
    applies_to = CLASSIFICATION
    # Fitted: 0.98 predictions plus 4.8 microseconds per row (21 runs).
    rowcost = 4.8e-6

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
        matrix = metrics.confusion_matrix(
            inputs.target, predicted, labels=labels
        )
        # scikit-learn's untyped return is inferred as a mix of float and
        # None; at runtime these are four per-class arrays.
        precision, recall, f1, support = cast(
            "tuple[Any, Any, Any, Any]",
            metrics.precision_recall_fscore_support(
                inputs.target,
                predicted,
                labels=labels,
                zero_division=0,  # pyright: ignore[reportArgumentType]  # scikit-learn infers zero_division as str from its "warn" default
            ),
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


class Calibration(Diagnostic):
    """Reliability curve, Brier score and expected calibration error."""

    name = "calibration"
    applies_to = BINARY_ONLY
    # Fitted: 0.99 predictions plus 0.34 microseconds per row (21 runs).
    rowcost = 3.4e-7

    def compute(self, inputs: DiagnosticInput) -> dict[str, Any]:
        """Bins predicted probabilities and compares them with outcomes.

        Args:
            inputs: The fitted candidate and evaluation data.

        Returns:
            The Brier score, the expected calibration error (bin-weighted
            mean absolute gap between predicted and observed frequency) and
            the curve of the non-empty equal-width bins; `skipped` when the
            deadline passed before the one prediction.
        """
        if not inputs.affords(self.estimate(inputs)):
            return skipped("the deadline leaves no time for one prediction")
        probability = inputs.estimator.predict_proba(inputs.features)[:, 1]
        observed = (inputs.target == inputs.positive_label).astype(float)
        edges = numpy.linspace(0.0, 1.0, CALIBRATION_BINS + 1)
        bins = numpy.clip(
            numpy.digitize(probability, edges[1:-1]), 0, CALIBRATION_BINS - 1
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
            "brier_score": float(numpy.mean((probability - observed) ** 2)),
            "expected_calibration_error": float(error),
            "curve": curve,
        }


class Reliability(Diagnostic):
    """Top-label reliability curve, Brier score and calibration error."""

    name = "calibration"
    applies_to = frozenset({tasks.Task.MULTICLASS})
    # Same fit as `Calibration` (the report shares its name and runs).
    rowcost = 3.4e-7

    def compute(self, inputs: DiagnosticInput) -> dict[str, Any]:
        """Compares confidence in the predicted class with its accuracy.

        Args:
            inputs: The fitted candidate and evaluation data.

        Returns:
            The multiclass Brier score, the top-label expected calibration
            error and the top-label reliability curve; `skipped` when the
            deadline passed before the one prediction.
        """
        if not inputs.affords(self.estimate(inputs)):
            return skipped("the deadline leaves no time for one prediction")
        probabilities = numpy.asarray(
            inputs.estimator.predict_proba(inputs.features)
        )
        classes = list(inputs.estimator.classes_)
        labels = numpy.asarray(inputs.target)
        error, curve = reliability.topclass(
            probabilities, labels, classes, CALIBRATION_BINS
        )
        truth = (labels[:, None] == numpy.asarray(classes)[None, :]).astype(
            float
        )
        return {
            "brier_score": float(
                numpy.mean(numpy.sum((probabilities - truth) ** 2, axis=1))
            ),
            "expected_calibration_error": error,
            "curve": curve,
        }


class Sweep(Diagnostic):
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


class Residuals(Diagnostic):
    """Bias, spread and error by segment for regression."""

    name = "residuals"
    applies_to = REGRESSION_ONLY
    # Fitted: 1.01 predictions plus 0.65 microseconds per row (10 runs).
    rowcost = 6.5e-7

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
        residual = inputs.target.to_numpy(dtype=float) - numpy.asarray(
            inputs.estimator.predict(inputs.features), dtype=float
        )
        absolute = numpy.abs(residual)
        report: dict[str, Any] = {
            "bias": float(residual.mean()),
            "residual_std": float(residual.std()),
            "mae": float(absolute.mean()),
            "abs_error_quantiles": {
                str(q): float(numpy.quantile(absolute, q))
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
            values = pandas.to_numeric(frame[column], errors="coerce")
            keys = (
                values
                if values.nunique() <= MAX_SEGMENT_LEVELS
                else pandas.qcut(values, SEGMENT_BANDS, duplicates="drop")
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


# The limitations of the error report: how rows are identified, and what
# each ranking's statistic can and cannot say.
IDENTITY = (
    "Rows are identified by their evaluation index only; raw feature values "
    "are not copied into the report."
)
CAVEATS = {
    "misclassification_then_low_confidence": (
        "Confidence is model probability, not correctness certainty or "
        "causality."
    ),
    "absolute_error": (
        "Residuals diagnose model behaviour and do not establish causal "
        "failure reasons."
    ),
}


def worst(keys: list[tuple[Any, ...]]) -> list[int]:
    """Positions of the `MAX_ERROR_ROWS` rows with the smallest sort keys.

    Args:
        keys: One sort key per row, in row order.

    Returns:
        The positions, smallest key first.
    """
    return sorted(range(len(keys)), key=keys.__getitem__)[:MAX_ERROR_ROWS]


class Errors(Diagnostic):
    """Identifies the highest-risk held-out rows without exposing features."""

    name = "error_analysis"
    applies_to = CLASSIFICATION | REGRESSION_ONLY
    # Fitted: 2.06 predictions plus 0.94 microseconds per row (31 runs).
    passes = 2.1
    rowcost = 9.4e-7

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
        predicted = numpy.asarray(inputs.estimator.predict(inputs.features))
        ids = [str(value) for value in inputs.features.index]
        if inputs.task in CLASSIFICATION:
            probabilities = numpy.asarray(
                inputs.estimator.predict_proba(inputs.features)
            )
            confidence = probabilities.max(axis=1)
            incorrect = predicted != inputs.target.to_numpy()
            positions = worst(
                [
                    (not bool(incorrect[k]), float(confidence[k]), k)
                    for k in range(len(predicted))
                ]
            )
            rows = [
                {
                    "row_id": ids[position],
                    "actual": str(inputs.target.iloc[position]),
                    "predicted": str(predicted[position]),
                    "confidence": float(confidence[position]),
                    "misclassified": bool(incorrect[position]),
                }
                for position in positions
            ]
            summary: dict[str, float] = {
                "misclassified_count": int(incorrect.sum())
            }
            ranking = "misclassification_then_low_confidence"
        else:
            actual = inputs.target.to_numpy(dtype=float)
            numeric_prediction = predicted.astype(float)
            residual = actual - numeric_prediction
            absolute = numpy.abs(residual)
            positions = worst(
                [(-float(absolute[k]), k) for k in range(len(predicted))]
            )
            rows = [
                {
                    "row_id": ids[position],
                    "actual": float(actual[position]),
                    "predicted": float(numeric_prediction[position]),
                    "residual": float(residual[position]),
                    "absolute_error": float(absolute[position]),
                }
                for position in positions
            ]
            summary = {"mean_absolute_error": float(absolute.mean())}
            ranking = "absolute_error"
        return {
            "task": inputs.task.value,
            "row_count": len(predicted),
            **summary,
            "rows": rows,
            "ranked_by": ranking,
            "limitations": [IDENTITY, CAVEATS[ranking]],
        }
