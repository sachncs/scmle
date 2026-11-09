"""Task-aware evaluation and release policy contracts.

The contract is intentionally independent from metric and diagnostic
implementations. It describes what a run promises to measure and what
evidence is required before a model may be released. A policy is not inferred
from the presence of an estimator or a diagnostic class: every supported
combination must be declared explicitly.
"""

from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import enum
import json
import math
from typing import Any

from scmle.contracts import errors
from scmle.contracts import tasks

POLICY_SCHEMA_VERSION = 1


class MetricDirection(enum.StrEnum):
    """Direction in which a metric improves."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class UncertaintyMethod(enum.StrEnum):
    """Uncertainty contract promised by an evaluation policy."""

    NONE = "none"
    CALIBRATED_PROBABILITY = "calibrated_probability"
    PREDICTION_INTERVAL = "prediction_interval"
    CONFORMAL = "conformal"
    SURVIVAL_CURVE = "survival_curve"
    ABSTENTION = "abstention"


class CalibrationMethod(enum.StrEnum):
    """Calibration procedure or diagnostic promised by a policy."""

    NONE = "none"
    RELIABILITY_CURVE = "reliability_curve"
    PLATT = "platt"
    ISOTONIC = "isotonic"
    DIRICHLET = "dirichlet"


class FairnessMethod(enum.StrEnum):
    """Fairness analysis or mitigation promised by a policy."""

    NONE = "none"
    GROUP_REPORT = "group_report"
    INTERSECTIONAL_REPORT = "intersectional_report"
    MITIGATION = "mitigation"


class ExplanationAvailability(enum.StrEnum):
    """Level of model explanation the evaluation must make available."""

    NONE = "none"
    GLOBAL = "global"
    LOCAL = "local"
    COUNTERFACTUAL = "counterfactual"
    FULL = "full"


class DiagnosticAction(enum.StrEnum):
    """Action taken when a required diagnostic does not pass."""

    WARN = "warn"
    BLOCK = "block"


class UnsupportedEvaluationPolicyError(errors.ConfigError):
    """Raised when a task and evaluation policy combination is unsupported."""

    category = "evaluation_policy"


class UndeclaredEvaluationPolicyError(UnsupportedEvaluationPolicyError):
    """Raised when no policy was declared for a requested task."""


SUPPORTED_METRICS: Mapping[tasks.Task, frozenset[str]] = {
    tasks.Task.BINARY: frozenset(
        {
            "roc_auc",
            "pr_auc",
            "log_loss",
            "f1",
            "f1_macro",
            "balanced_accuracy",
            "accuracy",
        }
    ),
    tasks.Task.MULTICLASS: frozenset(
        {"roc_auc_ovr", "log_loss", "f1_macro", "balanced_accuracy", "accuracy"}
    ),
    tasks.Task.REGRESSION: frozenset({"mae", "rmse", "r2"}),
    tasks.Task.SURVIVAL: frozenset({"c_index", "ibs", "td_auc"}),
    tasks.Task.ORDINAL: frozenset({"qwk", "mace", "accuracy_levels"}),
    tasks.Task.RANKING: frozenset(
        {"ndcg", "map", "mrr", "precision_at_10", "recall_at_10"}
    ),
    tasks.Task.MULTILABEL: frozenset(
        {"f1_macro_labels", "f1_micro_labels", "hamming", "subset_accuracy"}
    ),
    tasks.Task.MULTIOUTPUT: frozenset({"r2_mean", "mae_scaled"}),
    tasks.Task.FORECAST: frozenset({"mase", "forecast_mae", "forecast_smape"}),
    tasks.Task.UPLIFT: frozenset({"qini", "uplift_at_30"}),
}
"""Built-in metric names that the current evaluation catalog can score."""


METRIC_DIRECTIONS: Mapping[str, MetricDirection] = {
    "roc_auc": MetricDirection.MAXIMIZE,
    "roc_auc_ovr": MetricDirection.MAXIMIZE,
    "pr_auc": MetricDirection.MAXIMIZE,
    "log_loss": MetricDirection.MINIMIZE,
    "f1": MetricDirection.MAXIMIZE,
    "f1_macro": MetricDirection.MAXIMIZE,
    "balanced_accuracy": MetricDirection.MAXIMIZE,
    "accuracy": MetricDirection.MAXIMIZE,
    "mae": MetricDirection.MINIMIZE,
    "rmse": MetricDirection.MINIMIZE,
    "r2": MetricDirection.MAXIMIZE,
    "qwk": MetricDirection.MAXIMIZE,
    "mace": MetricDirection.MINIMIZE,
    "accuracy_levels": MetricDirection.MAXIMIZE,
    "f1_macro_labels": MetricDirection.MAXIMIZE,
    "f1_micro_labels": MetricDirection.MAXIMIZE,
    "hamming": MetricDirection.MINIMIZE,
    "subset_accuracy": MetricDirection.MAXIMIZE,
    "r2_mean": MetricDirection.MAXIMIZE,
    "mae_scaled": MetricDirection.MINIMIZE,
    "mase": MetricDirection.MINIMIZE,
    "forecast_mae": MetricDirection.MINIMIZE,
    "forecast_smape": MetricDirection.MINIMIZE,
    "qini": MetricDirection.MAXIMIZE,
    "uplift_at_30": MetricDirection.MAXIMIZE,
    "ndcg": MetricDirection.MAXIMIZE,
    "map": MetricDirection.MAXIMIZE,
    "mrr": MetricDirection.MAXIMIZE,
    "precision_at_10": MetricDirection.MAXIMIZE,
    "recall_at_10": MetricDirection.MAXIMIZE,
    "c_index": MetricDirection.MAXIMIZE,
    "ibs": MetricDirection.MINIMIZE,
    "td_auc": MetricDirection.MAXIMIZE,
}
"""Direction for every built-in metric declared in ``SUPPORTED_METRICS``."""


DEFAULT_METRICS: Mapping[tasks.Task, str] = {
    tasks.Task.BINARY: "roc_auc",
    tasks.Task.MULTICLASS: "log_loss",
    tasks.Task.REGRESSION: "rmse",
    tasks.Task.SURVIVAL: "c_index",
    tasks.Task.ORDINAL: "qwk",
    tasks.Task.RANKING: "ndcg",
    tasks.Task.MULTILABEL: "f1_macro_labels",
    tasks.Task.MULTIOUTPUT: "r2_mean",
    tasks.Task.FORECAST: "mase",
    tasks.Task.UPLIFT: "qini",
}


SUPPORTED_UNCERTAINTY: Mapping[tasks.Task, frozenset[UncertaintyMethod]] = {
    task: frozenset({UncertaintyMethod.NONE}) for task in tasks.Task
}
SUPPORTED_UNCERTAINTY = {
    **SUPPORTED_UNCERTAINTY,
    tasks.Task.BINARY: frozenset(
        {
            UncertaintyMethod.NONE,
            UncertaintyMethod.CALIBRATED_PROBABILITY,
            UncertaintyMethod.CONFORMAL,
            UncertaintyMethod.ABSTENTION,
        }
    ),
    tasks.Task.MULTICLASS: frozenset(
        {
            UncertaintyMethod.NONE,
            UncertaintyMethod.CALIBRATED_PROBABILITY,
            UncertaintyMethod.CONFORMAL,
            UncertaintyMethod.ABSTENTION,
        }
    ),
    tasks.Task.REGRESSION: frozenset(
        {
            UncertaintyMethod.NONE,
            UncertaintyMethod.PREDICTION_INTERVAL,
            UncertaintyMethod.CONFORMAL,
            UncertaintyMethod.ABSTENTION,
        }
    ),
    tasks.Task.SURVIVAL: frozenset(
        {UncertaintyMethod.NONE, UncertaintyMethod.SURVIVAL_CURVE}
    ),
}


SUPPORTED_CALIBRATION: Mapping[tasks.Task, frozenset[CalibrationMethod]] = {
    task: frozenset({CalibrationMethod.NONE}) for task in tasks.Task
}
SUPPORTED_CALIBRATION = {
    **SUPPORTED_CALIBRATION,
    tasks.Task.BINARY: frozenset(
        {
            CalibrationMethod.NONE,
            CalibrationMethod.RELIABILITY_CURVE,
            CalibrationMethod.PLATT,
            CalibrationMethod.ISOTONIC,
        }
    ),
    tasks.Task.MULTICLASS: frozenset(
        {
            CalibrationMethod.NONE,
            CalibrationMethod.RELIABILITY_CURVE,
            CalibrationMethod.DIRICHLET,
        }
    ),
}


SUPPORTED_FAIRNESS: Mapping[tasks.Task, frozenset[FairnessMethod]] = {
    task: frozenset({FairnessMethod.NONE}) for task in tasks.Task
}
SUPPORTED_FAIRNESS = {
    **SUPPORTED_FAIRNESS,
    tasks.Task.BINARY: frozenset(
        {
            FairnessMethod.NONE,
            FairnessMethod.GROUP_REPORT,
            FairnessMethod.MITIGATION,
        }
    ),
    tasks.Task.REGRESSION: frozenset(
        {
            FairnessMethod.NONE,
            FairnessMethod.GROUP_REPORT,
            FairnessMethod.MITIGATION,
        }
    ),
}


SUPPORTED_EXPLANATIONS: Mapping[
    tasks.Task, frozenset[ExplanationAvailability]
] = {
    task: frozenset(
        {ExplanationAvailability.NONE, ExplanationAvailability.GLOBAL}
    )
    for task in tasks.Task
}
SUPPORTED_EXPLANATIONS = {
    **SUPPORTED_EXPLANATIONS,
    tasks.Task.BINARY: frozenset(
        {
            ExplanationAvailability.NONE,
            ExplanationAvailability.GLOBAL,
            ExplanationAvailability.LOCAL,
            ExplanationAvailability.COUNTERFACTUAL,
            ExplanationAvailability.FULL,
        }
    ),
    tasks.Task.REGRESSION: frozenset(
        {
            ExplanationAvailability.NONE,
            ExplanationAvailability.GLOBAL,
            ExplanationAvailability.LOCAL,
            ExplanationAvailability.COUNTERFACTUAL,
            ExplanationAvailability.FULL,
        }
    ),
}


COMMON_DIAGNOSTICS = frozenset(
    {
        "metric_uncertainty",
        "slice_performance",
        "robustness",
        "adversarial_validation",
    }
)
SUPPORTED_DIAGNOSTICS: Mapping[tasks.Task, frozenset[str]] = dict.fromkeys(
    tasks.Task, COMMON_DIAGNOSTICS
)
SUPPORTED_DIAGNOSTICS = {
    **SUPPORTED_DIAGNOSTICS,
    tasks.Task.BINARY: COMMON_DIAGNOSTICS
    | frozenset(
        {
            "classification_report",
            "calibration",
            "threshold_analysis",
            "fairness",
        }
    ),
    tasks.Task.MULTICLASS: COMMON_DIAGNOSTICS
    | frozenset({"classification_report", "calibration"}),
    tasks.Task.REGRESSION: COMMON_DIAGNOSTICS
    | frozenset({"residuals", "fairness"}),
}


FAIRNESS_METRICS: Mapping[tasks.Task, frozenset[str]] = {
    tasks.Task.BINARY: frozenset(
        {
            "demographic_parity_difference",
            "equal_opportunity_difference",
            "equalized_odds_difference",
            "disparate_impact_ratio",
        }
    ),
    tasks.Task.REGRESSION: frozenset({"bias_gap"}),
}


def validate_text(value: str, field_name: str) -> str:
    """Returns non-empty canonical text for a contract field.

    Args:
        value: The text to check.
        field_name: The field name, for the error message.

    Returns:
        `value`, unchanged.

    Raises:
        TypeError: If `value` is not a string.
        ValueError: If the text is empty or has surrounding whitespace.
    """
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    result = value.strip()
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    if result != value:
        raise ValueError(f"{field_name} must not have surrounding whitespace")
    return result


def validate_evidence(
    values: tuple[str, ...], field_name: str
) -> tuple[str, ...]:
    """Validates and returns a deterministic evidence list.

    Args:
        values: The evidence references.
        field_name: The field name, for the error message.

    Returns:
        `values`, unchanged.

    Raises:
        TypeError: If `values` is not a tuple or an entry is not a string.
        ValueError: If an entry is blank or has surrounding whitespace, or a
            reference repeats.
    """
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be a tuple of strings")
    for value in values:
        validate_text(value, field_name)
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


def validate_finite(value: float, field_name: str) -> float:
    """Validates a finite numeric threshold.

    Args:
        value: The number to check.
        field_name: The field name, for the error message.

    Returns:
        The value as a float.

    Raises:
        TypeError: If the value is a bool or not numeric.
        ValueError: If the value is not finite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class MetricDeclaration:
    """One metric explicitly requested by an evaluation policy."""

    name: str
    direction: MetricDirection
    required: bool = True
    custom: bool = False
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validates the declaration before it enters a task policy.

        Raises:
            TypeError: If the name is not a string, the direction is not a
                `MetricDirection`, or the evidence is not a tuple of strings.
            ValueError: If the name is blank or padded with whitespace, the
                evidence is invalid or repeats, or a custom metric has no
                evidence.
        """
        validate_text(self.name, "metric.name")
        if not isinstance(self.direction, MetricDirection):
            raise TypeError("metric.direction must be MetricDirection")
        validate_evidence(self.evidence, "metric.evidence")
        if self.custom and not self.evidence:
            raise ValueError("custom metrics require evidence")

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible metric declaration."""
        return {
            "name": self.name,
            "direction": self.direction.value,
            "required": self.required,
            "custom": self.custom,
            "evidence": list(self.evidence),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class MetricGate:
    """Explicit release threshold for a declared model metric."""

    metric: str
    direction: MetricDirection
    threshold: float

    def __post_init__(self) -> None:
        """Validates the metric gate's name, direction and threshold.

        Raises:
            TypeError: If the metric name is not a string, the direction is
                not a `MetricDirection`, or the threshold is not numeric.
            ValueError: If the metric name is blank or padded with
                whitespace, or the threshold is not finite.
        """
        validate_text(self.metric, "metric_gate.metric")
        if not isinstance(self.direction, MetricDirection):
            raise TypeError("metric_gate.direction must be MetricDirection")
        validate_finite(self.threshold, "metric_gate.threshold")

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible metric gate."""
        return {
            "metric": self.metric,
            "direction": self.direction.value,
            "threshold": float(self.threshold),
        }


def check_open_unit_interval(value: float, field_name: str) -> None:
    """Validates a finite number strictly between zero and one.

    Args:
        value: The number to check.
        field_name: The field name, for the error message.

    Raises:
        ValueError: If the value is not strictly between 0 and 1 (or is not
            finite).
    """
    checked = validate_finite(value, field_name)
    if not 0 < checked < 1:
        raise ValueError(f"{field_name} must be between 0 and 1")


def check_explanation_threshold(
    threshold: float, name: str, required: bool, requirement: str
) -> None:
    """Validates a stability or fidelity threshold of an explanation policy.

    Args:
        threshold: The threshold to check.
        name: The threshold's field name.
        required: Whether its requirement flag is set.
        requirement: The name of the requirement flag.

    Raises:
        ValueError: If the threshold is outside [0, 1] or not finite, or its
            requirement flag is not set.
    """
    value = validate_finite(threshold, f"explanation.{name}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"explanation.{name} must be between 0 and 1")
    if not required:
        raise ValueError(f"{name} requires {requirement}")


@dataclasses.dataclass(frozen=True, slots=True)
class Uncertainty:
    """Task-specific uncertainty promise and statistical requirements."""

    method: UncertaintyMethod
    required: bool = False
    confidence_level: float | None = None
    coverage_target: float | None = None
    minimum_rows: int | None = None

    def __post_init__(self) -> None:
        """Validates confidence, coverage, and sample-size requirements.

        Raises:
            TypeError: If the method is not an `UncertaintyMethod`.
            ValueError: If uncertainty is required with method none, the
                confidence level or coverage target is not strictly between 0
                and 1, the minimum rows is below one, a required method has
                no confidence level, or a required prediction-interval,
                conformal or survival-curve method has no coverage target.
        """
        if not isinstance(self.method, UncertaintyMethod):
            raise TypeError("uncertainty.method must be UncertaintyMethod")
        if self.method is UncertaintyMethod.NONE and self.required:
            raise ValueError(
                "uncertainty cannot be required when method is none"
            )
        if self.confidence_level is not None:
            check_open_unit_interval(
                self.confidence_level, "uncertainty.confidence_level"
            )
        if self.coverage_target is not None:
            check_open_unit_interval(
                self.coverage_target, "uncertainty.coverage_target"
            )
        if self.minimum_rows is not None and self.minimum_rows < 1:
            raise ValueError("uncertainty.minimum_rows must be at least 1")
        if self.method is not UncertaintyMethod.NONE and self.required:
            if self.confidence_level is None:
                raise ValueError("required uncertainty needs confidence_level")
            if (
                self.method
                in {
                    UncertaintyMethod.PREDICTION_INTERVAL,
                    UncertaintyMethod.CONFORMAL,
                    UncertaintyMethod.SURVIVAL_CURVE,
                }
                and self.coverage_target is None
            ):
                raise ValueError(
                    "required interval uncertainty needs coverage_target"
                )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible uncertainty declaration."""
        return {
            "method": self.method.value,
            "required": self.required,
            "confidence_level": self.confidence_level,
            "coverage_target": self.coverage_target,
            "minimum_rows": self.minimum_rows,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Calibration:
    """Task-specific calibration method and release error threshold."""

    method: CalibrationMethod
    required: bool = False
    maximum_error: float | None = None

    def __post_init__(self) -> None:
        """Validates calibration requirements.

        Raises:
            TypeError: If the method is not a `CalibrationMethod`.
            ValueError: If requirements are stated without a method, a
                required calibration has no maximum error, or the maximum
                error is negative or not finite.
        """
        if not isinstance(self.method, CalibrationMethod):
            raise TypeError("calibration.method must be CalibrationMethod")
        if self.method is CalibrationMethod.NONE:
            if self.required or self.maximum_error is not None:
                raise ValueError(
                    "calibration requirements need a calibration method"
                )
        elif self.required and self.maximum_error is None:
            raise ValueError("required calibration needs maximum_error")
        if (
            self.maximum_error is not None
            and validate_finite(self.maximum_error, "calibration.maximum_error")
            < 0
        ):
            raise ValueError("calibration.maximum_error must not be negative")

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible calibration declaration."""
        return {
            "method": self.method.value,
            "required": self.required,
            "maximum_error": self.maximum_error,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Fairness:
    """Sensitive-group evaluation or mitigation declaration."""

    method: FairnessMethod
    sensitive_columns: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validates group columns and fairness metric names.

        Raises:
            TypeError: If the method is not a `FairnessMethod`.
            ValueError: If a column or metric entry is invalid or repeats,
                a method of none comes with columns or metrics, or any other
                method comes without sensitive columns or metrics.
        """
        if not isinstance(self.method, FairnessMethod):
            raise TypeError("fairness.method must be FairnessMethod")
        validate_evidence(self.sensitive_columns, "fairness.sensitive_columns")
        validate_evidence(self.metrics, "fairness.metrics")
        if self.method is FairnessMethod.NONE and (
            self.sensitive_columns or self.metrics
        ):
            raise ValueError(
                "fairness fields must be empty when method is none"
            )
        if self.method is not FairnessMethod.NONE:
            if not self.sensitive_columns:
                raise ValueError("fairness requires sensitive_columns")
            if not self.metrics:
                raise ValueError("fairness requires metrics")

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible fairness declaration."""
        return {
            "method": self.method.value,
            "sensitive_columns": list(self.sensitive_columns),
            "metrics": list(self.metrics),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class FairnessGate:
    """Explicit release threshold for one fairness diagnostic metric."""

    metric: str
    maximum: float | None = None
    minimum: float | None = None
    required_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Requires one explicit bound and its supporting evidence.

        Raises:
            ValueError: If not exactly one of the maximum and minimum is
                given, the maximum is negative, a bound is not finite, or
                there is no required evidence.
        """
        validate_text(self.metric, "fairness_gate.metric")
        if (self.maximum is None) == (self.minimum is None):
            raise ValueError("fairness gate needs exactly one bound")
        if (
            self.maximum is not None
            and validate_finite(self.maximum, "fairness_gate.maximum") < 0
        ):
            raise ValueError("fairness_gate.maximum must not be negative")
        if self.minimum is not None:
            validate_finite(self.minimum, "fairness_gate.minimum")
        validate_evidence(
            self.required_evidence, "fairness_gate.required_evidence"
        )
        if not self.required_evidence:
            raise ValueError("fairness gates require evidence")

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible fairness gate."""
        return {
            "metric": self.metric,
            "maximum": self.maximum,
            "minimum": self.minimum,
            "required_evidence": list(self.required_evidence),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class SmallGroups:
    """Power warning and release-blocking thresholds for subgroup reports."""

    warning_threshold: int
    blocking_threshold: int | None = None

    def __post_init__(self) -> None:
        """Validates subgroup thresholds and their ordering.

        Raises:
            ValueError: If the warning threshold is below one, or the blocking
                threshold is not between one and the warning threshold.
        """
        if self.warning_threshold < 1:
            raise ValueError("small-group warning_threshold must be at least 1")
        if (
            self.blocking_threshold is not None
            and not 1 <= self.blocking_threshold <= self.warning_threshold
        ):
            raise ValueError(
                "small-group blocking_threshold must be between one and "
                "warning_threshold"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible small-group policy."""
        return {
            "warning_threshold": self.warning_threshold,
            "blocking_threshold": self.blocking_threshold,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Explanations:
    """Availability and evidence requirements for model explanations."""

    availability: ExplanationAvailability
    required_methods: tuple[str, ...] = ()
    required_evidence: tuple[str, ...] = ()
    require_stability: bool = False
    require_fidelity: bool = False
    stability_threshold: float | None = None
    fidelity_threshold: float | None = None

    def __post_init__(self) -> None:
        """Validates explanation methods and evidence declarations.

        Raises:
            TypeError: If the availability is not an `ExplanationAvailability`.
            ValueError: If a stability or fidelity threshold is outside
                [0, 1] or lacks its requirement flag, a requirement lacks its
                threshold, requirements come with availability none, or an
                availability level other than none has no required methods.
        """
        if not isinstance(self.availability, ExplanationAvailability):
            raise TypeError(
                "explanation.availability must be ExplanationAvailability"
            )
        validate_evidence(self.required_methods, "explanation.required_methods")
        validate_evidence(
            self.required_evidence, "explanation.required_evidence"
        )
        if self.stability_threshold is not None:
            check_explanation_threshold(
                self.stability_threshold,
                "stability_threshold",
                self.require_stability,
                "require_stability",
            )
        if self.fidelity_threshold is not None:
            check_explanation_threshold(
                self.fidelity_threshold,
                "fidelity_threshold",
                self.require_fidelity,
                "require_fidelity",
            )
        if self.require_stability and self.stability_threshold is None:
            raise ValueError("require_stability needs stability_threshold")
        if self.require_fidelity and self.fidelity_threshold is None:
            raise ValueError("require_fidelity needs fidelity_threshold")
        if self.availability is ExplanationAvailability.NONE and (
            self.required_methods
            or self.required_evidence
            or self.require_stability
            or self.require_fidelity
            or self.stability_threshold is not None
            or self.fidelity_threshold is not None
        ):
            raise ValueError(
                "explanation requirements need an availability level"
            )
        if (
            self.availability is not ExplanationAvailability.NONE
            and not self.required_methods
        ):
            raise ValueError(
                "explanation availability requires required_methods"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible explanation declaration."""
        return {
            "availability": self.availability.value,
            "required_methods": list(self.required_methods),
            "required_evidence": list(self.required_evidence),
            "require_stability": self.require_stability,
            "require_fidelity": self.require_fidelity,
            "stability_threshold": self.stability_threshold,
            "fidelity_threshold": self.fidelity_threshold,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class DiagnosticGate:
    """Required diagnostic and the action taken when its evidence is absent."""

    name: str
    action: DiagnosticAction
    required_evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validates the diagnostic name, action, and evidence references.

        Raises:
            TypeError: If the action is not a `DiagnosticAction`.
            ValueError: If the name is blank, the evidence is invalid or
                repeats, or there is no required evidence.
        """
        validate_text(self.name, "diagnostic_gate.name")
        if not isinstance(self.action, DiagnosticAction):
            raise TypeError("diagnostic_gate.action must be DiagnosticAction")
        validate_evidence(
            self.required_evidence, "diagnostic_gate.required_evidence"
        )
        if not self.required_evidence:
            raise ValueError("diagnostic gates require evidence")

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible diagnostic gate."""
        return {
            "name": self.name,
            "action": self.action.value,
            "required_evidence": list(self.required_evidence),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Gates:
    """Evidence-backed thresholds required before a model is released."""

    metric_gates: tuple[MetricGate, ...]
    diagnostic_gates: tuple[DiagnosticGate, ...]
    required_evidence: tuple[str, ...]
    fairness_gates: tuple[FairnessGate, ...] = ()

    def __post_init__(self) -> None:
        """Requires explicit thresholds, diagnostics, and evidence.

        Raises:
            ValueError: If there are no metric gates, diagnostic gates or
                evidence, the evidence is invalid or repeats, no diagnostic
                gate blocks, or a metric, diagnostic or fairness gate names
                the same metric or diagnostic twice.
        """
        if not self.metric_gates:
            raise ValueError("release policy requires metric_gates")
        if not self.diagnostic_gates:
            raise ValueError("release policy requires diagnostic_gates")
        validate_evidence(self.required_evidence, "release.required_evidence")
        if not self.required_evidence:
            raise ValueError("release policy requires evidence")
        if not any(
            gate.action is DiagnosticAction.BLOCK
            for gate in self.diagnostic_gates
        ):
            raise ValueError("release policy requires a blocking diagnostic")
        metric_names = [gate.metric for gate in self.metric_gates]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("release metric gates must be unique")
        diagnostic_names = [gate.name for gate in self.diagnostic_gates]
        if len(diagnostic_names) != len(set(diagnostic_names)):
            raise ValueError("release diagnostic gates must be unique")
        fairness_names = [gate.metric for gate in self.fairness_gates]
        if len(fairness_names) != len(set(fairness_names)):
            raise ValueError("release fairness gates must be unique")

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible release policy."""
        return {
            "metric_gates": [gate.to_dict() for gate in self.metric_gates],
            "diagnostic_gates": [
                gate.to_dict() for gate in self.diagnostic_gates
            ],
            "fairness_gates": [gate.to_dict() for gate in self.fairness_gates],
            "required_evidence": list(self.required_evidence),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Criteria:
    """Complete task-aware evaluation and release declaration."""

    task: tasks.Task
    primary_metric: str
    metrics: tuple[MetricDeclaration, ...]
    uncertainty: Uncertainty
    calibration: Calibration
    fairness: Fairness
    small_groups: SmallGroups
    explanations: Explanations
    release: Gates
    schema_version: int = POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Rejects incomplete or unsupported policy combinations."""
        self.validate()

    def validate(self) -> None:
        """Validates task support and all cross-field policy references.

        Raises:
            TypeError: If the task is not a `Task`.
            UnsupportedEvaluationPolicyError: If a metric, uncertainty,
                calibration, fairness, explanation or diagnostic choice is not
                supported for the task.
            ValueError: If the schema version is below one, the primary
                metric is blank, not among the metrics or not required,
                metrics are missing or repeated, a built-in metric has the
                wrong direction, or a release gate is inconsistent with the
                declarations.
        """
        if not isinstance(self.task, tasks.Task):
            raise TypeError("policy.task must be Task")
        if self.schema_version < 1:
            raise ValueError("policy.schema_version must be at least 1")
        validate_text(self.primary_metric, "policy.primary_metric")
        if not self.metrics:
            raise ValueError("evaluation policy requires metrics")
        metric_names = [metric.name for metric in self.metrics]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("evaluation metrics must be unique")
        if self.primary_metric not in metric_names:
            raise ValueError("primary_metric must be one of metrics")
        primary = next(
            metric
            for metric in self.metrics
            if metric.name == self.primary_metric
        )
        if not primary.required:
            raise ValueError("primary_metric must be required")
        supported_metrics = SUPPORTED_METRICS[self.task]
        for metric in self.metrics:
            if not metric.custom and metric.name not in supported_metrics:
                raise UnsupportedEvaluationPolicyError(
                    f"metric '{metric.name}' is unsupported for task "
                    f"'{self.task.value}'"
                )
            if (
                not metric.custom
                and METRIC_DIRECTIONS[metric.name] is not metric.direction
            ):
                raise ValueError(
                    f"metric '{metric.name}' has direction "
                    f"{METRIC_DIRECTIONS[metric.name].value}, not "
                    f"{metric.direction.value}"
                )
        self.validate_uncertainty()
        self.validate_calibration()
        self.validate_fairness()
        self.validate_explanations()
        self.validate_release()

    def validate_uncertainty(self) -> None:
        """Rejects uncertainty methods not implemented for this task.

        Raises:
            UnsupportedEvaluationPolicyError: If the method is not supported.
        """
        method = self.uncertainty.method
        if method not in SUPPORTED_UNCERTAINTY[self.task]:
            raise UnsupportedEvaluationPolicyError(
                f"uncertainty method '{method.value}' is unsupported for task "
                f"'{self.task.value}'"
            )

    def validate_calibration(self) -> None:
        """Rejects calibration methods not implemented for this task.

        Raises:
            UnsupportedEvaluationPolicyError: If the method is not supported.
        """
        method = self.calibration.method
        if method not in SUPPORTED_CALIBRATION[self.task]:
            raise UnsupportedEvaluationPolicyError(
                f"calibration method '{method.value}' is unsupported for task "
                f"'{self.task.value}'"
            )

    def validate_fairness(self) -> None:
        """Rejects fairness methods and metrics outside the task boundary.

        Raises:
            UnsupportedEvaluationPolicyError: If the method or a metric is not
                supported.
        """
        method = self.fairness.method
        if method not in SUPPORTED_FAIRNESS[self.task]:
            raise UnsupportedEvaluationPolicyError(
                f"fairness method '{method.value}' is unsupported for task "
                f"'{self.task.value}'"
            )
        supported_metrics = FAIRNESS_METRICS.get(self.task, frozenset())
        for metric in self.fairness.metrics:
            if metric not in supported_metrics:
                raise UnsupportedEvaluationPolicyError(
                    f"fairness metric '{metric}' is unsupported for task "
                    f"'{self.task.value}'"
                )

    def validate_explanations(self) -> None:
        """Rejects explanation levels outside the task boundary.

        Raises:
            UnsupportedEvaluationPolicyError: If the level is not supported.
        """
        availability = self.explanations.availability
        if availability not in SUPPORTED_EXPLANATIONS[self.task]:
            raise UnsupportedEvaluationPolicyError(
                f"explanation level '{availability.value}' is unsupported for "
                f"task '{self.task.value}'"
            )

    def validate_metric_gates(self) -> None:
        """Validates release metric gates against the declared metrics.

        Raises:
            UnsupportedEvaluationPolicyError: If a gate names an undeclared
                metric that the task does not support.
            ValueError: If a gate names an undeclared supported metric or has
                a direction that differs from the declaration.
        """
        declared_metrics = {metric.name: metric for metric in self.metrics}
        for gate in self.release.metric_gates:
            declaration = declared_metrics.get(gate.metric)
            if declaration is None:
                if gate.metric not in SUPPORTED_METRICS[self.task]:
                    raise UnsupportedEvaluationPolicyError(
                        f"metric '{gate.metric}' is unsupported for task "
                        f"'{self.task.value}'"
                    )
                raise ValueError(
                    f"release metric gate '{gate.metric}' is undeclared"
                )
            if declaration.direction is not gate.direction:
                raise ValueError(
                    f"release metric gate '{gate.metric}' has a mismatched "
                    "direction"
                )

    def validate_release(self) -> None:
        """Validates release gates against declared metrics and diagnostics.

        Raises:
            UnsupportedEvaluationPolicyError: If a metric gate names an
                undeclared metric that the task does not support, or a
                diagnostic gate names a diagnostic the task does not support.
            ValueError: If a metric gate names an undeclared supported metric
                or has a direction that differs from the declaration, a
                fairness gate names a metric absent from the fairness
                declaration, or fairness gates exist without a fairness
                method or a fairness method has no gates.
        """
        self.validate_metric_gates()
        for fairness_gate in self.release.fairness_gates:
            if fairness_gate.metric not in self.fairness.metrics:
                raise ValueError(
                    f"release fairness gate '{fairness_gate.metric}' is "
                    "undeclared"
                )
        supported_diagnostics = SUPPORTED_DIAGNOSTICS[self.task]
        for diagnostic_gate in self.release.diagnostic_gates:
            if diagnostic_gate.name not in supported_diagnostics:
                raise UnsupportedEvaluationPolicyError(
                    f"diagnostic '{diagnostic_gate.name}' is unsupported for "
                    "task "
                    f"'{self.task.value}'"
                )
        if (
            self.fairness.method is FairnessMethod.NONE
            and self.release.fairness_gates
        ):
            raise ValueError(
                "fairness release gates need a fairness declaration"
            )
        if (
            self.fairness.method is not FairnessMethod.NONE
            and not self.release.fairness_gates
        ):
            raise ValueError(
                "fairness declaration needs release fairness gates"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns a deterministic JSON-compatible policy record."""
        return {
            "schema_version": self.schema_version,
            "task": self.task.value,
            "primary_metric": self.primary_metric,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "uncertainty": self.uncertainty.to_dict(),
            "calibration": self.calibration.to_dict(),
            "fairness": self.fairness.to_dict(),
            "small_groups": self.small_groups.to_dict(),
            "explanations": self.explanations.to_dict(),
            "release": self.release.to_dict(),
        }

    def canonical_json(self) -> str:
        """Returns stable JSON suitable for hashing and evidence storage."""
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


class Registry:
    """Explicit registry that fails closed for undeclared task policies."""

    def __init__(self, policies: Iterable[Criteria] = ()) -> None:
        """Creates a registry from complete policy declarations.

        Args:
            policies: The policies to register.
        """
        self.records: dict[tasks.Task, Criteria] = {}
        for policy in policies:
            self.register(policy)

    def register(self, policy: Criteria) -> None:
        """Registers a policy without silently replacing another policy.

        The policy is stored under its task.

        Args:
            policy: The policy to register.

        Raises:
            ConfigError: If the task already has a policy.
        """
        if policy.task in self.records:
            raise errors.ConfigError(
                f"evaluation policy already registered for task "
                f"'{policy.task.value}'"
            )
        self.records[policy.task] = policy

    def require(self, task: tasks.Task) -> Criteria:
        """Returns a task policy or rejects an undeclared task explicitly.

        Args:
            task: The task.

        Returns:
            The task's policy.

        Raises:
            TypeError: If `task` is not a `Task`.
            UndeclaredEvaluationPolicyError: If no policy is declared for the
                task.
        """
        if not isinstance(task, tasks.Task):
            raise TypeError("task must be Task")
        policy = self.records.get(task)
        if policy is None:
            raise UndeclaredEvaluationPolicyError(
                f"no evaluation policy declared for task '{task.value}'"
            )
        return policy

    def list_policies(self) -> tuple[Criteria, ...]:
        """Returns policies in deterministic task order."""
        return tuple(
            self.records[task]
            for task in sorted(self.records, key=lambda item: item.value)
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns a deterministic JSON-compatible registry record."""
        return {
            "schema_version": POLICY_SCHEMA_VERSION,
            "policies": [policy.to_dict() for policy in self.list_policies()],
        }

    def canonical_json(self) -> str:
        """Returns stable JSON for the complete policy registry."""
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
