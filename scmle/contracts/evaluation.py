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


class Orientation(enum.StrEnum):
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


class Procedure(enum.StrEnum):
    """Calibration procedure or diagnostic promised by a policy."""

    NONE = "none"
    RELIABILITY_CURVE = "reliability_curve"
    PLATT = "platt"
    ISOTONIC = "isotonic"
    DIRICHLET = "dirichlet"


class Technique(enum.StrEnum):
    """Fairness analysis or mitigation promised by a policy."""

    NONE = "none"
    GROUP_REPORT = "group_report"
    INTERSECTIONAL_REPORT = "intersectional_report"
    MITIGATION = "mitigation"


class Explicability(enum.StrEnum):
    """Level of model explanation the evaluation must make available."""

    NONE = "none"
    GLOBAL = "global"
    LOCAL = "local"
    COUNTERFACTUAL = "counterfactual"
    FULL = "full"


class Remedy(enum.StrEnum):
    """Action taken when a required diagnostic does not pass."""

    WARN = "warn"
    BLOCK = "block"


class UnsupportedError(errors.ConfigError):
    """Raised when a task and evaluation policy combination is unsupported."""

    category = "evaluation_policy"


class UndeclaredError(UnsupportedError):
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


METRIC_DIRECTIONS: Mapping[str, Orientation] = {
    "roc_auc": Orientation.MAXIMIZE,
    "roc_auc_ovr": Orientation.MAXIMIZE,
    "pr_auc": Orientation.MAXIMIZE,
    "log_loss": Orientation.MINIMIZE,
    "f1": Orientation.MAXIMIZE,
    "f1_macro": Orientation.MAXIMIZE,
    "balanced_accuracy": Orientation.MAXIMIZE,
    "accuracy": Orientation.MAXIMIZE,
    "mae": Orientation.MINIMIZE,
    "rmse": Orientation.MINIMIZE,
    "r2": Orientation.MAXIMIZE,
    "qwk": Orientation.MAXIMIZE,
    "mace": Orientation.MINIMIZE,
    "accuracy_levels": Orientation.MAXIMIZE,
    "f1_macro_labels": Orientation.MAXIMIZE,
    "f1_micro_labels": Orientation.MAXIMIZE,
    "hamming": Orientation.MINIMIZE,
    "subset_accuracy": Orientation.MAXIMIZE,
    "r2_mean": Orientation.MAXIMIZE,
    "mae_scaled": Orientation.MINIMIZE,
    "mase": Orientation.MINIMIZE,
    "forecast_mae": Orientation.MINIMIZE,
    "forecast_smape": Orientation.MINIMIZE,
    "qini": Orientation.MAXIMIZE,
    "uplift_at_30": Orientation.MAXIMIZE,
    "ndcg": Orientation.MAXIMIZE,
    "map": Orientation.MAXIMIZE,
    "mrr": Orientation.MAXIMIZE,
    "precision_at_10": Orientation.MAXIMIZE,
    "recall_at_10": Orientation.MAXIMIZE,
    "c_index": Orientation.MAXIMIZE,
    "ibs": Orientation.MINIMIZE,
    "td_auc": Orientation.MAXIMIZE,
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
    **{task: frozenset({UncertaintyMethod.NONE}) for task in tasks.Task},
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


SUPPORTED_CALIBRATION: Mapping[tasks.Task, frozenset[Procedure]] = {
    **{task: frozenset({Procedure.NONE}) for task in tasks.Task},
    tasks.Task.BINARY: frozenset(
        {
            Procedure.NONE,
            Procedure.RELIABILITY_CURVE,
            Procedure.PLATT,
            Procedure.ISOTONIC,
        }
    ),
    tasks.Task.MULTICLASS: frozenset(
        {
            Procedure.NONE,
            Procedure.RELIABILITY_CURVE,
            Procedure.DIRICHLET,
        }
    ),
}


SUPPORTED_FAIRNESS: Mapping[tasks.Task, frozenset[Technique]] = {
    **{task: frozenset({Technique.NONE}) for task in tasks.Task},
    tasks.Task.BINARY: frozenset(
        {
            Technique.NONE,
            Technique.GROUP_REPORT,
            Technique.MITIGATION,
        }
    ),
    tasks.Task.REGRESSION: frozenset(
        {
            Technique.NONE,
            Technique.GROUP_REPORT,
            Technique.MITIGATION,
        }
    ),
}


SUPPORTED_EXPLANATIONS: Mapping[tasks.Task, frozenset[Explicability]] = {
    **{
        task: frozenset({Explicability.NONE, Explicability.GLOBAL})
        for task in tasks.Task
    },
    tasks.Task.BINARY: frozenset(
        {
            Explicability.NONE,
            Explicability.GLOBAL,
            Explicability.LOCAL,
            Explicability.COUNTERFACTUAL,
            Explicability.FULL,
        }
    ),
    tasks.Task.REGRESSION: frozenset(
        {
            Explicability.NONE,
            Explicability.GLOBAL,
            Explicability.LOCAL,
            Explicability.COUNTERFACTUAL,
            Explicability.FULL,
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
SUPPORTED_DIAGNOSTICS: Mapping[tasks.Task, frozenset[str]] = {
    **dict.fromkeys(tasks.Task, COMMON_DIAGNOSTICS),
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
        raise errors.KindError(f"{field_name} must be a string")
    result = value.strip()
    if not result:
        raise errors.InputError(f"{field_name} must not be empty")
    if result != value:
        raise errors.InputError(
            f"{field_name} must not have surrounding whitespace"
        )
    return result


def vet(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
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
        raise errors.KindError(f"{field_name} must be a tuple of strings")
    for value in values:
        validate_text(value, field_name)
    if len(values) != len(set(values)):
        raise errors.InputError(f"{field_name} must not contain duplicates")
    return values


def quantify(value: float, field_name: str) -> float:
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
        raise errors.KindError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise errors.InputError(f"{field_name} must be finite")
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class Declaration:
    """One metric explicitly requested by an evaluation policy."""

    name: str
    direction: Orientation
    required: bool = True
    custom: bool = False
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validates the declaration before it enters a task policy.

        Raises:
            TypeError: If the name is not a string, the direction is not a
                `Orientation`, or the evidence is not a tuple of strings.
            ValueError: If the name is blank or padded with whitespace, the
                evidence is invalid or repeats, or a custom metric has no
                evidence.
        """
        validate_text(self.name, "metric.name")
        if not isinstance(self.direction, Orientation):
            raise errors.KindError("metric.direction must be Orientation")
        vet(self.evidence, "metric.evidence")
        if self.custom and not self.evidence:
            raise errors.InputError("custom metrics require evidence")

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
class Hurdle:
    """Explicit release threshold for a declared model metric."""

    metric: str
    direction: Orientation
    threshold: float

    def __post_init__(self) -> None:
        """Validates the metric gate's name, direction and threshold.

        Raises:
            TypeError: If the metric name is not a string, the direction is
                not a `Orientation`, or the threshold is not numeric.
            ValueError: If the metric name is blank or padded with
                whitespace, or the threshold is not finite.
        """
        validate_text(self.metric, "metric_gate.metric")
        if not isinstance(self.direction, Orientation):
            raise errors.KindError("metric_gate.direction must be Orientation")
        quantify(self.threshold, "metric_gate.threshold")

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible metric gate."""
        return {
            "metric": self.metric,
            "direction": self.direction.value,
            "threshold": float(self.threshold),
        }


def confine(value: float, field_name: str) -> None:
    """Validates a finite number strictly between zero and one.

    Args:
        value: The number to check.
        field_name: The field name, for the error message.

    Raises:
        ValueError: If the value is not strictly between 0 and 1 (or is not
            finite).
    """
    checked = quantify(value, field_name)
    if not 0 < checked < 1:
        raise errors.InputError(f"{field_name} must be between 0 and 1")


def qualify(
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
    value = quantify(threshold, f"explanation.{name}")
    if not 0.0 <= value <= 1.0:
        raise errors.InputError(f"explanation.{name} must be between 0 and 1")
    if not required:
        raise errors.InputError(f"{name} requires {requirement}")


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
            raise errors.KindError(
                "uncertainty.method must be UncertaintyMethod"
            )
        if self.method is UncertaintyMethod.NONE and self.required:
            raise errors.InputError(
                "uncertainty cannot be required when method is none"
            )
        if self.confidence_level is not None:
            confine(self.confidence_level, "uncertainty.confidence_level")
        if self.coverage_target is not None:
            confine(self.coverage_target, "uncertainty.coverage_target")
        if self.minimum_rows is not None and self.minimum_rows < 1:
            raise errors.InputError(
                "uncertainty.minimum_rows must be at least 1"
            )
        if self.method is not UncertaintyMethod.NONE and self.required:
            if self.confidence_level is None:
                raise errors.InputError(
                    "required uncertainty needs confidence_level"
                )
            if (
                self.method
                in {
                    UncertaintyMethod.PREDICTION_INTERVAL,
                    UncertaintyMethod.CONFORMAL,
                    UncertaintyMethod.SURVIVAL_CURVE,
                }
                and self.coverage_target is None
            ):
                raise errors.InputError(
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

    method: Procedure
    required: bool = False
    maximum_error: float | None = None

    def __post_init__(self) -> None:
        """Validates calibration requirements.

        Raises:
            TypeError: If the method is not a `Procedure`.
            ValueError: If requirements are stated without a method, a
                required calibration has no maximum error, or the maximum
                error is negative or not finite.
        """
        if not isinstance(self.method, Procedure):
            raise errors.KindError("calibration.method must be Procedure")
        if self.method is Procedure.NONE:
            if self.required or self.maximum_error is not None:
                raise errors.InputError(
                    "calibration requirements need a calibration method"
                )
        elif self.required and self.maximum_error is None:
            raise errors.InputError("required calibration needs maximum_error")
        if (
            self.maximum_error is not None
            and quantify(self.maximum_error, "calibration.maximum_error") < 0
        ):
            raise errors.InputError(
                "calibration.maximum_error must not be negative"
            )

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

    method: Technique
    sensitive_columns: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validates group columns and fairness metric names.

        Raises:
            TypeError: If the method is not a `Technique`.
            ValueError: If a column or metric entry is invalid or repeats,
                a method of none comes with columns or metrics, or any other
                method comes without sensitive columns or metrics.
        """
        if not isinstance(self.method, Technique):
            raise errors.KindError("fairness.method must be Technique")
        vet(self.sensitive_columns, "fairness.sensitive_columns")
        vet(self.metrics, "fairness.metrics")
        if self.method is Technique.NONE and (
            self.sensitive_columns or self.metrics
        ):
            raise errors.InputError(
                "fairness fields must be empty when method is none"
            )
        if self.method is not Technique.NONE:
            if not self.sensitive_columns:
                raise errors.InputError("fairness requires sensitive_columns")
            if not self.metrics:
                raise errors.InputError("fairness requires metrics")

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible fairness declaration."""
        return {
            "method": self.method.value,
            "sensitive_columns": list(self.sensitive_columns),
            "metrics": list(self.metrics),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Equity:
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
            raise errors.InputError("fairness gate needs exactly one bound")
        if (
            self.maximum is not None
            and quantify(self.maximum, "fairness_gate.maximum") < 0
        ):
            raise errors.InputError(
                "fairness_gate.maximum must not be negative"
            )
        if self.minimum is not None:
            quantify(self.minimum, "fairness_gate.minimum")
        vet(self.required_evidence, "fairness_gate.required_evidence")
        if not self.required_evidence:
            raise errors.InputError("fairness gates require evidence")

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible fairness gate."""
        return {
            "metric": self.metric,
            "maximum": self.maximum,
            "minimum": self.minimum,
            "required_evidence": list(self.required_evidence),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Subgroups:
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
            raise errors.InputError(
                "small-group warning_threshold must be at least 1"
            )
        if (
            self.blocking_threshold is not None
            and not 1 <= self.blocking_threshold <= self.warning_threshold
        ):
            raise errors.InputError(
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

    availability: Explicability
    required_methods: tuple[str, ...] = ()
    required_evidence: tuple[str, ...] = ()
    require_stability: bool = False
    require_fidelity: bool = False
    stability_threshold: float | None = None
    fidelity_threshold: float | None = None

    def __post_init__(self) -> None:
        """Validates explanation methods and evidence declarations.

        Raises:
            TypeError: If the availability is not an `Explicability`.
            ValueError: If a stability or fidelity threshold is outside
                [0, 1] or lacks its requirement flag, a requirement lacks its
                threshold, requirements come with availability none, or an
                availability level other than none has no required methods.
        """
        if not isinstance(self.availability, Explicability):
            raise errors.KindError(
                "explanation.availability must be Explicability"
            )
        vet(self.required_methods, "explanation.required_methods")
        vet(self.required_evidence, "explanation.required_evidence")
        if self.stability_threshold is not None:
            qualify(
                self.stability_threshold,
                "stability_threshold",
                self.require_stability,
                "require_stability",
            )
        if self.fidelity_threshold is not None:
            qualify(
                self.fidelity_threshold,
                "fidelity_threshold",
                self.require_fidelity,
                "require_fidelity",
            )
        if self.require_stability and self.stability_threshold is None:
            raise errors.InputError(
                "require_stability needs stability_threshold"
            )
        if self.require_fidelity and self.fidelity_threshold is None:
            raise errors.InputError("require_fidelity needs fidelity_threshold")
        if self.availability is Explicability.NONE and (
            self.required_methods
            or self.required_evidence
            or self.require_stability
            or self.require_fidelity
            or self.stability_threshold is not None
            or self.fidelity_threshold is not None
        ):
            raise errors.InputError(
                "explanation requirements need an availability level"
            )
        if (
            self.availability is not Explicability.NONE
            and not self.required_methods
        ):
            raise errors.InputError(
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
class Requirement:
    """Required diagnostic and the action taken when its evidence is absent.

    Attributes:
        name: The diagnostic.
        action: What a failed or missing report does to the release.
        required_evidence: Files the report rests on.
        decisive: Whether the report must carry a boolean pass verdict. A
            presence gate (a report without error, such as an interval) is
            not decisive; a gate on a test that passes or fails is, and a
            report that omits its verdict then fails it (ERR-5).
    """

    name: str
    action: Remedy
    required_evidence: tuple[str, ...]
    decisive: bool = False

    def __post_init__(self) -> None:
        """Validates the diagnostic name, action, and evidence references.

        Raises:
            TypeError: If the action is not a `Remedy`.
            ValueError: If the name is blank, the evidence is invalid or
                repeats, or there is no required evidence.
        """
        validate_text(self.name, "diagnostic_gate.name")
        if not isinstance(self.action, Remedy):
            raise errors.KindError("diagnostic_gate.action must be Remedy")
        vet(self.required_evidence, "diagnostic_gate.required_evidence")
        if not self.required_evidence:
            raise errors.InputError("diagnostic gates require evidence")
        if not isinstance(self.decisive, bool):
            raise errors.KindError("diagnostic_gate.decisive must be bool")

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible diagnostic gate."""
        return {
            "name": self.name,
            "action": self.action.value,
            "required_evidence": list(self.required_evidence),
            "decisive": self.decisive,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Gates:
    """Evidence-backed thresholds required before a model is released."""

    metric_gates: tuple[Hurdle, ...]
    diagnostic_gates: tuple[Requirement, ...]
    required_evidence: tuple[str, ...]
    fairness_gates: tuple[Equity, ...] = ()

    def __post_init__(self) -> None:
        """Requires explicit thresholds, diagnostics, and evidence.

        Raises:
            ValueError: If there are no metric gates, diagnostic gates or
                evidence, the evidence is invalid or repeats, no diagnostic
                gate blocks, or a metric, diagnostic or fairness gate names
                the same metric or diagnostic twice.
        """
        if not self.metric_gates:
            raise errors.InputError("release policy requires metric_gates")
        if not self.diagnostic_gates:
            raise errors.InputError("release policy requires diagnostic_gates")
        vet(self.required_evidence, "release.required_evidence")
        if not self.required_evidence:
            raise errors.InputError("release policy requires evidence")
        if not any(
            gate.action is Remedy.BLOCK for gate in self.diagnostic_gates
        ):
            raise errors.InputError(
                "release policy requires a blocking diagnostic"
            )
        metric_names = [gate.metric for gate in self.metric_gates]
        if len(metric_names) != len(set(metric_names)):
            raise errors.InputError("release metric gates must be unique")
        diagnostic_names = [gate.name for gate in self.diagnostic_gates]
        if len(diagnostic_names) != len(set(diagnostic_names)):
            raise errors.InputError("release diagnostic gates must be unique")
        fairness_names = [gate.metric for gate in self.fairness_gates]
        if len(fairness_names) != len(set(fairness_names)):
            raise errors.InputError("release fairness gates must be unique")

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
    metrics: tuple[Declaration, ...]
    uncertainty: Uncertainty
    calibration: Calibration
    fairness: Fairness
    small_groups: Subgroups
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
            UnsupportedError: If a metric, uncertainty,
                calibration, fairness, explanation or diagnostic choice is not
                supported for the task.
            ValueError: If the schema version is below one, the primary
                metric is blank, not among the metrics or not required,
                metrics are missing or repeated, a built-in metric has the
                wrong direction, or a release gate is inconsistent with the
                declarations.
        """
        if not isinstance(self.task, tasks.Task):
            raise errors.KindError("policy.task must be Task")
        if self.schema_version < 1:
            raise errors.InputError("policy.schema_version must be at least 1")
        validate_text(self.primary_metric, "policy.primary_metric")
        if not self.metrics:
            raise errors.InputError("evaluation policy requires metrics")
        metric_names = [metric.name for metric in self.metrics]
        if len(metric_names) != len(set(metric_names)):
            raise errors.InputError("evaluation metrics must be unique")
        if self.primary_metric not in metric_names:
            raise errors.InputError("primary_metric must be one of metrics")
        primary = next(
            metric
            for metric in self.metrics
            if metric.name == self.primary_metric
        )
        if not primary.required:
            raise errors.InputError("primary_metric must be required")
        supported_metrics = SUPPORTED_METRICS[self.task]
        for metric in self.metrics:
            if not metric.custom and metric.name not in supported_metrics:
                raise UnsupportedError(
                    f"metric '{metric.name}' is unsupported for task "
                    f"'{self.task.value}'"
                )
            if (
                not metric.custom
                and METRIC_DIRECTIONS[metric.name] is not metric.direction
            ):
                raise errors.InputError(
                    f"metric '{metric.name}' has direction "
                    f"{METRIC_DIRECTIONS[metric.name].value}, not "
                    f"{metric.direction.value}"
                )
        self.enforce()
        self.validate_calibration()
        self.scope()
        self.validate_explanations()
        self.audit()

    def enforce(self) -> None:
        """Rejects uncertainty methods not implemented for this task.

        Raises:
            UnsupportedError: If the method is not supported.
        """
        method = self.uncertainty.method
        if method not in SUPPORTED_UNCERTAINTY[self.task]:
            raise UnsupportedError(
                f"uncertainty method '{method.value}' is unsupported for task "
                f"'{self.task.value}'"
            )

    def validate_calibration(self) -> None:
        """Rejects calibration methods not implemented for this task.

        Raises:
            UnsupportedError: If the method is not supported.
        """
        method = self.calibration.method
        if method not in SUPPORTED_CALIBRATION[self.task]:
            raise UnsupportedError(
                f"calibration method '{method.value}' is unsupported for task "
                f"'{self.task.value}'"
            )

    def scope(self) -> None:
        """Rejects fairness methods and metrics outside the task boundary.

        Raises:
            UnsupportedError: If the method or a metric is not
                supported.
        """
        method = self.fairness.method
        if method not in SUPPORTED_FAIRNESS[self.task]:
            raise UnsupportedError(
                f"fairness method '{method.value}' is unsupported for task "
                f"'{self.task.value}'"
            )
        supported_metrics = FAIRNESS_METRICS.get(self.task, frozenset())
        for metric in self.fairness.metrics:
            if metric not in supported_metrics:
                raise UnsupportedError(
                    f"fairness metric '{metric}' is unsupported for task "
                    f"'{self.task.value}'"
                )

    def validate_explanations(self) -> None:
        """Rejects explanation levels outside the task boundary.

        Raises:
            UnsupportedError: If the level is not supported.
        """
        availability = self.explanations.availability
        if availability not in SUPPORTED_EXPLANATIONS[self.task]:
            raise UnsupportedError(
                f"explanation level '{availability.value}' is unsupported for "
                f"task '{self.task.value}'"
            )

    def gatekeep(self) -> None:
        """Validates release metric gates against the declared metrics.

        Raises:
            UnsupportedError: If a gate names an undeclared
                metric that the task does not support.
            ValueError: If a gate names an undeclared supported metric or has
                a direction that differs from the declaration.
        """
        declared_metrics = {metric.name: metric for metric in self.metrics}
        for gate in self.release.metric_gates:
            declaration = declared_metrics.get(gate.metric)
            if declaration is None:
                if gate.metric not in SUPPORTED_METRICS[self.task]:
                    raise UnsupportedError(
                        f"metric '{gate.metric}' is unsupported for task "
                        f"'{self.task.value}'"
                    )
                raise errors.InputError(
                    f"release metric gate '{gate.metric}' is undeclared"
                )
            if declaration.direction is not gate.direction:
                raise errors.InputError(
                    f"release metric gate '{gate.metric}' has a mismatched "
                    "direction"
                )

    def audit(self) -> None:
        """Validates release gates against declared metrics and diagnostics.

        Raises:
            UnsupportedError: If a metric gate names an
                undeclared metric that the task does not support, or a
                diagnostic gate names a diagnostic the task does not support.
            ValueError: If a metric gate names an undeclared supported metric
                or has a direction that differs from the declaration, a
                fairness gate names a metric absent from the fairness
                declaration, or fairness gates exist without a fairness
                method or a fairness method has no gates.
        """
        self.gatekeep()
        for fairness_gate in self.release.fairness_gates:
            if fairness_gate.metric not in self.fairness.metrics:
                raise errors.InputError(
                    f"release fairness gate '{fairness_gate.metric}' is "
                    "undeclared"
                )
        supported_diagnostics = SUPPORTED_DIAGNOSTICS[self.task]
        for diagnostic_gate in self.release.diagnostic_gates:
            if diagnostic_gate.name not in supported_diagnostics:
                raise UnsupportedError(
                    f"diagnostic '{diagnostic_gate.name}' is unsupported for "
                    "task "
                    f"'{self.task.value}'"
                )
        if (
            self.fairness.method is Technique.NONE
            and self.release.fairness_gates
        ):
            raise errors.InputError(
                "fairness release gates need a fairness declaration"
            )
        if (
            self.fairness.method is not Technique.NONE
            and not self.release.fairness_gates
        ):
            raise errors.InputError(
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
            UndeclaredError: If no policy is declared for the
                task.
        """
        if not isinstance(task, tasks.Task):
            raise errors.KindError("task must be Task")
        policy = self.records.get(task)
        if policy is None:
            raise UndeclaredError(
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
