"""The parts of an evaluation policy: metrics, promises and release gates.

Each part is nested in a `policy.Criteria` record and versioned with it, so
it carries no version of its own. A part whose stored keys are single words
while its fields keep their longer names is `records.Spelled`: its
`spelling` maps the field names to the stored keys. `former` maps the
version-1 keys of a part to the current ones, for `policy.Criteria`'s
migration.
"""

from typing import Annotated
from typing import ClassVar
from typing import Self

import pydantic

from scmle.contracts import errors
from scmle.contracts import records
from scmle.contracts.evaluation import support

Finite = Annotated[
    float, pydantic.Strict(), pydantic.AllowInfNan(allow_inf_nan=False)
]
"""A finite number; an integer is accepted and a bool is not."""

Count = Annotated[int, pydantic.Strict()]
"""An integer that is not a bool."""


def sanitize(value: str, field_name: str) -> str:
    """Returns non-empty canonical text for a contract field.

    Args:
        value: The text to check.
        field_name: The field name, for the error message.

    Returns:
        `value`, unchanged.

    Raises:
        InputError: If the text is empty or has surrounding whitespace.
    """
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
        InputError: If an entry is blank or has surrounding whitespace, or a
            reference repeats.
    """
    for value in values:
        sanitize(value, field_name)
    if len(values) != len(set(values)):
        raise errors.InputError(f"{field_name} must not contain duplicates")
    return values


def confine(value: float, field_name: str) -> None:
    """Validates a number strictly between zero and one.

    Args:
        value: The number to check.
        field_name: The field name, for the error message.

    Raises:
        InputError: If the value is not strictly between 0 and 1.
    """
    if not 0 < value < 1:
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
        InputError: If the threshold is outside [0, 1], or its requirement
            flag is not set.
    """
    if not 0.0 <= threshold <= 1.0:
        raise errors.InputError(f"explanation.{name} must be between 0 and 1")
    if not required:
        raise errors.InputError(f"{name} requires {requirement}")


class Declaration(records.Schema):
    """One metric explicitly requested by an evaluation policy."""

    name: str
    direction: support.Orientation
    required: pydantic.StrictBool = True
    custom: pydantic.StrictBool = False
    evidence: tuple[str, ...] = ()

    @pydantic.model_validator(mode="after")
    def check(self) -> Self:
        """Validates the declaration before it enters a task policy.

        Returns:
            The declaration, unchanged.

        Raises:
            InputError: If the name is blank or padded with whitespace, the
                evidence is invalid or repeats, or a custom metric has no
                evidence.
        """
        sanitize(self.name, "metric.name")
        vet(self.evidence, "metric.evidence")
        if self.custom and not self.evidence:
            raise errors.InputError("custom metrics require evidence")
        return self


class Hurdle(records.Schema):
    """Explicit release threshold for a declared model metric."""

    metric: str
    direction: support.Orientation
    threshold: Finite

    @pydantic.model_validator(mode="after")
    def check(self) -> Self:
        """Validates the metric gate's name.

        Returns:
            The gate, unchanged.

        Raises:
            InputError: If the metric name is blank or padded with whitespace.
        """
        sanitize(self.metric, "metric_gate.metric")
        return self


class Uncertainty(records.Spelled):
    """Task-specific uncertainty promise and statistical requirements."""

    former: ClassVar[records.Migration] = records.Migration(
        renames={
            "confidence_level": "confidence",
            "coverage_target": "coverage",
            "minimum_rows": "rows",
        }
    )

    method: support.Method
    required: pydantic.StrictBool = False
    confidence: Finite | None = None
    coverage: Finite | None = None
    rows: Count | None = None

    @pydantic.model_validator(mode="after")
    def check(self) -> Self:
        """Validates confidence, coverage, and sample-size requirements.

        Returns:
            The promise, unchanged.

        Raises:
            InputError: If uncertainty is required with method none, the
                confidence level or coverage target is not strictly between 0
                and 1, the minimum rows is below one, a required method has
                no confidence level, or a required prediction-interval,
                conformal or survival-curve method has no coverage target.
        """
        none = support.Method.NONE
        if self.method is none and self.required:
            raise errors.InputError(
                "uncertainty cannot be required when method is none"
            )
        if self.confidence is not None:
            confine(self.confidence, "uncertainty.confidence_level")
        if self.coverage is not None:
            confine(self.coverage, "uncertainty.coverage_target")
        if self.rows is not None and self.rows < 1:
            raise errors.InputError(
                "uncertainty.minimum_rows must be at least 1"
            )
        if self.method is not none and self.required:
            if self.confidence is None:
                raise errors.InputError(
                    "required uncertainty needs confidence_level"
                )
            if (
                self.method
                in {
                    support.Method.PREDICTION_INTERVAL,
                    support.Method.CONFORMAL,
                    support.Method.SURVIVAL_CURVE,
                }
                and self.coverage is None
            ):
                raise errors.InputError(
                    "required interval uncertainty needs coverage_target"
                )
        return self


class Calibration(records.Spelled):
    """Task-specific calibration method and release error threshold."""

    former: ClassVar[records.Migration] = records.Migration(
        renames={"maximum_error": "error"}
    )

    method: support.Procedure
    required: pydantic.StrictBool = False
    error: Finite | None = None

    @pydantic.model_validator(mode="after")
    def check(self) -> Self:
        """Validates calibration requirements.

        Returns:
            The declaration, unchanged.

        Raises:
            InputError: If requirements are stated without a method, a
                required calibration has no maximum error, or the maximum
                error is negative.
        """
        if self.method is support.Procedure.NONE:
            if self.required or self.error is not None:
                raise errors.InputError(
                    "calibration requirements need a calibration method"
                )
        elif self.required and self.error is None:
            raise errors.InputError("required calibration needs maximum_error")
        if self.error is not None and self.error < 0:
            raise errors.InputError(
                "calibration.maximum_error must not be negative"
            )
        return self


class Fairness(records.Spelled):
    """Sensitive-group evaluation or mitigation declaration."""

    former: ClassVar[records.Migration] = records.Migration(
        renames={"sensitive_columns": "sensitive"}
    )

    method: support.Technique
    sensitive: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()

    @pydantic.model_validator(mode="after")
    def check(self) -> Self:
        """Validates group columns and fairness metric names.

        Returns:
            The declaration, unchanged.

        Raises:
            InputError: If a column or metric entry is invalid or repeats,
                a method of none comes with columns or metrics, or any other
                method comes without sensitive columns or metrics.
        """
        vet(self.sensitive, "fairness.sensitive_columns")
        vet(self.metrics, "fairness.metrics")
        if self.method is support.Technique.NONE:
            if self.sensitive or self.metrics:
                raise errors.InputError(
                    "fairness fields must be empty when method is none"
                )
        elif not self.sensitive:
            raise errors.InputError("fairness requires sensitive_columns")
        elif not self.metrics:
            raise errors.InputError("fairness requires metrics")
        return self


class Equity(records.Spelled):
    """Explicit release threshold for one fairness diagnostic metric."""

    former: ClassVar[records.Migration] = records.Migration(
        renames={"required_evidence": "evidence"}
    )

    metric: str
    maximum: Finite | None = None
    minimum: Finite | None = None
    evidence: tuple[str, ...] = ()

    @pydantic.model_validator(mode="after")
    def check(self) -> Self:
        """Requires one explicit bound and its supporting evidence.

        Returns:
            The gate, unchanged.

        Raises:
            InputError: If not exactly one of the maximum and minimum is
                given, the maximum is negative, or there is no required
                evidence.
        """
        sanitize(self.metric, "fairness_gate.metric")
        if (self.maximum is None) == (self.minimum is None):
            raise errors.InputError("fairness gate needs exactly one bound")
        if self.maximum is not None and self.maximum < 0:
            raise errors.InputError(
                "fairness_gate.maximum must not be negative"
            )
        vet(self.evidence, "fairness_gate.required_evidence")
        if not self.evidence:
            raise errors.InputError("fairness gates require evidence")
        return self


class Subgroups(records.Spelled):
    """Power warning and release-blocking thresholds for subgroup reports."""

    former: ClassVar[records.Migration] = records.Migration(
        renames={
            "warning_threshold": "warning",
            "blocking_threshold": "blocking",
        }
    )

    warning: Count
    blocking: Count | None = None

    @pydantic.model_validator(mode="after")
    def check(self) -> Self:
        """Validates subgroup thresholds and their ordering.

        Returns:
            The policy, unchanged.

        Raises:
            InputError: If the warning threshold is below one, or the blocking
                threshold is not between one and the warning threshold.
        """
        if self.warning < 1:
            raise errors.InputError(
                "small-group warning_threshold must be at least 1"
            )
        if self.blocking is not None and not 1 <= self.blocking <= self.warning:
            raise errors.InputError(
                "small-group blocking_threshold must be between one and "
                "warning_threshold"
            )
        return self


GROUPS = {
    "stability": {
        "require_stability": "required",
        "stability_threshold": "threshold",
    },
    "fidelity": {
        "require_fidelity": "required",
        "fidelity_threshold": "threshold",
    },
}
"""The stability and fidelity requirements, stored as nested objects."""


class Explanations(records.Spelled):
    """Availability and evidence requirements for model explanations."""

    spelling: ClassVar[records.Migration] = records.nest({}, groups=GROUPS)
    former: ClassVar[records.Migration] = records.nest(
        renames={
            "required_methods": "methods",
            "required_evidence": "evidence",
        },
        groups=GROUPS,
    )

    availability: support.Explicability
    methods: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    require_stability: pydantic.StrictBool = False
    require_fidelity: pydantic.StrictBool = False
    stability_threshold: Finite | None = None
    fidelity_threshold: Finite | None = None

    @pydantic.model_validator(mode="after")
    def check(self) -> Self:
        """Validates explanation methods and evidence declarations.

        Returns:
            The declaration, unchanged.

        Raises:
            InputError: If a stability or fidelity threshold is outside
                [0, 1] or lacks its requirement flag, a requirement lacks its
                threshold, requirements come with availability none, or an
                availability level other than none has no required methods.
        """
        vet(self.methods, "explanation.required_methods")
        vet(self.evidence, "explanation.required_evidence")
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
        if self.availability is support.Explicability.NONE:
            if (
                self.methods
                or self.evidence
                or self.require_stability
                or self.require_fidelity
            ):
                raise errors.InputError(
                    "explanation requirements need an availability level"
                )
        elif not self.methods:
            raise errors.InputError(
                "explanation availability requires required_methods"
            )
        return self


class Requirement(records.Spelled):
    """Required diagnostic and the action taken when its evidence is absent.

    Attributes:
        name: The diagnostic.
        action: What a failed or missing report does to the release.
        evidence: Files the report rests on.
        decisive: Whether the report must carry a boolean pass verdict. A
            presence gate (a report without error, such as an interval) is
            not decisive; a gate on a test that passes or fails is, and a
            report that omits its verdict then fails it (ERR-5).
        former: Version-1 stored keys to the field names, for the
            policy's migration.
    """

    former: ClassVar[records.Migration] = records.Migration(
        renames={"required_evidence": "evidence"}
    )

    name: str
    action: support.Remedy
    evidence: tuple[str, ...]
    decisive: pydantic.StrictBool = False

    @pydantic.model_validator(mode="after")
    def check(self) -> Self:
        """Validates the diagnostic name and evidence references.

        Returns:
            The gate, unchanged.

        Raises:
            InputError: If the name is blank, the evidence is invalid or
                repeats, or there is no required evidence.
        """
        sanitize(self.name, "diagnostic_gate.name")
        vet(self.evidence, "diagnostic_gate.required_evidence")
        if not self.evidence:
            raise errors.InputError("diagnostic gates require evidence")
        return self


class Gates(records.Spelled):
    """Evidence-backed thresholds required before a model is released."""

    former: ClassVar[records.Migration] = records.Migration(
        renames={
            "metric_gates": "metrics",
            "diagnostic_gates": "diagnostics",
            "fairness_gates": "fairness",
            "required_evidence": "evidence",
        }
    )

    metrics: tuple[Hurdle, ...]
    diagnostics: tuple[Requirement, ...]
    fairness: tuple[Equity, ...] = ()
    evidence: tuple[str, ...]

    @pydantic.model_validator(mode="after")
    def check(self) -> Self:
        """Requires explicit thresholds, diagnostics, and evidence.

        Returns:
            The policy, unchanged.

        Raises:
            InputError: If there are no metric gates, diagnostic gates or
                evidence, the evidence is invalid or repeats, no diagnostic
                gate blocks, or a metric, diagnostic or fairness gate names
                the same metric or diagnostic twice.
        """
        if not self.metrics:
            raise errors.InputError("release policy requires metric_gates")
        if not self.diagnostics:
            raise errors.InputError("release policy requires diagnostic_gates")
        vet(self.evidence, "release.required_evidence")
        if not self.evidence:
            raise errors.InputError("release policy requires evidence")
        if not any(
            gate.action is support.Remedy.BLOCK for gate in self.diagnostics
        ):
            raise errors.InputError(
                "release policy requires a blocking diagnostic"
            )
        for kind, names in (
            ("metric", [gate.metric for gate in self.metrics]),
            ("diagnostic", [gate.name for gate in self.diagnostics]),
            ("fairness", [gate.metric for gate in self.fairness]),
        ):
            if len(names) != len(set(names)):
                raise errors.InputError(f"release {kind} gates must be unique")
        return self
