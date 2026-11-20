"""Behaviour of the evaluation policy parts, the policy and its registry."""

import json

import pydantic
import pytest

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.contracts.evaluation import parts
from scmle.contracts.evaluation import policy
from scmle.contracts.evaluation import support


def gates(**changes):
    values = {
        "metrics": (
            parts.Hurdle(
                metric="roc_auc",
                direction=support.Orientation.MAXIMIZE,
                threshold=0.7,
            ),
        ),
        "diagnostics": (
            parts.Requirement(
                name="metric_uncertainty",
                action=support.Remedy.BLOCK,
                evidence=("ci.json",),
            ),
        ),
        "evidence": ("run.json",),
    }
    values.update(changes)
    return parts.Gates(**values)


def criteria(**changes):
    values = {
        "task": tasks.Task.BINARY,
        "metric": "roc_auc",
        "metrics": (
            parts.Declaration(
                name="roc_auc", direction=support.Orientation.MAXIMIZE
            ),
            parts.Declaration(
                name="log_loss",
                direction=support.Orientation.MINIMIZE,
                required=False,
            ),
        ),
        "uncertainty": parts.Uncertainty(method=support.Method.NONE),
        "calibration": parts.Calibration(method=support.Procedure.NONE),
        "fairness": parts.Fairness(method=support.Technique.NONE),
        "small": parts.Subgroups(warning=30, blocking=10),
        "explanations": parts.Explanations(
            availability=support.Explicability.GLOBAL,
            methods=("permutation_importance",),
        ),
        "release": gates(),
    }
    values.update(changes)
    return policy.Criteria(**values)


def invalid(message, build, **values):
    with pytest.raises(pydantic.ValidationError, match=message):
        build(**values)


# ---- parts helpers ---------------------------------------------------------


def test_sanitize_refuses_blank_or_padded_text():
    assert parts.sanitize("a b", "f") == "a b"
    with pytest.raises(errors.InputError, match="f must not be empty"):
        parts.sanitize("  ", "f")
    with pytest.raises(errors.InputError, match="surrounding whitespace"):
        parts.sanitize(" a", "f")


def test_vet_refuses_duplicates_and_blank_entries():
    assert parts.vet(("a", "b"), "f") == ("a", "b")
    assert parts.vet((), "f") == ()
    with pytest.raises(
        errors.InputError, match="f must not contain duplicates"
    ):
        parts.vet(("a", "a"), "f")
    with pytest.raises(errors.InputError, match="must not be empty"):
        parts.vet(("",), "f")


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5])
def test_confine_needs_an_open_unit_interval(value):
    with pytest.raises(errors.InputError, match="f must be between 0 and 1"):
        parts.confine(value, "f")


def test_confine_accepts_the_inside():
    assert parts.confine(0.5, "f") is None
    assert parts.confine(1e-9, "f") is None


def test_qualify_needs_a_unit_threshold_and_its_flag():
    assert parts.qualify(0.0, "t", True, "r") is None
    assert parts.qualify(1.0, "t", True, "r") is None
    with pytest.raises(
        errors.InputError, match=r"explanation\.t must be between"
    ):
        parts.qualify(1.01, "t", True, "r")
    with pytest.raises(errors.InputError, match="t requires r"):
        parts.qualify(0.5, "t", False, "r")


# ---- parts records ---------------------------------------------------------


def test_a_declaration_needs_a_clean_name_and_custom_evidence():
    made = parts.Declaration(name="m", direction=support.Orientation.MAXIMIZE)
    assert (made.required, made.custom, made.evidence) == (True, False, ())
    invalid(
        "metric.name must not be empty",
        parts.Declaration,
        name="",
        direction=support.Orientation.MAXIMIZE,
    )
    invalid(
        "custom metrics require evidence",
        parts.Declaration,
        name="m",
        direction=support.Orientation.MAXIMIZE,
        custom=True,
    )
    assert parts.Declaration(
        name="m",
        direction=support.Orientation.MAXIMIZE,
        custom=True,
        evidence=("e",),
    )
    invalid(
        "valid boolean",
        parts.Declaration,
        name="m",
        direction=support.Orientation.MAXIMIZE,
        required=1,
    )


def test_a_hurdle_needs_a_finite_threshold():
    invalid(
        "metric_gate.metric",
        parts.Hurdle,
        metric=" m",
        direction=support.Orientation.MAXIMIZE,
        threshold=1.0,
    )
    with pytest.raises(pydantic.ValidationError):
        parts.Hurdle(
            metric="m",
            direction=support.Orientation.MAXIMIZE,
            threshold=float("inf"),
        )
    with pytest.raises(pydantic.ValidationError):
        parts.Hurdle(
            metric="m", direction=support.Orientation.MAXIMIZE, threshold=True
        )
    assert (
        parts.Hurdle(
            metric="m", direction=support.Orientation.MAXIMIZE, threshold=1
        ).threshold
        == 1
    )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            {"method": "none", "required": True},
            "cannot be required when method is none",
        ),
        (
            {"method": "conformal", "confidence": 1.0},
            "confidence_level must be",
        ),
        (
            {"method": "conformal", "coverage": 0.0},
            "coverage_target must be",
        ),
        (
            {"method": "conformal", "rows": 0},
            "minimum_rows must be at least 1",
        ),
        ({"method": "conformal", "required": True}, "needs confidence_level"),
        (
            {"method": "conformal", "required": True, "confidence": 0.9},
            "needs coverage_target",
        ),
        (
            {
                "method": "prediction_interval",
                "required": True,
                "confidence": 0.9,
            },
            "needs coverage_target",
        ),
        (
            {
                "method": "survival_curve",
                "required": True,
                "confidence": 0.9,
            },
            "needs coverage_target",
        ),
    ],
)
def test_uncertainty_requirements_are_complete(values, message):
    invalid(message, parts.Uncertainty, **values)


def test_complete_uncertainty_promises_are_accepted():
    assert parts.Uncertainty(
        method="abstention", required=True, confidence=0.9
    ).required
    assert parts.Uncertainty(method="conformal", rows=1).rows == 1
    made = parts.Uncertainty(
        method="conformal",
        required=True,
        confidence=0.9,
        coverage=0.8,
    )
    assert made.model_dump(mode="json")["coverage"] == 0.8


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"method": "none", "required": True}, "need a calibration method"),
        ({"method": "none", "error": 0.1}, "need a calibration method"),
        ({"method": "platt", "required": True}, "needs maximum_error"),
        ({"method": "platt", "error": -0.1}, "must not be negative"),
    ],
)
def test_calibration_requirements_are_complete(values, message):
    invalid(message, parts.Calibration, **values)


def test_a_zero_calibration_error_is_allowed():
    assert parts.Calibration(method="platt", required=True, error=0.0)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            {"method": "none", "metrics": ("m",)},
            "must be empty when method is none",
        ),
        ({"method": "none", "sensitive": ("s",)}, "must be empty"),
        ({"method": "group_report"}, "requires sensitive_columns"),
        (
            {"method": "group_report", "sensitive": ("s",)},
            "requires metrics",
        ),
        (
            {
                "method": "group_report",
                "sensitive": ("s", "s"),
                "metrics": ("m",),
            },
            "fairness.sensitive_columns must not contain duplicates",
        ),
    ],
)
def test_fairness_declarations_are_complete(values, message):
    invalid(message, parts.Fairness, **values)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"metric": "m", "evidence": ("e",)}, "exactly one bound"),
        (
            {
                "metric": "m",
                "maximum": 1.0,
                "minimum": 0.0,
                "evidence": ("e",),
            },
            "exactly one bound",
        ),
        (
            {"metric": "m", "maximum": -0.1, "evidence": ("e",)},
            "must not be negative",
        ),
        ({"metric": "m", "minimum": 0.8}, "fairness gates require evidence"),
    ],
)
def test_fairness_gates_need_one_bound_and_evidence(values, message):
    invalid(message, parts.Equity, **values)


def test_a_fairness_gate_with_a_zero_maximum_is_allowed():
    made = parts.Equity(metric="m", maximum=0.0, evidence=("e",))
    assert made.minimum is None


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"warning": 0}, "warning_threshold must be at least 1"),
        ({"warning": 10, "blocking": 0}, "between one and"),
        (
            {"warning": 10, "blocking": 11},
            "between one and",
        ),
    ],
)
def test_subgroup_thresholds_are_ordered(values, message):
    invalid(message, parts.Subgroups, **values)


def test_subgroup_thresholds_on_their_edges_are_allowed():
    assert parts.Subgroups(warning=1, blocking=1)
    assert parts.Subgroups(warning=10, blocking=10)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            {
                "availability": "global",
                "methods": ("x",),
                "stability_threshold": 0.5,
            },
            "stability_threshold requires require_stability",
        ),
        (
            {
                "availability": "global",
                "methods": ("x",),
                "require_fidelity": True,
                "fidelity_threshold": 1.5,
            },
            "explanation.fidelity_threshold must be between 0 and 1",
        ),
        (
            {
                "availability": "global",
                "methods": ("x",),
                "require_stability": True,
            },
            "require_stability needs stability_threshold",
        ),
        (
            {
                "availability": "global",
                "methods": ("x",),
                "require_fidelity": True,
            },
            "require_fidelity needs fidelity_threshold",
        ),
        (
            {"availability": "none", "methods": ("x",)},
            "need an availability",
        ),
        (
            {"availability": "none", "evidence": ("e",)},
            "need an availability",
        ),
        ({"availability": "local"}, "requires required_methods"),
    ],
)
def test_explanation_requirements_are_complete(values, message):
    invalid(message, parts.Explanations, **values)


def test_complete_explanation_requirements_are_stored_nested():
    made = parts.Explanations(
        availability="global",
        methods=("x",),
        require_stability=True,
        stability_threshold=0.8,
    )
    stored = made.model_dump(mode="json")
    assert stored["stability"] == {"required": True, "threshold": 0.8}
    assert stored["methods"] == ["x"]
    assert parts.Explanations(availability="none").methods == ()


def test_a_diagnostic_gate_needs_evidence():
    invalid(
        "diagnostic gates require evidence",
        parts.Requirement,
        name="d",
        action="block",
        evidence=(),
    )
    invalid(
        "diagnostic_gate.name",
        parts.Requirement,
        name="",
        action="warn",
        evidence=("e",),
    )
    assert (
        parts.Requirement(name="d", action="warn", evidence=("e",)).decisive
        is False
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"metrics": ()}, "requires metric_gates"),
        ({"diagnostics": ()}, "requires diagnostic_gates"),
        ({"evidence": ()}, "release policy requires evidence"),
        (
            {
                "diagnostics": (
                    parts.Requirement(name="d", action="warn", evidence=("e",)),
                )
            },
            "requires a blocking diagnostic",
        ),
        (
            {
                "metrics": (
                    parts.Hurdle(
                        metric="m",
                        direction=support.Orientation.MAXIMIZE,
                        threshold=1.0,
                    ),
                    parts.Hurdle(
                        metric="m",
                        direction=support.Orientation.MAXIMIZE,
                        threshold=2.0,
                    ),
                )
            },
            "release metric gates must be unique",
        ),
        (
            {
                "diagnostics": (
                    parts.Requirement(
                        name="d", action="block", evidence=("e",)
                    ),
                    parts.Requirement(name="d", action="warn", evidence=("e",)),
                )
            },
            "release diagnostic gates must be unique",
        ),
        (
            {
                "fairness": (
                    parts.Equity(metric="f", maximum=0.1, evidence=("e",)),
                    parts.Equity(metric="f", maximum=0.2, evidence=("e",)),
                )
            },
            "release fairness gates must be unique",
        ),
    ],
)
def test_release_gates_are_complete_and_unique(changes, message):
    invalid(message, gates, **changes)


# ---- policy ----------------------------------------------------------------


def test_a_complete_policy_is_accepted_and_stored_in_single_words():
    made = criteria()
    stored = made.model_dump(mode="json")
    assert stored["metric"] == "roc_auc"
    assert stored["small"]["warning"] == 30
    assert stored["release"]["metrics"][0]["metric"] == "roc_auc"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"metric": ""}, "policy.primary_metric must not be empty"),
        ({"metrics": ()}, "evaluation policy requires metrics"),
        (
            {
                "metrics": (
                    parts.Declaration(
                        name="roc_auc", direction=support.Orientation.MAXIMIZE
                    ),
                    parts.Declaration(
                        name="roc_auc", direction=support.Orientation.MAXIMIZE
                    ),
                )
            },
            "evaluation metrics must be unique",
        ),
        ({"metric": "f1"}, "primary_metric must be one of metrics"),
        (
            {
                "metrics": (
                    parts.Declaration(
                        name="roc_auc",
                        direction=support.Orientation.MAXIMIZE,
                        required=False,
                    ),
                )
            },
            "primary_metric must be required",
        ),
        (
            {
                "metrics": (
                    parts.Declaration(
                        name="roc_auc", direction=support.Orientation.MINIMIZE
                    ),
                )
            },
            "metric 'roc_auc' has direction maximize, not minimize",
        ),
    ],
)
def test_a_policy_refuses_inconsistent_metrics(changes, message):
    invalid(message, criteria, **changes)


def test_a_policy_refuses_a_metric_the_task_does_not_support():
    with pytest.raises(
        support.UnsupportedError, match="metric 'rmse' is unsupported"
    ):
        criteria(
            metrics=(
                parts.Declaration(
                    name="roc_auc", direction=support.Orientation.MAXIMIZE
                ),
                parts.Declaration(
                    name="rmse", direction=support.Orientation.MINIMIZE
                ),
            )
        )


def test_a_custom_metric_skips_the_support_table():
    made = criteria(
        metrics=(
            parts.Declaration(
                name="roc_auc", direction=support.Orientation.MAXIMIZE
            ),
            parts.Declaration(
                name="profit",
                direction=support.Orientation.MAXIMIZE,
                custom=True,
                evidence=("p",),
            ),
        )
    )
    assert made.metrics[1].custom


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"uncertainty": parts.Uncertainty(method="prediction_interval")},
            "uncertainty method 'prediction_interval' is unsupported for task 'binary_classification'",
        ),
        (
            {"calibration": parts.Calibration(method="dirichlet")},
            "calibration method 'dirichlet'",
        ),
        (
            {
                "fairness": parts.Fairness(
                    method="intersectional_report",
                    sensitive=("s",),
                    metrics=("demographic_parity_difference",),
                )
            },
            "fairness method 'intersectional_report'",
        ),
        (
            {
                "fairness": parts.Fairness(
                    method="group_report",
                    sensitive=("s",),
                    metrics=("odd",),
                )
            },
            "fairness metric 'odd' is unsupported",
        ),
        (
            {
                "release": gates(
                    metrics=(
                        parts.Hurdle(
                            metric="rmse",
                            direction=support.Orientation.MINIMIZE,
                            threshold=1.0,
                        ),
                    )
                )
            },
            "metric 'rmse' is unsupported",
        ),
        (
            {
                "release": gates(
                    diagnostics=(
                        parts.Requirement(
                            name="residuals",
                            action="block",
                            evidence=("e",),
                        ),
                    )
                )
            },
            "diagnostic 'residuals' is unsupported",
        ),
    ],
)
def test_a_policy_refuses_choices_outside_the_task(changes, message):
    with pytest.raises(support.UnsupportedError, match=message):
        criteria(**changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {
                "release": gates(
                    metrics=(
                        parts.Hurdle(
                            metric="f1",
                            direction=support.Orientation.MAXIMIZE,
                            threshold=0.5,
                        ),
                    )
                )
            },
            "release metric gate 'f1' is undeclared",
        ),
        (
            {
                "release": gates(
                    metrics=(
                        parts.Hurdle(
                            metric="roc_auc",
                            direction=support.Orientation.MINIMIZE,
                            threshold=0.5,
                        ),
                    )
                )
            },
            "release metric gate 'roc_auc' has a mismatched direction",
        ),
        (
            {
                "release": gates(
                    fairness=(
                        parts.Equity(metric="x", maximum=0.1, evidence=("e",)),
                    )
                )
            },
            "release fairness gate 'x' is undeclared",
        ),
        (
            {
                "fairness": parts.Fairness(
                    method="group_report",
                    sensitive=("s",),
                    metrics=("demographic_parity_difference",),
                )
            },
            "fairness declaration needs release fairness gates",
        ),
    ],
)
def test_a_policy_refuses_gates_that_disagree_with_declarations(
    changes, message
):
    invalid(message, criteria, **changes)


def test_fairness_gates_without_a_declaration_are_refused():
    with pytest.raises(pydantic.ValidationError, match="undeclared"):
        criteria(
            release=gates(
                fairness=(
                    parts.Equity(
                        metric="demographic_parity_difference",
                        maximum=0.1,
                        evidence=("e",),
                    ),
                )
            )
        )


def test_a_declared_fairness_policy_with_gates_is_accepted():
    made = criteria(
        fairness=parts.Fairness(
            method="group_report",
            sensitive=("s",),
            metrics=("demographic_parity_difference",),
        ),
        release=gates(
            fairness=(
                parts.Equity(
                    metric="demographic_parity_difference",
                    maximum=0.1,
                    evidence=("e",),
                ),
            )
        ),
    )
    assert made.fairness.method is support.Technique.GROUP_REPORT


def test_canonize_is_the_version_one_layout():
    text = criteria().canonize()
    layout = json.loads(text)
    assert layout["primary_metric"] == "roc_auc"
    assert layout["small_groups"]["warning_threshold"] == 30
    assert " " not in text
    assert text == criteria().canonize()


# ---- registry --------------------------------------------------------------


def regression():
    return criteria(
        task=tasks.Task.REGRESSION,
        metric="rmse",
        metrics=(
            parts.Declaration(
                name="rmse", direction=support.Orientation.MINIMIZE
            ),
        ),
        explanations=parts.Explanations(availability="none"),
        release=gates(
            metrics=(
                parts.Hurdle(
                    metric="rmse",
                    direction=support.Orientation.MINIMIZE,
                    threshold=1.0,
                ),
            )
        ),
    )


def test_a_registry_holds_one_policy_per_task():
    registry = policy.Registry([regression(), criteria()])
    assert registry.require(tasks.Task.BINARY).metric == "roc_auc"
    assert [p.task.value for p in registry.tabulate()] == [
        "binary_classification",
        "regression",
    ]
    with pytest.raises(
        errors.ConfigError,
        match="already registered for task 'binary_classification'",
    ):
        registry.register(criteria())


def test_a_registry_fails_closed_for_undeclared_tasks():
    registry = policy.Registry()
    assert registry.tabulate() == ()
    with pytest.raises(
        support.UndeclaredError,
        match="no evaluation policy declared for task 'binary_classification'",
    ):
        registry.require(tasks.Task.BINARY)
    with pytest.raises(errors.KindError, match="task must be Task"):
        registry.require("binary")


def test_a_registry_dumps_deterministically():
    registry = policy.Registry([criteria()])
    record = registry.to_dict()
    assert record["schema_version"] == policy.Criteria.latest()
    assert len(record["policies"]) == 1
    text = registry.canonize()
    assert json.loads(text) == record
    assert text == policy.Registry([criteria()]).canonize()
