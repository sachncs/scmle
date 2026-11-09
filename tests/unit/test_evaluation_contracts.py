"""Tests for task-aware evaluation policy contracts."""

import json

import pytest

from scmle.contracts import evaluation
from scmle.contracts import tasks


def release_policy(
    metric: str = "roc_auc",
    direction: evaluation.MetricDirection = evaluation.MetricDirection.MAXIMIZE,
    fairness_gates: tuple[evaluation.FairnessGate, ...] = (),
) -> evaluation.Gates:
    """Build a complete release policy for contract tests."""
    return evaluation.Gates(
        metric_gates=(evaluation.MetricGate(metric, direction, 0.75),),
        diagnostic_gates=(
            evaluation.DiagnosticGate(
                "metric_uncertainty",
                evaluation.DiagnosticAction.BLOCK,
                ("bootstrap-ci.json",),
            ),
            evaluation.DiagnosticGate(
                "robustness",
                evaluation.DiagnosticAction.WARN,
                ("robustness.json",),
            ),
        ),
        required_evidence=("run-manifest.json", "test-metrics.json"),
        fairness_gates=fairness_gates,
    )


def binary_policy(
    *,
    uncertainty: evaluation.Uncertainty | None = None,
    calibration: evaluation.Calibration | None = None,
    fairness: evaluation.Fairness | None = None,
    explanations: evaluation.Explanations | None = None,
    release: evaluation.Gates | None = None,
) -> evaluation.Criteria:
    """Build a complete valid binary policy for tests."""
    return evaluation.Criteria(
        task=tasks.Task.BINARY,
        primary_metric="roc_auc",
        metrics=(
            evaluation.MetricDeclaration(
                "roc_auc", evaluation.MetricDirection.MAXIMIZE
            ),
            evaluation.MetricDeclaration(
                "log_loss", evaluation.MetricDirection.MINIMIZE, required=False
            ),
        ),
        uncertainty=uncertainty
        or evaluation.Uncertainty(evaluation.UncertaintyMethod.NONE),
        calibration=calibration
        or evaluation.Calibration(evaluation.CalibrationMethod.NONE),
        fairness=fairness
        or evaluation.Fairness(evaluation.FairnessMethod.NONE),
        small_groups=evaluation.SmallGroups(30, 10),
        explanations=explanations
        or evaluation.Explanations(
            evaluation.ExplanationAvailability.GLOBAL,
            required_methods=("permutation_importance",),
            required_evidence=("explanation.json",),
        ),
        release=release or release_policy(),
    )


def test_valid_policy_contains_task_aware_contracts():
    policy = binary_policy(
        uncertainty=evaluation.Uncertainty(
            evaluation.UncertaintyMethod.CONFORMAL,
            required=True,
            confidence_level=0.95,
            coverage_target=0.90,
            minimum_rows=100,
        ),
        calibration=evaluation.Calibration(
            evaluation.CalibrationMethod.RELIABILITY_CURVE,
            required=True,
            maximum_error=0.05,
        ),
        fairness=evaluation.Fairness(
            evaluation.FairnessMethod.GROUP_REPORT,
            sensitive_columns=("sex",),
            metrics=("demographic_parity_difference",),
        ),
        release=release_policy(
            fairness_gates=(
                evaluation.FairnessGate(
                    "demographic_parity_difference",
                    maximum=0.10,
                    required_evidence=("fairness.json",),
                ),
            )
        ),
    )

    assert policy.task is tasks.Task.BINARY
    assert policy.to_dict()["release"]["metric_gates"][0]["threshold"] == 0.75
    assert (
        json.loads(policy.canonical_json())["task"] == "binary_classification"
    )


def test_explanation_quality_thresholds_are_explicit_and_serialized():
    policy = evaluation.Explanations(
        evaluation.ExplanationAvailability.GLOBAL,
        required_methods=("permutation_importance",),
        required_evidence=("explanation.json",),
        require_stability=True,
        require_fidelity=True,
        stability_threshold=0.8,
        fidelity_threshold=0.9,
    )

    assert policy.to_dict()["stability_threshold"] == 0.8
    assert policy.to_dict()["fidelity_threshold"] == 0.9


@pytest.mark.parametrize(
    "field",
    ("stability_threshold", "fidelity_threshold"),
)
def test_required_explanation_quality_needs_a_threshold(field):
    values = {
        "availability": evaluation.ExplanationAvailability.GLOBAL,
        "required_methods": ("permutation_importance",),
        "required_evidence": ("explanation.json",),
        "require_stability": field == "stability_threshold",
        "require_fidelity": field == "fidelity_threshold",
    }

    with pytest.raises(ValueError, match="threshold"):
        evaluation.Explanations(**values)


def test_unsupported_metric_and_direction_fail_closed():
    with pytest.raises(
        evaluation.UnsupportedEvaluationPolicyError, match="rmse"
    ):
        binary_policy(
            release=release_policy(
                metric="rmse",
                direction=evaluation.MetricDirection.MINIMIZE,
            )
        )

    with pytest.raises(ValueError, match="direction"):
        evaluation.Criteria(
            task=tasks.Task.BINARY,
            primary_metric="roc_auc",
            metrics=(
                evaluation.MetricDeclaration(
                    "roc_auc", evaluation.MetricDirection.MINIMIZE
                ),
            ),
            uncertainty=evaluation.Uncertainty(
                evaluation.UncertaintyMethod.NONE
            ),
            calibration=evaluation.Calibration(
                evaluation.CalibrationMethod.NONE
            ),
            fairness=evaluation.Fairness(evaluation.FairnessMethod.NONE),
            small_groups=evaluation.SmallGroups(30),
            explanations=evaluation.Explanations(
                evaluation.ExplanationAvailability.NONE
            ),
            release=release_policy(
                direction=evaluation.MetricDirection.MINIMIZE
            ),
        )


def test_unsupported_uncertainty_calibration_fairness_and_explanation_fail():
    with pytest.raises(
        evaluation.UnsupportedEvaluationPolicyError, match="uncertainty"
    ):
        evaluation.Criteria(
            task=tasks.Task.REGRESSION,
            primary_metric="rmse",
            metrics=(
                evaluation.MetricDeclaration(
                    "rmse", evaluation.MetricDirection.MINIMIZE
                ),
            ),
            uncertainty=evaluation.Uncertainty(
                evaluation.UncertaintyMethod.CALIBRATED_PROBABILITY
            ),
            calibration=evaluation.Calibration(
                evaluation.CalibrationMethod.NONE
            ),
            fairness=evaluation.Fairness(evaluation.FairnessMethod.NONE),
            small_groups=evaluation.SmallGroups(30),
            explanations=evaluation.Explanations(
                evaluation.ExplanationAvailability.GLOBAL,
                required_methods=("permutation_importance",),
            ),
            release=evaluation.Gates(
                metric_gates=(
                    evaluation.MetricGate(
                        "rmse", evaluation.MetricDirection.MINIMIZE, 1.0
                    ),
                ),
                diagnostic_gates=(
                    evaluation.DiagnosticGate(
                        "residuals",
                        evaluation.DiagnosticAction.BLOCK,
                        ("residuals.json",),
                    ),
                ),
                required_evidence=("run.json",),
            ),
        )

    with pytest.raises(
        evaluation.UnsupportedEvaluationPolicyError, match="calibration"
    ):
        binary_policy(
            calibration=evaluation.Calibration(
                evaluation.CalibrationMethod.DIRICHLET
            )
        )

    with pytest.raises(
        evaluation.UnsupportedEvaluationPolicyError, match="fairness"
    ):
        binary_policy(
            fairness=evaluation.Fairness(
                evaluation.FairnessMethod.INTERSECTIONAL_REPORT,
                sensitive_columns=("sex", "age_band"),
                metrics=("demographic_parity_difference",),
            ),
            release=release_policy(
                fairness_gates=(
                    evaluation.FairnessGate(
                        "demographic_parity_difference",
                        maximum=0.1,
                        required_evidence=("fairness.json",),
                    ),
                )
            ),
        )

    with pytest.raises(
        evaluation.UnsupportedEvaluationPolicyError, match="explanation"
    ):
        evaluation.Criteria(
            task=tasks.Task.SURVIVAL,
            primary_metric="c_index",
            metrics=(
                evaluation.MetricDeclaration(
                    "c_index", evaluation.MetricDirection.MAXIMIZE
                ),
            ),
            uncertainty=evaluation.Uncertainty(
                evaluation.UncertaintyMethod.NONE
            ),
            calibration=evaluation.Calibration(
                evaluation.CalibrationMethod.NONE
            ),
            fairness=evaluation.Fairness(evaluation.FairnessMethod.NONE),
            small_groups=evaluation.SmallGroups(30),
            explanations=evaluation.Explanations(
                evaluation.ExplanationAvailability.LOCAL,
                required_methods=("occlusion",),
            ),
            release=evaluation.Gates(
                metric_gates=(
                    evaluation.MetricGate(
                        "c_index", evaluation.MetricDirection.MAXIMIZE, 0.6
                    ),
                ),
                diagnostic_gates=(
                    evaluation.DiagnosticGate(
                        "metric_uncertainty",
                        evaluation.DiagnosticAction.BLOCK,
                        ("ci.json",),
                    ),
                ),
                required_evidence=("run.json",),
            ),
        )


def test_release_policy_requires_explicit_thresholds_evidence_and_blocker():
    with pytest.raises(ValueError, match="metric_gates"):
        evaluation.Gates(
            metric_gates=(),
            diagnostic_gates=(
                evaluation.DiagnosticGate(
                    "robustness", evaluation.DiagnosticAction.BLOCK, ("x",)
                ),
            ),
            required_evidence=("run.json",),
        )
    with pytest.raises(ValueError, match="evidence"):
        evaluation.Gates(
            metric_gates=(
                evaluation.MetricGate(
                    "roc_auc", evaluation.MetricDirection.MAXIMIZE, 0.7
                ),
            ),
            diagnostic_gates=(
                evaluation.DiagnosticGate(
                    "robustness", evaluation.DiagnosticAction.BLOCK, ("x",)
                ),
            ),
            required_evidence=(),
        )
    with pytest.raises(ValueError, match="blocking diagnostic"):
        evaluation.Gates(
            metric_gates=(
                evaluation.MetricGate(
                    "roc_auc", evaluation.MetricDirection.MAXIMIZE, 0.7
                ),
            ),
            diagnostic_gates=(
                evaluation.DiagnosticGate(
                    "robustness", evaluation.DiagnosticAction.WARN, ("x",)
                ),
            ),
            required_evidence=("run.json",),
        )


def test_release_gates_must_be_declared_and_task_supported():
    with pytest.raises(ValueError, match="undeclared"):
        binary_policy(release=release_policy(metric="accuracy"))

    with pytest.raises(
        evaluation.UnsupportedEvaluationPolicyError, match="diagnostic"
    ):
        binary_policy(
            release=evaluation.Gates(
                metric_gates=(
                    evaluation.MetricGate(
                        "roc_auc", evaluation.MetricDirection.MAXIMIZE, 0.7
                    ),
                ),
                diagnostic_gates=(
                    evaluation.DiagnosticGate(
                        "survival_curve",
                        evaluation.DiagnosticAction.BLOCK,
                        ("x",),
                    ),
                ),
                required_evidence=("run.json",),
            )
        )


def test_registry_fails_closed_for_undeclared_tasks_and_sorts_records():
    binary = binary_policy()
    registry = evaluation.Registry([binary])
    assert registry.require(tasks.Task.BINARY) is binary
    with pytest.raises(
        evaluation.UndeclaredEvaluationPolicyError, match="survival"
    ):
        registry.require(tasks.Task.SURVIVAL)
    assert registry.list_policies() == (binary,)
    assert json.loads(registry.canonical_json())["policies"][0]["task"] == (
        "binary_classification"
    )

    with pytest.raises(Exception, match="already registered"):
        registry.register(binary)


def test_custom_metric_requires_evidence_and_json_is_deterministic():
    custom = evaluation.MetricDeclaration(
        "business_cost",
        evaluation.MetricDirection.MINIMIZE,
        custom=True,
        evidence=("metric-reference.py",),
    )
    policy = evaluation.Criteria(
        task=tasks.Task.BINARY,
        primary_metric="business_cost",
        metrics=(custom,),
        uncertainty=evaluation.Uncertainty(evaluation.UncertaintyMethod.NONE),
        calibration=evaluation.Calibration(evaluation.CalibrationMethod.NONE),
        fairness=evaluation.Fairness(evaluation.FairnessMethod.NONE),
        small_groups=evaluation.SmallGroups(30),
        explanations=evaluation.Explanations(
            evaluation.ExplanationAvailability.NONE
        ),
        release=evaluation.Gates(
            metric_gates=(
                evaluation.MetricGate(
                    "business_cost", evaluation.MetricDirection.MINIMIZE, 0.2
                ),
            ),
            diagnostic_gates=(
                evaluation.DiagnosticGate(
                    "metric_uncertainty",
                    evaluation.DiagnosticAction.BLOCK,
                    ("ci.json",),
                ),
            ),
            required_evidence=("run.json",),
        ),
    )
    assert policy.canonical_json() == policy.canonical_json()
    assert "business_cost" in policy.canonical_json()
    with pytest.raises(ValueError, match="custom metrics require evidence"):
        evaluation.MetricDeclaration(
            "business_cost", evaluation.MetricDirection.MINIMIZE, custom=True
        )


def test_small_group_thresholds_are_ordered():
    with pytest.raises(ValueError, match="blocking_threshold"):
        evaluation.SmallGroups(30, 31)
    with pytest.raises(ValueError, match="warning_threshold"):
        evaluation.SmallGroups(0)
