"""Tests for task-aware evaluation policy contracts."""

import json

import pytest

from scml.contracts import evaluation
from scml.contracts import tasks


def release_policy(
    metric: str = "roc_auc",
    direction: evaluation.MetricDirection = evaluation.MetricDirection.MAXIMIZE,
    fairness_gates: tuple[evaluation.FairnessGate, ...] = (),
) -> evaluation.ReleasePolicy:
    """Build a complete release policy for contract tests."""
    return evaluation.ReleasePolicy(
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
    uncertainty: evaluation.UncertaintyPolicy | None = None,
    calibration: evaluation.CalibrationPolicy | None = None,
    fairness: evaluation.FairnessPolicy | None = None,
    explanations: evaluation.ExplanationPolicy | None = None,
    release: evaluation.ReleasePolicy | None = None,
) -> evaluation.EvaluationPolicy:
    """Build a complete valid binary policy for tests."""
    return evaluation.EvaluationPolicy(
        task=tasks.TaskType.BINARY,
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
        or evaluation.UncertaintyPolicy(evaluation.UncertaintyMethod.NONE),
        calibration=calibration
        or evaluation.CalibrationPolicy(evaluation.CalibrationMethod.NONE),
        fairness=fairness
        or evaluation.FairnessPolicy(evaluation.FairnessMethod.NONE),
        small_groups=evaluation.SmallGroupPolicy(30, 10),
        explanations=explanations
        or evaluation.ExplanationPolicy(
            evaluation.ExplanationAvailability.GLOBAL,
            required_methods=("permutation_importance",),
            required_evidence=("explanation.json",),
        ),
        release=release or release_policy(),
    )


def test_valid_policy_contains_task_aware_contracts():
    policy = binary_policy(
        uncertainty=evaluation.UncertaintyPolicy(
            evaluation.UncertaintyMethod.CONFORMAL,
            required=True,
            confidence_level=0.95,
            coverage_target=0.90,
            minimum_rows=100,
        ),
        calibration=evaluation.CalibrationPolicy(
            evaluation.CalibrationMethod.RELIABILITY_CURVE,
            required=True,
            maximum_error=0.05,
        ),
        fairness=evaluation.FairnessPolicy(
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

    assert policy.task is tasks.TaskType.BINARY
    assert policy.to_dict()["release"]["metric_gates"][0]["threshold"] == 0.75
    assert (
        json.loads(policy.canonical_json())["task"] == "binary_classification"
    )


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
        evaluation.EvaluationPolicy(
            task=tasks.TaskType.BINARY,
            primary_metric="roc_auc",
            metrics=(
                evaluation.MetricDeclaration(
                    "roc_auc", evaluation.MetricDirection.MINIMIZE
                ),
            ),
            uncertainty=evaluation.UncertaintyPolicy(
                evaluation.UncertaintyMethod.NONE
            ),
            calibration=evaluation.CalibrationPolicy(
                evaluation.CalibrationMethod.NONE
            ),
            fairness=evaluation.FairnessPolicy(evaluation.FairnessMethod.NONE),
            small_groups=evaluation.SmallGroupPolicy(30),
            explanations=evaluation.ExplanationPolicy(
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
        evaluation.EvaluationPolicy(
            task=tasks.TaskType.REGRESSION,
            primary_metric="rmse",
            metrics=(
                evaluation.MetricDeclaration(
                    "rmse", evaluation.MetricDirection.MINIMIZE
                ),
            ),
            uncertainty=evaluation.UncertaintyPolicy(
                evaluation.UncertaintyMethod.CALIBRATED_PROBABILITY
            ),
            calibration=evaluation.CalibrationPolicy(
                evaluation.CalibrationMethod.NONE
            ),
            fairness=evaluation.FairnessPolicy(evaluation.FairnessMethod.NONE),
            small_groups=evaluation.SmallGroupPolicy(30),
            explanations=evaluation.ExplanationPolicy(
                evaluation.ExplanationAvailability.GLOBAL,
                required_methods=("permutation_importance",),
            ),
            release=evaluation.ReleasePolicy(
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
            calibration=evaluation.CalibrationPolicy(
                evaluation.CalibrationMethod.DIRICHLET
            )
        )

    with pytest.raises(
        evaluation.UnsupportedEvaluationPolicyError, match="fairness"
    ):
        binary_policy(
            fairness=evaluation.FairnessPolicy(
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
        evaluation.EvaluationPolicy(
            task=tasks.TaskType.SURVIVAL,
            primary_metric="c_index",
            metrics=(
                evaluation.MetricDeclaration(
                    "c_index", evaluation.MetricDirection.MAXIMIZE
                ),
            ),
            uncertainty=evaluation.UncertaintyPolicy(
                evaluation.UncertaintyMethod.NONE
            ),
            calibration=evaluation.CalibrationPolicy(
                evaluation.CalibrationMethod.NONE
            ),
            fairness=evaluation.FairnessPolicy(evaluation.FairnessMethod.NONE),
            small_groups=evaluation.SmallGroupPolicy(30),
            explanations=evaluation.ExplanationPolicy(
                evaluation.ExplanationAvailability.LOCAL,
                required_methods=("occlusion",),
            ),
            release=evaluation.ReleasePolicy(
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
        evaluation.ReleasePolicy(
            metric_gates=(),
            diagnostic_gates=(
                evaluation.DiagnosticGate(
                    "robustness", evaluation.DiagnosticAction.BLOCK, ("x",)
                ),
            ),
            required_evidence=("run.json",),
        )
    with pytest.raises(ValueError, match="evidence"):
        evaluation.ReleasePolicy(
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
        evaluation.ReleasePolicy(
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
            release=evaluation.ReleasePolicy(
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
    registry = evaluation.EvaluationPolicyRegistry([binary])
    assert registry.require(tasks.TaskType.BINARY) is binary
    with pytest.raises(
        evaluation.UndeclaredEvaluationPolicyError, match="survival"
    ):
        registry.require(tasks.TaskType.SURVIVAL)
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
    policy = evaluation.EvaluationPolicy(
        task=tasks.TaskType.BINARY,
        primary_metric="business_cost",
        metrics=(custom,),
        uncertainty=evaluation.UncertaintyPolicy(
            evaluation.UncertaintyMethod.NONE
        ),
        calibration=evaluation.CalibrationPolicy(
            evaluation.CalibrationMethod.NONE
        ),
        fairness=evaluation.FairnessPolicy(evaluation.FairnessMethod.NONE),
        small_groups=evaluation.SmallGroupPolicy(30),
        explanations=evaluation.ExplanationPolicy(
            evaluation.ExplanationAvailability.NONE
        ),
        release=evaluation.ReleasePolicy(
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
        evaluation.SmallGroupPolicy(30, 31)
    with pytest.raises(ValueError, match="warning_threshold"):
        evaluation.SmallGroupPolicy(0)
