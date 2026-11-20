"""Tests for task-aware evaluation policy contracts."""

import json

import pytest

from scmle.contracts import evaluation
from scmle.contracts import tasks


def release_policy(
    metric: str = "roc_auc",
    direction: evaluation.Orientation = evaluation.Orientation.MAXIMIZE,
    fairness_gates: tuple[evaluation.Equity, ...] = (),
) -> evaluation.Gates:
    """Build a complete release policy for contract tests."""
    return evaluation.Gates(
        metrics=(
            evaluation.Hurdle(
                metric=metric, direction=direction, threshold=0.75
            ),
        ),
        diagnostics=(
            evaluation.Requirement(
                name="metric_uncertainty",
                action=evaluation.Remedy.BLOCK,
                evidence=("bootstrap-ci.json",),
            ),
            evaluation.Requirement(
                name="robustness",
                action=evaluation.Remedy.WARN,
                evidence=("robustness.json",),
            ),
        ),
        evidence=("run-manifest.json", "test-metrics.json"),
        fairness=fairness_gates,
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
        metric="roc_auc",
        metrics=(
            evaluation.Declaration(
                name="roc_auc", direction=evaluation.Orientation.MAXIMIZE
            ),
            evaluation.Declaration(
                name="log_loss",
                direction=evaluation.Orientation.MINIMIZE,
                required=False,
            ),
        ),
        uncertainty=uncertainty
        or evaluation.Uncertainty(method=evaluation.Method.NONE),
        calibration=calibration
        or evaluation.Calibration(method=evaluation.Procedure.NONE),
        fairness=fairness
        or evaluation.Fairness(method=evaluation.Technique.NONE),
        small=evaluation.Subgroups(warning=30, blocking=10),
        explanations=explanations
        or evaluation.Explanations(
            availability=evaluation.Explicability.GLOBAL,
            methods=("permutation_importance",),
            evidence=("explanation.json",),
        ),
        release=release or release_policy(),
    )


def test_valid_policy_contains_task_aware_contracts():
    policy = binary_policy(
        uncertainty=evaluation.Uncertainty(
            method=evaluation.Method.CONFORMAL,
            required=True,
            confidence=0.95,
            coverage=0.90,
            rows=100,
        ),
        calibration=evaluation.Calibration(
            method=evaluation.Procedure.RELIABILITY_CURVE,
            required=True,
            error=0.05,
        ),
        fairness=evaluation.Fairness(
            method=evaluation.Technique.GROUP_REPORT,
            sensitive=("sex",),
            metrics=("demographic_parity_difference",),
        ),
        release=release_policy(
            fairness_gates=(
                evaluation.Equity(
                    metric="demographic_parity_difference",
                    maximum=0.10,
                    evidence=("fairness.json",),
                ),
            )
        ),
    )

    assert policy.task is tasks.Task.BINARY
    assert (
        policy.model_dump(mode="json")["release"]["metrics"][0]["threshold"]
        == 0.75
    )
    assert json.loads(policy.canonize())["task"] == "binary_classification"


def test_explanation_quality_thresholds_are_explicit_and_serialized():
    policy = evaluation.Explanations(
        availability=evaluation.Explicability.GLOBAL,
        methods=("permutation_importance",),
        evidence=("explanation.json",),
        require_stability=True,
        require_fidelity=True,
        stability_threshold=0.8,
        fidelity_threshold=0.9,
    )

    assert policy.model_dump(mode="json")["stability"] == {
        "required": True,
        "threshold": 0.8,
    }
    assert policy.model_dump(mode="json")["fidelity"] == {
        "required": True,
        "threshold": 0.9,
    }


@pytest.mark.parametrize(
    "field",
    ["stability_threshold", "fidelity_threshold"],
)
def test_required_explanation_quality_needs_a_threshold(field):
    values = {
        "availability": evaluation.Explicability.GLOBAL,
        "methods": ("permutation_importance",),
        "evidence": ("explanation.json",),
        "require_stability": field == "stability_threshold",
        "require_fidelity": field == "fidelity_threshold",
    }

    with pytest.raises(ValueError, match="threshold"):
        evaluation.Explanations(**values)


def test_unsupported_metric_and_direction_fail_closed():
    with pytest.raises(evaluation.UnsupportedError, match="rmse"):
        binary_policy(
            release=release_policy(
                metric="rmse",
                direction=evaluation.Orientation.MINIMIZE,
            )
        )

    with pytest.raises(ValueError, match="direction"):
        evaluation.Criteria(
            task=tasks.Task.BINARY,
            metric="roc_auc",
            metrics=(
                evaluation.Declaration(
                    name="roc_auc", direction=evaluation.Orientation.MINIMIZE
                ),
            ),
            uncertainty=evaluation.Uncertainty(method=evaluation.Method.NONE),
            calibration=evaluation.Calibration(
                method=evaluation.Procedure.NONE
            ),
            fairness=evaluation.Fairness(method=evaluation.Technique.NONE),
            small=evaluation.Subgroups(warning=30),
            explanations=evaluation.Explanations(
                availability=evaluation.Explicability.NONE
            ),
            release=release_policy(direction=evaluation.Orientation.MINIMIZE),
        )


def test_unsupported_uncertainty_calibration_fairness_and_explanation_fail():
    with pytest.raises(evaluation.UnsupportedError, match="uncertainty"):
        evaluation.Criteria(
            task=tasks.Task.REGRESSION,
            metric="rmse",
            metrics=(
                evaluation.Declaration(
                    name="rmse", direction=evaluation.Orientation.MINIMIZE
                ),
            ),
            uncertainty=evaluation.Uncertainty(
                method=evaluation.Method.CALIBRATED_PROBABILITY
            ),
            calibration=evaluation.Calibration(
                method=evaluation.Procedure.NONE
            ),
            fairness=evaluation.Fairness(method=evaluation.Technique.NONE),
            small=evaluation.Subgroups(warning=30),
            explanations=evaluation.Explanations(
                availability=evaluation.Explicability.GLOBAL,
                methods=("permutation_importance",),
            ),
            release=evaluation.Gates(
                metrics=(
                    evaluation.Hurdle(
                        metric="rmse",
                        direction=evaluation.Orientation.MINIMIZE,
                        threshold=1.0,
                    ),
                ),
                diagnostics=(
                    evaluation.Requirement(
                        name="residuals",
                        action=evaluation.Remedy.BLOCK,
                        evidence=("residuals.json",),
                    ),
                ),
                evidence=("run.json",),
            ),
        )

    with pytest.raises(evaluation.UnsupportedError, match="calibration"):
        binary_policy(
            calibration=evaluation.Calibration(
                method=evaluation.Procedure.DIRICHLET
            )
        )

    with pytest.raises(evaluation.UnsupportedError, match="fairness"):
        binary_policy(
            fairness=evaluation.Fairness(
                method=evaluation.Technique.INTERSECTIONAL_REPORT,
                sensitive=("sex", "age_band"),
                metrics=("demographic_parity_difference",),
            ),
            release=release_policy(
                fairness_gates=(
                    evaluation.Equity(
                        metric="demographic_parity_difference",
                        maximum=0.1,
                        evidence=("fairness.json",),
                    ),
                )
            ),
        )

    with pytest.raises(evaluation.UnsupportedError, match="explanation"):
        evaluation.Criteria(
            task=tasks.Task.SURVIVAL,
            metric="c_index",
            metrics=(
                evaluation.Declaration(
                    name="c_index", direction=evaluation.Orientation.MAXIMIZE
                ),
            ),
            uncertainty=evaluation.Uncertainty(method=evaluation.Method.NONE),
            calibration=evaluation.Calibration(
                method=evaluation.Procedure.NONE
            ),
            fairness=evaluation.Fairness(method=evaluation.Technique.NONE),
            small=evaluation.Subgroups(warning=30),
            explanations=evaluation.Explanations(
                availability=evaluation.Explicability.LOCAL,
                methods=("occlusion",),
            ),
            release=evaluation.Gates(
                metrics=(
                    evaluation.Hurdle(
                        metric="c_index",
                        direction=evaluation.Orientation.MAXIMIZE,
                        threshold=0.6,
                    ),
                ),
                diagnostics=(
                    evaluation.Requirement(
                        name="metric_uncertainty",
                        action=evaluation.Remedy.BLOCK,
                        evidence=("ci.json",),
                    ),
                ),
                evidence=("run.json",),
            ),
        )


def test_release_policy_requires_explicit_thresholds_evidence_and_blocker():
    with pytest.raises(ValueError, match="metric_gates"):
        evaluation.Gates(
            metrics=(),
            diagnostics=(
                evaluation.Requirement(
                    name="robustness",
                    action=evaluation.Remedy.BLOCK,
                    evidence=("x",),
                ),
            ),
            evidence=("run.json",),
        )
    with pytest.raises(ValueError, match="evidence"):
        evaluation.Gates(
            metrics=(
                evaluation.Hurdle(
                    metric="roc_auc",
                    direction=evaluation.Orientation.MAXIMIZE,
                    threshold=0.7,
                ),
            ),
            diagnostics=(
                evaluation.Requirement(
                    name="robustness",
                    action=evaluation.Remedy.BLOCK,
                    evidence=("x",),
                ),
            ),
            evidence=(),
        )
    with pytest.raises(ValueError, match="blocking diagnostic"):
        evaluation.Gates(
            metrics=(
                evaluation.Hurdle(
                    metric="roc_auc",
                    direction=evaluation.Orientation.MAXIMIZE,
                    threshold=0.7,
                ),
            ),
            diagnostics=(
                evaluation.Requirement(
                    name="robustness",
                    action=evaluation.Remedy.WARN,
                    evidence=("x",),
                ),
            ),
            evidence=("run.json",),
        )


def test_release_gates_must_be_declared_and_task_supported():
    with pytest.raises(ValueError, match="undeclared"):
        binary_policy(release=release_policy(metric="accuracy"))

    with pytest.raises(evaluation.UnsupportedError, match="diagnostic"):
        binary_policy(
            release=evaluation.Gates(
                metrics=(
                    evaluation.Hurdle(
                        metric="roc_auc",
                        direction=evaluation.Orientation.MAXIMIZE,
                        threshold=0.7,
                    ),
                ),
                diagnostics=(
                    evaluation.Requirement(
                        name="survival_curve",
                        action=evaluation.Remedy.BLOCK,
                        evidence=("x",),
                    ),
                ),
                evidence=("run.json",),
            )
        )


def test_registry_fails_closed_for_undeclared_tasks_and_sorts_records():
    binary = binary_policy()
    registry = evaluation.Registry([binary])
    assert registry.require(tasks.Task.BINARY) is binary
    with pytest.raises(evaluation.UndeclaredError, match="survival"):
        registry.require(tasks.Task.SURVIVAL)
    assert registry.tabulate() == (binary,)
    assert json.loads(registry.canonize())["policies"][0]["task"] == (
        "binary_classification"
    )

    with pytest.raises(Exception, match="already registered"):
        registry.register(binary)


def test_custom_metric_requires_evidence_and_json_is_deterministic():
    custom = evaluation.Declaration(
        name="business_cost",
        direction=evaluation.Orientation.MINIMIZE,
        custom=True,
        evidence=("metric-reference.py",),
    )
    policy = evaluation.Criteria(
        task=tasks.Task.BINARY,
        metric="business_cost",
        metrics=(custom,),
        uncertainty=evaluation.Uncertainty(method=evaluation.Method.NONE),
        calibration=evaluation.Calibration(method=evaluation.Procedure.NONE),
        fairness=evaluation.Fairness(method=evaluation.Technique.NONE),
        small=evaluation.Subgroups(warning=30),
        explanations=evaluation.Explanations(
            availability=evaluation.Explicability.NONE
        ),
        release=evaluation.Gates(
            metrics=(
                evaluation.Hurdle(
                    metric="business_cost",
                    direction=evaluation.Orientation.MINIMIZE,
                    threshold=0.2,
                ),
            ),
            diagnostics=(
                evaluation.Requirement(
                    name="metric_uncertainty",
                    action=evaluation.Remedy.BLOCK,
                    evidence=("ci.json",),
                ),
            ),
            evidence=("run.json",),
        ),
    )
    assert policy.canonize() == policy.canonize()
    assert "business_cost" in policy.canonize()
    with pytest.raises(ValueError, match="custom metrics require evidence"):
        evaluation.Declaration(
            name="business_cost",
            direction=evaluation.Orientation.MINIMIZE,
            custom=True,
        )


def test_small_group_thresholds_are_ordered():
    with pytest.raises(ValueError, match="blocking_threshold"):
        evaluation.Subgroups(warning=30, blocking=31)
    with pytest.raises(ValueError, match="warning_threshold"):
        evaluation.Subgroups(warning=0)
