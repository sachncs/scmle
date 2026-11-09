"""Tests for project-to-release policy construction."""

import pytest

from scmle.contracts import evaluation
from scmle.contracts import tasks
from scmle.evaluation import policies


def test_empty_criteria_do_not_invent_release_thresholds():
    assert (
        policies.release_policy_from_criteria(
            tasks.Task.BINARY, "roc_auc", True, {}
        )
        is None
    )


def test_criteria_become_directional_metric_and_evidence_gates():
    policy = policies.release_policy_from_criteria(
        tasks.Task.BINARY,
        "roc_auc",
        True,
        {"roc_auc": 0.8, "log_loss": 0.4},
    )

    assert policy is not None
    assert policy.primary_metric == "roc_auc"
    assert [gate.metric for gate in policy.release.metric_gates] == [
        "roc_auc",
        "log_loss",
    ]
    assert policy.release.metric_gates[1].direction is (
        evaluation.MetricDirection.MINIMIZE
    )
    assert "diagnostics/metric_uncertainty.json" in (
        policy.release.required_evidence
    )


def test_unknown_metric_direction_fails_closed():
    with pytest.raises(
        evaluation.UnsupportedEvaluationPolicyError,
        match="no declared release direction",
    ):
        policies.release_policy_from_criteria(
            tasks.Task.BINARY, "roc_auc", True, {"unknown": 0.8}
        )
