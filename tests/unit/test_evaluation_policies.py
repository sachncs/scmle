"""Tests for project-to-release policy construction."""

import pytest

from scmle.contracts import evaluation
from scmle.contracts import tasks
from scmle.evaluation import policies


def test_empty_criteria_do_not_invent_release_thresholds():
    assert policies.derive(tasks.Task.BINARY, "roc_auc", True, {}) is None


def test_criteria_become_directional_metric_and_evidence_gates():
    policy = policies.derive(
        tasks.Task.BINARY,
        "roc_auc",
        True,
        {"roc_auc": 0.8, "log_loss": 0.4},
    )

    assert policy is not None
    assert policy.metric == "roc_auc"
    assert [gate.metric for gate in policy.release.metrics] == [
        "roc_auc",
        "log_loss",
    ]
    assert policy.release.metrics[1].direction is (
        evaluation.Orientation.MINIMIZE
    )
    assert "diagnostics/metric_uncertainty.json" in (policy.release.evidence)


def test_unknown_metric_direction_fails_closed():
    with pytest.raises(
        evaluation.UnsupportedError,
        match="no declared release direction",
    ):
        policies.derive(tasks.Task.BINARY, "roc_auc", True, {"unknown": 0.8})
