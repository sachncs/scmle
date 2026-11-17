"""Guarantees of docs/OPERATIONS.md that no other test enforced."""

import pytest

from scmle.operations import classify
from scmle.operations import continuous
from scmle.operations import signals
from tests import support_ops
from tests.unit import test_continuous_evaluation as evaluation
from tests.unit import test_operations_telemetry as telemetry


def test_only_feature_rows_are_sampled():
    """Enforces docs/OPERATIONS.md:29."""
    watch = signals.SignalMonitor(
        signals.Window(sample_rate=0.1, seed=3), profile=support_ops.profile()
    )
    support_ops.healthy_traffic(watch, count=200)
    watch.observe_outcomes(support_ops.START, [0.0] * 100)
    kept = watch.sense(support_ops.START, support_ops.frame(1000, seed=1))
    rows = watch.evaluate(support_ops.START + 20)
    assert 50 < kept < 150
    assert support_ops.named(rows, "server_error_rate").samples == 200
    assert support_ops.named(rows, "windowed_loss").samples == 100


@pytest.mark.parametrize(
    "name",
    [
        "invalid_request_rate",
        "feature_missing_increase",
        "unseen_category_rate",
        "feature_psi_max",
    ],
)
def test_each_data_signal_alone_is_a_data_quality_fault(name):
    """Enforces docs/OPERATIONS.md:39."""
    verdict = classify.classify(telemetry.window(**{name: "breach"})).verdict
    assert verdict is classify.Verdict.DATA_QUALITY


@pytest.mark.parametrize(
    ("changes", "overall"),
    [
        ({"shift_conv": 0.12}, "helped"),
        ({"shift_conv": 0.12, "shift_latency": 40.0}, "harmed"),
        ({"seed": 5}, "inconclusive"),
    ],
)
def test_every_answer_says_two_periods_cannot_separate_the_model(
    changes, overall
):
    """Enforces docs/OPERATIONS.md:95."""
    before, after = evaluation.periods(**changes)
    result = continuous.post_promotion_check(evaluation.SPECS, before, after)
    assert result["overall"] == overall
    assert "cannot separate the model" in result["statement"]
