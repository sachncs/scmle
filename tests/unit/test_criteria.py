"""Success criteria are compared in each metric's own direction."""

import pytest

from scml.contracts import errors
from scml.evaluation import criteria


def test_a_floor_for_higher_is_better_and_a_ceiling_for_lower_is_better():
    results = criteria.check(
        {"roc_auc": 0.8, "log_loss": 0.5},
        {"roc_auc": 0.85, "log_loss": 0.6},
    )

    by_name = {r.metric: r for r in results}
    assert by_name["roc_auc"].met and by_name["roc_auc"].higher_is_better
    assert not by_name["log_loss"].met
    assert not by_name["log_loss"].higher_is_better


def test_meeting_a_threshold_exactly_counts_as_met():
    assert criteria.check({"roc_auc": 0.8}, {"roc_auc": 0.8})[0].met
    assert criteria.check({"rmse": 3.0}, {"rmse": 3.0})[0].met


def test_an_unreported_metric_is_an_error():
    with pytest.raises(errors.ConfigError, match="not a metric"):
        criteria.check({"nope": 1.0}, {"roc_auc": 0.9})
