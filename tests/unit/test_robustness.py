"""Tests for the stress-test diagnostic."""

import numpy as np
import test_diagnostics

from mml.contracts import tasks
from mml.evaluation import robustness


def inputs():
    return test_diagnostics.make_inputs(tasks.TaskType.BINARY, rows=800)


def test_every_scenario_is_scored_and_noise_hurts_a_signal_model():
    report = robustness.RobustnessReport().compute(inputs())

    assert set(report["scenarios"]) == {
        "missing_10pct",
        "missing_30pct",
        "noise_0.1sd",
        "noise_0.5sd",
        "unseen_categories_20pct",
    }
    heavy = report["scenarios"]["noise_0.5sd"]
    light = report["scenarios"]["noise_0.1sd"]
    assert heavy["loss"] > light["loss"] >= -0.02
    assert heavy["relative_loss"] > 0.02


def test_corruptions_leave_the_input_untouched_and_apply_as_described():
    data = inputs()
    original = data.features.copy()
    rng = np.random.default_rng(0)

    blanked = robustness.MissingValues(0.3).corrupt(data.features, data, rng)
    unseen = robustness.UnseenCategories(0.5).corrupt(data.features, data, rng)

    assert data.features.equals(original)
    assert 0.2 < blanked["signal"].isna().mean() < 0.4
    assert (unseen["group"] == robustness.UNSEEN_TOKEN).mean() > 0.3


def test_a_model_ignoring_the_noise_feature_is_not_fragile_to_it():
    report = robustness.RobustnessReport(
        (robustness.UnseenCategories(0.2),)
    ).compute(inputs())

    assert report["fragile_under"] == []
