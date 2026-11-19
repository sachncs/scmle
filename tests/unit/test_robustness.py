"""Tests for the stress-test diagnostic."""

import dataclasses

import numpy
import test_diagnostics

from scmle.contracts import tasks
from scmle.evaluation import diagnostics
from scmle.evaluation import robustness


def inputs():
    return test_diagnostics.make_inputs(tasks.Task.BINARY, rows=800)


def test_every_scenario_is_scored_and_noise_hurts_a_signal_model():
    report = robustness.Robustness().compute(inputs())

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
    rng = numpy.random.default_rng(0)

    blanked = robustness.Blanking(0.3).corrupt(data.features, data, rng)
    unseen = robustness.UnseenCategories(0.5).corrupt(data.features, data, rng)

    assert data.features.equals(original)
    assert 0.2 < blanked["signal"].isna().mean() < 0.4
    assert (unseen["group"] == robustness.UNSEEN_TOKEN).mean() > 0.3


def test_a_model_ignoring_the_noise_feature_is_not_fragile_to_it():
    report = robustness.Robustness((robustness.UnseenCategories(0.2),)).compute(
        inputs()
    )

    assert report["fragile_under"] == []


def test_gaussian_noise_leaves_a_column_with_undefined_spread_unchanged():
    data = inputs()
    one = data.features.iloc[:1].copy()
    out = robustness.Noise(0.5).corrupt(one, data, numpy.random.default_rng(0))
    assert numpy.isfinite(out["signal"]).all()


def test_scenarios_that_cannot_fit_are_recorded_not_run(monkeypatch):
    clock = test_diagnostics.Clock()
    monkeypatch.setattr(diagnostics, "clock", clock)

    class Late(robustness.Noise):
        def corrupt(self, features, inputs, rng):
            clock.now += 1e6
            return super().corrupt(features, inputs, rng)

    data = dataclasses.replace(inputs(), deadline=clock.now + 10.0)
    report = robustness.Robustness(
        (Late(0.1), robustness.Blanking(0.1), robustness.Noise(0.5))
    ).compute(data)

    assert set(report["scenarios"]) == {"noise_0.1sd"}
    assert report["skipped_for_time"] == ["missing_10pct", "noise_0.5sd"]
    assert report["truncated"] and report["reason"]


def test_robustness_is_skipped_once_its_deadline_has_passed():
    data = dataclasses.replace(inputs(), deadline=diagnostics.clock() - 1.0)

    report = robustness.Robustness().compute(data)

    assert report["skipped"] and report["available"] is False


def test_noise_skips_numeric_features_the_pipeline_derives_from_a_date():
    data = test_diagnostics.make_dated_inputs()

    report = robustness.Robustness().compute(data)

    assert "noise_0.5sd" in report["scenarios"]
    noisy = robustness.Noise(0.5).corrupt(
        data.features, data, numpy.random.default_rng(0)
    )
    assert "when__month" not in noisy.columns
    assert not noisy["signal"].equals(data.features["signal"])
