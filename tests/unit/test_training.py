"""Tests for the torch-free training helpers."""

import pytest

from mml.models.deeplearning import training


def test_auto_batch_size_scales_with_the_data_within_bounds():
    sizes = [
        training.resolve_batch_size("auto", n) for n in (100, 2000, 100000)
    ]

    assert sizes == [32, 64, 512]
    assert training.resolve_batch_size("auto", 10) == 10
    assert training.resolve_batch_size(256, 100) == 100
    with pytest.raises(ValueError, match="batch size"):
        training.resolve_batch_size(0, 100)


def test_cosine_schedule_warms_up_then_decays_to_zero():
    rates = [
        training.learning_rate_at(s, 100, 0.1, 10, "cosine")
        for s in (0, 9, 10, 55, 99)
    ]

    assert rates[0] == pytest.approx(0.01) and rates[1] == pytest.approx(0.1)
    assert rates[2] == pytest.approx(0.1)
    assert 0.0 < rates[3] < 0.1 and rates[4] < 0.001
    assert training.learning_rate_at(80, 100, 0.1, 0, "constant") == 0.1
    with pytest.raises(ValueError, match="schedule"):
        training.learning_rate_at(0, 10, 0.1, 0, "linear")


def healthy(monitor):
    for epoch in range(6):
        for step in range(5):
            monitor.step(1.0 + 0 * step, 0.5, 5.0)
        monitor.epoch(1.0 - 0.12 * epoch)
    return monitor


def test_a_normal_run_is_ok():
    report = healthy(training.TrainingMonitor()).report()

    assert report["verdict"] == "ok" and report["clipped_steps"] == 0


def test_non_finite_values_mean_divergence_and_stop_the_run():
    monitor = training.TrainingMonitor()

    assert monitor.step(1.0, 0.5, None) is True
    assert monitor.step(float("nan"), 0.5, None) is False
    assert monitor.report()["verdict"] == "diverged"


def test_gradient_spikes_are_reported_as_exploding_and_counted_as_clipped():
    monitor = training.TrainingMonitor()
    for norm in [0.5] * 20 + [900.0]:
        monitor.step(1.0, norm, 5.0)
    for epoch in range(3):
        monitor.epoch(1.0 - 0.2 * epoch)

    report = monitor.report()

    assert report["verdict"] == "exploding" and report["clipped_steps"] == 1


def test_near_zero_gradients_are_vanishing():
    monitor = training.TrainingMonitor()
    for step in range(30):
        monitor.step(0.7 + 0 * step, 1e-10, 5.0)
    for epoch in range(4):
        monitor.epoch(0.7 + 0 * epoch)

    assert monitor.report()["verdict"] == "vanishing"


def test_a_flat_loss_with_live_gradients_is_stalled():
    monitor = training.TrainingMonitor()
    for step in range(30):
        monitor.step(0.7 + 0 * step, 0.4, 5.0)
    for epoch in range(5):
        monitor.epoch(0.7 + 0 * epoch)

    assert monitor.report()["verdict"] == "stalled"
