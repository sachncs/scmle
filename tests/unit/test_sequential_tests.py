import math

import numpy as np
import pytest

from scmle.monitoring import sequential_tests as seq


def brute_force_log_capital(values, null_mean, direction, alpha, cap=0.5):
    """The recurrence written out with plain loops."""
    sign = 1.0 if direction == seq.GREATER else -1.0
    room = null_mean if direction == seq.GREATER else 1.0 - null_mean
    log_capital, total, deviation = 0.0, 0.0, 0.0
    for step, value in enumerate(values, start=1):
        variance = (0.25 + deviation) / step
        raw = math.sqrt(
            2 * math.log(1 / alpha) / (variance * step * math.log(1 + step))
        )
        size = min(raw, cap / room)
        log_capital += math.log(1 + sign * size * (value - null_mean))
        total += value
        deviation += (value - (0.5 + total) / (step + 1)) ** 2
    return log_capital


@pytest.mark.parametrize("direction", [seq.GREATER, seq.LESS])
def test_betting_test_matches_a_plain_loop(direction):
    values = np.random.default_rng(1).beta(2, 6, 60)
    test = seq.BettingTest(0.3, direction, 0.05)
    for value in values:
        test.update(value)
    expected = brute_force_log_capital(values, 0.3, direction, 0.05)
    assert float(test.log_capital) == pytest.approx(expected, rel=1e-12)


def test_vector_test_equals_scalar_tests():
    rng = np.random.default_rng(2)
    data = rng.random((40, 3))
    vector = seq.BettingTest(np.full(3, 0.4), seq.GREATER, 0.05)
    scalars = [seq.BettingTest(0.4, seq.GREATER, 0.05) for _ in range(3)]
    for row in data:
        vector.update(row)
        for index, scalar in enumerate(scalars):
            scalar.update(row[index])
    assert np.allclose(
        vector.log_capital, [float(s.log_capital) for s in scalars]
    )


def test_false_alarm_rate_is_below_the_level_at_every_look():
    rng = np.random.default_rng(3)
    test = seq.BettingTest(np.full(1000, 0.15), seq.GREATER, 0.05)
    for _ in range(1500):
        test.update(rng.binomial(50, 0.15, 1000) / 50)
    assert test.ever_crossed.mean() <= 0.05


def test_a_rise_is_detected():
    rng = np.random.default_rng(4)
    alarm = seq.RiskAlarm(0.10, 0.05, 0.05)
    fired = [alarm.update(rng.binomial(50, 0.25) / 50) for _ in range(60)]
    assert any(fired)


def test_confidence_sequence_covers_and_tightens():
    rng = np.random.default_rng(5)
    sequence = seq.ConfidenceSequence(0.05)
    widths = []
    for _ in range(200):
        low, high = sequence.update(rng.binomial(50, 0.3) / 50)
        widths.append(high - low)
    assert low <= 0.3 <= high
    assert widths[-1] < widths[0]


def test_paired_process_proposes_for_a_better_challenger_only():
    rng = np.random.default_rng(6)
    better = seq.PairedEProcess(0.0, 0.05)
    same = seq.PairedEProcess(0.0, 0.05)
    for _ in range(80):
        latent = rng.random(200)
        champion = (latent < 0.2).astype(float)
        better.update(champion, (latent < 0.1).astype(float))
        same.update(champion, champion)
    assert better.crossed
    assert not same.crossed
    assert same.e_value <= 1.0 + 1e-9


def test_reverse_process_flags_a_worse_new_champion():
    rng = np.random.default_rng(7)
    process = seq.PairedEProcess(0.0, 0.05)
    for _ in range(80):
        latent = rng.random(200)
        process.update(
            (latent < 0.3).astype(float), (latent < 0.1).astype(float)
        )
    assert process.crossed


def test_state_round_trips():
    test = seq.BettingTest(0.2, seq.LESS, 0.1)
    for value in (0.1, 0.4, 0.2):
        test.update(value)
    again = seq.BettingTest.from_dict(test.to_dict())
    again.update(0.3)
    test.update(0.3)
    assert float(again.log_capital) == pytest.approx(float(test.log_capital))


def test_promotion_sequence_enforces_registration_order():
    sequence = seq.PromotionSequence("1/20")
    sequence.register("a")
    sequence.register("b")
    with pytest.raises(ValueError, match="registration order"):
        sequence.decide("b", 100.0)
    assert sequence.decide("a", 1000.0).rejected
    assert not sequence.decide("b", 1.0).rejected
    with pytest.raises(ValueError, match="already decided"):
        sequence.decide("a", 1.0)
    with pytest.raises(ValueError, match="not registered"):
        sequence.decide("c", 1.0)


@pytest.mark.parametrize(
    "bad",
    [
        {"null_mean": 0.0, "direction": "greater", "alpha": 0.05},
        {"null_mean": 0.5, "direction": "up", "alpha": 0.05},
        {"null_mean": 0.5, "direction": "greater", "alpha": 1.0},
    ],
)
def test_invalid_arguments_are_refused(bad):
    with pytest.raises(ValueError):
        seq.BettingTest(**bad)


def test_observations_outside_the_unit_interval_are_refused():
    with pytest.raises(ValueError, match="within"):
        seq.BettingTest(0.5, seq.GREATER, 0.05).update(1.5)


def test_scaled_gaps_and_state_round_trip_for_processes_and_alarms():
    process = seq.PairedEProcess(0.0, 0.05, scale=10.0)
    for gap in (2.0, 3.0, -1.0):
        process.update_gap(gap)
    again = seq.PairedEProcess.from_dict(process.to_dict())
    assert again.e_value == pytest.approx(process.e_value)
    with pytest.raises(ValueError, match="gap"):
        process.update_gap(11.0)
    with pytest.raises(ValueError, match="margin"):
        seq.PairedEProcess(10.0, 0.05, scale=10.0)
    alarm = seq.RiskAlarm(0.1, 0.05, 0.05, scale=2.0)
    alarm.update(0.5)
    restored = seq.RiskAlarm.from_dict(alarm.to_dict())
    assert restored.update(0.9) == alarm.update(0.9)
    assert float(restored.test.log_capital) == pytest.approx(
        float(alarm.test.log_capital)
    )
