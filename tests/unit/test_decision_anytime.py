"""The anytime-valid engine: martingale facts, Ville, bounds, adapters."""

import itertools
import math

import numpy as np
import pytest

from scmle.data.ledger import online_fdr
from scmle.decision import anytime


def test_the_wealth_of_a_fair_bet_matches_a_hand_computed_value():
    stream = anytime.Stream(0.0, 1.0, 0.5)
    values = [1.0, 0.0, 1.0, 1.0]
    got = anytime.log_wealth_paths(values, stream)
    grid = anytime.bet_grid(0.5)
    expected = []
    for t in range(1, 5):
        wealth = [
            math.prod(1 + lam * (v - 0.5) for v in values[:t]) for lam in grid
        ]
        expected.append(math.log(sum(wealth) / len(grid)))
    assert got == pytest.approx(expected, abs=1e-12)


def test_the_mixture_has_mean_exactly_one_under_the_null():
    stream = anytime.Stream(0.0, 1.0, 0.3)
    length = 10
    sequences = np.array(
        list(itertools.product([0.0, 1.0], repeat=length)), dtype=float
    )
    ones = sequences.sum(axis=1)
    probability = 0.3**ones * 0.7 ** (length - ones)
    final = np.exp(anytime.log_wealth_paths(sequences, stream)[:, -1])
    assert float(probability @ final) == pytest.approx(1.0, abs=1e-9)


def test_peeking_at_every_unit_stays_inside_the_level():
    rng = np.random.default_rng(1)
    stream = anytime.Stream(0.0, 1.0, 0.5)
    data = (rng.random((3000, 400)) < 0.5).astype(float)
    paths = anytime.log_wealth_paths(data, stream)
    ever = (paths.max(axis=1) >= math.log(20)).mean()
    assert ever <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / 3000)


def test_a_true_effect_is_found():
    rng = np.random.default_rng(2)
    stream = anytime.Stream(0.0, 1.0, 0.5)
    data = (rng.random(600) < 0.7).astype(float)
    assert anytime.log_wealth_paths(data, stream)[-1] > math.log(1000)


def test_direction_less_reflects_the_stream():
    rng = np.random.default_rng(3)
    data = (rng.random(600) < 0.3).astype(float)
    less = anytime.Stream(0.0, 1.0, 0.5, "less")
    greater = anytime.Stream(0.0, 1.0, 0.5, "greater")
    assert anytime.log_wealth_paths(data, less)[-1] > math.log(100)
    assert anytime.log_wealth_paths(data, greater)[-1] < 0.0


def test_batches_and_single_updates_agree_and_track_the_running_maximum():
    stream = anytime.Stream(0.0, 1.0, 0.5)
    rng = np.random.default_rng(4)
    data = (rng.random(200) < 0.6).astype(float)
    whole = anytime.EProcess(stream)
    whole.update(data)
    pieces = anytime.EProcess(stream)
    for chunk in np.array_split(data, 7):
        pieces.update(chunk)
    assert pieces.log_e == pytest.approx(whole.log_e, abs=1e-9)
    assert pieces.count == 200
    path = anytime.log_wealth_paths(data, stream)
    assert whole.max_log_e == pytest.approx(max(0.0, path.max()))
    assert whole.crossed(0.5) == (whole.log_e >= math.log(2))
    assert whole.ever_crossed(0.5)
    assert len(whole.path) == 200


def test_the_e_value_enters_the_ledger_as_evidence():
    stream = anytime.Stream(0.0, 1.0, 0.5)
    process = anytime.EProcess(stream)
    process.update([1.0] * 30)
    evidence = process.evidence()
    assert isinstance(evidence, online_fdr.Evidence)
    assert evidence.e == pytest.approx(math.exp(process.log_e))


def test_pseudo_outcomes_have_the_mean_difference_and_the_stated_bounds():
    arm = np.array([1.0, 0.0, 1.0, 0.0])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    z = anytime.pseudo_outcomes(arm, y, 0.25)
    assert z.tolist() == [4.0, -1 / 0.75, 0.0, 0.0]
    assert z.max() <= 1 / 0.25 and z.min() >= -1 / 0.75


def test_the_confidence_bound_covers_and_tightens_with_data():
    rng = np.random.default_rng(5)
    stream = anytime.Stream(0.0, 1.0, 0.5)
    small = (rng.random(100) < 0.7).astype(float)
    large = (rng.random(4000) < 0.7).astype(float)
    low_small = anytime.confidence_bound(small, stream, 0.05)
    low_large = anytime.confidence_bound(large, stream, 0.05)
    assert low_small < low_large < 0.7
    upper = anytime.confidence_bound(
        large, anytime.Stream(0.0, 1.0, 0.5, "less"), 0.05
    )
    assert upper > 0.7
    assert anytime.confidence_bound([], stream, 0.05) == 0.0


@pytest.mark.parametrize(
    "call",
    [
        lambda: anytime.Stream(1.0, 0.0, 0.5),
        lambda: anytime.Stream(0.0, 1.0, 1.0),
        lambda: anytime.Stream(0.0, 1.0, 0.5, "up"),
        lambda: anytime.Stream(0.0, 1.0, 0.5).rescale([2.0]),
        lambda: anytime.Stream(0.0, 1.0, 0.5).rescale([[1.0]]),
        lambda: anytime.Stream(0.0, 1.0, 0.5).rescale([math.nan]),
        lambda: anytime.pseudo_outcomes([1.0], [2.0], 0.5),
        lambda: anytime.pseudo_outcomes([2.0], [1.0], 0.5),
        lambda: anytime.pseudo_outcomes([1.0], [1.0], 1.0),
        lambda: anytime.bound_stream(-1.0),
        lambda: anytime.EProcess(anytime.Stream(0.0, 1.0, 0.5)).crossed(1.0),
    ],
)
def test_bad_inputs_are_refused(call):
    with pytest.raises(ValueError):
        call()


# --- centred pseudo-outcomes -------------------------------------------------


def test_the_running_centre_uses_only_earlier_outcomes():
    y = np.array([1.0, 0.0, 1.0, 1.0])
    centres = anytime.running_center(y, prior=0.5, strength=2.0)
    # (0.5 * 2 + sum so far) / (2 + units so far)
    assert centres.tolist() == pytest.approx([0.5, 2 / 3, 0.5, 0.6])
    later = anytime.running_center(y[2:], carry=(1.0, 2))
    assert later.tolist() == pytest.approx(centres[2:].tolist())
    changed = y.copy()
    changed[3] = 0.0
    assert anytime.running_center(changed)[:3].tolist() == pytest.approx(
        centres[:3].tolist()
    )


def test_a_centred_pseudo_outcome_keeps_the_mean_difference_exactly():
    q, length = 0.4, 6
    cells = np.array(
        list(itertools.product([0.0, 1.0], repeat=2 * length)), dtype=float
    )
    arm, y = cells[:, :length], cells[:, length:]
    p_arm, p_y = (0.4, 0.3)
    weight = (q ** arm.sum(1) * (1 - q) ** (length - arm.sum(1))) * (
        p_y ** y.sum(1) * (1 - p_y) ** (length - y.sum(1))
    )
    z = anytime.pseudo_outcomes(arm, y, q, anytime.running_center(y))
    # arm and outcome independent: the difference in means is zero
    assert float((weight * z[:, -1]).sum()) == pytest.approx(0.0, abs=1e-12)
    assert p_arm == q
    assert z.min() >= -max(1 / q, 1 / (1 - q)) - 1e-12
    assert z.max() <= max(1 / q, 1 / (1 - q)) + 1e-12


def test_the_centred_e_process_has_mean_one_at_the_boundary_null():
    q, length, p0 = 0.5, 5, 0.3
    effect = 0.1
    cells = np.array(
        list(itertools.product([0.0, 1.0], repeat=2 * length)), dtype=float
    )
    arm, y = cells[:, :length], cells[:, length:]
    p_y = p0 + effect * arm
    like = np.prod(np.where(y == 1, p_y, 1 - p_y), axis=1) * (
        q ** arm.sum(1) * (1 - q) ** (length - arm.sum(1))
    )
    paths = anytime.centered_paths(arm, y, q, effect)
    assert float(like @ np.exp(paths[:, -1])) == pytest.approx(1.0, abs=1e-9)


def test_centring_shortens_the_time_to_find_an_effect_on_a_high_mean_metric():
    rng = np.random.default_rng(10)
    runs, n = 200, 6000
    arm = (rng.random((runs, n)) < 0.5).astype(float)
    y = np.where(
        arm == 1,
        rng.beta(2.4, 7.6, arm.shape),
        rng.beta(2.0, 8.0, arm.shape),
    )
    target = math.log(50)
    plain = anytime.log_wealth_paths(
        anytime.pseudo_outcomes(arm, y, 0.5),
        anytime.difference_stream(0.5, 0.02),
    )
    centred = anytime.centered_paths(arm, y, 0.5, 0.02)
    assert (centred.max(axis=1) >= target).mean() > (
        (plain.max(axis=1) >= target).mean() + 0.3
    )


def test_the_difference_stream_refuses_an_assignment_probability_outside_0_1():
    for q in (0.0, 1.0, -0.2, float("nan")):
        with pytest.raises(ValueError, match="assignment_prob"):
            anytime.difference_stream(q, 0.0)
