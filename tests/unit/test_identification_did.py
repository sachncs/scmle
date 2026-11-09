"""Difference in differences: hand-computed cells, coverage, sensitivity."""

import numpy
import pandas
import pytest

from scmle.data.identification import did


def panel(
    seed: int,
    n_treated: int = 150,
    n_never: int = 150,
    periods: int = 6,
    adopt: int = 3,
    tau: float = 1.5,
    drift: float = 0.0,
    second: bool = False,
) -> pandas.DataFrame:
    """Linear differential trend `drift` per period, effect `tau` after."""
    rng = numpy.random.default_rng(seed)
    rows = []
    groups = [(adopt, n_treated)] + ([(adopt + 1, n_treated)] if second else [])
    units = [(g, n) for g, n in groups] + [(None, n_never)]
    first = 0
    for g, n in units:
        for unit_row in range(n):
            uid = first + unit_row
            level = rng.normal()
            for t in range(periods):
                y = level + 0.3 * t + rng.normal(scale=0.7)
                if g is not None:
                    y += drift * t
                    if t >= g:
                        y += tau
                rows.append((uid, t, g if g is not None else numpy.nan, y))
        first += n
    return pandas.DataFrame(rows, columns=["id", "t", "g", "y"])


def test_a_hand_computed_three_period_panel():
    rows = []
    # treated units 0, 1 adopt at t = 2; never-treated units 2, 3.
    outcomes = {
        0: [1.0, 2.0, 6.0],
        1: [3.0, 4.0, 8.0],
        2: [0.0, 1.0, 2.0],
        3: [2.0, 3.0, 3.0],
    }
    for unit, values in outcomes.items():
        for t, y in enumerate(values):
            rows.append((unit, t, 2.0 if unit < 2 else numpy.nan, y))
    frame = pandas.DataFrame(rows, columns=["id", "t", "g", "y"])
    out = did.estimate(frame, "id", "t", "g", "y")
    # treated change t1 -> t2: (4, 4) mean 4; control (1, 0) mean 0.5.
    assert out.estimate == pytest.approx(3.5)
    # pre-period t0 vs base t1: treated (1, 1) mean 1; control (1, 1) mean 1.
    k = out.events.index(-2)
    assert out.coefficients[k] == pytest.approx(0.0)
    assert out.groups == {"2.0": 2} and out.n_never == 2
    # variance: treated deltas (4, 4) -> 0; control deltas (1, 0) -> 0.5/2.
    assert out.se == pytest.approx(numpy.sqrt(0.5 / 2))


def test_the_effect_is_recovered_and_the_interval_covers():
    hits = 0
    for seed in range(60):
        out = did.estimate(panel(seed), "id", "t", "g", "y")
        hits += out.low <= 1.5 <= out.high
    assert hits >= 52


def test_event_study_pre_coefficients_are_zero_under_parallel_trends():
    out = did.estimate(
        panel(1, n_treated=400, n_never=400), "id", "t", "g", "y"
    )
    pre = [
        c for e, c in zip(out.events, out.coefficients, strict=True) if e < 0
    ]
    post = [
        c for e, c in zip(out.events, out.coefficients, strict=True) if e >= 0
    ]
    assert max(abs(c) for c in pre) < 0.25
    assert min(post) > 1.0
    assert out.pretrend_p is not None and out.pretrend_p > 0.001


def test_a_trend_violation_biases_the_estimate_and_sensitivity_covers_it():
    out = did.estimate(
        panel(2, n_treated=800, n_never=800, drift=0.2), "id", "t", "g", "y"
    )
    assert not out.low <= 1.5 <= out.high
    low, high = out.sensitivity(1.0)
    assert low <= 1.5 <= high
    assert out.breakdown() is not None
    assert out.sensitivity(0.0) == (out.low, out.high)


def test_staggered_adoption_uses_only_never_treated_controls():
    out = did.estimate(
        panel(3, n_treated=200, n_never=300, second=True), "id", "t", "g", "y"
    )
    assert set(out.groups) == {"3.0", "4.0"}
    assert out.low < 1.5 < out.high


def test_sensitivity_needs_two_pre_periods():
    out = did.estimate(panel(4, adopt=1, periods=4), "id", "t", "g", "y")
    assert out.max_pre_change is None and out.breakdown() is None
    with pytest.raises(ValueError, match="two pre-periods"):
        out.sensitivity(1.0)


def test_unusable_panels_are_refused():
    frame = panel(5)
    with pytest.raises(ValueError, match="balanced"):
        did.estimate(frame.iloc[1:], "id", "t", "g", "y")
    with pytest.raises(ValueError, match="periods"):
        did.estimate(frame[frame["t"] < 2], "id", "t", "g", "y")
    with pytest.raises(ValueError, match="never-treated"):
        did.estimate(frame.dropna(subset=["g"]), "id", "t", "g", "y")
    with pytest.raises(ValueError, match="constant"):
        bad = frame.copy()
        bad.loc[bad.index[0], "g"] = 5.0
        did.estimate(bad, "id", "t", "g", "y")
