"""AIPW score, overlap diagnostics and the difference in means."""

import numpy
import pandas
import pytest

from scmle.data.identification import aipw


def test_the_ate_score_equals_the_hand_computed_toy_table():
    t = numpy.array([1.0, 1.0, 0.0, 0.0])
    y = numpy.array([3.0, 5.0, 1.0, 2.0])
    e = numpy.array([0.5, 0.25, 0.5, 0.25])
    m0 = numpy.array([1.0, 1.5, 1.0, 1.5])
    m1 = numpy.array([2.0, 4.0, 2.5, 3.5])
    # psi_i by hand:
    #   1: 2-1   + 1*(3-2)/0.5            = 1 + 2           = 3
    #   2: 4-1.5 + 1*(5-4)/0.25           = 2.5 + 4         = 6.5
    #   3: 2.5-1 - 1*(1-1)/0.5            = 1.5 - 0         = 1.5
    #   4: 3.5-1.5 - 1*(2-1.5)/0.75       = 2 - 0.666667    = 1.333333
    expected = (3 + 6.5 + 1.5 + 4 / 3) / 4
    theta, influence = aipw.scores("ATE", t, y, e, m0, m1)
    assert theta == pytest.approx(expected, abs=1e-12)
    assert influence.sum() == pytest.approx(0.0, abs=1e-12)


def test_the_att_score_equals_the_hand_computed_toy_table():
    t = numpy.array([1.0, 1.0, 0.0, 0.0])
    y = numpy.array([3.0, 5.0, 1.0, 2.0])
    e = numpy.array([0.5, 0.25, 0.5, 0.25])
    m0 = numpy.array([1.0, 1.5, 1.0, 1.5])
    m1 = numpy.zeros(4)
    # raw: 2, 3.5, -(1*0)=0, -(1/3 * 0.5) = -1/6; p = 0.5
    expected = (2 + 3.5 + 0 - 0.5 / 3) / 4 / 0.5
    theta = aipw.scores("ATT", t, y, e, m0, m1)[0]
    assert theta == pytest.approx(expected, abs=1e-12)


def planted(n: int, seed: int, tau: float = 2.0, confound: float = 1.0):
    rng = numpy.random.default_rng(seed)
    x = rng.standard_normal((n, 3))
    e = 1 / (1 + numpy.exp(-confound * x[:, 0]))
    t = (rng.uniform(size=n) < e).astype(float)
    y = tau * t + x @ numpy.array([1.5, -1.0, 0.5]) + rng.standard_normal(n)
    return pandas.DataFrame(
        {"a": x[:, 0], "b": x[:, 1], "c": x[:, 2], "t": t, "y": y}
    )


@pytest.mark.parametrize("learner", aipw.LEARNERS)
def test_a_planted_effect_is_recovered_with_a_covering_interval(learner):
    nuis = aipw.fit_nuisances(
        planted(2000, 1), ("a", "b", "c"), "t", "y", 0, learner
    )
    for estimand in ("ATE", "ATT"):
        out = aipw.estimate(nuis, planted(3000, 2), "t", "y", estimand)
        assert abs(out.estimate - 2.0) < 4 * out.se
        assert out.se < 0.1
    assert out.nuisance_digest == nuis.digest


def test_the_naive_difference_is_biased_and_aipw_is_not():
    frame = planted(4000, 3, confound=1.5)
    naive = aipw.difference_in_means(frame, "t", "y")
    assert naive.high < 2.0 or naive.low > 2.0
    nuis = aipw.fit_nuisances(
        planted(3000, 4, confound=1.5), ("a", "b", "c"), "t", "y"
    )
    out = aipw.estimate(nuis, frame, "t", "y")
    assert out.low < 2.0 < out.high


def test_nuisances_are_deterministic():
    one = aipw.fit_nuisances(
        planted(500, 5), ("a", "b"), "t", "y", 3, "boosted"
    )
    two = aipw.fit_nuisances(
        planted(500, 5), ("a", "b"), "t", "y", 3, "boosted"
    )
    assert one.digest == two.digest


def test_trimming_changes_the_estimand_and_can_refuse():
    nuis = aipw.fit_nuisances(
        planted(1500, 6, confound=2.5), ("a", "b", "c"), "t", "y"
    )
    frame = planted(2000, 7, confound=2.5)
    trimmed = aipw.estimate(nuis, frame, "t", "y", trim=0.1)
    assert trimmed.n < trimmed.n_input
    assert "estimand changed" in trimmed.population
    with pytest.raises(ValueError, match="minimum share"):
        aipw.estimate(nuis, frame, "t", "y", trim=0.45, min_share=0.9)


def test_overlap_gate_flags_extreme_propensities_only():
    rng = numpy.random.default_rng(0)
    good = rng.uniform(0.2, 0.8, 1000)
    bad = numpy.concatenate([rng.uniform(0.001, 0.02, 300), good[:700]])
    t = (rng.uniform(size=1000) < 0.5).astype(float)
    assert aipw.overlap_report(good, t)["passed"]
    report = aipw.overlap_report(bad, t)
    assert not report["passed"] and report["share_outside"] >= 0.3
    assert report["ess_treated"] <= report["n"]


def test_categorical_covariates_are_encoded_with_frozen_levels():
    frame = planted(400, 8)
    frame["g"] = numpy.where(frame["a"] > 0, "up", "down")
    nuis = aipw.fit_nuisances(frame, ("a", "g"), "t", "y")
    fresh = frame.head(50).copy()
    fresh["g"] = "unseen"
    e, m0, m1 = nuis.predict(fresh)
    assert numpy.isfinite(e).all() and numpy.isfinite(m1 - m0).all()


def test_bad_inputs_are_refused():
    frame = planted(100, 9)
    with pytest.raises(ValueError, match="binary"):
        aipw.fit_nuisances(frame.assign(t=frame["t"] * 2), ("a",), "t", "y")
    with pytest.raises(ValueError, match="missing"):
        aipw.fit_nuisances(frame.assign(y=numpy.nan), ("a",), "t", "y")
    with pytest.raises(ValueError, match="learner"):
        aipw.fit_nuisances(frame, ("a",), "t", "y", learner="x")
    with pytest.raises(ValueError, match="at least"):
        aipw.fit_nuisances(frame.assign(t=0.0), ("a",), "t", "y")


def test_difference_in_means_matches_welch():
    frame = pandas.DataFrame(
        {"t": [1, 1, 1, 0, 0, 0], "y": [5.0, 6.0, 7.0, 1.0, 2.0, 4.0]}
    )
    out = aipw.difference_in_means(frame, "t", "y")
    assert out.estimate == pytest.approx(6.0 - 7.0 / 3)
    from scipy import stats

    test = stats.ttest_ind([5, 6, 7], [1, 2, 4], equal_var=False)
    assert out.se == pytest.approx(abs(out.estimate / test.statistic))
