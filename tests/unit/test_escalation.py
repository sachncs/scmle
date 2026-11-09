import math

import numpy
import pytest

from benchmarks.qualification import streams
from scmle.governance import escalation


def test_score_from_set_sizes_and_novelty():
    score = escalation.set_size_score(
        [1, 2, 4, 0, numpy.inf, numpy.nan, 1],
        [False, False, False, False, False, False, True],
    )
    assert score.tolist() == [1.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        escalation.set_size_score([1, 2], [True])


def test_binomial_p_value_is_the_exact_lower_tail():
    assert escalation.binomial_p_value(0, 10, 0.5) == pytest.approx(0.5**10)
    assert escalation.binomial_p_value(10, 10, 0.5) == pytest.approx(1.0)
    assert escalation.binomial_p_value(0, 0, 0.5) == 1.0


def test_threshold_stops_at_the_first_non_rejection():
    data = streams.draw_calibration(2000, 0.05, seed=1)
    result = escalation.select_threshold(
        data["score"], data["loss"], r_star=0.05, delta=0.1
    )
    assert not result.everything_escalates
    rejected = [s for s in result.steps if s.rejected]
    assert result.tau == rejected[-1].tau
    assert result.risk <= 0.05 + 0.02
    assert 0.0 < result.coverage <= 1.0


def test_validity_over_replicates_and_the_naive_cut_violates():
    delta, r_star, replicates = 0.1, 0.05, 300
    wrong, naive_wrong = 0, 0
    for seed in range(replicates):
        data = streams.draw_calibration(2000, r_star, seed=seed)
        result = escalation.select_threshold(
            data["score"], data["loss"], r_star, delta
        )
        if (
            not result.everything_escalates
            and streams.oracle_risk(result.tau) > r_star
        ):
            wrong += 1
        # naive: the loosest cut whose observed risk is at most r_star
        best = 1.0
        for tau in numpy.linspace(1.0, 0.0, 101):
            kept = data["score"] >= tau
            if kept.sum() and data["loss"][kept].mean() <= r_star:
                best = tau
        if streams.oracle_risk(best) > r_star:
            naive_wrong += 1
    interval = 2 * math.sqrt(delta * (1 - delta) / replicates)
    assert wrong / replicates <= delta + interval
    assert naive_wrong / replicates > delta + interval


def test_too_few_cases_escalate_everything():
    data = streams.draw_calibration(20, 0.05, seed=2)
    result = escalation.select_threshold(data["score"], data["loss"], 0.05, 0.1)
    assert result.everything_escalates
    assert result.to_dict()["tau"] is None


def test_weighted_selection_is_flagged_as_approximate():
    data = streams.draw_calibration(3000, 0.05, seed=3)
    weights = numpy.full(3000, 2.0)
    result = escalation.select_threshold(
        data["score"], data["loss"], 0.05, 0.1, weights=weights
    )
    assert result.weighted


def test_invalid_inputs_are_refused():
    ok = numpy.array([0.1, 0.9])
    with pytest.raises(ValueError, match="strictly between"):
        escalation.select_threshold(ok, ok, 0.0, 0.1)
    with pytest.raises(ValueError, match="aligned"):
        escalation.select_threshold(ok, ok[:1], 0.05, 0.1)
    with pytest.raises(ValueError, match="within"):
        escalation.select_threshold([2.0], [0.0], 0.05, 0.1)
    with pytest.raises(ValueError, match="decreasing"):
        escalation.select_threshold(ok, ok, 0.05, 0.1, grid=[0.1, 0.9])
    with pytest.raises(ValueError, match="weights"):
        escalation.select_threshold(ok, ok, 0.05, 0.1, weights=[1.0, -1.0])


def test_risk_coverage_curve_and_area():
    data = streams.draw_calibration(20_000, 0.05, seed=4)
    curve = escalation.risk_coverage(data["score"], data["loss"])
    assert curve.coverage[0] < curve.coverage[-1] == pytest.approx(1.0)
    assert curve.risk[-1] == pytest.approx(0.1, abs=0.01)
    assert 0.0 < curve.aurc < 0.1
    assert set(curve.to_dict()) == {"tau", "coverage", "risk", "aurc"}


def test_chow_check_reports_disagreement():
    confidence = numpy.linspace(0.5, 1.0, 1000)
    score = confidence
    same = escalation.chow_check(confidence, score, 0.8, 1.0, 5.0)
    assert same.cut == pytest.approx(0.2)
    assert same.consistent and same.disagreement < 0.01
    off = escalation.chow_check(confidence, score, 0.8, 1.0, 2.0)
    assert not off.consistent
    with pytest.raises(ValueError):
        escalation.chow_check(confidence, score, 0.8, 0.0, 1.0)


def test_route_escalates_low_scores_and_never_raises_tau():
    score = numpy.array([1.0, 0.9, 0.4, 0.3, 0.2, 0.1])
    routing = escalation.route(score, tau=0.5, c_max=1.0)
    assert routing.status.tolist() == ["answered"] * 2 + ["escalated"] * 4
    capped = escalation.route(score, tau=0.5, c_max=0.5)
    assert capped.backlog == 1
    assert capped.status.tolist().count("escalated") == 3
    assert capped.status[2] == "safe_default"  # highest of the escalated scores
    assert capped.answered.tolist() == routing.answered.tolist()


def test_halt_answers_nothing_and_blocked_cases_are_escalated():
    score = numpy.ones(10)
    assert escalation.route(score, 0.5, 1.0, halted=True).coverage == 0.0
    blocked = numpy.arange(10) < 3
    routing = escalation.route(score, 0.5, 1.0, blocked=blocked)
    assert routing.answered.sum() == 7
    with pytest.raises(ValueError):
        escalation.route(score, 0.5, 1.5)


def test_risk_coverage_of_an_empty_calibration_set_is_a_value_error():
    with pytest.raises(ValueError, match="empty"):
        escalation.risk_coverage([], [])
