"""Pins for survivors of mutation testing in the monitoring modules.

Golden values were computed from the committed code on fixed sequences; each
test names what the mutants it kills would change. See
`tests/unit/MUTANTS_loop.md`.
"""

from fractions import Fraction
import re

import numpy as np
import pandas as pd
import pytest

from scmle.monitoring import labels
from scmle.monitoring import sequential_tests as st

OBS = [0.3, 0.9, 0.1, 0.7, 0.5, 0.8, 0.2, 0.6, 0.95, 0.4, 0.85, 0.15]


# ------------------------------------------------------------- validation


@pytest.mark.parametrize("bad", [-0.01, 1.01, np.nan, np.inf])
def test_validate_unit_names_the_argument_and_the_range(bad):
    with pytest.raises(ValueError, match="x must be finite and within"):
        st.validate_unit(bad, "x")
    assert st.validate_unit([0.0, 1.0], "x").tolist() == [0.0, 1.0]
    assert st.validate_unit(0.5, "x").dtype == np.float64


def test_validate_unit_checks_each_side_of_the_range_on_its_own():
    with pytest.raises(ValueError):
        st.validate_unit([0.5, -0.1], "x")
    with pytest.raises(ValueError):
        st.validate_unit([0.5, 1.1], "x")
    with pytest.raises(ValueError):
        st.validate_unit([0.5, np.nan], "x")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("sideways", 0.1), "direction must be one of"),
        (("greater", 0.0), "alpha must be strictly between 0 and 1"),
        (("greater", 1.0), "alpha must be strictly between 0 and 1"),
        (("greater", 0.1, 0.0), "cap must be strictly between 0 and 1"),
        (("greater", 0.1, 1.0), "cap must be strictly between 0 and 1"),
        (("greater", 0.1, 1.5), "cap must be strictly between 0 and 1"),
    ],
)
def test_betting_test_rejects_invalid_arguments_with_the_right_message(
    arguments, message
):
    with pytest.raises(ValueError, match=message):
        st.BettingTest(0.4, *arguments)


@pytest.mark.parametrize("null", [0.0, 1.0, 1.5, [0.5, 1.0], [0.0, 0.5]])
def test_betting_test_rejects_a_null_mean_on_or_outside_the_boundary(null):
    with pytest.raises(ValueError, match="null_mean must be strictly between"):
        st.BettingTest(null, "greater", 0.1)


def test_betting_test_accepts_values_just_inside_the_boundaries():
    st.BettingTest(0.999, "greater", 0.999, cap=0.999)
    st.BettingTest(0.001, "less", 0.001, cap=0.001)


def test_betting_test_update_names_the_observation():
    test = st.BettingTest(0.4, "greater", 0.1)
    with pytest.raises(ValueError, match="observation must be finite"):
        test.update(2.0)


def test_array_tests_start_at_zero_capital_and_peak():
    test = st.BettingTest([0.2, 0.5], "less", 0.1)
    assert test.log_capital.tolist() == [0.0, 0.0]
    assert test.peak.tolist() == [0.0, 0.0]
    assert test.log_capital.shape == (2,) == test.peak.shape


def test_a_mixture_over_an_array_null_keeps_array_state():
    test = st.MixtureTest([0.2, 0.5], "greater", 0.1)
    assert test.peak.shape == (2,) and test.log_capital.shape == (2,)
    test.update(0.9)
    assert test.log_capital.shape == (2,)
    assert np.all(test.log_capital <= test.peak + 1e-15)


# ---------------------------------------------------------------- golden


def test_a_betting_test_reproduces_its_recorded_state():
    test = st.BettingTest(0.4, "greater", 0.05)
    for value in OBS[:5]:
        test.update(value)
    record = test.to_dict()
    assert record["count"] == 5 and record["total"] == pytest.approx(2.5)
    assert record["deviation"] == pytest.approx(0.2836111111111111, rel=1e-12)
    assert record["log_capital"] == pytest.approx(
        0.31820956068636064, rel=1e-12
    )
    assert record["peak"] == pytest.approx(0.3519764231571782, rel=1e-12)
    assert not test.crossed and not test.ever_crossed
    assert test.e_value == pytest.approx(np.exp(record["log_capital"]))


def test_the_less_direction_bets_the_other_way_with_its_own_cap():
    test = st.BettingTest(0.4, "less", 0.1, cap=0.3)
    for value in OBS[:4]:
        test.update(value)
    record = test.to_dict()
    assert record["cap"] == 0.3 and record["direction"] == "less"
    assert record["log_capital"] == pytest.approx(
        -0.26164889540496505, rel=1e-12
    )
    assert record["peak"] == pytest.approx(0.04879016416943205, rel=1e-12)


def test_a_betting_test_round_trips_through_its_record():
    test = st.BettingTest(0.4, "less", 0.1, cap=0.3)
    for value in OBS[:6]:
        test.update(value)
    again = st.BettingTest.from_dict(test.to_dict())
    assert again.to_dict() == test.to_dict()
    assert again.cap == 0.3 and again.alpha == 0.1
    again.update(0.5)
    test.update(0.5)
    assert again.to_dict() == test.to_dict()


def test_the_threshold_and_crossing_use_the_level():
    test = st.BettingTest(0.1, "greater", 0.05)
    assert test.threshold == pytest.approx(np.log(20.0))
    for ignored in range(60):
        test.update(1.0)
    assert test.crossed and test.ever_crossed


def test_a_serialized_array_test_is_refused():
    with pytest.raises(ValueError, match="only a single test"):
        st.BettingTest([0.2, 0.5], "greater", 0.1).to_dict()
    with pytest.raises(ValueError, match="only a single test"):
        st.MixtureTest([0.2, 0.5], "greater", 0.1).to_dict()


def test_a_mixture_keeps_its_newest_terms_and_weights():
    test = st.MixtureTest(0.4, "greater", 0.05, stride=3, terms=2)
    for value in OBS:
        test.update(value)
    record = test.to_dict()
    assert record["count"] == 12 and record["started"] == 4
    assert record["log_weights"] == pytest.approx(
        [-2.4849066497880004, -2.995732273553991], rel=1e-12
    )
    assert [t["count"] for t in record["inner"]] == [6, 3]
    assert record["inner"][0]["log_capital"] == pytest.approx(
        0.5303032758139856, rel=1e-12
    )
    assert record["log_capital"] == pytest.approx(
        -1.6330567875694775, rel=1e-12
    )
    assert record["peak"] == pytest.approx(0.34904781720202843, rel=1e-12)


def test_a_mixture_adds_a_term_exactly_every_stride_observations():
    test = st.MixtureTest(0.4, "greater", 0.05, stride=2, terms=64)
    sizes = []
    for value in OBS[:7]:
        test.update(value)
        sizes.append(len(test.inner))
    assert sizes == [1, 1, 2, 2, 3, 3, 4]
    assert test.started == 4 and test.count == 7


def test_a_mixture_defaults_are_stride_25_and_64_terms():
    test = st.MixtureTest(0.4, "greater", 0.05)
    assert (test.stride, test.terms) == (25, 64)
    assert test.cap == st.DEFAULT_CAP


@pytest.mark.parametrize(("stride", "terms"), [(0, 1), (1, 0), (-1, 5)])
def test_a_mixture_needs_positive_stride_and_terms(stride, terms):
    with pytest.raises(ValueError, match="stride and terms must be positive"):
        st.MixtureTest(0.4, "greater", 0.05, stride, terms)
    st.MixtureTest(0.4, "greater", 0.05, 1, 1)


def test_a_mixture_passes_its_level_and_cap_to_every_term():
    test = st.MixtureTest(0.4, "less", 0.2, stride=1, terms=3, cap=0.25)
    for value in OBS[:4]:
        test.update(value)
    assert {t.alpha for t in test.inner} == {0.2}
    assert {t.cap for t in test.inner} == {0.25}
    assert {t.direction for t in test.inner} == {"less"}
    assert test.threshold == pytest.approx(np.log(5.0))


def test_a_mixture_round_trips_and_continues_identically():
    test = st.MixtureTest(0.4, "greater", 0.05, stride=3, terms=2, cap=0.3)
    for value in OBS[:8]:
        test.update(value)
    again = st.MixtureTest.from_dict(test.to_dict())
    assert again.to_dict() == test.to_dict()
    assert (again.started, again.cap, again.terms) == (3, 0.3, 2)
    for value in OBS[8:]:
        again.update(value)
        test.update(value)
    assert again.to_dict() == test.to_dict()


# ------------------------------------------------------ confidence sequence


def test_a_confidence_sequence_needs_three_points():
    with pytest.raises(ValueError, match="points must be at least 3"):
        st.ConfidenceSequence(0.1, points=2)
    st.ConfidenceSequence(0.1, points=3)


def test_the_confidence_grid_sits_strictly_inside_the_unit_interval():
    sequence = st.ConfidenceSequence(0.1, points=9)
    assert sequence.alpha == 0.1
    assert sequence.grid.tolist() == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    assert sequence.upper_tests.alpha == pytest.approx(0.05)
    assert sequence.lower_tests.alpha == pytest.approx(0.05)
    assert sequence.upper_tests.direction == "greater"
    assert sequence.lower_tests.direction == "less"
    assert sequence.upper_tests.null_mean.shape == (9,)


def test_the_confidence_interval_tightens_and_never_widens():
    sequence = st.ConfidenceSequence(0.1, points=9)
    last = (0.0, 1.0)
    for value in [0.5, 0.55, 0.45, 0.6, 0.5, 0.52] * 10:
        low, high = sequence.update(value)
        assert last[0] - 1e-12 <= low and high <= last[1] + 1e-12
        last = (low, high)
    assert last == pytest.approx((0.4, 0.6))
    assert sequence.alive.tolist() == [False] * 4 + [True] + [False] * 4


def test_an_emptied_confidence_sequence_reports_the_empty_interval():
    sequence = st.ConfidenceSequence(0.5, points=3)
    for ignored in range(80):
        sequence.update(0.0)
    assert not sequence.alive.any()
    assert sequence.interval == (1.0, 0.0)


# ---------------------------------------------------------------- the alarm


def test_a_risk_alarm_rejects_a_null_risk_outside_the_scale():
    for risk, tolerance in [(0.0, 0.0), (1.0, 0.0), (1.5, 0.5), (0.5, -0.5)]:
        with pytest.raises(ValueError, match="must be in \\(0, scale\\)"):
            st.RiskAlarm(risk, tolerance, 0.05)
    st.RiskAlarm(0.9, 0.09, 0.05)
    st.RiskAlarm(0.1, 0.05, 0.05, scale=2.0)


def test_a_risk_alarm_clips_and_scales_each_mean_loss():
    alarm = st.RiskAlarm(0.1, 0.05, 0.05, scale=2.0)
    for value in [0.3, 0.1, 1.9, 0.5, 2.5, -1.0]:
        alarm.update(value)
    assert alarm.test.count == 6
    assert alarm.test.log_capital == pytest.approx(2.358465318891925, rel=1e-12)
    assert alarm.test.null_mean == pytest.approx(0.075)


def test_a_risk_alarm_round_trips_with_its_scale_and_null():
    alarm = st.RiskAlarm(0.1, 0.05, 0.05, scale=2.0)
    for value in [0.3, 0.1, 1.9]:
        alarm.update(value)
    again = st.RiskAlarm.from_dict(alarm.to_dict())
    assert again.to_dict() == alarm.to_dict()
    assert again.scale == 2.0
    assert again.test.null_mean == pytest.approx(0.075)
    again.update(1.0)
    alarm.update(1.0)
    assert again.to_dict() == alarm.to_dict()


def test_a_risk_alarm_alarms_on_a_sustained_rise_and_stays_quiet_below():
    quiet = st.RiskAlarm(0.1, 0.05, 0.05)
    loud = st.RiskAlarm(0.1, 0.05, 0.05)
    assert not any(quiet.update(0.05) for ignored in range(200))
    assert any(loud.update(0.6) for ignored in range(200))


# ------------------------------------------------------- paired e-process


def test_a_paired_process_needs_a_margin_inside_the_scale():
    for scale, margin in [(1.0, 1.0), (1.0, -1.0), (0.0, 0.0), (-1.0, 0.0)]:
        with pytest.raises(ValueError, match="margin must lie strictly"):
            st.PairedEProcess(margin, 0.05, scale)
    st.PairedEProcess(0.99, 0.05)
    st.PairedEProcess(-0.99, 0.05)
    st.PairedEProcess(0.0, 0.05, 1e-9)


def test_a_paired_process_maps_gaps_to_unit_bets_through_the_scale():
    process = st.PairedEProcess(0.1, 0.05, scale=2.0)
    assert process.test.null_mean == pytest.approx(0.525)
    for gap in [0.5, -1.0, 2.0, 0.1]:
        process.update_gap(gap)
    assert process.e_value == pytest.approx(0.5870451355145232, rel=1e-12)
    assert process.test.count == 4
    assert process.margin == 0.1 and process.scale == 2.0


def test_paired_losses_are_compared_as_first_minus_second():
    process = st.PairedEProcess(0.0, 0.05)
    process.update(np.array([0.5, 1.0]), np.array([0.0, 0.5]))
    assert process.e_value == pytest.approx(0.625, rel=1e-12)


@pytest.mark.parametrize(
    ("first", "second"), [([0.5], [0.5, 0.5]), ([], []), ([0.5], [2.0])]
)
def test_misaligned_empty_or_out_of_range_pairs_are_refused(first, second):
    with pytest.raises(ValueError):
        st.PairedEProcess(0.0, 0.05).update(first, second)


def test_a_single_pair_is_enough_and_names_both_losses():
    st.PairedEProcess(0.0, 0.05).update([0.5], [0.2])
    with pytest.raises(ValueError, match="champion_loss must be finite"):
        st.PairedEProcess(0.0, 0.05).update([2.0], [0.5])
    with pytest.raises(ValueError, match="other_loss must be finite"):
        st.PairedEProcess(0.0, 0.05).update([0.5], [2.0])


def test_a_gap_may_sit_exactly_on_the_scale_but_not_beyond():
    process = st.PairedEProcess(0.0, 0.05, scale=2.0)
    process.update_gap(2.0)
    process.update_gap(-2.0)
    for gap in (2.0 + 1e-6, np.nan, np.inf):
        with pytest.raises(ValueError, match="gap must be finite"):
            process.update_gap(gap)


def test_update_returns_whether_the_process_crossed():
    process = st.PairedEProcess(0.0, 0.2)
    crossed = [process.update_gap(0.9) for ignored in range(40)]
    assert crossed[-1] is True and process.crossed is True
    assert isinstance(crossed[0], bool)
    fresh = st.PairedEProcess(0.0, 0.2)
    assert fresh.update_gap(0.0) is False


def test_a_paired_process_round_trips_with_scale_and_level():
    process = st.PairedEProcess(0.1, 0.05, scale=2.0)
    for gap in [0.5, -1.0, 0.3]:
        process.update_gap(gap)
    again = st.PairedEProcess.from_dict(process.to_dict())
    assert again.to_dict() == process.to_dict()
    assert again.scale == 2.0 and again.test.alpha == 0.05


# ------------------------------------------------------- promotion sequence


def test_the_promotion_sequence_orders_and_refuses_repeats():
    sequence = st.PromotionSequence("1/20")
    assert [sequence.register(name) for name in "abc"] == [1, 2, 3]
    with pytest.raises(ValueError, match="'a' is already registered"):
        sequence.register("a")
    with pytest.raises(ValueError, match="'z' was not registered"):
        sequence.decide("z", 1.0)
    with pytest.raises(ValueError, match="registration order; 'a' is next"):
        sequence.decide("b", 1.0)
    step = sequence.decide("a", 30.0)
    assert step is sequence.decisions["a"]
    assert str(step.level) == "1/40" and not step.rejected
    assert str(sequence.wealth) == "1/40"
    with pytest.raises(ValueError, match="'a' is already decided"):
        sequence.decide("a", 1.0)
    with pytest.raises(ValueError, match="registration order; 'b' is next"):
        sequence.decide("c", 1.0)
    assert sequence.decide("b", 1e9).rejected
    assert sequence.state is sequence.decisions["b"].state


def test_the_promotion_sequence_accepts_fractions_and_floats():
    assert (
        st.PromotionSequence(0.05).config.alpha
        == st.PromotionSequence("1/20").config.alpha
    )


# ------------------------------------------------------------------ labels


def test_a_window_of_zero_is_allowed_and_a_negative_one_is_not():
    assert labels.is_matured(2, 2, 0)
    assert not labels.is_matured(3, 2, 0)
    with pytest.raises(ValueError, match="window must not be negative"):
        labels.is_matured(0, 1, -1)


def test_classify_needs_a_positive_epsilon_for_propensity_and_audit():
    seen = np.array([True, False])
    pi = np.array([0.5, 0.5])
    flags = np.array([True, False])
    assert labels.classify(seen, pi, flags, 0.0) is labels.Mechanism.UNVERIFIED
    assert (
        labels.classify(seen, pi, flags, 0.5)
        is labels.Mechanism.LOGGED_PROPENSITY
    )
    assert labels.classify(seen, pi, flags, 0.6) is labels.Mechanism.AUDIT_SLICE
    assert (
        labels.classify(seen, None, flags, 0.1) is labels.Mechanism.AUDIT_SLICE
    )
    assert (
        labels.classify(seen, None, np.array([False, False]), 0.1)
        is labels.Mechanism.UNVERIFIED
    )
    assert labels.classify(seen, None, None, 0.1) is labels.Mechanism.UNVERIFIED
    assert (
        labels.classify(seen, np.array([0.5, np.nan]), None, 0.1)
        is labels.Mechanism.UNVERIFIED
    )
    assert (
        labels.classify(np.array([True, True]), None, None, 0.0)
        is labels.Mechanism.COMPLETE
    )


def test_estimates_check_alignment_range_and_report_counts():
    with pytest.raises(ValueError, match="aligned and non-empty"):
        labels.estimate_batch(np.zeros(0), np.zeros(0, bool), None, None, 0.0)
    with pytest.raises(ValueError, match="aligned and non-empty"):
        labels.estimate_batch(np.zeros(2), np.zeros(3, bool), None, None, 0.0)
    estimate = labels.estimate_batch(
        np.array([0.5]), np.array([True]), None, None, 0.0
    )
    assert (estimate.cases, estimate.labelled) == (1, 1)
    with pytest.raises(ValueError, match="propensity must be aligned"):
        labels.estimate_batch(
            np.zeros(2), np.ones(2, bool), np.ones(3), None, 0.1
        )
    with pytest.raises(ValueError, match="audit must be aligned"):
        labels.estimate_batch(
            np.zeros(2), np.ones(2, bool), None, np.ones(3, bool), 0.1
        )
    for bad in (-0.1, 1.1, np.nan):
        with pytest.raises(ValueError, match="finite and within"):
            labels.estimate_batch(
                np.array([bad, 0.5]), np.ones(2, bool), None, None, 0.0
            )
    labels.estimate_batch(
        np.array([0.0, 1.0]), np.ones(2, bool), None, None, 0.0
    )


def test_signed_estimates_allow_negative_differences_down_to_minus_the_bound():
    ok = labels.estimate_batch(
        np.array([-2.0, 2.0, 0.0]),
        np.ones(3, bool),
        None,
        None,
        0.0,
        bound=2.0,
        signed=True,
    )
    assert ok.mean == 0.0 and ok.scale == 2.0
    for bad in (-2.1, 2.1):
        with pytest.raises(ValueError):
            labels.estimate_batch(
                np.array([bad]),
                np.ones(1, bool),
                None,
                None,
                0.0,
                bound=2.0,
                signed=True,
            )
    with pytest.raises(ValueError):
        labels.estimate_batch(
            np.array([-0.1]), np.ones(1, bool), None, None, 0.0
        )


def test_an_unverified_batch_reports_cases_zero_labelled_and_its_bound():
    estimate = labels.estimate_batch(
        np.array([np.nan, np.nan, 0.5]),
        np.array([False, False, True]),
        None,
        None,
        0.0,
        bound=3.0,
    )
    assert estimate == labels.BatchEstimate(
        labels.Mechanism.UNVERIFIED, 3, 0, None, None, 3.0
    )


def test_complete_and_weighted_estimates_report_their_fields():
    complete = labels.estimate_batch(
        np.array([0.5, 1.0]), np.ones(2, bool), None, None, 0.0, bound=2.0
    )
    assert complete == labels.BatchEstimate(
        labels.Mechanism.COMPLETE, 2, 2, 0.75, 0.375, 2.0
    )
    loss = np.array([1.0, 0.0, 1.0, 0.0])
    seen = np.array([True, True, False, False])
    weighted = labels.estimate_batch(
        loss, seen, np.array([0.5, 0.5, 0.25, 0.25]), None, 0.25
    )
    assert weighted == labels.BatchEstimate(
        labels.Mechanism.LOGGED_PROPENSITY, 4, 2, 0.5, 0.125, 4.0
    )
    audit = labels.estimate_batch(
        np.array([1.0, 1.0, 0.0, 0.0]),
        np.array([True, True, True, False]),
        None,
        np.array([True, False, True, False]),
        0.5,
    )
    assert audit == labels.BatchEstimate(
        labels.Mechanism.AUDIT_SLICE, 4, 2, 0.5, 0.25, 2.0
    )


def test_a_scaled_estimate_never_exceeds_one():
    estimate = labels.estimate_batch(
        np.ones(2), np.ones(2, bool), np.array([0.1, 0.1]), None, 0.5
    )
    assert estimate.scaled <= 1.0


def test_randomization_check_reports_its_counts_and_level():
    flags = np.zeros(1000, bool)
    flags[:100] = True
    ok = labels.check_randomization(flags, 0.1)
    assert (ok.audited, ok.cases, ok.rate) == (100, 1000, 0.1)
    assert ok.passed and ok.p_value > 0.9
    flags[:300] = True
    bad = labels.check_randomization(flags, 0.1)
    assert not bad.passed and bad.p_value < 1e-6
    for rate in (0.0, 1.0):
        with pytest.raises(ValueError, match="rate must be strictly between"):
            labels.check_randomization(flags, rate)
    tolerant = labels.check_randomization(flags, 0.1, alpha=1e-300)
    assert tolerant.passed == (tolerant.p_value > 1e-300)


def test_the_stream_mechanism_is_the_weakest_of_its_batches():
    def batch(mechanism):
        return labels.BatchEstimate(mechanism, 1, 1, 0.0, 0.0, 1.0)

    order = [
        labels.Mechanism.COMPLETE,
        labels.Mechanism.LOGGED_PROPENSITY,
        labels.Mechanism.AUDIT_SLICE,
        labels.Mechanism.UNVERIFIED,
    ]
    assert labels.stream_mechanism([]) is labels.Mechanism.COMPLETE
    for index, weakest in enumerate(order):
        mixed = [batch(order[0]), batch(weakest), batch(order[index // 2])]
        assert labels.stream_mechanism(mixed) is weakest
    assert (
        labels.stream_mechanism(
            [batch(order[2]), batch(order[1]), batch(order[2])]
        )
        is order[2]
    )


def test_permits_action_is_false_only_for_unverified():
    assert [labels.permits_action(m) for m in labels.Mechanism] == [
        True,
        True,
        True,
        False,
    ]
    assert tuple(m.value for m in labels.Mechanism) == labels.MECHANISMS


def test_the_feedback_verdict_threshold_is_five_percent_inclusive():
    class Result:
        def __init__(self, p_value):
            self.p_value = p_value

    assert labels.feedback_verdict(0.3, Result(0.05)) == "loop_detected"
    assert labels.feedback_verdict(0.3, Result(0.0501)) == "no_loop_detected"
    assert labels.feedback_verdict(0.3, Result(None)) == "no_loop_detected"
    assert labels.feedback_verdict(0.3, None) == labels.UNVERIFIED
    assert labels.feedback_verdict(0.0, Result(0.0)) == labels.UNVERIFIED
    assert "randomized bypass slice" in labels.FEEDBACK_UNVERIFIED


# ------------------------------------------------------------ second round


def test_a_stride_of_one_starts_a_term_on_every_observation_after_the_first():
    test = st.MixtureTest(0.4, "greater", 0.05, stride=1, terms=64)
    sizes = []
    for value in OBS[:4]:
        test.update(value)
        sizes.append(len(test.inner))
    assert sizes == [1, 2, 3, 4]


def test_array_mixtures_equal_the_scalar_mixtures_entry_by_entry():
    nulls = [0.2, 0.5, 0.8]
    joint = st.MixtureTest(nulls, "greater", 0.05, stride=3, terms=3)
    singles = [
        st.MixtureTest(null, "greater", 0.05, stride=3, terms=3)
        for null in nulls
    ]
    for value in OBS:
        joint.update(value)
        for single in singles:
            single.update(value)
    assert joint.log_capital.tolist() == pytest.approx(
        [float(single.log_capital) for single in singles], rel=1e-12
    )
    assert joint.peak.tolist() == pytest.approx(
        [float(single.peak) for single in singles], rel=1e-12
    )


def test_a_grid_point_stays_out_after_its_capital_falls_back():
    sequence = st.ConfidenceSequence(0.1, points=9)
    last = (0.0, 1.0)
    emptied = False
    for value in [0.0] * 40 + [1.0] * 200:
        low, high = sequence.update(value)
        emptied = emptied or not sequence.alive.any()
        assert (low, high) == (1.0, 0.0) or (
            last[0] - 1e-12 <= low and high <= last[1] + 1e-12
        )
        if (low, high) != (1.0, 0.0):
            last = (low, high)
    assert emptied


def test_a_round_trip_with_a_small_scale_rebuilds_the_same_alarm():
    alarm = st.RiskAlarm(0.1, 0.05, 0.05, scale=0.2)
    for value in [0.02, 0.15, 0.19]:
        alarm.update(value)
    again = st.RiskAlarm.from_dict(alarm.to_dict())
    assert again.to_dict() == alarm.to_dict()
    unit = st.RiskAlarm(0.1, 0.05, 0.05)
    assert st.RiskAlarm.from_dict(unit.to_dict()).to_dict() == unit.to_dict()


def test_a_gap_a_hair_beyond_the_scale_is_still_within_the_tolerance():
    process = st.PairedEProcess(0.0, 0.05, scale=2.0)
    process.update_gap(2.0 + 1e-12)
    process.update_gap(-2.0 - 1e-12)
    assert process.test.count == 2


def test_a_propensity_one_epsilon_below_the_floor_is_still_logged():
    seen = np.array([True, False])
    floor = 0.5 - 1e-12
    mechanism = labels.classify(seen, np.array([floor, 0.5]), None, 0.5)
    assert mechanism is labels.Mechanism.LOGGED_PROPENSITY
    below = labels.classify(seen, np.array([0.5 - 1e-9, 0.5]), None, 0.5)
    assert below is labels.Mechanism.UNVERIFIED


def test_unobserved_losses_never_enter_the_weighted_total():
    seen = np.array([True, False, True])
    weighted = labels.estimate_batch(
        np.array([0.5, np.nan, 0.25]),
        seen,
        np.array([0.5, 0.5, 0.5]),
        None,
        0.5,
    )
    assert weighted.mean == pytest.approx((1.0 + 0.5) / 3)


def test_randomization_is_judged_strictly_above_the_level():
    flags = np.zeros(200, bool)
    flags[:30] = True
    p_value = labels.check_randomization(flags, 0.1).p_value
    assert not labels.check_randomization(flags, 0.1, alpha=p_value).passed
    assert labels.check_randomization(flags, 0.1, alpha=p_value / 2).passed


def test_the_bypass_test_passes_its_permutations_and_level_on():
    rng = np.random.default_rng(0)
    bypass = pd.DataFrame({"z": rng.normal(size=60)})
    model = pd.DataFrame({"z": rng.normal(0.3, 1, size=60)})
    default = labels.bypass_test(bypass, model, seed=1)
    few = labels.bypass_test(bypass, model, seed=1, permutations=19)
    loose = labels.bypass_test(bypass, model, seed=1, alpha=0.999)
    assert default.p_value * 200 == pytest.approx(round(default.p_value * 200))
    assert few.p_value * 20 == pytest.approx(round(few.p_value * 20))
    assert default.decision != loose.decision


# ----------------------------------------------- exact messages and dtypes


def raises_exactly(message, call):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        call()


def test_every_error_message_is_exact():
    raises_exactly(
        "alpha must be strictly between 0 and 1",
        lambda: st.BettingTest(0.4, "greater", 1.0),
    )
    raises_exactly(
        "cap must be strictly between 0 and 1",
        lambda: st.BettingTest(0.4, "greater", 0.1, 1.0),
    )
    raises_exactly(
        "null_mean must be strictly between 0 and 1",
        lambda: st.BettingTest(1.0, "greater", 0.1),
    )
    raises_exactly(
        "only a single test can be serialized",
        lambda: st.BettingTest([0.2, 0.4], "greater", 0.1).to_dict(),
    )
    raises_exactly(
        "only a single test can be serialized",
        lambda: st.MixtureTest([0.2, 0.4], "greater", 0.1).to_dict(),
    )
    raises_exactly(
        "stride and terms must be positive",
        lambda: st.MixtureTest(0.4, "greater", 0.1, 0),
    )
    raises_exactly(
        "points must be at least 3",
        lambda: st.ConfidenceSequence(0.1, points=2),
    )
    raises_exactly(
        "reference_risk + tolerance must be in (0, scale)",
        lambda: st.RiskAlarm(1.0, 0.0, 0.05),
    )
    raises_exactly(
        "margin must lie strictly within (-scale, scale)",
        lambda: st.PairedEProcess(1.0, 0.05),
    )
    raises_exactly(
        "paired losses must be non-empty and aligned",
        lambda: st.PairedEProcess(0.0, 0.05).update([0.5], [0.5, 0.5]),
    )
    raises_exactly(
        "gap must be finite and within [-scale, scale]",
        lambda: st.PairedEProcess(0.0, 0.05).update_gap(2.0),
    )
    raises_exactly(
        "window must not be negative", lambda: labels.is_matured(0, 0, -1)
    )
    raises_exactly(
        "loss and observed must be aligned and non-empty",
        lambda: labels.estimate_batch(
            np.zeros(2), np.ones(3, bool), None, None, 0.0
        ),
    )
    raises_exactly(
        "propensity must be aligned with the cases",
        lambda: labels.estimate_batch(
            np.zeros(2), np.ones(2, bool), np.ones(3), None, 0.1
        ),
    )
    raises_exactly(
        "audit must be aligned with the cases",
        lambda: labels.estimate_batch(
            np.zeros(2), np.ones(2, bool), None, np.ones(3, bool), 0.1
        ),
    )
    raises_exactly(
        "observed losses must be finite and within [0, bound]",
        lambda: labels.estimate_batch(
            np.array([2.0]), np.ones(1, bool), None, None, 0.0
        ),
    )
    raises_exactly(
        "rate must be strictly between 0 and 1",
        lambda: labels.check_randomization(np.ones(3, bool), 1.0),
    )
    raises_exactly(
        "x must be finite and within [0, 1]",
        lambda: st.validate_unit(2, "x"),
    )


def test_inputs_are_converted_to_floats_and_masks_to_booleans():
    assert st.validate_unit([0, 1], "x").dtype == np.float64
    test = st.BettingTest(Fraction(2, 5), "greater", 0.1)
    assert test.null_mean.dtype == np.float64
    assert st.MixtureTest(Fraction(2, 5), "greater", 0.1).null_mean.dtype == (
        np.float64
    )
    estimate = labels.estimate_batch(
        np.array([1.0, 0.0, 0.5]), [1, 1, 1], None, None, 0.0
    )
    assert estimate.labelled == 3 and estimate.mean == pytest.approx(0.5)
