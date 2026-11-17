"""The leakage screen sees a target hidden in text, a date, a column pair
or an echo of the target, and leaves strong legitimate predictors alone."""

from fractions import Fraction

import numpy
import pandas
import pytest

from scmle.assurance.ledger import multiplicity
from scmle.data.admissibility import leakage
from scmle.data.admissibility import screen

CLS = "classification"
FAST = leakage.Config(
    skill_ceiling=0.95,
    association_ceiling=0.9,
    permutations=199,
    bootstrap=100,
)


def world(seed, n=400):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {"a": rng.normal(size=n), "b": rng.integers(0, 5, n).astype(float)}
    )
    frame["y"] = rng.integers(0, 2, n)
    return frame, rng


def suspects(report):
    return {(f.feature, f.mechanism) for f in report.suspects}


def test_a_target_written_into_free_text_is_a_suspect():
    frame = world(0)[0]
    words = frame.y.map({0: "renewed the plan", 1: "closed the account"})
    frame["note"] = [f"ticket {i}: customer {w}" for i, w in enumerate(words)]
    report = screen.screen(frame, "y", CLS, FAST)
    assert ("note", screen.SKILL) in suspects(report)


def test_a_target_encoded_as_the_year_of_a_date_is_a_suspect():
    frame, rng = world(1)
    days = rng.integers(0, 300, len(frame)) + 365 * frame.y.to_numpy()
    frame["closed_on"] = pandas.Timestamp("2020-01-01") + pandas.to_timedelta(
        days, unit="D"
    )
    report = screen.screen(frame, "y", CLS, FAST)
    assert ("closed_on", screen.SKILL) in suspects(report)


def test_a_target_split_over_two_columns_is_a_suspect_pair():
    frame, rng = world(2)
    mask = rng.integers(0, 2, len(frame))
    frame["p"] = mask
    frame["q"] = mask ^ frame.y.to_numpy()
    report = screen.screen(frame, "y", CLS, FAST)
    assert (screen.pair("p", "q"), screen.SKILL) in suspects(report)
    assert ("p", screen.SKILL) not in suspects(report)


def test_too_many_pairs_are_not_screened_and_the_notes_say_so():
    frame, rng = world(3, n=200)
    for k in range(6):
        frame[f"f{k}"] = rng.integers(0, 2, len(frame))
    names = [c for c in frame.columns if c != "y"]
    assert len(screen.pairs(frame, names)) > len(names)
    report = screen.screen(frame, "y", CLS, FAST)
    assert any("pairs" in note for note in report.notes)


def test_a_garbled_copy_of_the_label_is_an_echo_suspect():
    frame, rng = world(4, n=1000)
    labels = frame.y.map({0: "stay", 1: "churn"})
    frame["y"] = labels
    garbled = rng.random(len(frame)) < 0.1
    picks = rng.choice(["stay", "churn"], len(frame))
    frame["outcome_recorded"] = numpy.where(garbled, picks, labels)
    report = screen.screen(frame, "y", CLS, FAST)
    assert ("outcome_recorded", screen.ECHO) in suspects(report)


def test_a_noisy_copy_of_a_number_is_an_echo_suspect():
    rng = numpy.random.default_rng(5)
    n = 1000
    frame = pandas.DataFrame({"a": rng.normal(size=n)})
    frame["y"] = 50 + 10 * rng.normal(size=n)
    frame["recorded"] = frame.y + rng.normal(0, 2.5, n)
    report = screen.screen(frame, "y", "regression", FAST)
    assert ("recorded", screen.ECHO) in suspects(report)


def test_a_strong_legitimate_predictor_is_no_echo():
    rng = numpy.random.default_rng(6)
    n = 1000
    frame = pandas.DataFrame({"size": rng.normal(size=n)})
    frame["y"] = (frame["size"] + 0.4 * rng.normal(size=n) > 0).astype(int)
    report = screen.screen(frame, "y", CLS, FAST)
    assert not [f for f in report.suspects if f.mechanism == screen.ECHO]


@pytest.mark.parametrize("seed", range(3))
def test_the_measured_demand_counts_every_new_hypothesis(seed):
    frame, rng = world(seed)
    frame["p"] = rng.integers(0, 2, len(frame))
    frame["q"] = rng.integers(0, 2, len(frame))
    frame["e"] = frame.y ^ (rng.random(len(frame)) < 0.3)
    frame["t"] = pandas.Timestamp("2021-01-01") + pandas.to_timedelta(
        rng.integers(0, 99, len(frame)), unit="D"
    )
    names = [c for c in frame.columns if c != "y"]
    demand = screen.measure(frame, "y", CLS, FAST, names, None, None)
    report = screen.screen(frame, "y", CLS, FAST, names)
    assert demand is not None
    assert demand.family == report.family
    assert demand.permutations == report.permutations


def test_the_smallest_reachable_p_value_passes_a_whole_number_threshold():
    # 7 * H_7 / 0.05 is exactly 363, so B = 362 put 1 / (B + 1) on the
    # threshold, and the float p-value rounded to just above it.
    alpha = Fraction(1, 20)
    family = 7
    draws = screen.resamples(family, alpha)
    p = 1 / (draws + 1)
    assert Fraction(p) < alpha / (family * multiplicity.harmonics(family))


def test_the_run_screen_blocks_text_date_and_pair_leaks_together():
    frame, rng = world(7, n=600)
    words = frame.y.map({0: "renewed the plan", 1: "closed the account"})
    frame["note"] = [f"ticket {i}: customer {w}" for i, w in enumerate(words)]
    days = rng.integers(0, 300, len(frame)) + 365 * frame.y.to_numpy()
    frame["closed_on"] = pandas.Timestamp("2020-01-01") + pandas.to_timedelta(
        days, unit="D"
    )
    mask = rng.integers(0, 2, len(frame))
    frame["p"] = mask
    frame["q"] = mask ^ frame.y.to_numpy()
    names = [c for c in frame.columns if c != "y"]
    report = screen.screen_table(frame, "y", CLS, names, 0)
    blocking = {f.feature for f in report.blocking}
    assert {"note", "closed_on", screen.pair("p", "q")} <= blocking
