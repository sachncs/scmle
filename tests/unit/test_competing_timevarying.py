"""Competing risks and time-varying covariates against known truth."""

import numpy
import pandas
import pytest

from scmle.models.survival import competing
from scmle.models.survival import estimators
from scmle.models.survival import timevarying


def two_causes(rows=3000, seed=0):
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=rows)
    rate_a = 0.10 * numpy.exp(0.8 * x)
    rate_b = 0.15 * numpy.ones(rows)
    time_a = rng.exponential(1 / rate_a)
    time_b = rng.exponential(1 / rate_b)
    censor = rng.exponential(1 / 0.05, rows)
    time = numpy.minimum.reduce([time_a, time_b, censor])
    cause = numpy.where(time == censor, 0, numpy.where(time == time_a, 1, 2))
    return pandas.DataFrame({"x": x}), time, cause


def test_cumulative_incidence_matches_the_closed_form():
    time, cause = two_causes()[1:]

    curves = competing.cumulative_incidence(time, cause)
    values = curves[1][1]

    # The causes' incidences add up to one minus overall event-free survival.
    moments, survival = estimators.kaplan_meier(time, cause != 0)
    total = values[-1] + curves[2][1][-1]
    assert total == pytest.approx(1 - survival[-1], abs=1e-9)
    assert curves[1][0][-1] == moments[-1]
    assert values[-1] < 1 and numpy.all(numpy.diff(values) >= 0)


def test_the_naive_complement_overstates_incidence():
    time, cause = two_causes()[1:]
    exact = competing.cumulative_incidence(time, cause)[1][1][-1]

    naive = competing.naive_complement(time, cause, 1).iloc[-1]

    assert naive > exact + 0.1


def test_cause_specific_cox_recovers_the_effect_and_orders_risk():
    features, time, cause = two_causes()
    model = competing.CauseSpecificCox(penalty=0.1).fit(features, time, cause)

    assert model.models_[1].coefficients_[0] == pytest.approx(0.8, abs=0.12)
    assert abs(model.models_[2].coefficients_[0]) < 0.12
    low = pandas.DataFrame({"x": [-1.5]})
    high = pandas.DataFrame({"x": [1.5]})
    assert model.predict(high, 10.0, 1)[0] > model.predict(low, 10.0, 1)[0]
    both = model.predict(high, 10.0, 1) + model.predict(high, 10.0, 2)
    assert both[0] < 1.0
    with pytest.raises(ValueError, match="unknown cause"):
        model.predict(high, 10.0, 7)


def counting_process(people=600, seed=1):
    """Hazard rises with a covariate that switches on at a random time."""
    rng = numpy.random.default_rng(seed)
    rows = []
    for person in range(people):
        switch = rng.exponential(4.0)
        wait_before = rng.exponential(1 / 0.05)
        if wait_before <= switch:
            end, event = min(wait_before, 12.0), wait_before <= 12.0
            rows.append((person, 0.0, end, 0.0, event))
            continue
        rows.append((person, 0.0, switch, 0.0, False))
        wait_after = rng.exponential(1 / (0.05 * numpy.exp(1.2)))
        end = switch + min(wait_after, 12.0)
        rows.append((person, switch, end, 1.0, wait_after <= 12.0))
    return pandas.DataFrame(
        rows, columns=["person", "start", "stop", "x", "event"]
    )


def test_time_varying_cox_recovers_the_effect_of_a_switching_covariate():
    frame = counting_process()

    model = timevarying.TimeVaryingCox(penalty=0.01).fit(
        frame[["x"]], frame.start, frame.stop, frame.event
    )

    assert model.coefficients_[0] == pytest.approx(1.2, abs=0.3)
    assert numpy.all(numpy.diff(model.baseline_) >= 0)


def test_time_varying_cox_refuses_bad_rows():
    frame = counting_process(50)
    with pytest.raises(ValueError, match="stop > start"):
        timevarying.TimeVaryingCox().fit(
            frame[["x"]], frame.stop, frame.start, frame.event
        )
    with pytest.raises(ValueError, match="no event"):
        timevarying.TimeVaryingCox().fit(
            frame[["x"]], frame.start, frame.stop, frame.event & False
        )
