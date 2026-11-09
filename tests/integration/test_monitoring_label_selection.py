"""Maturity and propensity weights in `ModelMonitor.performance`."""

import numpy
import pandas
import pytest
import support
from test_monitoring import make_frame

import scmle.monitoring.monitor


@pytest.fixture(scope="module")
def monitor(tmp_path_factory):
    directory = tmp_path_factory.mktemp("label_selection")
    run = support.make_project(
        make_frame(800, 0), "y", time_budget=300, output_dir=directory
    ).run()
    return scmle.monitoring.monitor.ModelMonitor(run.bundle)


def test_defaults_are_unchanged_and_maturity_drops_rows(monitor):
    data = make_frame(600, 4)
    outcomes = data["y"].astype(float)
    features = data.drop(columns="y")
    plain = monitor.performance(features, outcomes)
    mature = pandas.Series(numpy.arange(600) < 200)
    cut = monitor.performance(features, outcomes, maturity=mature)
    assert plain.labeled_rows == 600
    assert cut.labeled_rows == 200


def test_propensity_weights_change_a_selected_sample(monitor):
    data = make_frame(1500, 6)
    features = data.drop(columns="y")
    outcomes = data["y"].astype(float)
    # Labels were collected mostly for rows the model would score well on.
    chance = numpy.where(data["a"] > 0, 0.9, 0.2)
    seen = numpy.random.default_rng(0).random(len(data)) < chance
    partial = outcomes.where(seen)
    naive = monitor.performance(features, partial)
    weighted = monitor.performance(
        features, partial, propensity=pandas.Series(chance)
    )
    assert naive.labeled_rows == weighted.labeled_rows
    assert naive.metrics != weighted.metrics
    again = monitor.performance(
        features, partial, propensity=pandas.Series(chance)
    )
    assert again.metrics == weighted.metrics


def test_invalid_propensity_is_refused(monitor):
    data = make_frame(600, 4)
    with pytest.raises(ValueError, match="propensity"):
        monitor.performance(
            data.drop(columns="y"),
            data["y"].astype(float),
            propensity=pandas.Series(numpy.zeros(600)),
        )
