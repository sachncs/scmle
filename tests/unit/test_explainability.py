"""Tests for the explanation layer."""

import dataclasses
import time

import numpy as np
import pandas as pd
import test_diagnostics

from mml.contracts import tasks
from mml.evaluation import diagnostics
from mml.evaluation import suite as full_suite
from mml.explainability import dependence
from mml.explainability import importance
from mml.explainability import local


def test_occlusion_attributes_the_prediction_to_the_feature_that_drives_it():
    rng = np.random.default_rng(0)
    rows = pd.DataFrame(
        {"signal": rng.normal(size=20), "noise": rng.normal(size=20)}
    )
    background = pd.DataFrame(
        {"signal": rng.normal(size=50), "noise": rng.normal(size=50)}
    )
    explainer = local.OcclusionExplainer(
        lambda frame: 3 * frame["signal"].to_numpy(), ["signal", "noise"]
    )

    table = explainer.explain(rows, background)

    assert np.allclose(table["noise"], 0.0)
    expected = 3 * (rows["signal"] - background["signal"].mean())
    assert np.allclose(table["signal"], expected)
    assert np.allclose(table["prediction"], 3 * rows["signal"])


def test_partial_dependence_follows_the_driving_feature():
    inputs = test_diagnostics.make_inputs(tasks.TaskType.REGRESSION)

    report = dependence.PartialDependence().compute(inputs)["features"]

    curve = report["signal"]["mean_prediction"]
    assert curve == sorted(curve) and curve[-1] - curve[0] > 3.0
    flat = report["noise"]["mean_prediction"]
    assert max(flat) - min(flat) < 1.0


def test_partial_dependence_for_binary_is_a_probability_curve():
    inputs = test_diagnostics.make_inputs(tasks.TaskType.BINARY)

    report = dependence.PartialDependence().compute(inputs)["features"]

    curve = report["signal"]["mean_prediction"]
    assert all(0.0 <= value <= 1.0 for value in curve)
    assert curve[-1] > curve[0] + 0.3


def test_importance_stops_at_its_deadline_and_says_it_was_partial():
    inputs = test_diagnostics.make_inputs(tasks.TaskType.BINARY)
    expired = dataclasses.replace(inputs, deadline=time.monotonic() - 1.0)

    report = importance.PermutationImportance().compute(expired)

    assert report["truncated"] is True and report["features_measured"] == 0
    full = importance.PermutationImportance().compute(inputs)
    assert full["truncated"] is False
    assert full["features_measured"] == full["features_total"]


def test_importance_is_deterministic_and_ranks_the_signal_first():
    inputs = test_diagnostics.make_inputs(tasks.TaskType.REGRESSION)

    first = importance.PermutationImportance().compute(inputs)
    second = importance.PermutationImportance().compute(inputs)

    assert first == second and first["features"][0]["feature"] == "signal"


def test_a_suite_shares_its_time_allowance_among_diagnostics():
    inputs = test_diagnostics.make_inputs(tasks.TaskType.BINARY)

    reports = diagnostics.run_suite(
        inputs, full_suite.full_suite(), total_seconds=0.0
    )

    assert reports["permutation_importance"]["truncated"] is True
