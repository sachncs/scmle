"""Tests for adversarial validation."""

import dataclasses

import numpy
import pandas
import test_diagnostics

from scmle.contracts import tasks
from scmle.evaluation import diagnostics
from scmle.evaluation import shift


def shifted_inputs(offset):
    base = test_diagnostics.make_inputs(tasks.Task.BINARY, rows=1200)
    reference = base.features.copy()
    moved = base.features.copy()
    moved["signal"] = moved["signal"] + offset
    return dataclasses.replace(
        base, features=moved, reference_features=reference
    )


def test_identical_distributions_are_indistinguishable():
    base = test_diagnostics.make_inputs(tasks.Task.BINARY, rows=1200)
    rng = numpy.random.default_rng(5)
    order = rng.permutation(len(base.features))
    half = len(order) // 2
    data = dataclasses.replace(
        base,
        features=base.features.iloc[order[:half]].reset_index(drop=True),
        reference_features=base.features.iloc[order[half:]].reset_index(
            drop=True
        ),
    )

    report = shift.Adversary().compute(data)

    assert report["auc"] < 0.6 and report["verdict"] == "no meaningful shift"


def test_a_moved_feature_is_detected_and_named():
    report = shift.Adversary().compute(shifted_inputs(2.0))

    assert report["auc"] > 0.75 and report["verdict"] == "strong shift"
    assert report["most_shifted"][0]["feature"] == "signal"


def test_category_frequency_changes_are_ranked_too():
    base = test_diagnostics.make_inputs(tasks.Task.BINARY, rows=1200)
    moved = base.features.copy()
    moved["group"] = "a"
    data = dataclasses.replace(
        base, features=moved, reference_features=base.features
    )

    report = shift.Adversary().compute(data)

    assert report["most_shifted"][0]["feature"] == "group"
    assert report["auc"] > 0.6


def test_without_reference_rows_the_report_says_so():
    base = test_diagnostics.make_inputs(tasks.Task.BINARY)

    assert shift.Adversary().compute(base) == {
        "available": False,
        "reason": "no reference rows",
    }


def test_the_report_gives_the_clipped_odds_the_weights_would_use():
    report = shift.Adversary().compute(shifted_inputs(2.0))

    odds = report["odds"]
    assert odds["clip"] == 20.0
    assert odds["size_correction"] == 1.0
    reference, evaluation = odds["reference"], odds["evaluation"]
    assert 0 < reference["effective_size"] <= report["rows"]["reference"]
    assert evaluation["mean"] > reference["mean"]
    assert reference["max"] <= odds["clip"]
    assert 0.0 <= reference["clipped_share"] <= 1.0
    assert odds["weighted_conformal"] in {"possible", "refused"}
    assert "out-of-fold" in odds["note"]


def test_a_strong_shift_leaves_too_few_effective_calibration_rows():
    strong = shift.Adversary().compute(shifted_inputs(4.0))
    none = no_shift()

    assert (
        strong["odds"]["reference"]["effective_size"]
        < (none["odds"]["reference"]["effective_size"])
    )


def test_without_a_shift_the_odds_stay_moderate_and_weights_are_possible():
    report = no_shift()
    odds = report["odds"]
    assert odds["reference"]["mean"] < 3.0
    assert odds["weighted_conformal"] == "possible"


def no_shift():
    base = test_diagnostics.make_inputs(tasks.Task.BINARY, rows=1200)
    rng = numpy.random.default_rng(5)
    order = rng.permutation(len(base.features))
    half = len(order) // 2
    data = dataclasses.replace(
        base,
        features=base.features.iloc[order[:half]].reset_index(drop=True),
        reference_features=base.features.iloc[order[half:]].reset_index(
            drop=True
        ),
    )
    return shift.Adversary().compute(data)


def test_adversary_is_skipped_once_its_deadline_has_passed():
    data = dataclasses.replace(
        shifted_inputs(0.0), deadline=diagnostics.clock() - 1.0
    )

    report = shift.Adversary().compute(data)

    assert report["skipped"] and report["available"] is False


def test_adversary_stops_between_folds_when_time_runs_out(monkeypatch):
    clock = test_diagnostics.Clock()
    monkeypatch.setattr(diagnostics, "clock", clock)
    fits = []
    original = shift.Adversary.fold

    def counted(self, *args):
        fits.append(1)
        clock.now += 100.0
        return original(self, *args)

    monkeypatch.setattr(shift.Adversary, "fold", counted)
    data = dataclasses.replace(shifted_inputs(1.0), deadline=clock.now + 150)

    report = shift.Adversary().compute(data)

    assert len(fits) == 1
    assert report["skipped"] and "fold" in report["reason"]


def test_adversary_cost_grows_with_rows_and_columns():
    small = shifted_inputs(0.0)
    adversary = shift.Adversary()
    more = dataclasses.replace(
        small,
        reference_features=pandas.concat([small.reference_features] * 2),
    )
    narrow = dataclasses.replace(
        small,
        feature_plan=dataclasses.replace(small.feature_plan, categorical=()),
    )

    assert adversary.estimate(more) > adversary.estimate(small)
    assert adversary.estimate(small) > adversary.estimate(narrow) > 0
