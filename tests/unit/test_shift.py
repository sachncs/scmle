"""Tests for adversarial validation."""

import dataclasses

import numpy as np
import test_diagnostics

from scml.contracts import tasks
from scml.evaluation import shift


def shifted_inputs(offset):
    base = test_diagnostics.make_inputs(tasks.TaskType.BINARY, rows=1200)
    reference = base.features.copy()
    moved = base.features.copy()
    moved["signal"] = moved["signal"] + offset
    return dataclasses.replace(
        base, features=moved, reference_features=reference
    )


def test_identical_distributions_are_indistinguishable():
    base = test_diagnostics.make_inputs(tasks.TaskType.BINARY, rows=1200)
    rng = np.random.default_rng(5)
    order = rng.permutation(len(base.features))
    half = len(order) // 2
    data = dataclasses.replace(
        base,
        features=base.features.iloc[order[:half]].reset_index(drop=True),
        reference_features=base.features.iloc[order[half:]].reset_index(
            drop=True
        ),
    )

    report = shift.AdversarialValidation().compute(data)

    assert report["auc"] < 0.6 and report["verdict"] == "no meaningful shift"


def test_a_moved_feature_is_detected_and_named():
    report = shift.AdversarialValidation().compute(shifted_inputs(2.0))

    assert report["auc"] > 0.75 and report["verdict"] == "strong shift"
    assert report["most_shifted"][0]["feature"] == "signal"


def test_category_frequency_changes_are_ranked_too():
    base = test_diagnostics.make_inputs(tasks.TaskType.BINARY, rows=1200)
    moved = base.features.copy()
    moved["group"] = "a"
    data = dataclasses.replace(
        base, features=moved, reference_features=base.features
    )

    report = shift.AdversarialValidation().compute(data)

    assert report["most_shifted"][0]["feature"] == "group"
    assert report["auc"] > 0.6


def test_without_reference_rows_the_report_says_so():
    base = test_diagnostics.make_inputs(tasks.TaskType.BINARY)

    assert shift.AdversarialValidation().compute(base) == {
        "available": False,
        "reason": "no reference rows",
    }
