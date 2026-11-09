"""Tests for greedy blending."""

import numpy
import pytest

from scmle.contracts import tasks
from scmle.models import conformance
from scmle.models import ensembles
import scmle.models.registry


def test_the_blender_puts_its_weight_on_the_member_that_predicts_best():
    rng = numpy.random.default_rng(0)
    truth = rng.normal(size=500)
    good = truth + rng.normal(0, 0.1, 500)
    bad = rng.normal(size=500)
    meta = numpy.column_stack([bad, good, bad])

    blender = ensembles.RegressorBlender(members=3).fit(meta, truth)

    assert blender.weights_.argmax() == 1 and blender.weights_[1] > 0.8
    assert blender.weights_.sum() == pytest.approx(1.0)
    assert numpy.mean((blender.predict(meta) - truth) ** 2) < 0.05


def test_binary_probabilities_are_blended_and_never_leave_zero_one():
    rng = numpy.random.default_rng(1)
    labels = rng.integers(0, 2, 600)
    sharp = numpy.clip(labels * 0.8 + 0.1 + rng.normal(0, 0.05, 600), 0, 1)
    noise = rng.random(600)
    meta = numpy.column_stack([noise, sharp])

    blender = ensembles.ClassifierBlender(members=2).fit(meta, labels)
    proba = blender.predict_proba(meta)

    assert proba.shape == (600, 2)
    numpy.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert blender.weights_[1] > blender.weights_[0]
    assert (blender.predict(meta) == labels).mean() > 0.95


def test_multiclass_members_contribute_one_column_per_class():
    rng = numpy.random.default_rng(2)
    labels = rng.integers(0, 3, 400)
    confident = numpy.eye(3)[labels] * 0.8 + 0.2 / 3
    uniform = numpy.full((400, 3), 1 / 3)
    meta = numpy.hstack([uniform, confident])

    blender = ensembles.ClassifierBlender(members=2).fit(meta, labels)

    assert blender.weights_[1] > 0.7
    assert (blender.predict(meta) == labels).all()
    assert list(blender.classes_) == [0, 1, 2]


def test_the_blending_adapter_builds_working_ensembles_and_passes_conformance():
    registry = scmle.models.registry.ModelRegistry.with_builtins()
    adapter = registry.get("blending")

    conformance.check_conformance(adapter)

    assert adapter.name == "blending" and adapter.family == "ensemble"
    estimator = adapter.build(
        tasks.TaskType.BINARY,
        0,
        {"members": [{"model": "decision_tree"}, {"model": "extra_trees"}]},
    )
    assert type(estimator.final_estimator).__name__ == "ClassifierBlender"
