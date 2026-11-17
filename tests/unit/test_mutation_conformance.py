"""The adapter conformance kit: its toy problems and the seeds it builds with."""

import types

import numpy
import pytest
from sklearn import dummy

from scmle.contracts import models
from scmle.contracts import search
from scmle.contracts import tasks
from scmle.models import conformance


class Recorder:
    """An adapter stub that builds a dummy regressor and records each seed.

    Attributes:
        seeds: The seed of every `build` call.
    """

    def __init__(self) -> None:
        self.seeds: list[int] = []

    def build(self, task, seed, params=None):
        """Records the seed and returns a dummy regressor."""
        self.seeds.append(seed)
        return dummy.DummyRegressor(**(params or {}))


@pytest.mark.parametrize("task", list(tasks.Task))
def test_toy_has_two_varying_features(task) -> None:
    features, target = conformance.toy(task)
    assert list(features.columns) == ["a", "b"]
    assert features["b"].nunique() == conformance.ROWS
    assert len(target) == conformance.ROWS


def test_toy_regression_follows_its_feature() -> None:
    features, target = conformance.toy(tasks.Task.REGRESSION)
    assert numpy.corrcoef(features["a"], target)[0, 1] > 0.99


def test_toy_ranking_is_noisy_in_its_feature() -> None:
    features, target = conformance.toy(tasks.Task.RANKING)
    ordered = target[features["a"].argsort()].to_numpy()
    assert (numpy.diff(ordered) < 0).any()
    assert set(target) == {0.0, 1.0, 2.0}


def test_smoke_builds_every_check_with_seed_zero() -> None:
    recorder = Recorder()
    adapter = types.SimpleNamespace(
        name="recorder",
        capabilities=models.Capabilities(
            frozenset({tasks.Task.REGRESSION}),
            handles_missing=True,
            sparse_input=True,
        ),
        conformance_params=None,
        build=recorder.build,
    )
    features, target = conformance.toy(tasks.Task.REGRESSION)
    conformance.smoke(adapter, tasks.Task.REGRESSION, features, target)
    assert recorder.seeds == [0, 0, 0]


def test_searchable_builds_with_seed_zero() -> None:
    recorder = Recorder()
    adapter = types.SimpleNamespace(
        name="recorder",
        search_space=search.Space(
            (search.Choice("strategy", ("mean", "median")),)
        ),
        build=recorder.build,
    )
    features, target = conformance.toy(tasks.Task.REGRESSION)
    conformance.searchable(adapter, tasks.Task.REGRESSION, features, target)
    assert recorder.seeds == [0]
