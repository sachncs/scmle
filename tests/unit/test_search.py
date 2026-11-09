"""Tests for search-space contracts and tunable adapters."""

import random

import pytest

from scmle.contracts import errors
from scmle.contracts import search
from scmle.contracts import tasks
from scmle.models import conformance
import scmle.models.registry


def test_parameters_sample_inside_their_domains():
    rng = random.Random(0)
    floats = search.FloatParameter("x", 0.1, 10.0, log=True)
    ints = search.IntParameter("n", 2, 9)
    logs = search.IntParameter("m", 1, 100, log=True)
    choices = search.ChoiceParameter("c", ("a", "b"))
    rounds = range(200)
    assert all(0.1 <= floats.sample(rng) <= 10.0 for index in rounds)
    assert all(2 <= ints.sample(rng) <= 9 for index in rounds)
    assert all(1 <= logs.sample(rng) <= 100 for index in rounds)
    assert all(choices.sample(rng) in ("a", "b") for index in rounds)


def test_log_sampling_covers_orders_of_magnitude():
    rng = random.Random(1)
    draws = [
        search.FloatParameter("x", 1e-3, 1e3, log=True).sample(rng)
        for index in range(400)
    ]
    assert sum(d < 1 for d in draws) > 100 and sum(d > 1 for d in draws) > 100


def test_space_sampling_is_deterministic_per_seed():
    space = search.SearchSpace(
        (search.FloatParameter("x", 0, 1), search.IntParameter("n", 1, 5))
    )
    assert space.sample(random.Random(3)) == space.sample(random.Random(3))
    assert space.names() == ("x", "n")


def test_parameters_are_immutable_and_comparable():
    param = search.IntParameter("n", 1, 3)
    with pytest.raises(AttributeError):
        param.low = 2
    assert param == search.IntParameter("n", 1, 3)


def test_adapters_apply_sampled_params():
    registry = scmle.models.registry.Catalog.with_builtins()
    forest = registry.get("random_forest")
    sampled = forest.search_space.sample(random.Random(0))
    estimator = forest.build(tasks.Task.BINARY, 0, sampled)
    for key, value in sampled.items():
        assert estimator.get_params()[key] == value
    assert registry.get("dummy").search_space is None


def test_every_tunable_builtin_samples_valid_configurations():
    registry = scmle.models.registry.Catalog.with_builtins()
    for adapter in registry.adapters.values():
        if not adapter.is_available():
            continue  # Optional library not installed in this environment.
        conformance.check_conformance(adapter)
        assert (
            adapter.search_space is None
            or len(adapter.search_space.names()) > 0
        )


def test_conformance_rejects_ignored_search_parameter():
    from sklearn import dummy

    from scmle.contracts import models

    class Ignoring(models.Model):
        name = "ignoring"
        family = "x"
        capabilities = models.Capabilities(frozenset({tasks.Task.BINARY}))
        search_space = search.SearchSpace(
            (search.FloatParameter("alpha", 0.1, 1.0),)
        )

        def build(self, task, seed, params=None):
            return dummy.DummyClassifier(strategy="prior")

    with pytest.raises(errors.ConfigError, match="not applied"):
        conformance.check_conformance(Ignoring())


def test_search_spec_defaults_and_hash_sensitivity():
    base = tasks.Config(data=tasks.Dataset(uri="x", target="y"))
    off = tasks.Config(
        data=tasks.Dataset(uri="x", target="y"),
        search=tasks.Search(top_models=0),
    )
    assert base.search.top_models == 2 and base.search.optimizer == "random"
    assert base.config_hash() != off.config_hash()
