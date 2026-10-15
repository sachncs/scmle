"""Tests for the optimizers."""

import pytest

from mml.contracts import errors
from mml.contracts import search
from mml.optimization import optimizers

SPACE = search.SearchSpace(
    (
        search.FloatParameter("x", 0.0, 1.0),
        search.IntParameter("n", 1, 10),
        search.ChoiceParameter("c", ("a", "b", "c")),
    )
)


def drive(optimizer, rounds=12):
    proposals = []
    for params in (optimizer.ask() for round_number in range(rounds)):
        proposals.append(params)
        optimizer.tell(params, params["x"] + params["n"] / 10)
    return proposals


def assert_valid(params):
    assert 0.0 <= params["x"] <= 1.0
    assert 1 <= params["n"] <= 10
    assert params["c"] in ("a", "b", "c")


def test_random_optimizer_is_seeded_and_valid():
    first = drive(optimizers.RandomOptimizer(SPACE, 1))
    second = drive(optimizers.RandomOptimizer(SPACE, 1))
    other = drive(optimizers.RandomOptimizer(SPACE, 2))
    assert first == second and first != other
    assert all(assert_valid(p) is None for p in first)


def test_random_optimizer_records_history():
    optimizer = optimizers.RandomOptimizer(SPACE, 0)
    params = optimizer.ask()
    optimizer.tell(params, 0.5)
    assert optimizer.history == [(params, 0.5)]


@pytest.mark.parametrize("greater", [True, False])
def test_optuna_optimizer_proposes_valid_configurations(greater):
    pytest.importorskip("optuna")
    optimizer = optimizers.OptunaOptimizer(SPACE, 0, greater)
    proposals = drive(optimizer)
    assert all(assert_valid(p) is None for p in proposals)
    assert optimizer.pending == []
    assert len(optimizer.study.trials) == len(proposals)


def test_optuna_is_deterministic_per_seed():
    pytest.importorskip("optuna")
    first = drive(optimizers.OptunaOptimizer(SPACE, 7))
    second = drive(optimizers.OptunaOptimizer(SPACE, 7))
    assert first == second


def test_optuna_exploits_a_clear_signal():
    pytest.importorskip("optuna")
    space = search.SearchSpace((search.FloatParameter("x", 0.0, 1.0),))
    optimizer = optimizers.OptunaOptimizer(space, 0, greater_is_better=True)
    proposals = (optimizer.ask() for round_number in range(40))
    for params in proposals:
        optimizer.tell(params, -((params["x"] - 0.8) ** 2))
    best = max(t.value for t in optimizer.study.trials)
    assert best > -0.01


def test_factory_and_errors(monkeypatch):
    assert isinstance(
        optimizers.create_optimizer("random", SPACE, 0, True),
        optimizers.RandomOptimizer,
    )
    with pytest.raises(errors.ConfigError, match="unknown optimizer"):
        optimizers.create_optimizer("nope", SPACE, 0, True)
    monkeypatch.setattr(
        optimizers.OptunaOptimizer, "requires", ("not_installed_pkg",)
    )
    with pytest.raises(errors.ConfigError, match="optimization"):
        optimizers.OptunaOptimizer(SPACE, 0)


def test_unsupported_parameter_type_is_rejected():
    class Odd(search.Parameter):
        def sample(self, rng):
            return 1

    with pytest.raises(errors.ConfigError, match="unsupported"):
        optimizers.suggest(Odd("odd"), object())
