"""The declared grid: enumeration, subsets, caching and the time budget."""

import json

import pytest

from scmle.contracts import qualification
from scmle.data.charter import contracts
from scmle.data.robustness import multiverse
from tests.unit.test_charter_contracts import make_charter


def choices():
    return (
        contracts.Choice("imputation", ("median", "mean", "zero"), "median"),
        contracts.Choice("outlier_rule", ("none", "winsorize"), "none"),
    )


def test_the_full_grid_is_the_product_with_the_default_first():
    grid = multiverse.enumerate_grid(choices(), seed=0, max_specs=100)
    assert grid.total == 6 and len(grid.specs) == 6 and not grid.subset
    assert grid.default.to_dict() == {
        "imputation": "median",
        "outlier_rule": "none",
    }
    assert len({s.sha256 for s in grid.specs}) == 6


def test_a_large_grid_becomes_a_seeded_subset_of_fixed_size():
    first = multiverse.enumerate_grid(choices(), seed=1, max_specs=3)
    again = multiverse.enumerate_grid(choices(), seed=1, max_specs=3)
    others = [
        multiverse.enumerate_grid(choices(), seed=k, max_specs=3)
        for k in range(2, 8)
    ]
    assert first == again
    assert first.subset and len(first.specs) == 3 and first.total == 6
    assert first.specs[0] == first.default
    mine = {s.sha256 for s in first.specs}
    assert any(mine != {s.sha256 for s in o.specs} for o in others)


def test_enumeration_is_independent_of_the_order_of_the_declaration():
    forward = multiverse.enumerate_grid(choices(), 0)
    backward = multiverse.enumerate_grid(tuple(reversed(choices())), 0)
    assert forward == backward


def test_repeated_names_and_an_empty_budget_are_refused():
    with pytest.raises(ValueError, match="distinct"):
        multiverse.enumerate_grid(choices() + choices()[:1], 0)
    with pytest.raises(ValueError, match="max_specs"):
        multiverse.enumerate_grid(choices(), 0, max_specs=0)


def test_no_choices_give_a_single_default_specification():
    grid = multiverse.enumerate_grid((), 0)
    assert grid.total == 1 and grid.default.values == ()


def test_a_specification_added_later_changes_both_hashes_and_the_tier():
    base = make_charter()
    amended = make_charter(
        multiverse=(
            contracts.Choice(
                "imputation", ("median", "mice", "mean"), "median"
            ),
        )
    )
    assert base.charter_hash != amended.charter_hash
    declared = multiverse.declaration_sha256(base.multiverse)
    used = multiverse.declaration_sha256(amended.multiverse)
    assert declared != used
    keep = qualification.Tier.ATTESTED
    assert multiverse.tier_after_declaration(declared, declared, keep) is keep
    assert (
        multiverse.tier_after_declaration(declared, used, keep)
        is qualification.Tier.UNVERIFIED
    )


def test_the_declaration_hash_ignores_the_order_of_choices():
    assert multiverse.declaration_sha256(
        choices()
    ) == multiverse.declaration_sha256(tuple(reversed(choices())))


def test_results_are_cached_by_finding_specification_and_seed(tmp_path):
    grid = multiverse.enumerate_grid(choices(), 0)
    cache = multiverse.ResultCache()
    calls = []

    def evaluate(spec):
        calls.append(spec.sha256)
        return {"effect": 0.5}

    first = multiverse.run_grid(
        grid, evaluate, finding="a", seed=0, cache=cache
    )
    assert len(calls) == 6 and first.cached == 0
    second = multiverse.run_grid(
        grid, evaluate, finding="a", seed=0, cache=cache
    )
    assert len(calls) == 6 and second.cached == 6
    multiverse.run_grid(grid, evaluate, finding="b", seed=0, cache=cache)
    multiverse.run_grid(grid, evaluate, finding="a", seed=1, cache=cache)
    assert len(calls) == 18
    path = tmp_path / "cache.json"
    cache.save(path)
    assert json.loads(path.read_text())
    resumed = multiverse.ResultCache.load(path)
    assert resumed.items == cache.items
    assert multiverse.ResultCache.load(tmp_path / "missing.json").items == {}


def test_the_time_budget_skips_and_records_the_rest():
    grid = multiverse.enumerate_grid(choices(), 0)
    ticks = iter(range(100))
    run = multiverse.run_grid(
        grid,
        lambda spec: {"effect": 1.0},
        finding="a",
        seed=0,
        budget_seconds=2.5,
        clock=lambda: float(next(ticks)),
    )
    assert 0 < len(run.results) < 6
    assert len(run.results) + len(run.skipped) == 6
    assert all(item[1] == "time budget spent" for item in run.skipped)


def test_the_run_is_deterministic_under_a_fixed_seed():
    grid = multiverse.enumerate_grid(choices(), 3)
    a = multiverse.run_grid(grid, lambda s: {"v": 1.0}, finding="x", seed=3)
    b = multiverse.run_grid(grid, lambda s: {"v": 1.0}, finding="x", seed=3)
    assert a.results == b.results and a.skipped == ()
