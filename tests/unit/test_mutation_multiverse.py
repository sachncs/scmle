"""The multiverse sweep's clock and the cache's stable file."""

import json

from scmle.assurance.charter import contracts
from scmle.assurance.robustness import multiverse


def grid() -> multiverse.Grid:
    return multiverse.universe(
        (
            contracts.Choice("imputation", ("median", "mean"), "median"),
            contracts.Choice("outlier_rule", ("none", "winsorize"), "none"),
        ),
        0,
    )


def test_the_sweep_reports_the_seconds_between_its_clock_readings() -> None:
    ticks = iter([10.0, 17.5])
    run = multiverse.sweep(
        grid(),
        lambda spec: {"effect": 1.0},
        finding="a",
        seed=0,
        clock=lambda: next(ticks),
    )
    assert run.seconds == 7.5


def test_the_cache_file_lists_keys_in_sorted_order(tmp_path) -> None:
    cache = multiverse.Memo({"b": {"y": 2.0, "x": 1.0}, "a": {"z": 3.0}})
    path = tmp_path / "cache.json"
    cache.save(path)
    expected = {"a": {"z": 3.0}, "b": {"x": 1.0, "y": 2.0}}
    assert path.read_text(encoding="utf-8") == (
        json.dumps(expected, indent=2) + "\n"
    )
