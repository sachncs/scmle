"""Keyed random streams: reproducible, independent, order-free."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy
import pytest

from scmle.contracts import qualification
from scmle.runtime import randomness

ROOT = Path(__file__).parents[2]


def draw(rng, size=1000):
    return rng.random(size)


def test_same_key_gives_a_bit_identical_stream():
    first = draw(randomness.spawn(7, "missingness", 0))
    second = draw(randomness.spawn(7, "missingness", 0))
    assert first.tobytes() == second.tobytes()


def test_adding_a_component_never_changes_another_stream():
    alone = draw(randomness.spawn(7, "alpha", 0))
    randomness.spawn(7, "beta", 0).random(50)
    after_another = draw(randomness.spawn(7, "alpha", 0))
    assert alone.tobytes() == after_another.tobytes()


def test_request_order_does_not_matter():
    forward = [
        draw(randomness.spawn(3, name)).tobytes() for name in ("a", "b", "c")
    ]
    backward = [
        draw(randomness.spawn(3, name)).tobytes() for name in ("c", "b", "a")
    ][::-1]
    assert forward == backward


@pytest.mark.parametrize(
    "other",
    [
        {"run_seed": 8, "component": "alpha", "index": 0},
        {"run_seed": 7, "component": "beta", "index": 0},
        {"run_seed": 7, "component": "alpha", "index": 1},
    ],
)
def test_any_key_change_changes_the_stream(other):
    base = draw(randomness.spawn(7, "alpha", 0))
    changed = draw(randomness.spawn(**other))
    assert base.tobytes() != changed.tobytes()


def test_distinct_components_are_uncorrelated():
    size = 20000
    left = randomness.spawn(11, "alpha").standard_normal(size)
    right = randomness.spawn(11, "beta").standard_normal(size)
    # Under independence |r| has sd 1/sqrt(n) ~ 0.007; 0.05 is ~7 sd.
    assert abs(float(numpy.corrcoef(left, right)[0, 1])) < 0.05


def test_component_names_are_hashed_stably_not_with_python_hash():
    first = randomness.sequence(1, "x").generate_state(2)
    assert (
        first.tolist() == randomness.sequence(1, "x").generate_state(2).tolist()
    )
    assert randomness.derive("x") == randomness.derive("x")
    assert randomness.derive("x") != randomness.derive("y")


def test_seed_path_records_the_key():
    assert randomness.seed(7, "alpha", 2) == (7, "alpha", 2)


def test_integer_seed_is_deterministic_and_in_range():
    value = randomness.narrow(7, "alpha", 0)
    assert value == randomness.narrow(7, "alpha", 0)
    assert 0 <= value < 2**32
    assert value != randomness.narrow(7, "beta", 0)


@pytest.mark.parametrize(
    "arguments",
    [(-1, "a", 0), (True, "a", 0), (1, "  ", 0), (1, "a", -1), (1, "a", 1.5)],
)
def test_invalid_keys_are_rejected(arguments):
    with pytest.raises(ValueError):
        randomness.spawn(*arguments)


SCRIPT = """
import json
from scmle.contracts import qualification as q
from scmle.runtime import randomness

checks = []
for index, name in enumerate(("alpha", "beta", "gamma")):
    rng = randomness.spawn(42, name, index)
    checks.append(
        q.Check(
            identifier=name,
            null="x is independent of y",
            statistic=float(rng.standard_normal()),
            p_value=float(rng.random()),
            interval=None,
            effect=float(rng.random()),
            n=100,
            decision=q.Decision.PASS,
            subset=q.Subset.EXPLORATION,
            seeds=randomness.seed(42, name, index),
            tier=q.Tier.UNVERIFIED,
        ).to_dict()
    )
print(json.dumps(checks, sort_keys=True))
"""


def run_in_new_process(hash_seed):
    completed = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
    )
    return completed.stdout.strip()


def test_check_lists_are_bit_identical_across_processes():
    first = run_in_new_process("1")
    second = run_in_new_process("999")
    assert first == second
    records = json.loads(first)
    assert [record["identifier"] for record in records] == [
        "alpha",
        "beta",
        "gamma",
    ]
    assert all(qualification.Check.from_dict(record) for record in records)
