"""Keyed random streams: reproducible, independent, order-free."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy
import pytest

from scmle.contracts import qualification
from scmle.core import randomness

ROOT = Path(__file__).parents[2]


def draw(rng, size=1000):
    return rng.random(size)


def test_same_key_gives_a_bit_identical_stream():
    first = draw(randomness.component_rng(7, "missingness", 0))
    second = draw(randomness.component_rng(7, "missingness", 0))
    assert first.tobytes() == second.tobytes()


def test_adding_a_component_never_changes_another_stream():
    alone = draw(randomness.component_rng(7, "alpha", 0))
    randomness.component_rng(7, "beta", 0).random(50)
    after_another = draw(randomness.component_rng(7, "alpha", 0))
    assert alone.tobytes() == after_another.tobytes()


def test_request_order_does_not_matter():
    forward = [
        draw(randomness.component_rng(3, name)).tobytes()
        for name in ("a", "b", "c")
    ]
    backward = [
        draw(randomness.component_rng(3, name)).tobytes()
        for name in ("c", "b", "a")
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
    base = draw(randomness.component_rng(7, "alpha", 0))
    changed = draw(randomness.component_rng(**other))
    assert base.tobytes() != changed.tobytes()


def test_distinct_components_are_uncorrelated():
    size = 20000
    left = randomness.component_rng(11, "alpha").standard_normal(size)
    right = randomness.component_rng(11, "beta").standard_normal(size)
    # Under independence |r| has sd 1/sqrt(n) ~ 0.007; 0.05 is ~7 sd.
    assert abs(float(numpy.corrcoef(left, right)[0, 1])) < 0.05


def test_component_names_are_hashed_stably_not_with_python_hash():
    first = randomness.component_seed_sequence(1, "x").generate_state(2)
    assert (
        first.tolist()
        == randomness.component_seed_sequence(1, "x").generate_state(2).tolist()
    )
    assert randomness.name_words("x") == randomness.name_words("x")
    assert randomness.name_words("x") != randomness.name_words("y")


def test_seed_path_records_the_key():
    assert randomness.seed_path(7, "alpha", 2) == (7, "alpha", 2)


def test_integer_seed_is_deterministic_and_in_range():
    value = randomness.derive_int_seed(7, "alpha", 0)
    assert value == randomness.derive_int_seed(7, "alpha", 0)
    assert 0 <= value < 2**32
    assert value != randomness.derive_int_seed(7, "beta", 0)


@pytest.mark.parametrize(
    "arguments",
    [(-1, "a", 0), (True, "a", 0), (1, "  ", 0), (1, "a", -1), (1, "a", 1.5)],
)
def test_invalid_keys_are_rejected(arguments):
    with pytest.raises(ValueError):
        randomness.component_rng(*arguments)


SCRIPT = """
import json
from scmle.contracts import qualification as q
from scmle.core import randomness

checks = []
for index, name in enumerate(("alpha", "beta", "gamma")):
    rng = randomness.component_rng(42, name, index)
    checks.append(
        q.Check(
            check_id=name,
            null_hypothesis="x is independent of y",
            statistic=float(rng.standard_normal()),
            p_value=float(rng.random()),
            interval=None,
            effect=float(rng.random()),
            n=100,
            decision=q.Decision.PASS,
            set_id=q.SetId.EXPLORATION,
            seed_path=randomness.seed_path(42, name, index),
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
    assert [record["check_id"] for record in records] == [
        "alpha",
        "beta",
        "gamma",
    ]
    assert all(qualification.Check.from_dict(record) for record in records)
