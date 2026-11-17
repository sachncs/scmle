"""The metrics spool's batch-size bound, at and below its edge."""

import pytest

from scmle.contracts import errors
from scmle.operations import spool


def send(samples) -> int:
    return len(samples)


def test_the_default_batch_holds_one_hundred(tmp_path) -> None:
    assert spool.MetricsSpool(tmp_path / "s.jsonl", send).batch_size == 100


def test_a_batch_of_one_is_allowed(tmp_path) -> None:
    assert spool.MetricsSpool(tmp_path / "s.jsonl", send, 1).batch_size == 1


def test_an_empty_batch_is_refused_by_name(tmp_path) -> None:
    with pytest.raises(
        errors.InputError, match=r"^batch_size must be positive$"
    ):
        spool.MetricsSpool(tmp_path / "s.jsonl", send, 0)
