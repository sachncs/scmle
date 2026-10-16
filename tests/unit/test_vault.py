"""Tests for the test-set vault."""

import os

import pandas as pd
import pytest

from mml.contracts import errors
from mml.governance import vault


@pytest.fixture
def sealed():
    features = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    target = pd.Series([0, 1, 0])
    locked = vault.SealedTestSet(features, target, (10, 11, 12))
    yield locked
    locked.close()


def test_the_rows_are_released_once_to_the_token_holder(sealed):
    features, target = sealed.open(sealed.issue_token())

    assert list(features["a"]) == [1.0, 2.0, 3.0] and list(target) == [0, 1, 0]
    assert sealed.opened is True
    with pytest.raises(errors.LeakageError, match="already used"):
        sealed.open(sealed.issue_token())


def test_a_wrong_token_is_refused_by_the_vault_process(sealed):
    with pytest.raises(errors.LeakageError, match="invalid token"):
        sealed.open("guess")

    assert sealed.opened is False
    released = sealed.open(sealed.issue_token())
    assert len(released[0]) == 3


def test_this_process_keeps_only_the_ids(sealed):
    assert sealed.ids == (10, 11, 12) and len(sealed) == 3
    attributes = vars(sealed)
    assert not any(
        isinstance(value, pd.DataFrame | pd.Series)
        for value in attributes.values()
    )


def test_the_vault_pipe_is_not_inherited_by_child_processes(sealed):
    assert os.get_inheritable(sealed.channel.fileno()) is False


def test_the_vault_process_ends_with_its_owner(sealed):
    sealed.channel.close()

    sealed.process.wait(timeout=15)

    assert sealed.process.returncode == 0


def test_a_vault_that_is_gone_is_reported_not_hung():
    locked = vault.SealedTestSet(
        pd.DataFrame({"a": [1.0]}), pd.Series([1]), (0,)
    )
    locked.process.kill()
    locked.process.wait()

    with pytest.raises(errors.LeakageError, match=r"gone|locked"):
        locked.open(locked.issue_token())
