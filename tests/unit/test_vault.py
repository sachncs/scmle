"""Tests for the test-set vault."""

import os
import sys

import pandas
import pytest

from scmle.assurance.custody import vault
from scmle.contracts import errors


@pytest.fixture
def sealed():
    features = pandas.DataFrame({"a": [1.0, 2.0, 3.0]})
    target = pandas.Series([0, 1, 0])
    locked = vault.SealedTestSet(features, target, (10, 11, 12))
    yield locked
    locked.close()


def test_the_rows_are_released_once_to_the_token_holder(sealed):
    sealed.bind("b")
    features, target = sealed.open(sealed.mint(), "b")

    assert list(features["a"]) == [1.0, 2.0, 3.0] and list(target) == [0, 1, 0]
    assert sealed.opened is True
    with pytest.raises(errors.LeakageError, match="already used"):
        sealed.open(sealed.mint(), "b")


def test_a_wrong_token_is_refused_by_the_vault_process(sealed):
    with pytest.raises(errors.LeakageError, match="invalid token"):
        sealed.open("guess")

    assert sealed.opened is False
    sealed.bind("b")
    released = sealed.open(sealed.mint(), "b")
    assert len(released[0]) == 3


def test_this_process_keeps_only_the_ids(sealed):
    assert sealed.ids == (10, 11, 12) and len(sealed) == 3
    attributes = vars(sealed)
    assert not any(
        isinstance(value, pandas.DataFrame | pandas.Series)
        for value in attributes.values()
    )


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows handles are not file descriptors"
)
def test_the_vault_pipe_is_not_inherited_by_child_processes(sealed):
    assert os.get_inheritable(sealed.channel.fileno()) is False


def test_the_vault_process_ends_with_its_owner(sealed):
    sealed.channel.close()

    sealed.process.wait(timeout=15)

    assert sealed.process.returncode == 0


def test_a_vault_that_is_gone_is_reported_not_hung():
    locked = vault.SealedTestSet(
        pandas.DataFrame({"a": [1.0]}), pandas.Series([1]), (0,)
    )
    locked.process.kill()
    locked.process.wait()

    with pytest.raises(errors.LeakageError, match=r"gone|locked"):
        locked.open(locked.mint())


def test_a_bound_vault_refuses_a_release_for_another_selection(sealed):
    # Enforces website/src/pages/docs/threat-model.astro:32.
    sealed.bind("selection-a")
    token = sealed.mint()

    with pytest.raises(errors.LeakageError, match="binding mismatch"):
        sealed.open(token)
    with pytest.raises(errors.LeakageError, match="binding mismatch"):
        sealed.open(token, "selection-b")
    assert sealed.opened is False
    features, ignored = sealed.open(token, "selection-a")
    assert len(features) == 3 and sealed.opened is True


def test_a_vault_binds_once(sealed):
    sealed.bind("selection-a")

    with pytest.raises(errors.LeakageError, match="already bound"):
        sealed.bind("selection-b")


def test_an_unbound_vault_refuses(sealed):
    # GATE-3: an unbound vault used to release with the token alone.
    with pytest.raises(errors.LeakageError, match="not bound"):
        sealed.open(sealed.mint())
    sealed.bind("b")
    assert len(sealed.open(sealed.mint(), "b")[0]) == 3


def test_a_bound_vault_checks_the_token_before_the_binding(sealed):
    sealed.bind("selection-a")

    with pytest.raises(errors.LeakageError, match="invalid token"):
        sealed.open("guess", "selection-a")
