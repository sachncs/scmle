"""Never-list items 1 and 2: the checks that already existed stay in force."""

import pandas
import pytest

from scmle.contracts import errors
from scmle.governance import registry
from scmle.governance import vault


def test_the_champion_alias_does_not_move_without_a_signed_approval(tmp_path):
    store = registry.ModelRegistryStore(tmp_path / "registry")
    with pytest.raises(errors.ArtifactError):
        store.promote("model", 1)


def test_a_sealed_set_opens_once_even_with_the_right_token():
    locked = vault.SealedTestSet(
        pandas.DataFrame({"a": [1.0]}), pandas.Series([1]), (0,)
    )
    try:
        locked.open(locked.issue_token())
        with pytest.raises(errors.LeakageError, match="already used"):
            locked.open(locked.token)
    finally:
        locked.close()
