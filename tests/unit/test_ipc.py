"""The helper-process channel works on every platform."""

import pandas
import pytest

from scmle.core import ipc
from scmle.governance import vault


def test_vault_releases_rows_once_over_the_channel():
    features = pandas.DataFrame({"a": [1, 2, 3]})
    target = pandas.Series([0, 1, 0])
    sealed = vault.SealedTestSet(features, target, [10, 11, 12])
    rows = sealed.open(sealed.issue_token())
    assert rows[0].equals(features)
    assert rows[1].equals(target)


def test_a_helper_that_exits_before_connecting_is_reported():
    with pytest.raises(ipc.StartError):
        ipc.start_helper("scmle.nonexistent_module")
