"""An unsigned custody log says so, naming where it lives."""

import logging

from scmle.assurance.custody import sealed
from scmle.contracts import signing


def test_an_unsigned_in_memory_log_warns_by_name(monkeypatch, caplog) -> None:
    monkeypatch.delenv(signing.SIGNING_KEY_ENV, raising=False)
    with caplog.at_level(logging.WARNING):
        sealed.Journal()
    assert [r.getMessage() for r in caplog.records] == [
        (
            "WARNING custody log (in memory) is not signed: set "
            f"{signing.SIGNING_KEY_ENV} to MAC it"
        )
    ]
