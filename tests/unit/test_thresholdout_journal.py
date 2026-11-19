"""A reuse journal of other rows blocks the opening, not just its claim.

`opening.evaluate` turns a crashing claim runner into `failed` claims, but a
`LeakageError` is a gate's refusal (`confirm.reuse_journal`: a budget is
never reset by starting over), so it is logged and propagated, as a
`LeakageError` from opening `C` is.
"""

import pandas
import pytest

from scmle.assurance.ledger import confirm
from scmle.assurance.ledger.confirm import opening
from scmle.assurance.ledger.confirm import records
from scmle.assurance.ledger.confirm import thresholdout
from scmle.contracts import errors
from tests.unit import test_thresholdout


def test_a_foreign_journal_blocks_the_confirmation(tmp_path):
    ledger, x, y = test_thresholdout.explored()
    rows, labels = test_thresholdout.world(20000, 2)
    journal = tmp_path / "journal.json"
    other = pandas.Series(-labels, name="y")
    thresholdout.Thresholdout((x, y), (rows, other), journal=journal).query(
        lambda f, t: (f["n0"].to_numpy() * t.to_numpy() > 0).astype(float)
    )
    source = test_thresholdout.locker(rows, labels)
    session = thresholdout.Session(
        (x, y), test_thresholdout.above, journal=journal
    )
    with pytest.raises(errors.LeakageError, match="other rows or terms"):
        confirm.confirm(
            ledger,
            source,
            alpha_confirm=test_thresholdout.ALPHA,
            policy=confirm.THRESHOLDOUT,
            reuse=session,
        )
    assert [e["event"] for e in source.log.entries][-1] == "claims_failed"


def test_evaluate_logs_and_propagates_a_refusal():
    def refused(features, target):
        raise errors.LeakageError("refused")

    runner = records.Docket(
        (records.Commitment("a", "reuse", "s", "0" * 64),), refused
    )
    log = test_thresholdout.sealed.Journal()
    with pytest.raises(errors.LeakageError, match="refused"):
        opening.evaluate(runner, (pandas.DataFrame(), pandas.Series()), log)
    assert log.entries[-1]["event"] == "claims_failed"
