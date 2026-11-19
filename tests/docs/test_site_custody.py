"""Custody statements of the site that no other test enforced."""

import pandas

from scmle.assurance.custody import sealed
from scmle.assurance.ledger import confirm
from tests.leakage.test_custody_access import build
from tests.leakage.test_custody_access import SENTINEL
from tests.unit.test_ledger_confirm import ALPHA
from tests.unit.test_ledger_confirm import c_rows
from tests.unit.test_ledger_confirm import explored
from tests.unit.test_ledger_confirm import source_with


def test_the_callers_frame_and_file_keep_every_row_after_sealing(tmp_path):
    """The vault seals `C` but cannot take back the caller's copies.

    Sealing moves the confirmation rows into the vault process; the frame
    the caller passed in, and the file it was read from, still hold every
    row, including the sealed ones.

    Also enforces website/src/pages/docs/architecture.astro:104.
    Also enforces website/src/pages/docs/threat-model.astro:58.
    """
    built = build()
    path = tmp_path / "rows.csv"
    built.frame.to_csv(path, index=False)
    sealed = list(built.vault.sets["C"].ids)
    assert sealed
    assert len(built.frame) == 2000
    assert built.frame[SENTINEL].iloc[sealed].notna().all()
    again = pandas.read_csv(path)
    assert len(again) == 2000
    assert (
        again[SENTINEL].iloc[sealed] == built.frame[SENTINEL].iloc[sealed]
    ).all()
    built.vault.close()


def test_the_confirmation_stage_binds_the_release_before_it_opens_c():
    """The stage commits `C` to the frozen claims and ledger head first.

    Also enforces website/src/pages/docs/threat-model.astro:32.
    Also enforces docs/SECURITY.md:71.
    """
    ledger = explored()
    source = source_with(*c_rows())
    confirm.confirm(ledger, source, alpha_confirm=ALPHA)
    assert source.committed
    events = [entry["event"] for entry in source.log.entries]
    assert events.index("bound") < events.index("open")
    bound = source.log.entries[events.index("bound")]["detail"]
    assert source.binding == sealed.digest(
        bound["claims_hash"], bound["ledger_hash"]
    )
