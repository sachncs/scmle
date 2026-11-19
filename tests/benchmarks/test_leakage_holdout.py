"""The C2 corpora read no copy of the sealed TabArena hold-out."""

from benchmarks.evidence import leakage
from benchmarks.evidence import refusal

HOLDOUT = {31, 37, 1111, 4135, 4541}


def test_no_clean_table_is_a_copy_of_the_sealed_holdout():
    sources = {t.source for t in leakage.clean()}
    assert not sources & set(refusal.SEALED)
    assert not sources & HOLDOUT


def test_no_leak_case_is_planted_on_the_sealed_holdout():
    cases = leakage.catalog().values()
    assert not {c.table.source for c in cases} & set(refusal.SEALED)
    names = {t.name for t in leakage.clean()}
    assert set(leakage.BASES) <= names
    assert len(leakage.BASES) == len(set(leakage.BASES)) == 16
    planted = [c for c in cases if c.provenance == leakage.PLANTED]
    assert len(planted) == len(leakage.BASES) * len(leakage.KINDS)
