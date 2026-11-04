"""The oracle comparison table covers every oracle and is not stale."""

import pathlib

from benchmarks.qualification import oracle_table
from benchmarks.qualification import oracles

TABLE = pathlib.Path(__file__).resolve().parents[2] / "docs" / "ORACLES.md"


def test_every_oracle_has_a_row_with_a_producer_or_an_explicit_gap():
    found = {row["oracle"]: row for row in oracle_table.rows()}
    assert set(found) == set(oracles.NAMES)
    for name, row in found.items():
        if name in oracle_table.WIRED:
            assert row["producer"] and row["within"] is True
        else:
            assert row["producer"] is None and row["difference"] is None


def test_every_wired_producer_names_an_oracle_that_exists():
    assert set(oracle_table.WIRED) <= set(oracles.NAMES)


def test_the_committed_table_equals_a_fresh_run():
    assert oracle_table.main([str(TABLE), "--check"]) == 0


def test_a_missing_or_edited_table_fails_the_check(tmp_path):
    assert oracle_table.main([str(tmp_path / "none.md"), "--check"]) == 1
    edited = tmp_path / "t.md"
    edited.write_text(TABLE.read_text().replace("| yes |", "| NO |", 1))
    assert oracle_table.main([str(edited), "--check"]) == 1


def test_the_difference_bound_is_a_stable_power_of_ten():
    assert oracle_table.bound(0.0) == "<= 1e-16"
    assert oracle_table.bound(3e-13) == "<= 1e-12"
    assert oracle_table.bound(1e-12) == "<= 1e-12"
    assert oracle_table.gap({"a": [1.0, 2.0]}, {"a": [1.0, 2.0]}) == 0.0
    assert oracle_table.gap(1.5, 1.0) == 0.5
    assert oracle_table.gap(0.25, 0.0) == 0.25
