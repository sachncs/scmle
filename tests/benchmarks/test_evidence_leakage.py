"""Claim C2 evidence: leaks caught, clean real tables not blocked."""

import json
from typing import Any

import numpy
import pandas
import pytest

from benchmarks import acceptance
from benchmarks.evidence import leakage
from scmle.contracts import errors


def record(corpus: str, outcome: str, kind: str = "") -> dict[str, Any]:
    """One synthetic case record."""
    provenance = leakage.PLANTED if corpus == "leak" else "clean real data"
    return {
        "name": f"{corpus}/{outcome}/{kind}",
        "corpus": corpus,
        "kind": kind,
        "provenance": provenance,
        "outcome": outcome,
    }


def document(
    caught: int, leaks: int, blocks: int, cleans: int
) -> dict[str, Any]:
    """A current document with the given counts."""
    records = [record("leak", "blocked", "post_outcome")] * caught
    records += [record("leak", "passed", "duplicates")] * (leaks - caught)
    records += [record("clean", "blocked")] * blocks
    records += [record("clean", "passed")] * (cleans - blocks)
    recall, blocks = leakage.decide(records)
    return {
        "code": leakage.digest(),
        "recall": recall,
        "false_blocks": blocks,
        "kinds": leakage.breakdown(records),
        "cases": records,
    }


def test_the_minimum_samples_come_from_acceptance() -> None:
    # Enforces docs/preregistration/c2.md:60-61 (first Holm step).
    variables = acceptance.load()
    alpha = variables["alpha"].value
    recall = variables[leakage.RECALL].rule
    blocks = variables[leakage.BLOCKS].rule
    assert recall is not None and blocks is not None
    assert recall.minimum(alpha / leakage.FAMILY) == 42
    assert blocks.minimum(alpha / leakage.FAMILY) == 86
    cases = leakage.catalog().values()
    assert sum(c.corpus == "leak" for c in cases) >= 42
    assert document(0, 0, 0, 0)["false_blocks"]["minimum"] == 86


def test_enough_catches_and_no_blocks_pass() -> None:
    found = document(60, 60, 0, 90)
    assert leakage.problems(found) == []
    text, verdict = leakage.summary(found)
    assert verdict == "pass"
    assert "planted on real data" in text


def test_too_many_false_blocks_fail() -> None:
    # Enforces docs/preregistration/c2.md:46.
    assert leakage.summary(document(60, 60, 10, 90))[1] == "fail"


def test_low_recall_fails() -> None:
    # Enforces docs/preregistration/c2.md:46.
    assert leakage.summary(document(30, 60, 0, 90))[1] == "fail"


def test_a_pass_needs_the_minimum_sample() -> None:
    found = document(20, 20, 0, 90)
    assert any("none can pass" in p for p in leakage.problems(found))
    assert leakage.summary(found)[1] == "not measured"


def test_stale_code_is_not_measured() -> None:
    found = document(60, 60, 0, 90)
    found["code"] = "0" * 64
    assert any("other code" in p for p in leakage.problems(found))
    assert leakage.summary(found)[1] == "not measured"


def test_a_clean_refusal_crash_or_missing_table_counts_against_the_row() -> (
    None
):
    # Enforces docs/preregistration/c2.md:42.
    # Enforces docs/preregistration/c2.md:77-78.
    records = [record("clean", "refused"), record("clean", "failed")]
    records += [record("clean", "passed"), record("clean", "missing")]
    row = leakage.tally(records, "clean", leakage.blocked)
    assert (row["hits"], row["trials"]) == (3, 4)
    missed = leakage.tally([record("leak", "missing")], "leak", leakage.caught)
    assert (missed["hits"], missed["trials"]) == (0, 1)


def test_holm_steps_down_over_the_four_stage_d_rows() -> None:
    # Enforces docs/preregistration/c2.md:46-50.
    assert leakage.holm([0.001, 0.01], 0.05) == [
        (0.0125, True),
        (0.05 / 3, True),
    ]
    assert leakage.holm([0.02, 0.001], 0.05) == [
        (0.05 / 3, True),
        (0.0125, True),
    ]
    assert leakage.holm([0.02, 0.5], 0.05) == [
        (0.0125, True),
        (0.05 / 3, False),
    ]
    # The second row passes at its own level only once the first passed.
    passing = document(60, 60, 0, 90)
    assert passing["recall"]["level"] == 0.0125
    assert passing["false_blocks"]["level"] == pytest.approx(0.05 / 3)
    # Too few catches fail recall; the clean row, first by p-value, passes.
    mixed = document(30, 60, 0, 90)
    assert not mixed["recall"]["passed"] and mixed["false_blocks"]["passed"]


def test_each_planted_kind_adds_the_leak_it_names() -> None:
    rng = numpy.random.default_rng(0)
    frame = pandas.DataFrame(
        {"x": rng.normal(size=200), "y": rng.integers(0, 2, 200)}
    )
    table = leakage.Table("toy", "toy", "y", leakage.CLASSIFY)
    encoded = leakage.plant(frame, table, "target_encoding", 0)
    assert encoded["level_target_mean"].nunique() > 2
    recorded = leakage.plant(frame, table, "post_outcome", 0)
    agree = (recorded["outcome_recorded"] == frame["y"]).mean()
    assert 1 - leakage.GARBLE <= agree < 1
    doubled = leakage.plant(frame, table, "duplicates", 0)
    assert len(doubled) == round(200 * (1 + leakage.COPIES))
    dated = leakage.plant(frame, table, "future_timestamp", 0)
    late = dated.groupby("y")["closed_at"].mean()
    assert late[1] > late[0]
    with pytest.raises(errors.InputError):
        leakage.plant(frame, table, "unknown", 0)


def test_a_planted_classification_leak_needs_a_binary_target() -> None:
    frame = pandas.DataFrame({"x": [1.0, 2.0, 3.0], "y": ["a", "b", "c"]})
    table = leakage.Table("toy", "toy", "y", leakage.CLASSIFY)
    with pytest.raises(errors.InputError):
        leakage.score(frame, table)


def test_the_stored_measurement_is_current() -> None:
    stored = json.loads(leakage.PATH.read_text(encoding="utf-8"))
    # At the first Holm step a ceiling of 0.05 needs 86 clean tables; the
    # corpus has fewer (docs/preregistration/c2.md, "Where the current
    # module differs"), which is the only problem allowed.
    issues = leakage.problems(stored)
    assert all("none can pass" in p for p in issues), issues
    expected = ("not measured",) if issues else ("pass", "fail")
    assert leakage.summary(stored)[1] in expected


def test_the_clean_corpus_reaches_the_first_holm_step_minimum() -> None:
    # Enforces docs/preregistration/c2.md:60-61 (86 clean tables at alpha/4).
    variables = acceptance.load()
    blocks = variables[leakage.BLOCKS].rule
    assert blocks is not None
    least = blocks.minimum(variables["alpha"].value / leakage.FAMILY)
    cases = leakage.catalog().values()
    assert sum(c.corpus == "clean" for c in cases) >= least


def test_the_added_clean_tables_follow_the_stated_rule() -> None:
    # Every C14 real table not already here, not a copy of one here, with
    # its C14 task: the rule of the module docstring, fixed before screening.
    from benchmarks.evidence import refusal

    tables = {t.source: t for t in leakage.clean()}
    for source, entry in {**refusal.EXTRA, **refusal.SUITE}.items():
        target, task = entry[2], entry[3]
        if source in leakage.SAME:
            assert source not in tables
            continue
        assert source in tables, source
        assert (tables[source].target, tables[source].task) == (target, task)


def test_a_planted_closing_date_is_screened_with_the_users_cutoff() -> None:
    for case in leakage.catalog().values():
        if case.kind == "future_timestamp":
            assert case.cutoff == leakage.START.isoformat()
        else:
            assert case.cutoff == ""
