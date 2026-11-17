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
    return {
        "code": leakage.digest(),
        "recall": leakage.tally(records, "leak", leakage.caught),
        "false_blocks": leakage.tally(records, "clean", leakage.blocked),
        "kinds": leakage.breakdown(records),
        "cases": records,
    }


def test_the_minimum_samples_come_from_acceptance() -> None:
    variables = acceptance.load()
    alpha = variables["alpha"].value
    recall = variables[leakage.RECALL].rule
    blocks = variables[leakage.BLOCKS].rule
    assert recall is not None and blocks is not None
    assert recall.minimum(alpha) == 29
    assert blocks.minimum(alpha) == 59
    cases = leakage.catalog().values()
    assert sum(c.corpus == "leak" for c in cases) >= 29
    assert sum(c.corpus == "clean" for c in cases) >= 59


def test_enough_catches_and_no_blocks_pass() -> None:
    found = document(60, 60, 0, 80)
    assert leakage.problems(found) == []
    text, verdict = leakage.summary(found)
    assert verdict == "pass"
    assert "planted on real data" in text


def test_too_many_false_blocks_fail() -> None:
    assert leakage.summary(document(60, 60, 10, 80))[1] == "fail"


def test_low_recall_fails() -> None:
    assert leakage.summary(document(30, 60, 0, 80))[1] == "fail"


def test_a_pass_needs_the_minimum_sample() -> None:
    found = document(20, 20, 0, 80)
    assert any("none can pass" in p for p in leakage.problems(found))
    assert leakage.summary(found)[1] == "not measured"


def test_stale_code_and_missing_data_are_not_measured() -> None:
    found = document(60, 60, 0, 80)
    found["code"] = "0" * 64
    found["cases"].append(record("clean", "missing"))
    issues = leakage.problems(found)
    assert any("other code" in p for p in issues)
    assert any("data missing" in p for p in issues)
    assert leakage.summary(found)[1] == "not measured"


def test_a_clean_refusal_or_crash_counts_against_the_row() -> None:
    records = [record("clean", "refused"), record("clean", "failed")]
    records += [record("clean", "passed"), record("clean", "missing")]
    row = leakage.tally(records, "clean", leakage.blocked)
    assert (row["hits"], row["trials"]) == (2, 3)


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
    assert leakage.problems(stored) == []
    assert leakage.summary(stored)[1] in ("pass", "fail")
