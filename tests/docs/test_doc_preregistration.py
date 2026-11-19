"""Guarantees of the pre-registrations that no other test enforced."""

import json
import pathlib

import numpy
import pandas
import pytest

from benchmarks import acceptance
from benchmarks.evidence import leakage
from scmle.assurance.admissibility import screen

ROOT = pathlib.Path(__file__).resolve().parents[2]
SYLVINE = "clean/openml_41146"


@pytest.fixture(scope="module")
def stored():
    """The stored C2 measurement.

    Returns:
        The parsed `benchmarks/evidence/leakage.json`.
    """
    path = ROOT / "benchmarks" / "evidence" / "leakage.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_closing_date_is_screened_with_the_declared_prediction_time():
    """Enforces docs/preregistration/c2.md:106-109."""
    planted = [
        case
        for case in leakage.catalog().values()
        if case.kind == "future_timestamp"
    ]
    assert planted
    assert {case.cutoff for case in planted} == {leakage.START.isoformat()}


def test_availability_is_declared_only_from_the_sources_documentation():
    """Enforces docs/preregistration/c2.md:110-114."""
    declared = {
        name: case.known
        for name, case in leakage.catalog().items()
        if case.known
    }
    assert declared == {
        "leak/bank_duration": ("V12",),
        "leak/taxi_total": ("total_amount",),
    }
    for name in ("titanic_boat", "titanic_body", "tecator_moisture"):
        assert leakage.catalog()[f"leak/{name}"].known == ()


def test_a_column_declared_outcome_blocks_whatever_its_statistics():
    """Enforces docs/preregistration/c2.md:119-121."""
    rng = numpy.random.default_rng(0)
    frame = pandas.DataFrame(
        {"age": rng.normal(size=300), "y": rng.integers(0, 2, 300)}
    )
    # Pure noise: no statistic ties it to the label.
    frame["duration"] = rng.exponential(size=300)
    names = ["age", "duration"]
    declared = screen.screen_table(
        frame, "y", "classification", names, 0, known={"duration": "outcome"}
    )
    assert [f.feature for f in declared.blocking] == ["duration"]
    plain = screen.screen_table(frame, "y", "classification", names, 0)
    assert plain.blocking == ()


def test_sylvine_still_blocks_and_nothing_declares_it_a_leak(stored):
    """Enforces docs/preregistration/c2.md:128-130."""
    (record,) = [c for c in stored["cases"] if c["name"] == SYLVINE]
    assert (record["corpus"], record["outcome"]) == ("clean", "blocked")
    assert record["blocking"] == ["V7"]
    case = leakage.catalog()[SYLVINE]
    assert case.table.drop == () and case.known == ()
    assert 41146 not in leakage.DOCUMENTED


def test_a_split_gate_refusal_is_a_false_block_counted_apart(stored):
    """Enforces docs/preregistration/c2.md:131-133."""
    refused = {"corpus": "clean", "outcome": "refused"}
    assert leakage.blocked(refused)
    outcomes = stored["false_blocks"]["outcomes"]
    assert stored["false_blocks"]["hits"] == (
        outcomes.get("blocked", 0) + outcomes.get("refused", 0)
    )


def test_holm_counts_the_c14_rows_as_not_rejected():
    """Enforces docs/preregistration/c2.md:136-138."""
    alpha = acceptance.load()["alpha"].value
    assert leakage.FAMILY == 4
    # The two C2 rows take the two smallest Holm levels; a C14 row entered
    # as rejected first could only have raised them.
    steps = leakage.holm([0.0, 0.0], alpha)
    assert [step[0] for step in steps] == [alpha / 4, alpha / 3]


def test_the_last_recorded_result_fails_both_rows(stored):
    """Enforces docs/preregistration/c2.md:139-141."""
    recall, blocks = stored["recall"], stored["false_blocks"]
    assert (recall["hits"], recall["trials"]) == (66, 70)
    assert round(recall["bound"], 3) == 0.847
    assert recall["threshold"] == 0.9
    assert (blocks["hits"], blocks["trials"]) == (7, 167)
    assert (blocks["outcomes"]["blocked"], blocks["outcomes"]["refused"]) == (
        2,
        5,
    )
    assert round(blocks["bound"], 3) == 0.088
    assert blocks["threshold"] == 0.05
    assert not recall["passed"] and not blocks["passed"]
