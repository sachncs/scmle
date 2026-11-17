"""Claim C5 evidence: awkward files and point-in-time features."""

import json

import numpy
import pandas
import pytest

from benchmarks.evidence import reading
from scmle.contracts import errors

TOL = 1e-8


def document(wrong: int = 0, unequal: int = 0, worlds: int = 1000) -> dict:
    """A current document with the given failures."""
    cases = reading.corpus()
    files = [
        {"name": c.name, "outcome": "correct", "expects_table": True}
        for c in cases
    ]
    return {
        "code": reading.digest(),
        "tolerance": reading.tolerance(),
        "files": files,
        "corpus": {
            "cases": len(files),
            "correct": len(files) - wrong,
            "refused": 0,
            "over_refused": 0,
            "wrong": wrong,
        },
        "point_in_time": {"worlds": worlds, "equal": worlds - unequal},
        "real_datasets": 0,
    }


def test_the_tolerance_comes_from_acceptance():
    assert reading.tolerance() == TOL


def test_a_clean_simulated_run_is_not_a_pass_without_real_data():
    text, verdict = reading.summary(document())
    assert reading.problems(document()) == []
    assert verdict == "not measured"
    assert "simulated only" in text


@pytest.mark.parametrize(("wrong", "unequal"), [(1, 0), (0, 1)])
def test_one_silent_error_or_mismatch_fails(wrong, unequal):
    assert reading.summary(document(wrong, unequal))[1] == "fail"


def test_too_few_worlds_is_not_measured():
    found = document(worlds=999)
    assert any("fewer than the 1000" in i for i in reading.problems(found))
    assert reading.summary(found)[1] == "not measured"


def test_stale_code_or_missing_cases_is_not_measured():
    stale = document() | {"code": "0" * 64}
    assert reading.problems(stale) == [
        "reading: measured with other code or corpus"
    ]
    short = document()
    short["files"] = short["files"][1:]
    assert any("cases not read" in i for i in reading.problems(short))
    assert reading.summary(stale)[1] == "not measured"


def test_an_empty_document_is_not_measured():
    assert reading.summary({}) == ("no measurement", "not measured")


def test_cells_must_carry_the_value_without_loss():
    big = 2**53 + 1
    assert reading.matches(big, big, TOL)
    assert not reading.matches(float(big), big, TOL)
    assert reading.matches(str(big), big, TOL)
    assert not reading.matches(2134, "02134", TOL)
    assert reading.matches(numpy.nan, None, TOL)
    assert reading.matches(pandas.NA, None, TOL)
    assert not reading.matches(pandas.NA, 1, TOL)
    assert not reading.matches(True, 1, TOL)
    assert reading.matches(1.0 + 1e-12, 1.0, TOL)
    date = reading.Date("03/01/2024", "2024-03-01")
    assert reading.matches("03/01/2024", date, TOL)
    assert reading.matches(pandas.Timestamp("2024-03-01"), date, TOL)
    assert not reading.matches(pandas.Timestamp("2024-01-03"), date, TOL)


def test_judge_catches_renamed_columns_and_a_table_for_a_refusal():
    case = reading.text("d", "g", "", ("a", "b"), ((1, 2),))
    assert (
        reading.judge(case, pandas.DataFrame({"a": [1], "b": [2]}), TOL) == ""
    )
    renamed = pandas.DataFrame([[1, 2]], columns=["a", "a.1"])
    assert "columns" in reading.judge(case, renamed, TOL)
    refusal = reading.text("r", "g", "", None)
    assert "refusal" in reading.judge(refusal, renamed, TOL)


def test_read_classifies_the_product_outcome(tmp_path):
    good = reading.text("ok", "g", "a\n1\n", ("a",), ((1,),))
    assert reading.read(good, tmp_path, TOL)["outcome"] == "correct"
    empty = reading.text("none", "g", "", None)
    assert reading.read(empty, tmp_path, TOL)["outcome"] == "refused"


def test_one_world_agrees_with_the_naive_reference():
    assert reading.world(0, TOL)["equal"]


def test_bad_arguments_are_refused():
    with pytest.raises(errors.InputError):
        reading.measure(worlds=0)


def test_the_stored_reading_is_for_the_current_code():
    stored = json.loads(reading.PATH.read_text(encoding="utf-8"))
    assert stored["code"] == reading.digest()
    assert reading.problems() == []
    assert reading.summary()[1] in {"fail", "not measured"}
