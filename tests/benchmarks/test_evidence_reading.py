"""Claim C5 evidence: awkward files and point-in-time features."""

import json

import numpy
import pandas
import pytest

from benchmarks.evidence import reading
from benchmarks.evidence import relational
from scmle.contracts import errors

TOL = 1e-8


def real(names=None, bad=()) -> list[dict]:
    """Real-dataset results; the names in `bad` disagree."""
    found = []
    for name in names or [d.name for d in relational.DATASETS]:
        equal = 9 if name in bad else 10
        found.append(
            {"name": name, "features": 10, "equal": equal, "error": ""}
        )
    return found


def splits(crossing: str = "", error: str = "") -> list[dict]:
    """Split records for every table; the first crosses `crossing`."""
    found = []
    for index, (table, declared) in enumerate(reading.tables()):
        found.append(
            {
                "table": table,
                "declared": declared,
                "strategy": "random",
                "splits": 6,
                "entity": 0,
                "duplicate": 0,
                "copy": 0,
                "time": 0,
                "refused": "",
                "error": "",
                "entities": "known",
                "grouped": False,
            }
        )
        if index == 0 and crossing:
            found[0][crossing] = 1
        if index == 0 and error:
            found[0]["error"] = error
    return found


def document(
    wrong: int = 0,
    unequal: int = 0,
    worlds: int = 1000,
    results=None,
    hygiene=None,
) -> dict:
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
        "real": real() if results is None else results,
        "splits": splits() if hygiene is None else hygiene,
    }


def test_the_tolerance_comes_from_acceptance():
    assert reading.tolerance() == TOL


def test_a_clean_run_with_clean_splits_passes():
    text, verdict = reading.summary(document())
    assert reading.problems(document()) == []
    assert verdict == "pass"
    assert "real relational datasets 5 of 5 equal" in text
    assert "0 cross" in text


def test_a_run_without_the_split_measurement_is_not_measured():
    found = document()
    del found["splits"]
    assert any("split" in i for i in reading.problems(found))
    assert reading.summary(found)[1] == "not measured"


@pytest.mark.parametrize("crossing", ["entity", "copy", "time"])
def test_one_split_with_an_entity_copy_or_timestamp_across_fails(crossing):
    # Enforces docs/preregistration/c5.md:63.
    text, verdict = reading.summary(document(hygiene=splits(crossing)))
    assert verdict == "fail"
    assert f"{crossing} 1" in text


def test_a_duplicate_crossing_alone_is_reported_and_passes():
    # Enforces docs/preregistration/c5.md:65.
    hygiene = splits("duplicate")
    text, verdict = reading.summary(document(hygiene=hygiene))
    assert not reading.crossed(hygiene[0])
    assert verdict == "pass"
    assert "0 cross" in text
    assert "1 runs share a duplicate record (reported" in text


def test_a_duplicate_of_one_declared_entity_fails_where_it_may_not_cross():
    # Enforces docs/preregistration/c5.md:64.
    cases = [(s, "known") for s in ("random", "group")]
    cases += [(s, "unseen") for s in ("temporal", "rolling")]
    for strategy, entities in cases:
        hygiene = splits("copy")
        hygiene[0] |= {
            "strategy": strategy,
            "duplicate": 1,
            "grouped": True,
            "entities": entities,
        }
        assert reading.crossed(hygiene[0])
        assert reading.summary(document(hygiene=hygiene))[1] == "fail"


def test_a_duplicate_of_one_known_entity_may_cross_a_forecast_split():
    for strategy in ("temporal", "rolling"):
        hygiene = splits("copy")
        hygiene[0] |= {"strategy": strategy, "duplicate": 1, "grouped": True}
        assert not reading.crossed(hygiene[0])
        assert reading.summary(document(hygiene=hygiene))[1] == "pass"


def test_a_record_measured_before_copies_were_counted_has_none():
    record = splits()[0]
    del record["copy"]
    assert not reading.breached(record, "copy")


@pytest.mark.parametrize(
    ("strategy", "entities", "violates"),
    [
        ("temporal", "known", False),
        ("rolling", "known", False),
        ("temporal", "unseen", True),
        ("rolling", "unseen", True),
        ("group", "known", True),
    ],
)
def test_an_entity_crossing_violates_only_where_entities_must_be_unseen(
    strategy, entities, violates
):
    hygiene = splits("entity")
    hygiene[0] |= {"strategy": strategy, "entities": entities, "grouped": True}
    text, verdict = reading.summary(document(hygiene=hygiene))
    assert reading.crossed(hygiene[0]) == violates
    assert verdict == ("fail" if violates else "pass")
    assert f"entity {int(violates)}" in text


def test_the_summary_counts_the_runs_declared_each_way():
    hygiene = splits()
    hygiene[0] |= {"entities": "unseen", "grouped": True}
    hygiene[1] |= {"grouped": True}
    hygiene[2] |= {"grouped": True}
    text = reading.summary(document(hygiene=hygiene))[0]
    assert "entity declared on 3 runs: 2 known, 1 unseen" in text


def test_a_table_that_could_not_be_split_fails():
    found = document(hygiene=splits(error="OSError: no cache"))
    assert reading.summary(found)[1] == "fail"


def test_a_missing_table_is_not_measured():
    found = document(hygiene=splits()[1:])
    assert any("tables not split" in i for i in reading.problems(found))


def test_the_split_tables_are_the_relational_and_the_refusal_tables():
    names = [pair[0] for pair in reading.tables()]
    for dataset in relational.DATASETS:
        for declared in reading.DECLARATIONS:
            assert (f"{dataset.name}/{declared}", declared) in (
                reading.tables()
            )
    assert any(name.startswith("refusal/") for name in names)


def test_the_relational_tables_are_split_declared_known_and_unseen():
    dataset = relational.DATASETS[0]
    for declared in ("both", "rolling"):
        assert "entities" not in reading.declare(dataset, declared)
    for declared in ("unseen", "unseen_rolling"):
        options = reading.declare(dataset, declared)
        assert options["entities"] == "unseen"
        assert options["group_column"] and options["time_column"]
    assert reading.declare(dataset, "unseen_rolling")["validation"] == "rolling"


def test_hygiene_counts_what_crosses_a_product_split():
    rng = numpy.random.default_rng(0)
    rows = 300
    frame = pandas.DataFrame(
        {
            "who": rng.integers(0, 30, rows),
            "when": pandas.Timestamp("2024-01-01")
            + pandas.to_timedelta(rng.integers(0, 40, rows), unit="D"),
            "x": rng.integers(0, 3, rows).astype(float),
            "y": rng.normal(size=rows),
        }
    )
    plain = reading.hygiene(frame, "y", {})
    assert plain["strategy"] == "random" and plain["splits"] == 6
    # 3 feature values: duplicates fall on both sides of a random split;
    # entity and time are not declared, so they are not checked.
    assert plain["duplicate"] > 0
    assert plain["entity"] == plain["time"] == plain["copy"] == 0
    assert not reading.crossed(plain)  # identical rows are independent
    grouped = reading.hygiene(frame, "y", {"group_column": "who"})
    assert grouped["strategy"] == "group" and grouped["entity"] == 0
    assert grouped["copy"] == 0  # a group split keeps every entity apart
    timed = reading.hygiene(frame, "y", {"time_column": "when"})
    assert timed["strategy"] == "temporal" and timed["time"] == 0
    both = reading.hygiene(
        frame, "y", {"time_column": "when", "group_column": "who"}
    )
    assert both["entity"] > 0  # an entity's past and future straddle
    # 3 feature values over 30 entities: one entity's copies straddle too.
    assert both["copy"] > 0 and both["entities"] == "known" and both["grouped"]
    assert not reading.breached(both, "entity")  # known entities may cross
    assert not reading.breached(both, "copy")  # and so may their copies
    unseen = reading.hygiene(
        frame,
        "y",
        {"time_column": "when", "group_column": "who", "entities": "unseen"},
    )
    assert unseen["entities"] == "unseen" and unseen["entity"] == 0
    assert unseen["strategy"] == "temporal" and unseen["splits"] > 1
    refused = reading.hygiene(
        frame.assign(when="x"), "y", {"time_column": "when"}
    )
    assert refused["refused"] and refused["splits"] == 0


def test_one_real_dataset_that_disagrees_fails():
    # Enforces docs/preregistration/c5.md:61.
    found = document(results=real(bad=("sakila",)))
    text, verdict = reading.summary(found)
    assert verdict == "fail"
    assert "4 of 5" in text


def test_a_real_dataset_that_could_not_be_read_fails():
    results = real()
    results[0] |= {"error": "OSError: SHA-256 differs", "equal": 0}
    assert reading.summary(document(results=results))[1] == "fail"


def test_missing_real_datasets_are_not_measured():
    found = document(results=real()[1:])
    assert any("real datasets not run" in i for i in reading.problems(found))
    assert reading.summary(found)[1] == "not measured"


@pytest.mark.parametrize(("wrong", "unequal"), [(1, 0), (0, 1)])
def test_one_silent_error_or_mismatch_fails(wrong, unequal):
    # Enforces docs/preregistration/c5.md:61.
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
    # Enforces docs/preregistration/c5.md:61.
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
