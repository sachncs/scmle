"""Exact evidence of the representativeness check, branch by branch."""

import numpy
import pandas

from scmle.data.profiling import representativeness

THRESHOLDS = {
    "minimum_rows_per_side": 50,
    "maximum_rows_per_side": 20000,
    "unrepresentative_auc": 0.65,
    "cross_validation_folds": 5,
}
METHOD = "out_of_fold_reference_classifier_and_column_gaps"
LIMITATIONS = [
    (
        "Classifier distinguishability is a shift signal, not proof of "
        "population bias."
    ),
    "Column gaps are univariate diagnostics and do not establish causality.",
    "Only the leading bounded rows from each side are checked.",
]


def bounds(
    sample: int, reference: int, checked: tuple[int, int], truncated: bool
) -> dict:
    return {
        "sample_rows": sample,
        "reference_rows": reference,
        "checked_sample_rows": checked[0],
        "checked_reference_rows": checked[1],
        "maximum_rows_per_side": 20000,
        "truncated": truncated,
    }


def test_no_shared_column_is_not_evaluated() -> None:
    found = representativeness.assess_representativeness(
        pandas.DataFrame({"a": [1.0, 2.0]}),
        pandas.DataFrame({"b": [1.0]}),
        seed=7,
    )
    assert found.note == (
        "representativeness: the reference shares no feature columns"
    )
    assert found.evidence == {
        "observation": "representativeness",
        "status": "not_evaluated",
        "method": METHOD,
        "thresholds": THRESHOLDS,
        "sample_bounds": bounds(2, 1, (0, 0), False),
        "seed": 7,
        "metrics": {"shared_columns": []},
        "limitations": [
            "No common feature columns were available for comparison."
        ],
    }


def test_too_few_rows_is_not_evaluated_with_the_error() -> None:
    found = representativeness.assess_representativeness(
        pandas.DataFrame({"a": [1.0, 2.0], "c": ["x", "y"]}),
        pandas.DataFrame({"a": [1.0, 2.0, 3.0]}),
        seed=3,
    )
    assert found.note is None
    assert found.evidence == {
        "observation": "representativeness",
        "status": "not_evaluated",
        "method": METHOD,
        "thresholds": THRESHOLDS,
        "sample_bounds": bounds(2, 3, (2, 3), False),
        "seed": 3,
        "metrics": {
            "shared_columns": ["a"],
            "error": "each side needs at least 50 rows",
        },
        "limitations": [
            (
                "The comparison was not evaluated because its data contract "
                "was not met."
            )
        ],
    }


def test_quality_evidence_of_a_flagged_comparison() -> None:
    gap = representativeness.Gap("a", 0.5, 0.01)
    report = representativeness.Representativeness(0.9, False, (gap,))
    found = report.quality_evidence(100, 80, 100, 60, 5, ("a", "b"))
    assert found == {
        "observation": "representativeness",
        "status": "flagged",
        "method": METHOD,
        "thresholds": THRESHOLDS,
        "sample_bounds": bounds(100, 80, (100, 60), True),
        "seed": 5,
        "metrics": {
            "auc": 0.9,
            "representative": False,
            "shared_columns": ["a", "b"],
            "column_gaps": [{"column": "a", "statistic": 0.5, "p_value": 0.01}],
        },
        "limitations": LIMITATIONS,
    }


def test_quality_evidence_of_a_clear_comparison() -> None:
    report = representativeness.Representativeness(0.5, True, ())
    found = report.quality_evidence(10, 10, 10, 10, 0, ())
    assert found["status"] == "clear"
    assert found["sample_bounds"]["truncated"] is False


def test_distinguishable_rows_name_the_largest_gap() -> None:
    rng = numpy.random.default_rng(0)
    sample = pandas.DataFrame(
        {"a": rng.normal(size=60), "b": rng.normal(size=60)}
    )
    reference = pandas.DataFrame(
        {"a": rng.normal(size=60) + 5, "b": rng.normal(size=60)}
    )
    found = representativeness.assess_representativeness(
        sample, reference, seed=2
    )
    assert found.note is not None
    assert found.note.endswith("the largest gap is in 'a'")
    evidence = found.evidence
    assert evidence["seed"] == 2
    assert evidence["metrics"]["shared_columns"] == ["a", "b"]
    assert evidence["sample_bounds"] == bounds(60, 60, (60, 60), False)
