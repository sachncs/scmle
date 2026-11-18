"""The coverage claim's documents, key for key, and its floor."""

import numpy
import pandas

from scmle.assurance import coverage_confirm
from scmle.assurance.ledger import confirm
from tests.unit.test_coverage_confirm import Band

CHOICE = {"method": "residual", "level": 0.9, "tolerance": 0.1, "alpha": 0.05}


def released() -> tuple[pandas.DataFrame, pandas.Series]:
    rng = numpy.random.default_rng(0)
    features = pandas.DataFrame({"x": rng.normal(size=50)})
    return features, features["x"] + rng.uniform(-1.0, 1.0, 50)


def test_the_registration_names_the_floor_below_the_level() -> None:
    found = coverage_confirm.registration("a" * 64, CHOICE)
    assert found.statement == (
        "the residual interval or set at level 0.9 covers at least 0.8 of "
        "the rows of C"
    )


def test_the_prefreeze_keeps_the_choice_under_lower_case_keys() -> None:
    found = coverage_confirm.prefreeze("a" * 64, CHOICE)
    assert found["primary"] == CHOICE
    assert set(found) == {"coverage_claim_sha256", "primary", "rule"}


def test_no_rows_have_a_lower_limit_of_zero() -> None:
    found = coverage_confirm.decide(0, 0, CHOICE)
    assert found["one_sided_lower_limit"] == 0.0
    assert found["coverage"] is None


def test_the_not_evaluated_document_word_for_word() -> None:
    assert coverage_confirm.not_evaluated("no rows") == {
        "schema_version": 1,
        "claim_id": "coverage_nominal",
        "status": confirm.NOT_EVALUATED,
        "reasons": ["no rows"],
    }


def test_held_reads_numeric_text_as_numbers() -> None:
    features = pandas.DataFrame({"x": [0.0, 0.0]})
    inside = coverage_confirm.held(Band(), features, ["0.5", "3"], CHOICE)
    assert inside.tolist() == [True, False]


def test_the_document_describes_every_method_at_the_four_levels() -> None:
    features, target = released()
    found = coverage_confirm.assess(Band(), features, target, "c" * 64, CHOICE)
    assert found["schema_version"] == 1
    described = found["described_on_the_same_rows"]
    assert set(described) == {"note", "methods", "exchangeability"}
    methods = described["methods"]
    assert methods
    for levels in methods.values():
        assert list(levels) == ["0.5", "0.8", "0.9", "0.95"]
