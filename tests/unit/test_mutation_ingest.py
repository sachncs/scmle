"""The charter ingest block's refusals, word for word, and its fold."""

import pytest

from scmle.assurance.charter import ingest
from scmle.contracts import errors

CHECKSUM = "a" * 64


def test_fold_nests_the_version_one_tolerances() -> None:
    assert ingest.fold({"eps_key": 0.1, "eps_fk": 0.2, "lag": 1.0}) == {
        "lag": 1.0,
        "tolerances": {"key": 0.1, "link": 0.2},
    }


def test_a_blank_attribute_name_is_refused_by_name() -> None:
    with pytest.raises(errors.InputError, match=r"^name must not be empty$"):
        ingest.Attribute(" ")


def test_a_repeated_entity_field_and_a_blank_column_are_refused() -> None:
    passes = (ingest.Pass("name"),)
    with pytest.raises(errors.InputError, match=r"^duplicate entity field$"):
        ingest.Entity((ingest.Attribute("a"), ingest.Attribute("a")), passes)
    with pytest.raises(errors.InputError, match=r"^column must not be empty$"):
        ingest.Entity((ingest.Attribute("a"),), passes, column=" ")


def test_an_empty_reconcile_block_is_refused_word_for_word() -> None:
    with pytest.raises(
        errors.InputError, match=r"^a reconcile block needs at least one table$"
    ):
        ingest.Expectation(())


def test_a_wall_time_is_kept_and_windows_must_be_a_tuple() -> None:
    assert ingest.Ingestion(seconds=5.0).seconds == 5.0
    with pytest.raises(errors.InputError, match=r"^windows must be a tuple$"):
        ingest.Ingestion(windows=[7.0])


def test_table_totals_refuse_by_lower_case_name_word_for_word() -> None:
    with pytest.raises(errors.InputError, match=r"^view must not be empty$"):
        ingest.Totals(" ", rows=1)
    with pytest.raises(errors.InputError, match=r"^rows must be non-negative$"):
        ingest.Totals("v", rows=-1)
    with pytest.raises(
        errors.InputError,
        match=r"^checksum must be 64 lower-case hex digits$",
    ):
        ingest.Totals("v", column="id", checksum="A" * 64)
    assert ingest.Totals("v", column="id", checksum=CHECKSUM).checksum
