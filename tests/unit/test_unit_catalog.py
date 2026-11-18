"""Behavioural tests for the catalog evidence parts and data observers."""

import datetime
import hashlib

import numpy
import pandas
import pytest

from scmle.contracts import errors
from scmle.data.catalog import evidence
from scmle.data.catalog import inspection

DIGEST = "ab" * 32


def test_a_field_strips_whitespace_and_round_trips():
    field = evidence.Field("  age ", " int64 ")
    assert field.name == "age"
    assert field.kind == "int64"
    assert field.to_dict() == {"name": "age", "kind": "int64"}
    assert evidence.Field.from_dict({"name": "age", "kind": "int64"}) == field


@pytest.mark.parametrize(
    ("name", "kind", "message"),
    [
        ("", "int", "field name must not be empty"),
        ("  ", "int", "field name must not be empty"),
        (3, "int", "field name must not be empty"),
        ("a", "", "field kind must not be empty"),
        ("a", None, "field kind must not be empty"),
    ],
)
def test_a_field_with_a_blank_part_is_rejected(name, kind, message):
    with pytest.raises(errors.InputError, match=message):
        evidence.Field(name, kind)


def test_a_field_records_its_legacy_rename():
    assert evidence.Field.migrations[0].renames == {"data_type": "kind"}


def test_missingness_sorts_fields_and_reports_its_fraction():
    record = evidence.Missingness("observed", 8, 2, 1, {"b": 1, "a": 1})
    assert list(record.fields) == ["a", "b"]
    assert record.fraction == 0.25
    assert record.to_dict() == {
        "status": "observed",
        "total": 8,
        "missing": 2,
        "missing_fraction": 0.25,
        "incomplete": 1,
        "fields": {"a": 1, "b": 1},
    }


def test_missingness_with_no_values_has_a_zero_fraction():
    assert evidence.Missingness("observed", 0, 0, 0, {}).fraction == 0.0


def test_missingness_may_have_all_values_missing():
    record = evidence.Missingness("observed", 2, 2, 2, {"a": 2})
    assert record.fraction == 1.0


@pytest.mark.parametrize(
    ("arguments", "kind", "message"),
    [
        (("", 1, 0, 0, {}), errors.InputError, "status must not be empty"),
        (("s", 1.0, 0, 0, {}), errors.KindError, "total must be an integer"),
        (("s", True, 0, 0, {}), errors.KindError, "total must be an integer"),
        (("s", 1, -1, 0, {}), errors.InputError, "missing must not be neg"),
        (("s", 1, 0, -1, {}), errors.InputError, "incomplete must not be neg"),
        (("s", 1, 0, "0", {}), errors.KindError, "incomplete must be an int"),
        (("s", 1, 2, 0, {}), errors.InputError, "missing cannot exceed total"),
        (("s", 1, 0, 0, {"": 0}), errors.InputError, "non-negative named"),
        (("s", 1, 0, 0, {"a": -1}), errors.InputError, "non-negative named"),
        (("s", 1, 0, 0, {"a": True}), errors.InputError, "non-negative named"),
        (("s", 1, 0, 0, {"a": 1.0}), errors.InputError, "non-negative named"),
        (("s", 1, 0, 0, {1: 1}), errors.InputError, "non-negative named"),
    ],
)
def test_missingness_rejects_inconsistent_counts(arguments, kind, message):
    with pytest.raises(kind, match=message):
        evidence.Missingness(*arguments)


def test_missingness_round_trips_and_checks_its_recorded_fraction():
    record = evidence.Missingness("observed", 4, 1, 1, {"a": 1})
    assert evidence.Missingness.from_dict(record.to_dict()) == record
    loose = {"status": "s", "total": 4, "missing": 1, "incomplete": 1}
    assert evidence.Missingness.from_dict(loose).fields == {}
    with pytest.raises(errors.InputError, match="missing_fraction"):
        evidence.Missingness.from_dict(loose | {"missing_fraction": 0.5})


def test_missingness_records_its_legacy_renames():
    assert evidence.Missingness.migrations[0].renames == {
        "total_values": "total",
        "missing_values": "missing",
        "rows_with_missing": "incomplete",
        "by_field": "fields",
    }


def test_duplication_round_trips():
    record = evidence.Duplication("observed", 2, 0.5, "m")
    assert record.to_dict() == {
        "status": "observed",
        "rows": 2,
        "fraction": 0.5,
        "method": "m",
    }
    assert evidence.Duplication.from_dict(record.to_dict()) == record
    assert evidence.Duplication("s", 0, 0.0, "m").fraction == 0.0
    assert evidence.Duplication("s", 1, 1.0, "m").fraction == 1.0
    assert evidence.Duplication.migrations[0].renames == {
        "duplicate_rows": "rows",
        "duplicate_fraction": "fraction",
    }


@pytest.mark.parametrize(
    ("arguments", "kind", "message"),
    [
        (("", 0, 0.0, "m"), errors.InputError, "status and method"),
        (("s", 0, 0.0, " "), errors.InputError, "status and method"),
        (("s", 0.0, 0.0, "m"), errors.KindError, "rows must be an integer"),
        (("s", False, 0.0, "m"), errors.KindError, "rows must be an integer"),
        (("s", -1, 0.0, "m"), errors.InputError, "rows must not be negative"),
        (("s", 0, -0.1, "m"), errors.InputError, "between zero and one"),
        (("s", 0, 1.1, "m"), errors.InputError, "between zero and one"),
        (("s", 0, float("nan"), "m"), errors.InputError, "between zero"),
    ],
)
def test_duplication_rejects_bad_accounting(arguments, kind, message):
    with pytest.raises(kind, match=message):
        evidence.Duplication(*arguments)


def test_labeling_sorts_its_distribution_and_round_trips():
    record = evidence.Labeling("observed", "y", 3, 1, 2, {"b": 1, "a": 2})
    assert list(record.distribution) == ["a", "b"]
    assert record.to_dict() == {
        "status": "observed",
        "field": "y",
        "observed": 3,
        "missing": 1,
        "distinct": 2,
        "distribution": {"a": 2, "b": 1},
    }
    assert evidence.Labeling.from_dict(record.to_dict()) == record
    bare = {"status": "s", "observed": 0, "missing": 0, "distinct": 0}
    assert evidence.Labeling.from_dict(bare).field is None
    assert evidence.Labeling.from_dict(bare).distribution == {}
    assert evidence.Labeling.migrations[0].renames == {
        "observed_count": "observed",
        "missing_count": "missing",
        "distinct_count": "distinct",
    }


def test_labeling_may_have_as_many_distinct_labels_as_observed():
    record = evidence.Labeling("s", None, 2, 0, 2, {"a": 1, "b": 1})
    assert record.distinct == 2


@pytest.mark.parametrize(
    ("arguments", "kind", "message"),
    [
        (("", None, 0, 0, 0, {}), errors.InputError, "status must not be"),
        (("s", " ", 0, 0, 0, {}), errors.InputError, "field must not be"),
        (("s", None, 1.0, 0, 0, {}), errors.KindError, "observed must be"),
        (("s", None, 0, True, 0, {}), errors.KindError, "missing must be"),
        (("s", None, 0, 0, -1, {}), errors.InputError, "distinct must not"),
        (("s", None, 0, -1, 0, {}), errors.InputError, "missing must not"),
        (("s", None, 1, 0, 2, {"a": 1}), errors.InputError, "cannot exceed"),
        (("s", None, 0, 0, 0, {"": 0}), errors.InputError, "non-negative"),
        (("s", None, 0, 0, 0, {"a": -1}), errors.InputError, "non-negative"),
        (("s", None, 1, 0, 0, {"a": True}), errors.InputError, "non-neg"),
        (("s", None, 2, 0, 1, {"a": 1}), errors.InputError, "does not match"),
    ],
)
def test_labeling_rejects_inconsistent_counts(arguments, kind, message):
    with pytest.raises(kind, match=message):
        evidence.Labeling(*arguments)


def test_sequential_accepts_ordered_containers_but_not_text_or_mappings():
    assert inspection.sequential([1])
    assert inspection.sequential((1,))
    assert inspection.sequential(numpy.array([1]))
    assert not inspection.sequential("ab")
    assert not inspection.sequential(b"ab")
    assert not inspection.sequential(bytearray(b"ab"))
    assert not inspection.sequential({"a": 1})
    assert not inspection.sequential(3)


def test_measure_returns_plain_integers():
    shape = inspection.measure(pandas.DataFrame({"a": [1, 2]}))
    assert shape == (2, 1)
    assert all(type(value) is int for value in shape)


def test_a_named_series_becomes_a_one_column_frame():
    frame, kind, shape = inspection.frame_for_pandas(
        pandas.Series([1, 2], name="x")
    )
    assert list(frame.columns) == ["x"]
    assert (kind, shape) == ("pandas_series", (2, 1))
    frame = inspection.frame_for_pandas(pandas.Series([1]))[0]
    assert list(frame.columns) == ["value"]
    data = pandas.DataFrame({"a": [1]})
    assert inspection.frame_for_pandas(data) == (
        data,
        "pandas_dataframe",
        (1, 1),
    )


def test_tabulate_reads_each_array_rank():
    frame, kind, shape = inspection.tabulate(numpy.array(5))
    assert frame["value"].tolist() == [5]
    assert (kind, shape) == ("numpy_scalar", ())
    frame, kind, shape = inspection.tabulate(numpy.array([1, 2, 3]))
    assert frame["value"].tolist() == [1, 2, 3]
    assert (kind, shape) == ("numpy_array", (3,))
    frame, kind, shape = inspection.tabulate(numpy.zeros((2, 3)))
    assert list(frame.columns) == ["0", "1", "2"]
    assert shape == (2, 3)
    assert inspection.tabulate(numpy.zeros((1, 2, 3))) == (
        None,
        "numpy_array",
        (1, 2, 3),
    )
    assert inspection.MATRIX_DIMENSIONS == 2


def test_a_mapping_is_columns_when_every_value_is_a_sequence():
    frame, kind, shape = inspection.frame_for_mapping(
        {"a": [1, 2], "b": (3, 4)}
    )
    assert (kind, shape) == ("column_mapping", (2, 2))
    assert frame["b"].tolist() == [3, 4]
    frame, kind, shape = inspection.frame_for_mapping({"a": [1], "b": 2})
    assert (kind, shape) == ("record_mapping", (1, 2))
    kind, shape = inspection.frame_for_mapping({})[1:]
    assert (kind, shape) == ("record_mapping", (1, 0))
    with pytest.raises(errors.InputError, match="equal lengths"):
        inspection.frame_for_mapping({"a": [1], "b": [1, 2]})


def test_coerce_routes_each_supported_form():
    assert inspection.coerce(pandas.Series([1]))[1] == "pandas_series"
    assert inspection.coerce(numpy.array([1]))[1] == "numpy_array"
    assert inspection.coerce({"a": 1})[1] == "record_mapping"
    frame, kind, shape = inspection.coerce([{"a": 1}, {"a": 2}])
    assert (kind, shape) == ("record_sequence", (2, 1))
    frame, kind, shape = inspection.coerce((1, 2, 3))
    assert (kind, shape) == ("sequence", (3, 1))
    assert frame["value"].tolist() == [1, 2, 3]
    frame, kind, shape = inspection.coerce([])
    assert (kind, shape) == ("sequence", (0, 1))
    assert inspection.coerce([{"a": 1}, 2])[1] == "sequence"
    assert inspection.coerce(b"abc") == (None, "bytes", (3,))


def test_fallback_describes_bytes_scalars_and_opaque_objects():
    assert inspection.fallback(memoryview(b"ab")) == (None, "bytes", (2,))
    assert inspection.fallback(bytearray(b"a")) == (None, "bytes", (1,))
    for value in ("s", 1, 1.5, True, datetime.date(2020, 1, 1), None):
        frame, kind, shape = inspection.fallback(value)
        assert (kind, shape) == ("scalar", ())
        assert frame["value"].tolist() == [value]
    assert inspection.fallback(object()) == (None, "object", ())


def test_describe_keeps_duplicate_columns_in_order():
    frame = pandas.DataFrame([[1, 2.5]], columns=["x", "x"])
    assert inspection.describe(frame) == (
        evidence.Field("x", "int64"),
        evidence.Field("x", "float64"),
    )


def test_tally_counts_missing_values_and_names_duplicate_columns():
    frame = pandas.DataFrame(
        [[1, None, 3], [None, None, 4], [5, 6, 7]], columns=["a", "a", "b"]
    )
    record = inspection.tally(frame)
    assert record.status == "observed"
    assert record.total == 9
    assert record.missing == 3
    assert record.incomplete == 2
    assert record.fields == {"a[0]": 1, "a[1]": 2, "b": 0}


def test_dedupe_counts_every_copy_of_a_duplicate_row():
    frame = pandas.DataFrame({"a": [1, 1, 2, 3]})
    record = inspection.dedupe(frame)
    assert record == evidence.Duplication(
        "observed", 2, 0.5, "pandas.DataFrame.duplicated(keep=False)"
    )
    empty = inspection.dedupe(pandas.DataFrame({"a": []}))
    assert (empty.rows, empty.fraction) == (0, 0.0)


def test_dedupe_reports_unhashable_rows_as_not_observed():
    class Unhashable(pandas.DataFrame):
        def duplicated(self, keep="first"):
            raise TypeError("unhashable")

    record = inspection.dedupe(Unhashable({"a": [1, 1]}))
    assert record.status == "not_observed"
    assert record.rows == 0
    assert record.fraction == 0.0
    assert "not supported" in record.method


def test_summarize_without_a_declared_label_blocks_supervised_use():
    label, limitations, blockers = inspection.summarize(None, None)
    assert label == evidence.Labeling("not_declared", None, 0, 0, 0, {})
    assert limitations == ("label_was_not_declared",)
    assert blockers == ("label_contract_not_declared_for_supervised_use",)


def test_summarize_reports_an_absent_or_ambiguous_label():
    label, limitations, blockers = inspection.summarize(None, "y")
    assert label.status == "not_observed"
    assert label.field == "y"
    assert limitations == ("declared_label_could_not_be_observed",)
    assert blockers == ("declared_label_field_is_not_available",)
    frame = pandas.DataFrame({"x": [1]})
    assert inspection.summarize(frame, "y")[0].status == "not_observed"
    twin = pandas.DataFrame([[1, 2]], columns=["y", "y"])
    label, limitations, blockers = inspection.summarize(twin, "y")
    assert label.status == "not_observed"
    assert limitations == blockers == ("declared_label_field_is_ambiguous",)


def test_summarize_counts_labels_by_canonical_value():
    frame = pandas.DataFrame({"y": ["a", "b", "a", None]})
    label, limitations, blockers = inspection.summarize(frame, "y")
    assert label == evidence.Labeling(
        "observed", "y", 3, 1, 2, {'"a"': 2, '"b"': 1}
    )
    assert (limitations, blockers) == ((), ())


def test_summarize_flags_a_constant_label_and_an_all_missing_label():
    constant = pandas.DataFrame({"y": [1, 1]})
    label, blockers = inspection.summarize(constant, "y")[::2]
    assert (label.status, label.distribution) == ("constant", {"1": 2})
    assert blockers == ("label_is_constant",)
    empty = pandas.DataFrame({"y": [None, None]})
    label, blockers = inspection.summarize(empty, "y")[::2]
    assert (label.status, label.missing, label.observed) == (
        "missing_all",
        2,
        0,
    )
    assert blockers == ()


def test_frame_content_lists_kind_shape_schema_and_rows():
    frame = pandas.DataFrame({"a": [1, 2], "b": [0.5, 1.5]})
    assert inspection.frame_content(frame, "k", (2, 2)) == {
        "data_kind": "k",
        "shape": [2, 2],
        "schema": [
            {"name": "a", "kind": "int64"},
            {"name": "b", "kind": "float64"},
        ],
        "rows": [[1, 0.5], [2, 1.5]],
    }


def test_examine_reports_missing_and_duplicate_rows_as_blockers():
    frame = pandas.DataFrame({"a": [1, 1, None], "y": [0, 0, 1]})
    observed = inspection.examine(frame, "y", None)
    assert observed.data_kind == "pandas_dataframe"
    assert observed.shape == (3, 2)
    assert observed.sample_count == 3
    assert observed.limitations == (
        "missing_values_are_reported_not_repaired",
        "duplicate_rows_are_reported_not_removed",
    )
    assert observed.release_blockers == (
        "missingness_policy_not_declared",
        "duplicate_policy_not_declared",
    )
    assert observed.content_identity.startswith("sha256:")
    assert len(observed.content_identity) == 71
    again = inspection.examine(frame.copy(), "y", None)
    assert again.content_identity == observed.content_identity


def test_examine_of_clean_data_has_no_quality_blockers():
    observed = inspection.examine({"a": [1, 2], "y": [0, 1]}, "y", None)
    assert observed.limitations == ()
    assert observed.release_blockers == ()
    assert observed.label.status == "observed"


def test_examine_uses_a_supplied_identity_in_normal_form():
    observed = inspection.examine([1, 2], None, DIGEST.upper())
    assert observed.content_identity == f"sha256:{DIGEST}"
    assert observed.limitations == (
        "label_was_not_declared",
        "content_identity_was_supplied_by_caller",
    )
    with pytest.raises(errors.InputError, match="content_identity"):
        inspection.examine([1], None, "abc")


def test_examine_digests_bytes_without_decoding_them():
    observed = inspection.examine(b"abc", None, None)
    assert observed.content_identity == (
        f"sha256:{hashlib.sha256(b'abc').hexdigest()}"
    )
    assert observed.schema == (evidence.Field("payload", "bytes"),)
    assert observed.sample_count == 1
    assert observed.shape == (3,)
    assert observed.missingness.status == "not_observed"
    assert observed.duplicates.method == "raw bytes have no row identity"
    assert observed.limitations == (
        "modality_specific_decoding_was_not_attempted",
        "label_was_not_declared",
    )
    assert observed.release_blockers == (
        "label_contract_not_declared_for_supervised_use",
        "modality_specific_decoder_not_declared",
    )
    supplied = inspection.examine(b"abc", "y", DIGEST)
    assert supplied.content_identity == f"sha256:{DIGEST}"
    assert supplied.limitations[0] == "content_identity_was_supplied_by_caller"
    assert supplied.label.status == "not_observed"


def test_examine_needs_an_identity_for_opaque_data():
    with pytest.raises(errors.KindError, match="opaque data"):
        inspection.examine(object(), None, None)
    observed = inspection.examine(numpy.zeros((1, 1, 1)), None, DIGEST)
    assert observed.schema == (evidence.Field("value", "ndarray"),)
    assert observed.shape == (1, 1, 1)
    assert observed.limitations == (
        "schema_and_quality_metrics_were_not_observed",
        "content_identity_was_supplied_by_caller",
        "label_was_not_declared",
    )
    assert observed.release_blockers == (
        "schema_adapter_not_declared",
        "label_contract_not_declared_for_supervised_use",
    )
    assert observed.duplicates.method == "opaque object has no row identity"
