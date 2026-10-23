"""Focused tests for the model-agnostic dataset catalog contract."""

import datetime
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import scml
from scml.data import catalog


def test_dataframe_observation_is_deterministic_and_reviewable():
    """Tabular evidence records quality state without fitting a model."""
    frame = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "value": [10.0, 10.0, np.nan],
            "label": ["keep", "keep", None],
        }
    )
    source = catalog.DatasetSource(
        uri="s3://bucket/events?version=7",
        version="7",
        owner="analytics",
        acquisition_method="warehouse-export",
        privacy_classification="internal",
    )

    first = catalog.DatasetCatalog.from_data(
        frame, source=source, label_column="label"
    )
    second = catalog.DatasetCatalog.from_data(
        frame.copy(), source=source, label_column="label"
    )

    assert first == second
    assert first.source.source_identity.startswith("sha256:")
    assert first.data_kind == "pandas_dataframe"
    assert first.shape == (3, 3)
    assert first.sample_count == 3
    assert [field.name for field in first.schema] == [
        "id",
        "value",
        "label",
    ]
    assert first.missingness.missing_values == 2
    assert first.missingness.rows_with_missing == 1
    assert first.duplicates.duplicate_rows == 2
    assert first.label.status == "constant"
    assert first.label.missing_count == 1
    assert first.label.distribution['"keep"'] == 2
    assert "curation_only_no_model_fit_or_performance_claim" in (
        first.limitations
    )
    assert "missingness_policy_not_declared" in first.release_blockers
    assert "duplicate_policy_not_declared" in first.release_blockers


@pytest.mark.parametrize(
    ("data", "expected_kind", "expected_shape"),
    [
        ({"feature": [1, 2], "target": [0, 1]}, "column_mapping", (2, 2)),
        ([{"feature": 1}, {"feature": 2}], "record_sequence", (2, 1)),
        (np.asarray([[1, 2], [3, 4]]), "numpy_array", (2, 2)),
        ([1, 2, 3], "sequence", (3, 1)),
    ],
)
def test_common_caller_owned_shapes_are_cataloged(
    data, expected_kind, expected_shape
):
    """The entry point accepts several data shapes without a model claim."""
    result = catalog.DatasetCatalog.from_data(data)

    assert result.data_kind == expected_kind
    assert result.shape == expected_shape
    assert result.sample_count == expected_shape[0]
    assert result.content_identity.startswith("sha256:")


def test_raw_bytes_report_decoder_boundary_and_direct_identity():
    """Raw bytes receive identity but no invented modality understanding."""
    payload = b"opaque dataset payload"
    result = catalog.DatasetCatalog.from_data(payload)

    expected = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    assert result.content_identity == expected
    assert result.data_kind == "bytes"
    assert result.sample_count == 1
    assert result.missingness.status == "not_observed"
    assert result.duplicates.status == "not_observed"
    assert "modality_specific_decoder_not_declared" in result.release_blockers
    assert "modality_specific_decoding_was_not_attempted" in result.limitations


def test_opaque_data_requires_explicit_content_identity():
    """An opaque object cannot silently receive a process-dependent hash."""
    opaque = object()
    identity = "sha256:" + "a" * 64

    with pytest.raises(TypeError, match="caller-supplied"):
        catalog.DatasetCatalog.from_data(opaque)

    result = catalog.DatasetCatalog.from_data(opaque, content_identity=identity)

    assert result.content_identity == identity
    assert result.data_kind == "object"
    assert "schema_adapter_not_declared" in result.release_blockers
    assert "content_identity_was_supplied_by_caller" in result.limitations


def test_source_metadata_removes_governance_blockers_for_clean_data():
    """Declared source context is retained instead of inferred from bytes."""
    source = catalog.DatasetSource(
        uri="memory://customer-sample",
        version="2026-10-23",
        owner="data-platform",
        acquisition_method="approved-export",
        privacy_classification="internal",
    )
    result = catalog.DatasetCatalog.from_data(
        {"x": [1, 2], "label": [0, 1]},
        source=source,
        label_column="label",
    )

    assert result.release_blockers == ()
    assert result.label.status == "observed"
    assert result.source.to_dict()["uri"] == "memory://customer-sample"


@pytest.mark.parametrize(
    ("data", "label_status"),
    [
        ({"label": [1, 1]}, "constant"),
        ({"label": [None, None]}, "missing_all"),
    ],
)
def test_label_state_is_explicit_and_does_not_infer_task(data, label_status):
    """Label presence is described without deciding classification or regression."""
    result = catalog.DatasetCatalog.from_data(data, label_column="label")

    assert result.label.status == label_status
    assert result.label.field == "label"
    assert "curation_only_no_model_fit_or_performance_claim" in (
        result.limitations
    )


def test_catalog_persists_atomically_and_verifies_read_back(tmp_path):
    """Persistence verifies both nested source identity and catalog digest."""
    result = catalog.DatasetCatalog.from_data(
        [{"text": "a"}, {"text": "b"}],
        source="file:///tmp/records.jsonl",
    )
    path = tmp_path / "catalog.json"

    result.write(path)
    restored = catalog.DatasetCatalog.read_verified(path)

    assert restored == result
    assert json.loads(path.read_text(encoding="utf-8"))["catalog_digest"] == (
        result.catalog_digest
    )

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["sample_count"] = 99
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="catalog digest"):
        catalog.DatasetCatalog.read_verified(path)


def test_adapter_owned_observation_supports_unknown_shapes():
    """Adapters can publish evidence without routing through a model API."""
    result = catalog.DatasetCatalog.from_observation(
        source="warehouse://events",
        data_kind="custom_event_stream",
        shape=(12, 4),
        schema=(catalog.DatasetField("event", "custom"),),
        sample_count=12,
        missingness=catalog.MissingnessSummary("not_observed", 0, 0, 0, {}),
        duplicates=catalog.DuplicateSummary(
            "not_observed", 0, 0.0, "adapter did not expose row identity"
        ),
        label=catalog.LabelSummary("not_declared", None, 0, 0, 0, {}),
        content_identity="sha256:" + "b" * 64,
        limitations=("adapter_summary_only",),
        release_blockers=("adapter_schema_review_required",),
    )

    assert result.data_kind == "custom_event_stream"
    assert result.shape == (12, 4)
    assert result.release_blockers == ("adapter_schema_review_required",)


def test_catalog_is_a_top_level_public_entry_point():
    """The general catalog is discoverable without importing implementation files."""
    assert scml.DatasetCatalog is catalog.DatasetCatalog
    assert "DatasetCatalog" in scml.__all__


def test_invalid_mapping_lengths_and_tampered_source_fail_closed():
    """Invalid structural input and source identity tampering are rejected."""
    with pytest.raises(ValueError, match="equal lengths"):
        catalog.DatasetCatalog.from_data({"a": [1], "b": [1, 2]})

    source = catalog.DatasetSource(uri="memory://source")
    record = source.to_dict()
    record["uri"] = "memory://other"
    with pytest.raises(ValueError, match="source_identity"):
        catalog.DatasetSource.from_dict(record)


def test_contract_components_reject_invalid_accounting():
    """Nested summaries fail closed instead of publishing contradictory counts."""
    with pytest.raises(ValueError, match="uri"):
        catalog.DatasetSource(uri=" ")
    with pytest.raises(TypeError, match="version"):
        catalog.DatasetSource(version=3)
    with pytest.raises(ValueError, match="field name"):
        catalog.DatasetField(" ", "int64")
    with pytest.raises(ValueError, match="data_type"):
        catalog.DatasetField("x", " ")
    with pytest.raises(ValueError, match="status"):
        catalog.MissingnessSummary(" ", 0, 0, 0, {})
    with pytest.raises(TypeError, match="total_values"):
        catalog.MissingnessSummary("observed", True, 0, 0, {})
    with pytest.raises(ValueError, match="missing_values"):
        catalog.MissingnessSummary("observed", 1, 2, 0, {})
    with pytest.raises(ValueError, match="by_field"):
        catalog.MissingnessSummary("observed", 1, 0, 0, {"x": -1})
    with pytest.raises(ValueError, match="missing_fraction"):
        catalog.MissingnessSummary.from_dict(
            {
                "status": "observed",
                "total_values": 2,
                "missing_values": 1,
                "rows_with_missing": 1,
                "missing_fraction": 0.0,
            }
        )
    assert catalog.MissingnessSummary("observed", 0, 0, 0, {}).fraction == 0
    with pytest.raises(ValueError, match="duplicate status"):
        catalog.DuplicateSummary(" ", 0, 0.0, "method")
    with pytest.raises(TypeError, match="duplicate_rows"):
        catalog.DuplicateSummary("observed", True, 0.0, "method")
    with pytest.raises(ValueError, match="duplicate_fraction"):
        catalog.DuplicateSummary("observed", 0, 1.1, "method")
    with pytest.raises(ValueError, match="label status"):
        catalog.LabelSummary(" ", None, 0, 0, 0, {})
    with pytest.raises(ValueError, match="label field"):
        catalog.LabelSummary("observed", " ", 0, 0, 0, {})
    with pytest.raises(ValueError, match="distinct_count"):
        catalog.LabelSummary("observed", "y", 0, 0, 1, {})
    with pytest.raises(ValueError, match="distribution"):
        catalog.LabelSummary("observed", "y", 1, 0, 1, {"a": -1})
    with pytest.raises(ValueError, match="does not match"):
        catalog.LabelSummary("observed", "y", 1, 0, 1, {})


def test_scalar_series_and_nested_value_observers_cover_type_boundaries():
    """Common scalar and nested values remain deterministic and JSON-native."""
    series = pd.Series([1, 2], name="score")
    scalar = catalog.DatasetCatalog.from_data(np.asarray(4))
    dated = catalog.DatasetCatalog.from_data(datetime.date(2026, 10, 23))
    series_result = catalog.DatasetCatalog.from_data(series)

    assert series_result.data_kind == "pandas_series"
    assert scalar.shape == () and scalar.sample_count == 1
    assert dated.data_kind == "scalar"
    assert catalog.canonical_value(b"x")["bytes_length"] == 1
    assert catalog.canonical_value(np.int64(4)) == 4
    assert catalog.canonical_value(datetime.date(2026, 10, 23)) == (
        "2026-10-23"
    )
    assert catalog.canonical_value({"b": 2, "a": 1}) == {"a": 1, "b": 2}
    assert catalog.canonical_value((1, 2)) == [1, 2]
    assert catalog.canonical_value({1, 2}) == [1, 2]
    assert catalog.canonical_value(float("nan")) == {"non_finite": "nan"}
    assert catalog.canonical_value(float("-inf")) == {"non_finite": "-inf"}
    assert catalog.canonical_value(np.asarray([1, 2])) == [1, 2]
    assert catalog.canonical_value(pd.NA) is None
    with pytest.raises(TypeError, match="deterministically catalog"):
        catalog.canonical_value(object())


def test_label_and_source_boundaries_are_explicit():
    """Unavailable and ambiguous labels become reviewable evidence."""
    unavailable = catalog.DatasetCatalog.from_data([1, 2], label_column="label")
    duplicate_columns = pd.DataFrame([[1, 2]], columns=["label", "label"])
    ambiguous = catalog.DatasetCatalog.from_data(
        duplicate_columns, label_column="label"
    )
    path_source = catalog.DatasetSource(uri="relative/catalog.json")

    assert unavailable.label.status == "not_observed"
    assert "declared_label_field_is_not_available" in (
        unavailable.release_blockers
    )
    assert ambiguous.label.status == "not_observed"
    assert "declared_label_field_is_ambiguous" in ambiguous.release_blockers
    assert path_source.uri.startswith("file://")


def test_raw_and_opaque_identity_overrides_are_validated():
    """Caller identity overrides are accepted only as SHA-256 values."""
    identity = "a" * 64
    raw = catalog.DatasetCatalog.from_data(
        b"payload", content_identity=identity, label_column="label"
    )
    high_dimensional = catalog.DatasetCatalog.from_data(
        np.zeros((2, 2, 2)), content_identity=identity
    )

    assert raw.content_identity == "sha256:" + identity
    assert raw.label.status == "not_observed"
    assert high_dimensional.shape == (2, 2, 2)
    with pytest.raises(ValueError, match="SHA-256"):
        catalog.DatasetCatalog.from_data(object(), content_identity="bad")
    with pytest.raises(ValueError, match="SHA-256"):
        catalog.DatasetCatalog.from_observation(
            source="memory://bad",
            data_kind="custom",
            shape=(),
            schema=(),
            sample_count=1,
            missingness=catalog.MissingnessSummary("not_observed", 0, 0, 0, {}),
            duplicates=catalog.DuplicateSummary("not_observed", 0, 0.0, "none"),
            label=catalog.LabelSummary("not_declared", None, 0, 0, 0, {}),
            content_identity="bad",
            limitations=("adapter_only",),
        )


def test_verified_reader_rejects_malformed_envelopes(tmp_path):
    """Read verification distinguishes invalid JSON, shape, schema, and digest."""
    path = tmp_path / "catalog.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        catalog.DatasetCatalog.read_verified(path)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        catalog.DatasetCatalog.read_verified(path)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="catalog_digest"):
        catalog.DatasetCatalog.read_verified(path)
    path.write_text(json.dumps({"catalog_digest": "bad"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema is invalid"):
        catalog.DatasetCatalog.read_verified(path)
