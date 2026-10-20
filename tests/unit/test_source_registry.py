"""Tests for source snapshot identity and schema evolution."""

import dataclasses
import json

import pandas as pd
import pytest

from scml.contracts import data
from scml.data.readers import registry


def make_manifest(snapshot_id: str, data_hash: str) -> data.DatasetManifest:
    """Build a small manifest for registry tests."""
    return data.DatasetManifest(
        dataset_id=f"dataset-{data_hash}",
        schema_hash="schema-hash",
        data_hash=data_hash,
        row_count=2,
        column_count=2,
        target="label",
        uri="file:///tmp/events.csv",
        snapshot_id=snapshot_id,
    )


def make_snapshot(
    snapshot_id: str = "snapshot-1",
    data_hash: str = "data-1",
    schema: dict[str, str] | None = None,
) -> registry.SourceSnapshot:
    """Build a deterministic source snapshot."""
    frame_schema = schema or {"value": "int64", "label": "int64"}
    return registry.SourceSnapshot.from_manifest(
        make_manifest(snapshot_id, data_hash),
        frame_schema,
        "2026-10-20T10:00:00+00:00",
    )


def test_registry_round_trips_and_repeats_are_idempotent():
    source = make_snapshot()
    observed_again = dataclasses.replace(
        source, captured_at="2026-10-20T10:01:00+00:00"
    )
    registry_value = registry.SourceRegistry()
    registry_value.register(source)
    registry_value.register(observed_again)

    serialized = registry_value.to_json()
    restored = registry.SourceRegistry.from_json(serialized)

    assert len(restored.snapshots[source.source_id]) == 1
    assert json.loads(serialized) == json.loads(restored.to_json())


def test_schema_compatibility_allows_additive_and_rejects_breaking_changes():
    first = make_snapshot()
    additive = make_snapshot(
        "snapshot-2",
        "data-2",
        {"value": "int64", "label": "int64", "region": "string"},
    )
    breaking = make_snapshot(
        "snapshot-3",
        "data-3",
        {"value": "float64", "label": "int64"},
    )

    additive_result = registry.compare_schemas(first, additive)
    breaking_result = registry.compare_schemas(first, breaking)

    assert additive_result.compatible
    assert additive_result.status == "additive"
    assert additive_result.added == ("region",)
    assert not breaking_result.compatible
    assert breaking_result.status == "breaking"
    assert breaking_result.changed == ("value",)


def test_snapshot_from_frame_preserves_ordered_source_types():
    frame = pd.DataFrame({"value": [1, 2], "label": [0, 1]})
    manifest = make_manifest("snapshot-1", "data-1")

    source = registry.SourceSnapshot.from_manifest(
        manifest,
        {str(name): str(dtype) for name, dtype in frame.dtypes.items()},
        "2026-10-20T10:00:00+00:00",
    )

    assert list(source.schema) == ["value", "label"]
    assert source.source_id.startswith("src_")


def test_registry_rejects_conflicting_repeated_snapshot():
    first = make_snapshot()
    conflict = make_snapshot(data_hash="different")
    registry_value = registry.SourceRegistry()
    registry_value.register(first)

    with pytest.raises(ValueError, match="different evidence"):
        registry_value.register(conflict)
