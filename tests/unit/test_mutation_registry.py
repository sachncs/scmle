"""The fault-plan and source registries' refusals, word for word."""

import dataclasses
import json

import pytest

from scmle.contracts import errors
from scmle.contracts import reliability
from scmle.contracts.data import provenance
from scmle.data.readers import registry
from tests.unit.test_reliability_contracts import make_evidence
from tests.unit.test_source_registry import make_manifest
from tests.unit.test_source_registry import make_snapshot

VERSION_MESSAGE = r"^unsupported fault-injection plan schema version$"


@pytest.mark.parametrize("version", [True, "1", 1.0])
def test_only_an_integer_schema_version_is_read(version) -> None:
    with pytest.raises(errors.InputError, match=VERSION_MESSAGE):
        reliability.Registry.from_dict(
            {"schema_version": version, "scenarios": []}
        )


def test_each_scenario_must_be_a_mapping() -> None:
    with pytest.raises(
        errors.KindError, match=r"^each scenario must be a mapping$"
    ):
        reliability.Registry.from_dict(
            {"schema_version": reliability.VERSION, "scenarios": ["plan"]}
        )


def test_serialized_text_must_hold_an_object() -> None:
    with pytest.raises(
        errors.KindError,
        match=r"^serialized plan must contain a JSON object$",
    ):
        reliability.Registry.parse("[]")


def test_only_a_contingency_can_be_logged() -> None:
    with pytest.raises(
        errors.KindError, match=r"^scenario must be Contingency$"
    ):
        reliability.Registry().log("crash", ())


@pytest.mark.parametrize("name", ["execution", "observation"])
def test_blank_evidence_text_is_refused_by_name(name) -> None:
    with pytest.raises(errors.InputError, match=rf"^{name} must not be empty$"):
        dataclasses.replace(make_evidence("crash"), **{name: " "})


def test_flatten_lifts_the_digests_and_drops_their_object() -> None:
    assert registry.flatten(
        {"rows": 1, "digests": {"schema": "s", "data": "d", "raw": "r"}}
    ) == {
        "rows": 1,
        "schema_hash": "s",
        "data_hash": "d",
        "raw_source_hash": "r",
    }
    assert registry.flatten({"rows": 1}) == {"rows": 1}


def test_shift_leaves_entries_that_are_not_sources_with_snapshot_lists() -> (
    None
):
    layout = {"sources": ["text", {"source_id": "a"}]}
    assert registry.shift(layout, dict) == layout


def test_a_missing_watermark_reads_as_empty() -> None:
    record = make_snapshot().to_dict()
    del record["watermark"]
    assert registry.Observation.from_dict(record).watermark == ""


def test_restore_keeps_the_raw_source_hash_of_the_lineage() -> None:
    manifest = dataclasses.replace(
        make_manifest("snapshot-1", "data-1"),
        lineage=provenance.Ancestry(digests={"raw-source": "raw-1"}),
    )
    found = registry.Observation.restore(
        manifest, {"value": "int64"}, "2026-10-20T10:00:00+00:00"
    )
    assert found.digests["raw"] == "raw-1"


def test_the_registry_json_is_ascii_with_sorted_keys() -> None:
    sources = registry.Sources()
    sources.register(make_snapshot(schema={"café": "int64"}))
    text = sources.to_json()
    assert text == json.dumps(
        sources.to_dict(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert "caf\\u00e9" in text


def test_a_registry_whose_sources_are_not_a_list_is_refused() -> None:
    record = {"schema_version": 2, "sources": {}}
    with pytest.raises(errors.InputError, match=r"^sources must be a list$"):
        registry.Sources.from_dict(record)


def test_saving_something_that_is_not_sources_is_refused(tmp_path) -> None:
    store = registry.Registry(tmp_path / "registry.json")
    with pytest.raises(errors.KindError, match=r"^registry must be Sources$"):
        store.save({})
