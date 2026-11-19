"""The data records write single-word keys (stage B, phase 5, rest F4).

Each write-only record routes `to_dict` through a `KEYS` migration (or its
versioned `migrations`): the new keys are written, and the downgrade gives
back the version-1 layout. Where a reader exists, a version-1 sample still
reads; the identities hashed over a record (the image augmentation identity
inside every image dataset hash) are computed on the version-1 form, so
stored hashes still verify.
"""

import dataclasses
import json

import pandas
import pytest

from scmle.contracts import data
from scmle.contracts import records
from scmle.data import reference
from scmle.data.catalog import canon
from scmle.data.catalog import identity
from scmle.data.catalog import manifest
from scmle.data.modalities import curation
from scmle.data.modalities import graph
from scmle.data.modalities import image
from scmle.data.modalities.audio import metadata
from scmle.data.modalities.document import elements
from scmle.data.modalities.document import report
from scmle.data.modalities.geospatial import evidence
from scmle.data.profiling import missingness
from scmle.data.readers import incremental
from scmle.data.readers import registry
from scmle.data.readers.incremental import timeline
from scmle.data.splitting.splitters import partitions
from scmle.data.store import processing
from scmle.data.store import quality
from scmle.data.store import snapshot
from scmle.preparation import pseudo_labeling

DIGEST = "a" * 64
OTHER = "b" * 64


def filled(kind, **given):
    """An instance whose every field holds its own name, unless given."""
    values = {f.name: f.name for f in dataclasses.fields(kind)}
    values.update(given)
    return kind(**values)


# record, the written keys it gains, the version-1 keys it loses
CASES = {
    "transcript": (
        metadata.Transcript("t", "s", "p", "1.0", "en"),
        {"version"},
        {"provider_version"},
    ),
    "citation": (
        elements.Citation("d", "file:///d", DIGEST, 0, 3, "l", 2),
        {"document", "uri", "digests", "page"},
        {"document_id", "source_uri", "source_digest", "page_number"},
    ),
    "layout": (
        elements.Layout("s", "p", "3.2", (), 0.5, DIGEST),
        {"version", "confidence", "digests"},
        {"provider_version", "ocr_confidence", "source_digest"},
    ),
    "match": (
        evidence.Match("l", "r"),
        {"left", "right"},
        {"left_feature_id", "right_feature_id"},
    ),
    "provenance": (
        evidence.Provenance(DIGEST, OTHER, "EPSG:4326", "EPSG:3857", "t", 1),
        {"digests", "crs", "transformer", "features"},
        {
            "source_dataset_hash",
            "target_dataset_hash",
            "source_crs",
            "target_crs",
            "transformer_name",
            "feature_count",
        },
    ),
    "profile": (
        evidence.Profile(
            1, ("Point",), "EPSG:4326", (0.0, 0.0, 1.0, 1.0), DIGEST
        ),
        {"features", "geometries", "digests"},
        {"feature_count", "geometry_types", "content_hash"},
    ),
    "autocorrelation": (
        evidence.Autocorrelation(5, 2, 6, 0.3, 99, 0.04, 7),
        {"features", "neighbors", "edges", "permutations", "p_value"},
        {"feature_count", "neighbor_count", "edge_count", "permutation_count"},
    ),
    "degrees": (
        graph.evidence.Degrees(3, 1, 2, 4, 0, 2, 1.0),
        {"nodes", "isolated", "labeled", "dimension", "degree"},
        {
            "node_count",
            "isolated_node_count",
            "labeled_node_count",
            "feature_dimension",
            "degree_min",
            "degree_max",
            "degree_mean",
        },
    ),
    "connectivity": (
        graph.evidence.Connectivity(2, 0, 1, 0.5, 1.5, 1.0),
        {"edges", "duplicates", "loops", "weight"},
        {
            "edge_count",
            "duplicate_edge_count",
            "self_loop_count",
            "weight_min",
            "weight_max",
            "weight_mean",
        },
    ),
    "image": (
        filled(image.evidence.Profile, modes=(), channels=()),
        {
            "samples",
            "profiled",
            "size",
            "unique",
            "duplicates",
            "width",
            "height",
            "pixel",
        },
        {
            "dataset_count",
            "profiled_count",
            "profiled_bytes",
            "unique_content_count",
            "duplicate_count",
            "width_min",
            "mean_pixel",
            "pixel_std",
        },
    ),
    "missingness": (
        filled(missingness.Finding),
        {"missing", "p_value"},
        {"missing_rate"},
    ),
    "manifest": (
        partitions.Manifest("random", 1, (0, 1), (2,), (((0,), (1,)),)),
        {"training", "test"},
        {"train_ids", "test_ids"},
    ),
    "processing": (
        processing.Processing(("a",), (), ("b",), 2, 5, ("o",), None, None, ()),
        {"duplicates", "rows"},
        {"duplicate_objects", "duplicate_rows", "rows_out"},
    ),
    "outcome": (
        pseudo_labeling.Outcome(0.1, 5, 12.0, True),
        {"pseudo"},
        {"pseudo_rows"},
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_record_writes_its_new_keys_and_downgrades_to_version_one(name):
    record, gained, lost = CASES[name]
    written = json.loads(json.dumps(record.to_dict()))
    assert gained <= set(written)
    assert not lost & set(written)
    old = record.KEYS.downgrade(written)
    assert record.KEYS.upgrade(old) == written
    assert lost <= set(old)


def test_nested_keys_hold_the_old_values():
    degrees = CASES["degrees"][0].to_dict()
    assert degrees["degree"] == {"low": 0, "high": 2, "mean": 1.0}
    weight = CASES["connectivity"][0].to_dict()["weight"]
    assert weight == {"low": 0.5, "high": 1.5, "mean": 1.0}
    pixel = CASES["image"][0].to_dict()["pixel"]
    assert pixel == {"mean": "mean_pixel", "std": "pixel_std"}
    found = CASES["provenance"][0].to_dict()
    assert found["digests"] == {"source": DIGEST, "target": OTHER}
    assert found["crs"] == {"source": "EPSG:4326", "target": "EPSG:3857"}
    assert CASES["processing"][0].to_dict()["rows"] == {
        "dropped": 2,
        "used": 5,
    }


def test_a_join_nests_its_sides_and_limits():
    join = evidence.Join(
        DIGEST,
        OTHER,
        "EPSG:4326",
        "p/v1",
        2,
        2,
        4,
        1,
        1,
        1,
        10,
        5,
        (evidence.Match("a", "b"),),
    )
    written = join.to_dict()
    assert written["digests"] == {"left": DIGEST, "right": OTHER}
    assert written["features"] == {"left": 2, "right": 2}
    assert written["unmatched"] == {"left": 1, "right": 1}
    assert written["limits"] == {"comparisons": 10, "matches": 5}
    assert (written["comparisons"], written["matched"]) == (4, 1)
    assert written["matches"] == [{"left": "a", "right": "b"}]
    old = evidence.Join.KEYS.downgrade(written)
    assert old["left_dataset_hash"] == DIGEST
    assert old["match_limit"] == 5


def test_a_leakage_record_writes_single_words():
    leakage = evidence.Leakage(1.0, 2, (("a", 2), ("b", 1)), ("a",), (), 4)
    written = leakage.to_dict()
    assert set(written) == {
        "status",
        "block",
        "features",
        "blocks",
        "crossing",
        "truncated",
        "spread",
    }
    assert (
        evidence.Leakage.KEYS.downgrade(written)["max_blocks_per_feature"] == 4
    )


def test_a_graph_profile_and_validation_write_single_words():
    profile = graph.evidence.Profile(
        "g" * 64,
        True,
        1,
        CASES["degrees"][0],
        CASES["connectivity"][0],
    )
    written = profile.to_dict()
    assert written["digests"] == {"graph": "g" * 64}
    assert written["components"] == 1
    assert written["node"]["nodes"] == 3
    assert "graph_identity" not in written
    check = graph.elements.Validation(())
    assert check.to_dict() == {"valid": True, "codes": [], "issues": []}
    assert graph.elements.Validation.KEYS.downgrade(check.to_dict())["is_valid"]


def test_an_augmented_image_keeps_its_version_one_identity():
    augmentation = image.sample.Augmentation(
        "s3://b/a.png", "flip", {"axis": 1}
    )
    written = augmentation.to_dict()
    assert written["parent"] == "s3://b/a.png"
    assert "parent_source" not in written
    version_one = {
        "operation": "flip",
        "parameters": {"axis": 1},
        "parent_source": "s3://b/a.png",
        "tool": "",
    }
    assert augmentation.identity == json.dumps(
        version_one, sort_keys=True, separators=(",", ":")
    )


def test_an_image_sample_nests_its_digests():
    sample = image.sample.Sample(
        source="s",
        encoded=b"xy",
        file_hash=DIGEST,
        content_hash=OTHER,
        width=1,
        height=1,
        channels=3,
        mode="RGB",
        format="PNG",
        size=2,
        group="g",
        tags=("1",),
        metadata=("GPS",),
    )
    written = sample.to_dict()
    assert written["digests"] == {"file": DIGEST, "content": OTHER}
    assert (written["size"], written["group"]) == (2, "g")
    assert (written["exif"], written["sensitive"]) == (["1"], ["GPS"])
    assert "encoded_bytes" not in written
    old = image.sample.Sample.KEYS.downgrade(written)
    assert old["file_hash"] == DIGEST
    assert old["sensitive_metadata"] == ["GPS"]


def finding(identifier, blocking):
    """A modality curation finding."""
    return curation.flaw(
        identifier,
        data.FindingSeverity.ERROR,
        {"source": "s"},
        {},
        "fix it",
        blocking=blocking,
    )


def test_a_modality_curation_is_version_two_and_reads_version_one():
    report = curation.Curation(
        "image",
        DIGEST,
        3,
        {"k": 1},
        (finding("quality.a", True),),
        (finding("risk.b", False),),
    )
    written = report.to_dict()
    assert written[records.KEY] == report.schema_version == 2
    assert written["digests"] == {"dataset": DIGEST}
    assert written["samples"] == 3
    assert set(written["findings"]) == {"quality", "risk"}
    assert written["ready"] is False
    assert written["blockers"] == ["quality.a"]
    content = {k: v for k, v in written.items() if k != records.KEY}
    old = records.downgrade(curation.Curation.migrations, content)
    assert {"dataset_hash", "sample_count", "release_ready"} <= set(old)
    assert {"quality_findings", "unresolved_blocking_findings"} <= set(old)
    assert curation.identify(old) == curation.identify(written) == (DIGEST, 3)
    assert curation.findings(old) == curation.findings(written)
    for layout in (written, {**old, records.KEY: 1}):
        card = curation.Datasheet.draft("image", layout, ["small"])
        assert card.samples == 3
        assert len(card.manifest.findings) == 2


def test_a_dataset_card_is_version_two():
    report = curation.Curation("image", DIGEST, 3, {}).to_dict()
    card = curation.Datasheet.draft("image", report, ["small"])
    written = card.to_dict()
    assert written[records.KEY] == 2
    assert written["digests"] == {"dataset": DIGEST}
    assert (written["samples"], written["use"]) == (3, card.use)
    assert not {"dataset_hash", "sample_count", "intended_use"} & set(written)
    content = {k: v for k, v in written.items() if k != records.KEY}
    old = records.downgrade(curation.Datasheet.migrations, content)
    assert old["dataset_hash"] == DIGEST
    with pytest.raises(ValueError, match="dataset_hash"):
        dataclasses.replace(card, dataset_hash=OTHER)


def test_a_document_curation_is_version_two_with_groups():
    found = report.Curation(
        DIGEST,
        2,
        2,
        (),
        (("a", "b"),),
        {"en": 2},
        {"text": 2},
        {"unlabeled": 2},
        (),
        (),
        100,
        50,
        0.5,
        ("ocr",),
        3,
        1,
        0.2,
        0.9,
        {"email": 1},
        1.0,
        4,
    )
    written = found.to_dict()
    assert written[records.KEY] == 2
    assert written["counts"] == {
        "languages": {"en": 2},
        "formats": {"text": 2},
        "labels": {"unlabeled": 2},
        "sensitive": {"email": 1},
    }
    assert written["layout"] == {
        "coverage": 0.5,
        "providers": ["ocr"],
        "elements": 3,
    }
    assert written["ocr"] == {"low": 0.2, "high": 0.9}
    assert (written["tables"], written["citations"]) == (1, 4)
    assert (written["characters"], written["tokens"]) == (100, 50)
    assert written["duplicates"] == [["a", "b"]]
    assert written["ready"] is True
    content = {k: v for k, v in written.items() if k != records.KEY}
    old = records.downgrade(report.Curation.migrations, content)
    assert old["ocr_confidence_min"] == 0.2
    assert old["source_traceability_coverage"] == 1.0
    assert curation.identify(written) == (DIGEST, 2)


def test_incremental_evidence_is_version_three_and_downgrades():
    point = timeline.Checkpoint("s", 2)
    receipt = incremental.evidence.Ack("s", ("a",), ("b",), point)
    written = receipt.to_dict()
    assert written[records.KEY] == 3
    content = {k: v for k, v in written.items() if k != records.KEY}
    two = records.downgrade(incremental.evidence.Ack.migrations, content, 2)
    assert two["source_identity"] == "s"
    assert two["checkpoint"] == point.to_dict()
    done = incremental.evidence.Consumption("s", "g", receipt, ("a",))
    stored = done.to_dict()
    assert stored[records.KEY] == 3
    assert (stored["consumer"], stored["acknowledged"]) == ("g", ["a"])


def test_a_compatibility_record_nests_its_snapshots():
    compatibility = registry.Compatibility("p", "c", ("x",), (), ())
    written = compatibility.to_dict()
    assert written["snapshots"] == {"previous": "p", "current": "c"}
    assert written["status"] == "additive"
    old = registry.Compatibility.KEYS.downgrade(written)
    assert old["previous_snapshot_id"] == "p"


def test_store_records_write_single_words():
    rule = quality.Finding(
        "r", quality.Severity.QUARANTINE, False, 1, True, (3,), "d"
    )
    assert rule.to_dict()["granular"] is True
    assessment = quality.Assessment("train", "s1", "d", (rule,), "x", 9, 8)
    written = assessment.to_dict()
    assert written["snapshot"] == "s1"
    assert written["rows"] == {"received": 9, "used": 8}
    assert written["findings"][0]["granular"] is True
    read = filled(
        snapshot.Record,
        size=1,
        rows=1,
        age=2.0,
        staleness=None,
        lineage=("a",),
    )
    stored = read.to_dict()
    assert stored["identifier"] == "identifier"
    assert stored["digests"] == {"schema": "schema_digest"}
    assert stored["schema_version"] == "schema_version"
    assert (stored["age"], stored["staleness"]) == (2.0, None)
    old = snapshot.Record.KEYS.downgrade(stored)
    assert old["read_at"] == "read"


def test_a_baseline_profile_writes_missing_and_reads_versions_one_and_two():
    frame = pandas.DataFrame({"x": [1.0, 2.0, None, 4.0]})
    plan = data.Blueprint(numeric=("x",), categorical=())
    output = pandas.Series(["a", "b", "a", "a"])
    profile = reference.Baseline.build(frame, plan, output, "label", {})
    written = profile.to_dict()
    assert written[records.KEY] == 3
    assert written["features"]["x"]["missing"] == 0.25
    assert "missing_rate" not in written["output"]
    content = {k: v for k, v in written.items() if k != records.KEY}
    two = records.downgrade(reference.Baseline.migrations, content, 2)
    assert two["features"]["x"]["missing_rate"] == 0.25
    loaded = reference.Baseline.from_dict({**two, records.KEY: 2})
    assert loaded == profile
    assert reference.rebuild(two["output"]) == profile.output


def test_a_catalog_writes_identity_and_digest_and_reads_version_two(tmp_path):
    result = manifest.Catalog.observe(
        [{"text": "a"}, {"text": "b"}], source="file:///tmp/r.jsonl"
    )
    written = result.to_dict()
    assert written["schema_version"] == manifest.CATALOG_SCHEMA_VERSION == 3
    assert written["digest"] == result.digest
    assert written["source"]["identity"] == result.source.identity
    assert "catalog_digest" not in written
    content = {k: v for k, v in written.items() if k != "digest"}
    content.pop(records.KEY)
    two = records.downgrade(manifest.Catalog.migrations, content, 2)
    assert "source_identity" in two["source"]
    stored = {**two, records.KEY: 2}
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps({**stored, "catalog_digest": canon.fingerprint(stored)}),
        encoding="utf-8",
    )
    assert manifest.Catalog.read(path) == result
    assert identity.Source.from_dict(two["source"]) == result.source
