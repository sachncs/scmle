"""Version-2 data-plumbing records: renamed keys, old records still load.

Each fixture in `tests/fixtures/records/v1/plumbing` was written by the
writer that existed before the keys of the catalog, source registry,
incremental and streaming readers and detector certificates were renamed.
It must load through the kind's migration, and every digest it carries
(the catalog digest, the source identity, the certificate code hash) must
still verify against the bytes as stored.
"""

import json
import pathlib

import hypothesis
from hypothesis import strategies
import pytest

from scmle.assurance.detectors import certificates
from scmle.contracts import records
from scmle.data import catalog
from scmle.data.catalog import canon
from scmle.data.readers import registry
from scmle.data.readers.incremental import evidence
from scmle.data.readers.incremental import timeline
from scmle.data.readers.streaming import progress
from scmle.data.readers.streaming import transaction

FIXTURES = (
    pathlib.Path(__file__).parents[1] / "fixtures" / "records" / "v1"
) / "plumbing"
TEXT = strategies.text(
    strategies.characters(min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=12,
)
COUNT = strategies.integers(0, 10**6)
DIGEST = strategies.text("0123456789abcdef", min_size=64, max_size=64)


def stored(name):
    """The parsed version-1 fixture `name`."""
    return json.loads((FIXTURES / f"{name}.json").read_text("utf-8"))


def back(kind, layout):
    """Moves a current layout of `kind` down to version 1.

    Args:
        kind: A record kind with `migrations`.
        layout: The kind's current layout, with its version key.

    Returns:
        The version-1 layout, with `schema_version` 1.
    """
    rest = {key: value for key, value in layout.items() if key != records.KEY}
    return {records.KEY: 1, **records.downgrade(kind.migrations, rest)}


def test_every_renamed_kind_is_at_version_two_and_injective():
    kinds = (
        catalog.Catalog,
        catalog.Source,
        catalog.Field,
        catalog.Missingness,
        catalog.Duplication,
        catalog.Labeling,
        registry.Observation,
        registry.Sources,
        timeline.Event,
        timeline.Checkpoint,
        evidence.Acknowledgement,
        evidence.Settlement,
        progress.Checkpoint,
        transaction.Acknowledgement,
        transaction.Voucher,
        transaction.Commitment,
        certificates.Standard,
        certificates.Outcome,
        certificates.Certification,
    )
    for kind in kinds:
        step = kind.migrations[-1]
        assert len(set(step.renames.values())) == len(step.renames)
        assert (step.change is None) == (step.revert is None)
    assert catalog.CATALOG_SCHEMA_VERSION == 3
    assert (
        records.ORIGIN + len(transaction.Settlement.migrations)
        == transaction.STREAM_STATE_SCHEMA
    )


def test_a_version_one_catalog_verifies_its_digest_then_migrates(tmp_path):
    layout = stored("catalog")
    assert layout["schema_version"] == 1
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(layout), encoding="utf-8")
    loaded = catalog.Catalog.read(path)
    assert loaded.schema_version == 3
    assert loaded.kind == layout["data_kind"]
    assert loaded.samples == layout["sample_count"]
    assert loaded.digests == {"content": layout["content_identity"]}
    assert loaded.blockers == tuple(layout["release_blockers"])
    assert [field.kind for field in loaded.columns] == [
        field["data_type"] for field in layout["schema"]
    ]
    assert (
        loaded.missingness.incomplete
        == (layout["missingness"]["rows_with_missing"])
    )
    assert loaded.label.distinct == layout["label"]["distinct_count"]
    assert loaded.duplicates.rows == layout["duplicates"]["duplicate_rows"]
    assert loaded.source.acquisition == "warehouse-export"
    loaded.write(path)
    assert catalog.Catalog.read(path) == loaded


def test_a_tampered_version_one_catalog_is_refused(tmp_path):
    layout = stored("catalog")
    layout["sample_count"] += 1
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(layout), encoding="utf-8")
    with pytest.raises(ValueError, match="catalog digest"):
        catalog.Catalog.read(path)


def test_the_source_identity_is_anchored_at_version_one():
    layout = stored("catalog")["source"]
    source = catalog.Source.from_dict(
        catalog.Source.migrations[0].upgrade(layout)
    )
    assert source.identity == layout["source_identity"]
    fields = {k: v for k, v in layout.items() if k != "source_identity"}
    assert source.identity == canon.fingerprint(fields)


@hypothesis.given(samples=COUNT, kind=TEXT, owner=TEXT)
def test_a_catalog_survives_both_migration_directions(samples, kind, owner):
    base = catalog.Catalog.from_dict(stored("catalog"))
    layout = base.contents()
    layout["samples"] = samples
    layout["kind"] = kind
    source = catalog.Source(uri="memory://x", owner=owner)
    layout["source"] = source.to_dict()
    record = catalog.Catalog.from_dict(layout)
    assert catalog.Catalog.from_dict(record.contents()) == record
    older = back(catalog.Catalog, record.contents())
    assert "data_kind" in older and "kind" not in older
    assert older["source"]["acquisition_method"] == ""
    assert catalog.Catalog.from_dict(older) == record


def test_a_version_one_source_registry_loads():
    layout = stored("sources")
    restored = registry.Sources.from_dict(layout)
    old = layout["sources"][0]["snapshots"][0]
    snapshot = restored.latest(old["source_id"])
    assert snapshot is not None
    assert snapshot.digests == {
        "schema": old["schema_hash"],
        "data": old["data_hash"],
        "raw": old["raw_source_hash"],
    }
    assert (snapshot.rows, snapshot.layout, snapshot.captured) == (
        old["row_count"],
        old["schema"],
        old["captured_at"],
    )
    assert restored.to_dict()["schema_version"] == 2


@hypothesis.given(rows=COUNT, snapshot=TEXT, digest=DIGEST)
def test_a_source_registry_survives_both_directions(rows, snapshot, digest):
    observation = registry.Observation(
        source="src_a",
        uri="file:///a.csv",
        snapshot=snapshot,
        digests={"schema": digest, "data": digest, "raw": ""},
        rows=rows,
        layout={"a": "int64"},
        captured="2026-01-01T00:00:00+00:00",
    )
    sources = registry.Sources()
    sources.register(observation)
    layout = sources.to_dict()
    assert registry.Sources.from_dict(layout).snapshots == sources.snapshots
    older = back(registry.Sources, layout)
    assert older["sources"][0]["snapshots"][0]["row_count"] == rows
    assert registry.Sources.from_dict(older).snapshots == sources.snapshots


def test_a_version_one_event_loads():
    layout = stored("event")
    event = timeline.Event.from_dict(layout)
    assert (event.identifier, event.occurred, event.ingested) == (
        layout["event_id"],
        layout["event_time"],
        layout["ingest_time"],
    )
    assert back(timeline.Event, event.to_dict()) == {
        records.KEY: 1,
        **layout,
    }


@hypothesis.given(identifier=TEXT, sequence=strategies.integers(1, 10**6))
def test_an_event_survives_both_directions(identifier, sequence):
    event = timeline.Event(
        identifier, sequence, "2026-01-01T00:00:00+00:00", {"v": sequence}
    )
    assert timeline.Event.from_dict(event.to_dict()) == event
    assert timeline.Event.from_dict(back(timeline.Event, event.to_dict())) == (
        event
    )


def test_version_one_incremental_state_and_acknowledgement_load():
    state = stored("incremental_settlement")
    settled = evidence.Settlement.from_dict(state)
    assert settled.seen == tuple(state["seen_event_ids"])
    assert settled.checkpoint.source == state["checkpoint"]["source_identity"]
    pending = stored("incremental_acknowledgement")
    acknowledgement = evidence.Acknowledgement.from_dict(pending)
    assert (acknowledgement.consumer, acknowledgement.events) == (
        pending["consumer_group"],
        tuple(pending["event_ids"]),
    )
    assert back(evidence.Settlement, settled.to_dict()) == state
    assert back(evidence.Acknowledgement, acknowledgement.to_dict()) == (
        pending
    )


@hypothesis.given(source=TEXT, sequence=COUNT, seen=strategies.sets(TEXT))
def test_incremental_state_survives_both_directions(source, sequence, seen):
    state = evidence.Settlement(
        timeline.Checkpoint(source, sequence), tuple(sorted(seen))
    )
    for layout in (state.to_dict(), back(evidence.Settlement, state.to_dict())):
        assert evidence.Settlement.from_dict(layout) == state


def test_version_one_streaming_state_and_acknowledgement_load():
    state = stored("streaming_settlement")
    settled = transaction.Settlement.from_dict(state)
    commitment = settled.commitments[0]
    old = state["transaction_evidence"][0]
    assert commitment.batch == old["batch_id"]
    assert commitment.receipt.receipt == old["sink_receipt"]["receipt"]
    assert settled.checkpoint.digests == {
        "snapshot": state["checkpoint"]["snapshot_sha256"]
    }
    assert settled.checkpoint.column == state["checkpoint"]["watermark_column"]
    assert settled.committed == tuple(state["committed_batch_ids"])
    pending = stored("streaming_acknowledgement")
    acknowledgement = transaction.Acknowledgement.from_dict(pending)
    assert acknowledgement.batch == pending["batch_id"]
    assert back(transaction.Acknowledgement, acknowledgement.to_dict()) == (
        pending
    )
    assert (
        back(transaction.Commitment, commitment.to_dict())["sink_receipt"]
        == old["sink_receipt"]
    )


@hypothesis.given(offset=COUNT, digest=DIGEST, consumer=TEXT)
def test_streaming_state_survives_both_directions(offset, digest, consumer):
    point = progress.Checkpoint("s", {"snapshot": digest}, offset, "", "t")
    batch = "batch_" + digest
    acknowledgement = transaction.Acknowledgement("s", consumer, batch, point)
    voucher = transaction.Voucher("s", batch, point, "r")
    commitment = transaction.Commitment("s", batch, voucher, acknowledgement)
    state = transaction.Settlement(point, (batch,), (commitment,))
    older = back(transaction.Settlement, state.to_dict())
    assert "committed_batch_ids" in older
    assert older["checkpoint"]["snapshot_sha256"] == digest
    assert transaction.Settlement.from_dict(older) == state
    assert transaction.Settlement.from_dict(state.to_dict()) == state


def test_a_version_one_certificate_store_keeps_its_code_hash(tmp_path):
    document = stored("detector_certificates")
    old = document["certificates"][0]
    store = certificates.Store.from_dict(document)
    (certificate,) = store.certificates
    assert certificate.digests == {
        "code": old["code_hash"],
        "corpus": old["corpus_hash"],
    }
    assert store.state(old["detector_id"], old["code_hash"]) is (
        certificates.Freshness.CURRENT
    )
    assert certificate.policy.recall == old["policy"]["recall_floor"]
    path = tmp_path / "store.json"
    store.save(path)
    assert certificates.Store.load(path).certificates == store.certificates


def test_a_tampered_version_one_certificate_is_refused():
    document = stored("detector_certificates")
    outcome = document["certificates"][0]["outcomes"][0]
    outcome["recall_interval"] = [0.0, 0.5]
    with pytest.raises(certificates.CertificateError, match="interval"):
        certificates.Store.from_dict(document)


@hypothesis.given(created=strategies.none() | TEXT)
def test_a_certificate_survives_both_directions(created):
    document = stored("detector_certificates")
    first = certificates.Store.from_dict(document).certificates[0]
    layout = {**first.to_dict(), "created": created}
    certificate = certificates.Certification.from_dict(layout)
    older = back(certificates.Certification, certificate.to_dict())
    assert older["code_hash"] == first.digests["code"]
    assert certificates.Certification.from_dict(older) == certificate
    policy = back(certificates.Standard, certificate.policy.to_dict())
    assert certificates.Standard.from_dict(policy) == certificate.policy
