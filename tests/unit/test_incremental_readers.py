"""Tests for incremental event identity, watermarks, and durable commits."""

import json

import pytest

from scml.data.readers import incremental


def event(event_id, sequence, minute, value=None):
    return incremental.IncrementalEvent(
        event_id=event_id,
        sequence=sequence,
        event_time=f"2026-10-20T10:{minute:02d}:00+00:00",
        payload={"value": sequence if value is None else value},
    )


def test_incremental_source_batches_are_bounded_and_deduplicated():
    source = incremental.SequenceEventSource(
        "memory://events",
        [
            event("a", 1, 0),
            event("b", 2, 1),
            event("b", 3, 2),
            event("c", 4, 3),
        ],
    )

    first = source.read_events(limit=2)
    second = source.read_events(first.next_checkpoint, first.event_ids, limit=2)

    assert first.event_ids == ("a", "b")
    assert first.next_checkpoint.sequence == 2
    assert first.next_checkpoint.watermark.endswith("01:00+00:00")
    assert first.has_more
    assert second.event_ids == ("c",)
    assert second.duplicate_event_ids == ("a", "b")
    assert second.next_checkpoint.sequence == 4
    assert not second.has_more


def test_ingestion_ledger_commits_atomically_and_retries_idempotently(tmp_path):
    source = incremental.SequenceEventSource(
        "memory://events", [event("a", 1, 0), event("b", 2, 1)]
    )
    ledger = incremental.IncrementalIngestionLedger(
        tmp_path / "ingestion.json", source.source_identity
    )

    batch = ledger.read_batch(source, limit=1)
    committed = ledger.commit(batch)
    retried = ledger.commit(batch)
    next_batch = ledger.read_batch(source)
    reopened = incremental.IncrementalIngestionLedger(
        ledger.path, source.source_identity
    ).load()

    assert committed == retried == reopened
    assert committed.checkpoint.sequence == 1
    assert next_batch.event_ids == ("b",)
    assert ledger.lock_path.exists()
    assert json.loads(ledger.path.read_text())["schema_version"] == 1


def test_late_data_policy_is_explicit():
    records = [event("late", 1, 0), event("fresh", 3, 2)]
    source_identity = incremental.SequenceEventSource(
        "memory://events", records
    ).source_identity
    checkpoint = incremental.EventCheckpoint(
        source_identity, sequence=2, watermark="2026-10-20T10:01:00+00:00"
    )

    dropped = incremental.SequenceEventSource(
        "memory://events", records, incremental.LateDataPolicy.DROP
    ).read_events(checkpoint)
    accepted = incremental.SequenceEventSource(
        "memory://events", records, incremental.LateDataPolicy.ACCEPT
    ).read_events(checkpoint)

    assert dropped.event_ids == ("fresh",)
    assert dropped.late_event_ids == ("late",)
    assert accepted.event_ids == ("late", "fresh")
    assert accepted.next_checkpoint.sequence == 3
    with pytest.raises(incremental.IncrementalError, match="older"):
        incremental.SequenceEventSource(
            "memory://events", records, incremental.LateDataPolicy.REJECT
        ).read_events(checkpoint)


def test_incremental_state_rejects_cross_source_reads_and_commits(tmp_path):
    first = incremental.SequenceEventSource(
        "memory://first", [event("a", 1, 0)]
    )
    second = incremental.SequenceEventSource(
        "memory://second", [event("b", 1, 0)]
    )
    ledger = incremental.IncrementalIngestionLedger(
        tmp_path / "ingestion.json", first.source_identity
    )

    with pytest.raises(incremental.IncrementalError, match="does not match"):
        ledger.read_batch(second)
    batch = ledger.read_batch(first)
    with pytest.raises(incremental.IncrementalError, match="does not match"):
        ledger.commit(
            incremental.EventBatch(
                second.source_identity,
                batch.events,
                incremental.EventCheckpoint(second.source_identity, 1),
            )
        )


def test_idempotent_sink_commit_filters_duplicates_and_persists_receipt(
    tmp_path,
):
    source = incremental.SequenceEventSource(
        "memory://events", [event("a", 1, 0), event("b", 2, 1)]
    )
    ledger = incremental.IncrementalIngestionLedger(
        tmp_path / "sink.json", source.source_identity
    )
    written = []

    first = ledger.commit_with_writer(
        (event("b", 2, 1), event("a", 1, 0)),
        lambda values: written.extend(values),
    )
    second = ledger.commit_with_writer(
        (event("a", 1, 0), event("b", 2, 1)),
        lambda values: written.extend(values),
    )

    assert first.accepted_event_ids == ("a", "b")
    assert second.accepted_event_ids == ()
    assert second.duplicate_event_ids == ("a", "b")
    assert [item.event_id for item in written] == ["a", "b"]
    assert first.to_dict()["checkpoint"]["sequence"] == 2


def test_idempotent_sink_keeps_state_unchanged_when_writer_fails(tmp_path):
    source = incremental.SequenceEventSource(
        "memory://events", [event("a", 1, 0)]
    )
    ledger = incremental.IncrementalIngestionLedger(
        tmp_path / "sink.json", source.source_identity
    )

    def fail(_: tuple[incremental.IncrementalEvent, ...]):
        raise OSError("sink unavailable")

    with pytest.raises(OSError, match="sink unavailable"):
        ledger.commit_with_writer((event("a", 1, 0),), fail)

    assert not ledger.path.exists()
    assert ledger.load().seen_event_ids == ()


def test_event_contract_rejects_naive_timestamps_and_invalid_sequences():
    with pytest.raises(ValueError, match="timezone"):
        event_value = incremental.IncrementalEvent(
            "a", 1, "2026-10-20T10:00:00", {"value": 1}
        )
        del event_value
    with pytest.raises(ValueError, match="positive"):
        incremental.IncrementalEvent(
            "a", 0, "2026-10-20T10:00:00+00:00", {"value": 1}
        )
