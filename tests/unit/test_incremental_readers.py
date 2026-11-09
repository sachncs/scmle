"""Tests for incremental event identity, watermarks, and durable commits."""

import json

import pytest

from scmle.data.readers import incremental


def event(event_id, sequence, minute, value=None):
    return incremental.Event(
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
    ledger = incremental.IngestionLedger(
        tmp_path / "ingestion.json", source.source_identity
    )

    batch = ledger.read_batch(source, limit=1)
    committed = ledger.commit(batch)
    retried = ledger.commit(batch)
    next_batch = ledger.read_batch(source)
    reopened = incremental.IngestionLedger(
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
        "memory://events", records, incremental.LateData.DROP
    ).read_events(checkpoint)
    accepted = incremental.SequenceEventSource(
        "memory://events", records, incremental.LateData.ACCEPT
    ).read_events(checkpoint)

    assert dropped.event_ids == ("fresh",)
    assert dropped.late_event_ids == ("late",)
    assert accepted.event_ids == ("late", "fresh")
    assert accepted.next_checkpoint.sequence == 3
    with pytest.raises(incremental.IncrementalError, match="older"):
        incremental.SequenceEventSource(
            "memory://events", records, incremental.LateData.REJECT
        ).read_events(checkpoint)


def test_accepting_only_late_events_never_moves_the_checkpoint_back():
    source = incremental.SequenceEventSource(
        "memory://events",
        [event("late", 1, 0)],
        incremental.LateData.ACCEPT,
    )
    checkpoint = incremental.EventCheckpoint(
        source.source_identity,
        sequence=5,
        watermark="2026-10-20T10:01:00+00:00",
    )

    batch = source.read_events(checkpoint)

    assert batch.event_ids == ("late",)
    assert batch.next_checkpoint.sequence == 5


def test_event_stream_source_polls_bounded_batches_with_retry_and_checkpoint():
    """Broker callbacks receive bounded, resumable transport positions."""
    calls = []
    records = {
        0: (event("a", 1, 0), event("b", 2, 1)),
        2: (event("c", 3, 2),),
    }

    def read(uri, checkpoint, limit):
        calls.append((uri, checkpoint.sequence, limit))
        if len(calls) == 1:
            raise OSError("broker temporarily unavailable")
        return records.get(checkpoint.sequence, ())

    source = incremental.EventStreamSource(
        "kafka://cluster/orders",
        read,
        retry_budget=incremental.connectors.RetryBudget(max_retries=1),
    )

    first = source.read_events(limit=2)
    second = source.read_events(first.next_checkpoint, first.event_ids, limit=2)

    assert first.event_ids == ("a", "b")
    assert second.event_ids == ("c",)
    assert calls == [
        (source.source_uri, 0, 2),
        (source.source_uri, 0, 2),
        (source.source_uri, 2, 2),
    ]


def test_event_stream_source_fails_closed_for_unbounded_or_oversized_reads():
    """Transport callbacks cannot bypass the bounded event contract."""
    source = incremental.EventStreamSource(
        "pulsar://cluster/topic",
        lambda uri, checkpoint, limit: (
            event("a", 1, 0),
            event("b", 2, 1),
        ),
    )

    with pytest.raises(
        incremental.IncrementalError, match="require read_events"
    ):
        source.available_events()
    with pytest.raises(incremental.IncrementalError, match="more events"):
        source.read_events(limit=1)


def test_event_stream_source_rejects_wrong_checkpoint_before_polling():
    """A broker callback is never called with another source checkpoint."""
    calls = []

    def read(uri, checkpoint, limit):
        calls.append((uri, checkpoint, limit))
        return ()

    source = incremental.EventStreamSource("nats://cluster/topic", read)
    other = incremental.SequenceEventSource(
        "memory://other", ()
    ).source_identity

    with pytest.raises(incremental.IncrementalError, match="different source"):
        source.read_events(incremental.EventCheckpoint(other))
    assert calls == []


def test_incremental_state_rejects_cross_source_reads_and_commits(tmp_path):
    first = incremental.SequenceEventSource(
        "memory://first", [event("a", 1, 0)]
    )
    second = incremental.SequenceEventSource(
        "memory://second", [event("b", 1, 0)]
    )
    ledger = incremental.IngestionLedger(
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
    ledger = incremental.IngestionLedger(
        tmp_path / "sink.json", source.source_identity
    )
    written = []

    first = ledger.commit_with_writer(
        (event("b", 2, 1), event("a", 1, 0)),
        written.extend,
    )
    second = ledger.commit_with_writer(
        (event("a", 1, 0), event("b", 2, 1)),
        written.extend,
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
    ledger = incremental.IngestionLedger(
        tmp_path / "sink.json", source.source_identity
    )

    def fail(events: tuple[incremental.Event, ...]):
        del events
        raise OSError("sink unavailable")

    with pytest.raises(OSError, match="sink unavailable"):
        ledger.commit_with_writer((event("a", 1, 0),), fail)

    assert not ledger.path.exists()
    assert ledger.load().seen_event_ids == ()


def test_event_consumer_commits_before_acknowledging(tmp_path):
    """A broker ack is delivered only after the sink commit is durable."""
    source = incremental.SequenceEventSource(
        "memory://events", [event("a", 1, 0), event("b", 2, 1)]
    )
    ledger = incremental.IngestionLedger(
        tmp_path / "consumer.json", source.source_identity
    )
    written = []
    acknowledgements = []

    consumer = incremental.EventConsumer(
        source,
        ledger,
        "orders-worker",
        written.extend,
        acknowledgements.append,
    )

    result = consumer.consume(limit=2)

    assert [item.event_id for item in written] == ["a", "b"]
    assert result.receipt.accepted_event_ids == ("a", "b")
    assert result.acknowledged_event_ids == ("a", "b")
    assert acknowledgements[0].event_ids == ("a", "b")
    assert not ledger.pending_acknowledgement_path.exists()


def test_event_consumer_retries_ack_without_replaying_the_sink(tmp_path):
    """An ack outage leaves durable evidence for a later idempotent retry."""
    available = [event("a", 1, 0)]
    source = incremental.SequenceEventSource(
        "memory://events", lambda: tuple(available)
    )
    ledger = incremental.IngestionLedger(
        tmp_path / "consumer.json", source.source_identity
    )
    written = []
    attempts = []

    def first_ack_fails(acknowledgement):
        attempts.append(acknowledgement.event_ids)
        if len(attempts) == 1:
            raise OSError("broker unavailable")
        available.clear()

    consumer = incremental.EventConsumer(
        source,
        ledger,
        "orders-worker",
        written.extend,
        first_ack_fails,
    )

    with pytest.raises(incremental.IncrementalError, match="acknowledgement"):
        consumer.consume()
    assert [item.event_id for item in written] == ["a"]
    assert ledger.load().seen_event_ids == ("a",)
    assert ledger.pending_acknowledgement_path.exists()

    result = consumer.consume()

    assert result.receipt.accepted_event_ids == ()
    assert attempts == [("a",), ("a",)]
    assert [item.event_id for item in written] == ["a"]
    assert not ledger.pending_acknowledgement_path.exists()


def test_event_contract_rejects_naive_timestamps_and_invalid_sequences():
    with pytest.raises(ValueError, match="timezone"):
        incremental.Event("a", 1, "2026-10-20T10:00:00", {"value": 1})
    with pytest.raises(ValueError, match="positive"):
        incremental.Event("a", 0, "2026-10-20T10:00:00+00:00", {"value": 1})
