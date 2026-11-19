"""Tests for incremental event identity, watermarks, and durable commits."""

import json

import pytest

from scmle.data.readers import incremental


def event(event_id, sequence, minute, value=None):
    return incremental.Event(
        identifier=event_id,
        sequence=sequence,
        occurred=f"2026-10-20T10:{minute:02d}:00+00:00",
        payload={"value": sequence if value is None else value},
    )


def test_incremental_source_batches_are_bounded_and_deduplicated():
    source = incremental.Replay(
        "memory://events",
        [
            event("a", 1, 0),
            event("b", 2, 1),
            event("b", 3, 2),
            event("c", 4, 3),
        ],
    )

    first = source.read(limit=2)
    second = source.read(first.checkpoint, first.identities, limit=2)

    assert first.identities == ("a", "b")
    assert first.checkpoint.sequence == 2
    assert first.checkpoint.watermark.endswith("01:00+00:00")
    assert first.more
    assert second.identities == ("c",)
    assert second.duplicates == ("a", "b")
    assert second.checkpoint.sequence == 4
    assert not second.more


def test_ingestion_ledger_commits_atomically_and_retries_idempotently(tmp_path):
    source = incremental.Replay(
        "memory://events", [event("a", 1, 0), event("b", 2, 1)]
    )
    ledger = incremental.Ledger(tmp_path / "ingestion.json", source.identity)

    batch = ledger.read(source, limit=1)
    committed = ledger.commit(batch)
    retried = ledger.commit(batch)
    next_batch = ledger.read(source)
    reopened = incremental.Ledger(ledger.path, source.identity).load()

    assert committed == retried == reopened
    assert committed.checkpoint.sequence == 1
    assert next_batch.identities == ("b",)
    assert ledger.lockfile.exists()
    assert json.loads(ledger.path.read_text())["schema_version"] == 2


def test_late_data_policy_is_explicit():
    records = [event("late", 1, 0), event("fresh", 3, 2)]
    source_identity = incremental.Replay("memory://events", records).identity
    checkpoint = incremental.Checkpoint(
        source_identity, sequence=2, watermark="2026-10-20T10:01:00+00:00"
    )

    dropped = incremental.Replay(
        "memory://events", records, incremental.Lateness.DROP
    ).read(checkpoint)
    accepted = incremental.Replay(
        "memory://events", records, incremental.Lateness.ACCEPT
    ).read(checkpoint)

    assert dropped.identities == ("fresh",)
    assert dropped.late == ("late",)
    assert accepted.identities == ("late", "fresh")
    assert accepted.checkpoint.sequence == 3
    with pytest.raises(incremental.IncrementalError, match="older"):
        incremental.Replay(
            "memory://events", records, incremental.Lateness.REJECT
        ).read(checkpoint)


def test_accepting_only_late_events_never_moves_the_checkpoint_back():
    source = incremental.Replay(
        "memory://events",
        [event("late", 1, 0)],
        incremental.Lateness.ACCEPT,
    )
    checkpoint = incremental.Checkpoint(
        source.identity,
        sequence=5,
        watermark="2026-10-20T10:01:00+00:00",
    )

    batch = source.read(checkpoint)

    assert batch.identities == ("late",)
    assert batch.checkpoint.sequence == 5


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

    source = incremental.Stream(
        "kafka://cluster/orders",
        read,
        budget=incremental.connectors.Retries(limit=1),
    )

    first = source.read(limit=2)
    second = source.read(first.checkpoint, first.identities, limit=2)

    assert first.identities == ("a", "b")
    assert second.identities == ("c",)
    assert calls == [
        (source.uri, 0, 2),
        (source.uri, 0, 2),
        (source.uri, 2, 2),
    ]


def test_event_stream_source_fails_closed_for_unbounded_or_oversized_reads():
    """Transport callbacks cannot bypass the bounded event contract.

    Also enforces website/src/pages/docs/data.astro:17.
    """
    source = incremental.Stream(
        "pulsar://cluster/topic",
        lambda uri, checkpoint, limit: (
            event("a", 1, 0),
            event("b", 2, 1),
        ),
    )

    with pytest.raises(
        incremental.IncrementalError, match="require read_events"
    ):
        source.unconsumed()
    with pytest.raises(incremental.IncrementalError, match="more events"):
        source.read(limit=1)


def test_event_stream_source_rejects_wrong_checkpoint_before_polling():
    """A broker callback is never called with another source checkpoint."""
    calls = []

    def read(uri, checkpoint, limit):
        calls.append((uri, checkpoint, limit))
        return ()

    source = incremental.Stream("nats://cluster/topic", read)
    other = incremental.Replay("memory://other", ()).identity

    with pytest.raises(incremental.IncrementalError, match="different source"):
        source.read(incremental.Checkpoint(other))
    assert calls == []


def test_incremental_state_rejects_cross_source_reads_and_commits(tmp_path):
    first = incremental.Replay("memory://first", [event("a", 1, 0)])
    second = incremental.Replay("memory://second", [event("b", 1, 0)])
    ledger = incremental.Ledger(tmp_path / "ingestion.json", first.identity)

    with pytest.raises(incremental.IncrementalError, match="does not match"):
        ledger.read(second)
    batch = ledger.read(first)
    with pytest.raises(incremental.IncrementalError, match="does not match"):
        ledger.commit(
            incremental.Batch(
                second.identity,
                batch.events,
                incremental.Checkpoint(second.identity, 1),
            )
        )


def test_idempotent_sink_commit_filters_duplicates_and_persists_receipt(
    tmp_path,
):
    # Enforces website/src/pages/docs/data.astro:152.
    source = incremental.Replay(
        "memory://events", [event("a", 1, 0), event("b", 2, 1)]
    )
    ledger = incremental.Ledger(tmp_path / "sink.json", source.identity)
    written = []

    first = ledger.finalize(
        (event("b", 2, 1), event("a", 1, 0)),
        written.extend,
    )
    second = ledger.finalize(
        (event("a", 1, 0), event("b", 2, 1)),
        written.extend,
    )

    assert first.accepted == ("a", "b")
    assert second.accepted == ()
    assert second.duplicates == ("a", "b")
    assert [item.identifier for item in written] == ["a", "b"]
    assert first.to_dict()["checkpoint"]["sequence"] == 2


def test_idempotent_sink_keeps_state_unchanged_when_writer_fails(tmp_path):
    source = incremental.Replay("memory://events", [event("a", 1, 0)])
    ledger = incremental.Ledger(tmp_path / "sink.json", source.identity)

    def fail(events: tuple[incremental.Event, ...]):
        del events
        raise OSError("sink unavailable")

    with pytest.raises(OSError, match="sink unavailable"):
        ledger.finalize((event("a", 1, 0),), fail)

    assert not ledger.path.exists()
    assert ledger.load().seen == ()


def test_event_consumer_commits_before_acknowledging(tmp_path):
    """A broker ack is delivered only after the sink commit is durable."""
    source = incremental.Replay(
        "memory://events", [event("a", 1, 0), event("b", 2, 1)]
    )
    ledger = incremental.Ledger(tmp_path / "consumer.json", source.identity)
    written = []
    acknowledgements = []

    consumer = incremental.Ingestor(
        source,
        ledger,
        "orders-worker",
        written.extend,
        acknowledgements.append,
    )

    result = consumer.consume(limit=2)

    assert [item.identifier for item in written] == ["a", "b"]
    assert result.receipt.accepted == ("a", "b")
    assert result.acknowledged == ("a", "b")
    assert acknowledgements[0].events == ("a", "b")
    assert not ledger.pending.exists()


def test_event_consumer_retries_ack_without_replaying_the_sink(tmp_path):
    """An ack outage leaves durable evidence for a later idempotent retry."""
    available = [event("a", 1, 0)]
    source = incremental.Replay("memory://events", lambda: tuple(available))
    ledger = incremental.Ledger(tmp_path / "consumer.json", source.identity)
    written = []
    attempts = []

    def first_ack_fails(acknowledgement):
        attempts.append(acknowledgement.events)
        if len(attempts) == 1:
            raise OSError("broker unavailable")
        available.clear()

    consumer = incremental.Ingestor(
        source,
        ledger,
        "orders-worker",
        written.extend,
        first_ack_fails,
    )

    with pytest.raises(incremental.IncrementalError, match="acknowledgement"):
        consumer.consume()
    assert [item.identifier for item in written] == ["a"]
    assert ledger.load().seen == ("a",)
    assert ledger.pending.exists()

    result = consumer.consume()

    assert result.receipt.accepted == ()
    assert attempts == [("a",), ("a",)]
    assert [item.identifier for item in written] == ["a"]
    assert not ledger.pending.exists()


def test_event_contract_rejects_naive_timestamps_and_invalid_sequences():
    with pytest.raises(ValueError, match="timezone"):
        incremental.Event("a", 1, "2026-10-20T10:00:00", {"value": 1})
    with pytest.raises(ValueError, match="positive"):
        incremental.Event("a", 0, "2026-10-20T10:00:00+00:00", {"value": 1})
