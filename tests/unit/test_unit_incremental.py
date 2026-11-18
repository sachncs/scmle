"""Behavioural tests for incremental sources, the ledger and the consumer."""

import json

import pytest

from scmle.contracts import errors
from scmle.data.readers import connectors
from scmle.data.readers.incremental import consumer
from scmle.data.readers.incremental import evidence
from scmle.data.readers.incremental import sources
from scmle.data.readers.incremental import store
from scmle.data.readers.incremental import timeline

URI = "memory://events"


def event(identifier, sequence, minute=0):
    return timeline.Event(
        identifier=identifier,
        sequence=sequence,
        occurred=f"2026-01-01T00:{minute:02d}:00+00:00",
        payload={"v": sequence},
    )


def replay(*events, policy=timeline.Lateness.REJECT):
    return sources.SequenceEventSource(URI, events, policy)


def identity():
    return connectors.source_identity(URI)


def ledger(tmp_path, source_identity=None):
    return store.Ledger(tmp_path / "state.json", source_identity or identity())


def test_a_source_canonicalises_its_uri_and_policy():
    source = sources.SequenceEventSource("MEMORY://events", [], "drop")
    assert source.source_uri == "memory://events"
    assert source.source_identity == identity()
    assert source.late_data_policy is timeline.Lateness.DROP
    with pytest.raises(ValueError, match="nope"):
        sources.SequenceEventSource(URI, [], "nope")


def test_rebind_defaults_and_rejects_foreign_checkpoints():
    source = replay()
    fresh = source.rebind(None)
    assert (fresh.source, fresh.sequence, fresh.watermark) == (
        identity(),
        0,
        "",
    )
    with pytest.raises(timeline.IncrementalError) as raised:
        source.rebind(timeline.Checkpoint("src_other", 1))
    assert raised.value.code == "checkpoint_mismatch"


def test_select_events_orders_bounds_and_advances():
    source = replay(event("b", 2, 5), event("a", 1, 9), event("c", 3, 1))
    batch = source.read_events(limit=2)
    assert batch.event_ids == ("a", "b")
    assert batch.has_more
    assert batch.next_checkpoint.sequence == 2
    assert batch.next_checkpoint.watermark == "2026-01-01T00:09:00+00:00"
    rest = source.read_events(batch.next_checkpoint, batch.event_ids, 5)
    assert rest.event_ids == ("c",)
    assert not rest.has_more
    assert rest.duplicate_event_ids == ("a", "b")
    assert rest.next_checkpoint.watermark == "2026-01-01T00:09:00+00:00"


def test_select_events_keeps_the_checkpoint_when_nothing_is_new():
    source = replay(event("a", 1))
    checkpoint = timeline.Checkpoint(identity(), 1, "2026-01-01T00:30:00+00:00")
    batch = source.read_events(checkpoint, ("a",))
    assert batch.events == ()
    assert batch.next_checkpoint == checkpoint
    assert batch.duplicate_event_ids == ("a",)
    assert not batch.has_more


def test_select_events_marks_repeats_in_one_poll_as_duplicates():
    batch = replay(event("a", 1), event("a", 2)).read_events()
    assert batch.event_ids == ("a",)
    assert batch.duplicate_event_ids == ("a",)


def test_select_events_validates_inputs():
    source = replay()
    with pytest.raises(errors.InputError, match="limit"):
        source.select_events([], limit=0)
    with pytest.raises(errors.KindError, match="Event values"):
        source.select_events(["x"])


@pytest.mark.parametrize(
    ("policy", "selected", "late"),
    [
        (timeline.Lateness.DROP, ("b",), ("a",)),
        (timeline.Lateness.ACCEPT, ("a", "b"), ("a",)),
    ],
)
def test_late_events_follow_the_policy(policy, selected, late):
    source = replay(event("a", 2), event("b", 3), policy=policy)
    checkpoint = timeline.Checkpoint(identity(), 2)
    batch = source.read_events(checkpoint)
    assert batch.event_ids == selected
    assert batch.late_event_ids == late
    assert batch.next_checkpoint.sequence == 3


def test_a_late_event_is_rejected_by_default():
    source = replay(event("a", 2))
    with pytest.raises(timeline.IncrementalError) as raised:
        source.read_events(timeline.Checkpoint(identity(), 2))
    assert raised.value.code == "late_event"
    assert raised.value.details["checkpoint_sequence"] == 2
    accepted = replay(event("a", 3)).read_events(
        timeline.Checkpoint(identity(), 2)
    )
    assert accepted.event_ids == ("a",)


def test_an_accepted_late_event_never_moves_the_checkpoint_back():
    source = replay(event("a", 1), policy=timeline.Lateness.ACCEPT)
    batch = source.read_events(timeline.Checkpoint(identity(), 5))
    assert batch.next_checkpoint.sequence == 5


def test_a_sequence_source_can_use_a_factory():
    produced = [event("a", 1)]
    source = sources.SequenceEventSource(URI, lambda: tuple(produced))
    assert source.read_events().event_ids == ("a",)
    produced.append(event("b", 2))
    assert source.read_events().event_ids == ("a", "b")


def test_a_stream_polls_through_its_reader():
    calls = []

    def poll(uri, checkpoint, limit):
        calls.append((uri, checkpoint.sequence, limit))
        return [event("a", 1), event("b", 2)]

    source = sources.Stream(
        " kafka://t ", poll, retry_budget=connectors.Retries(0)
    )
    batch = source.read_events(limit=2)
    assert batch.event_ids == ("a", "b")
    assert calls == [("kafka://t", 0, 2)]
    with pytest.raises(timeline.IncrementalError) as raised:
        source.read_events(limit=1)
    assert raised.value.code == "batch_limit_exceeded"
    with pytest.raises(errors.InputError, match="limit"):
        source.poll_events(source.rebind(None), 0)
    with pytest.raises(timeline.IncrementalError) as raised:
        source.unconsumed()
    assert raised.value.code == "bounded_poll_required"
    assert source.retry_budget == connectors.Retries(0)
    assert (
        sources.Stream("kafka://t", poll).retry_budget == connectors.Retries()
    )


def test_a_stream_validates_its_configuration():
    with pytest.raises(connectors.SchemeError):
        sources.Stream("http://t", lambda *a: [])
    with pytest.raises(errors.KindError, match="callable"):
        sources.Stream("kafka://t", None)


def test_a_failing_stream_poll_is_retried_then_reported():
    calls = []

    def poll(uri, checkpoint, limit):
        calls.append(1)
        raise OSError("down")

    source = sources.Stream(
        "kafka://t", poll, retry_budget=connectors.Retries(1)
    )
    with pytest.raises(connectors.RetryError):
        source.read_events()
    assert len(calls) == 2


def test_publish_writes_sorted_json_and_cleans_up(tmp_path):
    target = tmp_path / "deep" / "state.json"
    store.publish(target, {"b": 1, "a": 2})
    assert target.read_text() == '{"a": 2, "b": 1}\n'
    with pytest.raises(TypeError):
        store.publish(target, {"a": object()})
    assert sorted(p.name for p in target.parent.iterdir()) == ["state.json"]
    assert json.loads(target.read_text()) == {"a": 2, "b": 1}


def test_partition_separates_unseen_and_duplicate_events():
    accepted, duplicates = store.partition(
        [event("b", 2), event("a", 1), event("b", 3), event("c", 4)], ["c"]
    )
    assert [e.identifier for e in accepted] == ["a", "b"]
    assert [e.sequence for e in accepted] == [1, 2]
    assert duplicates == {"c"}


def test_a_ledger_loads_an_empty_state(tmp_path):
    state = ledger(tmp_path).load()
    assert state.checkpoint == timeline.Checkpoint(identity())
    assert state.seen == ()


def test_a_ledger_commits_and_refuses_regressions(tmp_path):
    book = ledger(tmp_path)
    batch = replay(event("a", 1), event("b", 2)).read_events()
    state = book.commit(batch)
    assert state.seen == ("a", "b")
    assert state.checkpoint.sequence == 2
    assert book.load() == state
    assert book.commit(batch) == state
    behind = timeline.Batch(
        identity(), (event("z", 1),), timeline.Checkpoint(identity(), 1)
    )
    with pytest.raises(timeline.IncrementalError) as raised:
        book.commit(behind)
    assert raised.value.code == "checkpoint_regression"
    foreign = timeline.Batch("src_x", (), timeline.Checkpoint("src_x"))
    with pytest.raises(timeline.IncrementalError) as raised:
        book.commit(foreign)
    assert raised.value.code == "batch_source_mismatch"


def test_a_ledger_rejects_foreign_state_and_sources(tmp_path):
    book = ledger(tmp_path)
    with pytest.raises(timeline.IncrementalError) as raised:
        book.write_atomically(
            evidence.Settlement(timeline.Checkpoint("src_other"))
        )
    assert raised.value.code == "state_source_mismatch"
    ledger(tmp_path, "src_other").write_atomically(
        evidence.Settlement(timeline.Checkpoint("src_other"))
    )
    with pytest.raises(timeline.IncrementalError) as raised:
        book.load()
    assert raised.value.code == "state_source_mismatch"
    with pytest.raises(timeline.IncrementalError) as raised:
        book.read_batch(sources.SequenceEventSource("memory://other", []))
    assert raised.value.code == "source_mismatch"


def test_read_batch_uses_the_durable_state(tmp_path):
    book = ledger(tmp_path)
    source = replay(event("a", 1), event("b", 2))
    book.commit(source.read_events(limit=1))
    batch = book.read_batch(source)
    assert batch.event_ids == ("b",)
    assert batch.duplicate_event_ids == ("a",)


def test_finalize_writes_only_unseen_events(tmp_path):
    book = ledger(tmp_path)
    written = []
    receipt = book.finalize(
        [event("b", 2, 3), event("a", 1, 1)], written.append
    )
    assert [e.identifier for e in written[0]] == ["a", "b"]
    assert receipt.accepted_event_ids == ("a", "b")
    assert receipt.duplicate_event_ids == ()
    assert receipt.checkpoint.sequence == 2
    assert receipt.checkpoint.watermark == "2026-01-01T00:03:00+00:00"
    again = book.finalize([event("a", 1), event("c", 3)], written.append)
    assert [e.identifier for e in written[1]] == ["c"]
    assert again.accepted_event_ids == ("c",)
    assert again.duplicate_event_ids == ("a",)
    idle = book.finalize([event("a", 1)], written.append)
    assert len(written) == 2
    assert idle.accepted_event_ids == ()
    assert idle.checkpoint == again.checkpoint


def test_finalize_accepts_a_repeat_within_one_call(tmp_path):
    written = []
    book = ledger(tmp_path)
    receipt = book.finalize([event("a", 1), event("a", 1)], written.append)
    assert receipt.accepted_event_ids == ("a",)
    assert receipt.duplicate_event_ids == ()
    assert [e.identifier for e in written[0]] == ["a"]
    again = book.finalize([event("a", 1), event("a", 2)], written.append)
    assert again.accepted_event_ids == ()
    assert again.duplicate_event_ids == ("a",)


def test_finalize_leaves_the_ledger_unchanged_when_the_writer_fails(tmp_path):
    book = ledger(tmp_path)

    def failing(events):
        raise RuntimeError("sink down")

    with pytest.raises(RuntimeError, match="sink down"):
        book.finalize([event("a", 1)], failing)
    assert not book.path.exists()


def test_finalize_validates_its_inputs(tmp_path):
    book = ledger(tmp_path)
    with pytest.raises(errors.KindError, match="Event values"):
        book.finalize(["x"], print)
    with pytest.raises(errors.KindError, match="writer"):
        book.finalize([], None)
    foreign = evidence.EventAcknowledgement(
        "src_x", "g", ("a",), timeline.Checkpoint("src_x")
    )
    with pytest.raises(timeline.IncrementalError) as raised:
        book.finalize([], print, foreign)
    assert raised.value.code == "acknowledgement_source_mismatch"


def test_finalize_stages_an_acknowledgement_at_the_new_checkpoint(tmp_path):
    book = ledger(tmp_path)
    ack = evidence.EventAcknowledgement(
        identity(), "g", ("a",), timeline.Checkpoint(identity())
    )
    book.finalize([event("a", 4)], lambda events: None, ack)
    staged = book.reclaim()
    assert staged.events == ("a",)
    assert staged.checkpoint.sequence == 4
    book.retract()
    assert book.reclaim() is None
    book.retract()
    empty = evidence.EventAcknowledgement(
        identity(), "g", (), timeline.Checkpoint(identity())
    )
    book.finalize([], lambda events: None, empty)
    assert book.reclaim() is None


def test_reclaim_rejects_bad_staged_files(tmp_path):
    book = ledger(tmp_path)
    book.pending_acknowledgement_path.write_text("{not json")
    with pytest.raises(timeline.IncrementalError) as raised:
        book.reclaim()
    assert raised.value.code == "invalid_pending_acknowledgement"
    other = evidence.EventAcknowledgement(
        "src_x", "g", ("a",), timeline.Checkpoint("src_x")
    )
    store.publish(book.pending_acknowledgement_path, other.to_dict())
    with pytest.raises(timeline.IncrementalError) as raised:
        book.reclaim()
    assert raised.value.code == "acknowledgement_source_mismatch"
    with pytest.raises(timeline.IncrementalError) as raised:
        book.earmark(other)
    assert raised.value.code == "acknowledgement_source_mismatch"


def test_an_ingestor_validates_its_configuration(tmp_path):
    source = replay()
    book = ledger(tmp_path)
    with pytest.raises(errors.InputError, match="consumer_group"):
        consumer.Ingestor(source, book, " ", print, print)
    with pytest.raises(timeline.IncrementalError) as raised:
        consumer.Ingestor(source, ledger(tmp_path, "src_x"), "g", print, print)
    assert raised.value.code == "source_mismatch"
    with pytest.raises(errors.KindError, match="writer"):
        consumer.Ingestor(source, book, "g", None, print)
    with pytest.raises(errors.KindError, match="acknowledge"):
        consumer.Ingestor(source, book, "g", print, None)
    assert consumer.Ingestor(
        source, book, " g ", print, print
    ).consumer_group == ("g")


def test_consume_commits_writes_and_acknowledges(tmp_path):
    written, acknowledged = [], []
    ingestor = consumer.Ingestor(
        replay(event("a", 1), event("b", 2)),
        ledger(tmp_path),
        "g",
        written.append,
        acknowledged.append,
    )
    result = ingestor.consume(limit=1)
    assert result.receipt.accepted_event_ids == ("a",)
    assert result.acknowledged_event_ids == ("a",)
    assert result.consumer_group == "g"
    assert acknowledged[0].consumer == "g"
    assert acknowledged[0].checkpoint.sequence == 1
    second = ingestor.consume()
    assert second.receipt.accepted_event_ids == ("b",)
    assert second.acknowledged_event_ids == ("a", "b")
    assert len(written) == 2
    idle = ingestor.consume()
    assert idle.receipt.accepted_event_ids == ()
    assert idle.acknowledged_event_ids == ("a", "b")


def test_a_failed_acknowledgement_is_kept_for_retry(tmp_path):
    state = {"fail": True}
    delivered = []

    def acknowledge(ack):
        if state["fail"]:
            raise ConnectionError
        delivered.append(ack.events)

    book = ledger(tmp_path)
    ingestor = consumer.Ingestor(
        replay(event("a", 1)), book, "g", lambda events: None, acknowledge
    )
    with pytest.raises(timeline.IncrementalError) as raised:
        ingestor.consume()
    assert raised.value.code == "acknowledgement_failed"
    assert raised.value.details["event_ids"] == ["a"]
    assert book.reclaim().events == ("a",)
    state["fail"] = False
    assert ingestor.handoff() == ("a",)
    assert delivered == [("a",)]
    assert ingestor.handoff() == ()


def test_handoff_refuses_another_groups_acknowledgement(tmp_path):
    book = ledger(tmp_path)
    book.earmark(
        evidence.EventAcknowledgement(
            identity(), "other", ("a",), timeline.Checkpoint(identity())
        )
    )
    ingestor = consumer.Ingestor(replay(), book, "g", print, print)
    with pytest.raises(timeline.IncrementalError) as raised:
        ingestor.handoff()
    assert raised.value.code == "consumer_group_mismatch"
    assert raised.value.details["pending_group"] == "other"
