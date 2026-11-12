"""Targeted tests for incremental-reader behaviours the main suite left open."""

import datetime
import json

import pytest

from scmle.data.readers import incremental

URI = "kafka://broker/topic"


def make_event(
    event_id="a", sequence=1, event_time="2026-01-01T00:00:00Z", **kw
):
    return incremental.Event(event_id, sequence, event_time, {"v": 1}, **kw)


def make_source(events=(), policy=incremental.LateData.REJECT):
    return incremental.SequenceEventSource(URI, events, policy)


def test_incremental_error_keeps_code_uri_and_ordered_details():
    error = incremental.IncrementalError(
        "bad_code", "the message", "kafka://x/y", {"extra": 3, "code": "over"}
    )

    assert error.code == "bad_code"
    assert error.source_uri == "kafka://x/y"
    assert str(error) == "the message"
    assert error.details == {
        "code": "over",
        "source_uri": "kafka://x/y",
        "extra": 3,
    }
    assert error.category == "data"
    plain = incremental.IncrementalError("c", "m", "u")
    assert plain.details == {"code": "c", "source_uri": "u"}
    assert plain.details is not None


def test_parse_timestamp_converts_to_utc_and_names_the_field():
    parsed = incremental.parse_timestamp("2026-01-01T05:30:00+05:30", "when")

    assert parsed == datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    assert parsed.utcoffset() == datetime.timedelta(0)
    with pytest.raises(ValueError, match=r"^when must not be empty$"):
        incremental.parse_timestamp("  ", "when")
    with pytest.raises(ValueError, match=r"^when must not be empty$"):
        incremental.parse_timestamp(5, "when")
    with pytest.raises(ValueError, match=r"^when must be ISO-8601$"):
        incremental.parse_timestamp("yesterday", "when")
    with pytest.raises(ValueError, match=r"^when must include a timezone$"):
        incremental.parse_timestamp("2026-01-01T00:00:00", "when")


def test_event_validation_messages_and_types():
    with pytest.raises(ValueError, match=r"^event_id must not be empty$"):
        make_event(event_id="  ")
    with pytest.raises(ValueError, match=r"^event_id must not be empty$"):
        make_event(event_id=4)
    with pytest.raises(TypeError, match=r"^sequence must be an integer$"):
        make_event(sequence=True)
    with pytest.raises(TypeError, match=r"^sequence must be an integer$"):
        make_event(sequence=1.0)
    with pytest.raises(
        ValueError, match=r"^sequence must be a positive integer$"
    ):
        make_event(sequence=0)
    assert make_event(sequence=1).sequence == 1
    with pytest.raises(ValueError, match=r"^event_time must be ISO-8601$"):
        make_event(event_time="nope")
    with pytest.raises(ValueError, match=r"^ingest_time must be ISO-8601$"):
        make_event(ingest_time="nope")
    with pytest.raises(TypeError, match=r"^payload must be a mapping$"):
        incremental.Event("a", 1, "2026-01-01T00:00:00Z", [1])
    original = {"k": 1}
    event = incremental.Event("a", 1, "2026-01-01T00:00:00Z", original)
    assert event.payload == original
    assert event.payload is not original


def test_event_dict_round_trip_includes_ingest_time_only_when_set():
    plain = make_event()
    assert plain.to_dict() == {
        "event_id": "a",
        "sequence": 1,
        "event_time": "2026-01-01T00:00:00Z",
        "payload": {"v": 1},
    }
    stamped = make_event(ingest_time="2026-01-02T00:00:00Z")
    assert stamped.to_dict()["ingest_time"] == "2026-01-02T00:00:00Z"
    assert incremental.Event.from_dict(stamped.to_dict()) == stamped
    assert incremental.Event.from_dict(plain.to_dict()).ingest_time == ""
    rebuilt = incremental.Event.from_dict(
        {
            "event_id": 7,
            "sequence": "3",
            "event_time": "2026-01-01T00:00:00Z",
            "payload": {},
        }
    )
    assert (rebuilt.event_id, rebuilt.sequence) == ("7", 3)


def test_checkpoint_validation_and_round_trip():
    with pytest.raises(
        ValueError, match=r"^source_identity must not be empty$"
    ):
        incremental.EventCheckpoint("  ")
    with pytest.raises(TypeError, match=r"^sequence must be an integer$"):
        incremental.EventCheckpoint("s", True)
    with pytest.raises(TypeError, match=r"^sequence must be an integer$"):
        incremental.EventCheckpoint("s", "1")
    with pytest.raises(
        ValueError, match=r"^sequence must be a non-negative integer$"
    ):
        incremental.EventCheckpoint("s", -1)
    with pytest.raises(ValueError, match=r"^watermark must be ISO-8601$"):
        incremental.EventCheckpoint("s", 0, "later")
    assert incremental.EventCheckpoint("s", 0).sequence == 0

    point = incremental.EventCheckpoint("s", 4, "2026-01-01T00:00:00+00:00")
    assert point.to_dict() == {
        "source_identity": "s",
        "sequence": 4,
        "watermark": "2026-01-01T00:00:00+00:00",
    }
    assert incremental.EventCheckpoint.from_dict(point.to_dict()) == point
    loose = incremental.EventCheckpoint.from_dict(
        {"source_identity": 12, "sequence": "5"}
    )
    assert loose == incremental.EventCheckpoint("12", 5, "")
    with pytest.raises(ValueError, match=r"different source"):
        point.bind("other")
    assert point.bind("s") is point


def test_batch_validation_messages():
    point = incremental.EventCheckpoint("s")
    with pytest.raises(ValueError, match=r"^batch and checkpoint source"):
        incremental.EventBatch("other", (), point)
    with pytest.raises(TypeError, match=r"^events must contain Event values$"):
        incremental.EventBatch("s", ("x",), point)
    with pytest.raises(
        ValueError, match=r"^duplicate_event_ids must be sorted"
    ):
        incremental.EventBatch("s", (), point, ("b", "a"))
    with pytest.raises(ValueError, match=r"^late_event_ids must be sorted"):
        incremental.EventBatch("s", (), point, (), ("b", "a"))
    with pytest.raises(ValueError, match=r"^late_event_ids must be sorted"):
        incremental.EventBatch("s", (), point, (), ("a", "a"))
    batch = incremental.EventBatch("s", (make_event("z"),), point)
    assert batch.event_ids == ("z",)


def test_receipt_to_dict_and_validation():
    point = incremental.EventCheckpoint("s", 2)
    receipt = incremental.EventCommitReceipt("s", ("a",), ("b",), point)
    assert receipt.to_dict() == {
        "schema_version": 1,
        "source_identity": "s",
        "accepted_event_ids": ["a"],
        "duplicate_event_ids": ["b"],
        "checkpoint": point.to_dict(),
    }
    with pytest.raises(
        ValueError, match=r"^source_identity must not be empty$"
    ):
        incremental.EventCommitReceipt(
            " ", (), (), incremental.EventCheckpoint(" x")
        )
    with pytest.raises(ValueError, match=r"^receipt and checkpoint source"):
        incremental.EventCommitReceipt("t", (), (), point)
    with pytest.raises(
        ValueError, match=r"^accepted_event_ids must be sorted and unique$"
    ):
        incremental.EventCommitReceipt("s", ("b", "a"), (), point)
    with pytest.raises(
        ValueError, match=r"^duplicate_event_ids must be sorted and unique$"
    ):
        incremental.EventCommitReceipt("s", (), ("b", "a"), point)
    with pytest.raises(ValueError, match=r"^accepted and duplicate event IDs"):
        incremental.EventCommitReceipt("s", ("a",), ("a",), point)


def test_acknowledgement_validation_and_round_trip():
    point = incremental.EventCheckpoint("s", 1)
    ack = incremental.EventAcknowledgement("s", "g", ("a", "b"), point)
    assert ack.to_dict() == {
        "schema_version": 1,
        "source_identity": "s",
        "consumer_group": "g",
        "event_ids": ["a", "b"],
        "checkpoint": point.to_dict(),
    }
    assert incremental.EventAcknowledgement.from_dict(ack.to_dict()) == ack
    for version in (2, None):
        record = ack.to_dict()
        record["schema_version"] = version
        with pytest.raises(ValueError, match=r"^unsupported acknowledgement"):
            incremental.EventAcknowledgement.from_dict(record)
    with pytest.raises(
        ValueError, match=r"^source_identity must not be empty$"
    ):
        incremental.EventAcknowledgement(
            " ", "g", (), incremental.EventCheckpoint("x")
        )
    with pytest.raises(ValueError, match=r"^consumer_group must not be empty$"):
        incremental.EventAcknowledgement("s", " ", (), point)
    with pytest.raises(
        ValueError, match=r"^event_ids must be sorted and unique$"
    ):
        incremental.EventAcknowledgement("s", "g", ("b", "a"), point)
    with pytest.raises(ValueError, match=r"^acknowledgement and checkpoint"):
        incremental.EventAcknowledgement(
            "s", "g", (), incremental.EventCheckpoint("t")
        )


def test_consumption_validation_and_dict():
    point = incremental.EventCheckpoint("s", 1)
    receipt = incremental.EventCommitReceipt("s", ("a",), (), point)
    done = incremental.Consumption("s", "g", receipt, ("a",))
    assert done.to_dict() == {
        "schema_version": 1,
        "source_identity": "s",
        "consumer_group": "g",
        "receipt": receipt.to_dict(),
        "acknowledged_event_ids": ["a"],
    }
    with pytest.raises(ValueError, match=r"^result and receipt source"):
        incremental.Consumption("t", "g", receipt, ())
    with pytest.raises(ValueError, match=r"^consumer_group must not be empty$"):
        incremental.Consumption("s", "  ", receipt, ())
    with pytest.raises(
        ValueError, match=r"^acknowledged_event_ids must be sorted and unique$"
    ):
        incremental.Consumption("s", "g", receipt, ("b", "a"))


def test_event_source_accepts_a_policy_string_and_rejects_unknown():
    source = incremental.SequenceEventSource(URI, (), "drop")
    assert source.late_data_policy is incremental.LateData.DROP
    explicit = make_source(policy=incremental.LateData.ACCEPT)
    assert explicit.late_data_policy is incremental.LateData.ACCEPT
    with pytest.raises(ValueError, match=r"not a valid LateData"):
        incremental.SequenceEventSource(URI, (), "ignore")
    default = incremental.SequenceEventSource(URI, ())
    assert default.late_data_policy is incremental.LateData.REJECT
    assert default.source_uri == "kafka://broker/topic"


def test_checkpoint_mismatch_reports_the_checkpoint():
    source = make_source()
    foreign = incremental.EventCheckpoint("other", 3)
    with pytest.raises(incremental.IncrementalError) as caught:
        source.validate_checkpoint(foreign)
    assert caught.value.code == "checkpoint_mismatch"
    assert caught.value.details["checkpoint"] == foreign.to_dict()
    assert caught.value.source_uri == source.source_uri
    assert str(caught.value) == "checkpoint belongs to a different source"
    fresh = source.validate_checkpoint(None)
    assert fresh == incremental.EventCheckpoint(source.source_identity)


def test_select_events_orders_deduplicates_and_limits():
    events = [
        make_event("c", 3, "2026-01-03T00:00:00Z"),
        make_event("a", 1, "2026-01-01T00:00:00Z"),
        make_event("b", 2, "2026-01-02T00:00:00Z"),
        make_event("a", 1, "2026-01-01T00:00:00Z"),
    ]
    source = make_source()
    batch = source.select_events(events, limit=2, seen_event_ids=["c"])

    assert batch.event_ids == ("a", "b")
    assert batch.duplicate_event_ids == ("a", "c")
    assert batch.has_more is False
    assert batch.next_checkpoint.sequence == 2
    assert batch.next_checkpoint.watermark == "2026-01-02T00:00:00+00:00"
    more = source.select_events(events, limit=1)
    assert more.has_more is True
    assert more.event_ids == ("a",)
    with pytest.raises(ValueError, match=r"^limit must be positive$"):
        source.select_events(events, limit=0)
    with pytest.raises(TypeError, match=r"^transport records must yield Event"):
        source.select_events(["x"])


def test_select_events_late_policies():
    point = incremental.EventCheckpoint(
        make_source().source_identity, 5, "2026-02-01T00:00:00+00:00"
    )
    old = make_event("old", 3, "2026-01-01T00:00:00Z")
    new = make_event("new", 6, "2026-03-01T00:00:00Z")

    with pytest.raises(incremental.IncrementalError) as caught:
        make_source().select_events([old, new], point)
    assert caught.value.code == "late_event"
    assert str(caught.value) == "event 'old' is older than checkpoint"
    assert caught.value.details["event_id"] == "old"
    assert caught.value.details["event_sequence"] == 3
    assert caught.value.details["checkpoint_sequence"] == 5

    dropped = make_source(policy=incremental.LateData.DROP).select_events(
        [old, new], point
    )
    assert dropped.event_ids == ("new",)
    assert dropped.late_event_ids == ("old",)

    kept = make_source(policy=incremental.LateData.ACCEPT).select_events(
        [old, new], point
    )
    assert kept.event_ids == ("old", "new")
    assert kept.late_event_ids == ("old",)
    assert kept.next_checkpoint.sequence == 6


def test_select_events_keeps_the_checkpoint_when_nothing_is_selected():
    point = incremental.EventCheckpoint(
        make_source().source_identity, 5, "2026-02-01T00:00:00Z"
    )
    batch = make_source().select_events([], point)
    assert batch.events == ()
    assert batch.next_checkpoint.sequence == 5
    assert batch.next_checkpoint.watermark == "2026-02-01T00:00:00+00:00"
    empty = make_source().select_events([])
    assert empty.next_checkpoint.sequence == 0
    assert empty.next_checkpoint.watermark == ""
    earlier = make_source(policy=incremental.LateData.ACCEPT).select_events(
        [make_event("e", 2, "2026-01-01T00:00:00Z")], point
    )
    assert earlier.next_checkpoint.sequence == 5
    assert earlier.next_checkpoint.watermark == "2026-02-01T00:00:00+00:00"


def test_read_events_uses_the_available_events():
    source = make_source([make_event("a", 1)])
    batch = source.read_events()
    assert batch.event_ids == ("a",)
    calls = make_source(lambda: [make_event("b", 2)])
    assert calls.read_events().event_ids == ("b",)


def test_stream_source_validation_and_polling():
    seen = []

    def reader(uri, checkpoint, limit):
        seen.append((uri, checkpoint, limit))
        return [make_event("a", 1), make_event("b", 2)]

    with pytest.raises(incremental.connectors.UnsupportedUriSchemeError):
        incremental.EventStreamSource("http://x/y", reader)
    with pytest.raises(TypeError, match=r"^event_reader must be callable$"):
        incremental.EventStreamSource(URI, "nope")
    stream = incremental.EventStreamSource("  KAFKA://broker/topic ", reader)
    assert stream.source_uri == "kafka://broker/topic"
    assert isinstance(stream.retry_budget, incremental.connectors.RetryBudget)
    budget = incremental.connectors.RetryBudget()
    assert (
        incremental.EventStreamSource(
            URI, reader, retry_budget=budget
        ).retry_budget
        is budget
    )

    with pytest.raises(incremental.IncrementalError) as caught:
        stream.available_events()
    assert caught.value.code == "bounded_poll_required"
    assert caught.value.source_uri == stream.source_uri

    with pytest.raises(ValueError, match=r"^limit must be positive$"):
        stream.poll_events(incremental.EventCheckpoint("s"), 0)
    with pytest.raises(incremental.IncrementalError) as exceeded:
        stream.poll_events(incremental.EventCheckpoint("s"), 1)
    assert exceeded.value.code == "batch_limit_exceeded"
    assert exceeded.value.details["limit"] == 1
    assert exceeded.value.details["received"] == 2

    batch = stream.read_events(limit=5)
    assert batch.event_ids == ("a", "b")
    assert seen[-1][0] == "kafka://broker/topic"
    assert seen[-1][2] == 5
    assert seen[-1][1] == incremental.EventCheckpoint(stream.source_identity)


def test_committed_state_round_trip_and_validation():
    point = incremental.EventCheckpoint("s", 2)
    state = incremental.CommittedState(point, ("a", "b"))
    assert state.to_dict() == {
        "schema_version": 1,
        "checkpoint": point.to_dict(),
        "seen_event_ids": ["a", "b"],
    }
    assert incremental.CommittedState.from_dict(state.to_dict()) == state
    for version in (2, None):
        record = state.to_dict()
        record["schema_version"] = version
        with pytest.raises(ValueError, match=r"^unsupported incremental state"):
            incremental.CommittedState.from_dict(record)
    with pytest.raises(ValueError, match=r"^seen_event_ids must be sorted"):
        incremental.CommittedState(point, ("b", "a"))


def make_ledger(tmp_path, source=None):
    source = source or make_source()
    ledger = incremental.IngestionLedger(
        tmp_path / "state.json", source.source_identity
    )
    return source, ledger


def test_ledger_paths_and_load_defaults(tmp_path):
    source, ledger = make_ledger(tmp_path)
    assert ledger.lock_path.name == "state.json.lock"
    assert ledger.pending_acknowledgement_path.name == "state.json.ack"
    assert ledger.load() == incremental.CommittedState(
        incremental.EventCheckpoint(source.source_identity)
    )
    ledger.path.write_text(
        json.dumps(
            incremental.CommittedState(
                incremental.EventCheckpoint("other")
            ).to_dict()
        ),
        encoding="utf-8",
    )
    with pytest.raises(incremental.IncrementalError) as caught:
        ledger.load()
    assert caught.value.code == "state_source_mismatch"
    assert caught.value.source_uri == source.source_identity
    assert caught.value.details["state_source_identity"] == "other"


def test_ledger_source_and_batch_mismatches(tmp_path):
    ledger = make_ledger(tmp_path)[1]
    other = incremental.SequenceEventSource("kafka://elsewhere/t", ())
    with pytest.raises(incremental.IncrementalError) as caught:
        ledger.read_batch(other)
    assert caught.value.code == "source_mismatch"
    assert caught.value.details["source_identity"] == other.source_identity
    foreign = incremental.EventBatch(
        "other", (), incremental.EventCheckpoint("other")
    )
    with pytest.raises(incremental.IncrementalError) as batch_error:
        ledger.commit(foreign)
    assert batch_error.value.code == "batch_source_mismatch"
    assert batch_error.value.details["batch_source_identity"] == "other"
    with pytest.raises(incremental.IncrementalError) as state_error:
        ledger.write_atomically(
            incremental.CommittedState(incremental.EventCheckpoint("other"))
        )
    assert state_error.value.code == "state_source_mismatch"
    assert not ledger.path.exists()
    ack = incremental.EventAcknowledgement(
        "other", "g", (), incremental.EventCheckpoint("other")
    )
    with pytest.raises(incremental.IncrementalError) as ack_error:
        ledger.write_pending_acknowledgement(ack)
    assert ack_error.value.code == "acknowledgement_source_mismatch"
    assert ack_error.value.details["ack_source_identity"] == "other"
    with pytest.raises(incremental.IncrementalError) as writer_error:
        ledger.commit_with_writer([], lambda events: None, ack)
    assert writer_error.value.code == "acknowledgement_source_mismatch"


def test_write_atomically_persists_sorted_json_and_cleans_up(tmp_path):
    source, ledger = make_ledger(tmp_path / "nested")
    state = incremental.CommittedState(
        incremental.EventCheckpoint(source.source_identity, 1), ("a",)
    )
    ledger.write_atomically(state)
    text = ledger.path.read_text(encoding="utf-8")
    assert text == json.dumps(state.to_dict(), sort_keys=True) + "\n"
    assert [p.name for p in ledger.path.parent.iterdir()] == ["state.json"]


def test_commit_is_idempotent_and_rejects_regression(tmp_path):
    source, ledger = make_ledger(tmp_path)
    ident = source.source_identity
    first = incremental.EventBatch(
        ident,
        (make_event("a", 4),),
        incremental.EventCheckpoint(ident, 4, "2026-01-01T00:00:00+00:00"),
    )
    state = ledger.commit(first)
    assert state.seen_event_ids == ("a",)
    assert ledger.load() == state
    before = ledger.path.stat().st_mtime_ns
    assert ledger.commit(first) == state
    assert ledger.path.stat().st_mtime_ns == before

    behind = incremental.EventBatch(
        ident,
        (make_event("b", 2),),
        incremental.EventCheckpoint(ident, 2, "2026-01-01T00:00:00+00:00"),
    )
    with pytest.raises(incremental.IncrementalError) as caught:
        ledger.commit(behind)
    assert caught.value.code == "checkpoint_regression"
    assert caught.value.details["current"]["sequence"] == 4
    assert caught.value.details["candidate"]["sequence"] == 2

    same_point = incremental.EventBatch(
        ident,
        (make_event("c", 3),),
        incremental.EventCheckpoint(ident, 4, "2026-01-01T00:00:00+00:00"),
    )
    assert ledger.commit(same_point).seen_event_ids == ("a", "c")


def test_commit_with_writer_receipt_and_checkpoint(tmp_path):
    ledger = make_ledger(tmp_path)[1]
    written = []
    events = [
        make_event("b", 2, "2026-01-02T00:00:00Z"),
        make_event("a", 1, "2026-01-05T00:00:00Z"),
    ]
    receipt = ledger.commit_with_writer(events, written.append)

    assert [tuple(e.event_id for e in call) for call in written] == [("a", "b")]
    assert receipt.accepted_event_ids == ("a", "b")
    assert receipt.duplicate_event_ids == ()
    assert receipt.checkpoint.sequence == 2
    assert receipt.checkpoint.watermark == "2026-01-05T00:00:00+00:00"
    assert ledger.load().seen_event_ids == ("a", "b")

    again = ledger.commit_with_writer(
        [make_event("a", 1), make_event("c", 9, "2026-01-03T00:00:00Z")],
        written.append,
    )
    assert again.accepted_event_ids == ("c",)
    assert again.duplicate_event_ids == ("a",)
    assert again.checkpoint.sequence == 9
    assert again.checkpoint.watermark == "2026-01-05T00:00:00+00:00"

    noop = ledger.commit_with_writer([make_event("a", 1)], written.append)
    assert noop.accepted_event_ids == ()
    assert noop.checkpoint == again.checkpoint
    assert len(written) == 2


def test_commit_with_writer_type_checks_and_failed_writer(tmp_path):
    ledger = make_ledger(tmp_path)[1]
    with pytest.raises(TypeError, match=r"^events must contain Event values$"):
        ledger.commit_with_writer(["x"], lambda events: None)
    with pytest.raises(TypeError, match=r"^writer must be callable$"):
        ledger.commit_with_writer([], "nope")

    def failing(events):
        raise RuntimeError("sink down")

    with pytest.raises(RuntimeError, match=r"sink down"):
        ledger.commit_with_writer([make_event("a", 1)], failing)
    assert not ledger.path.exists()


def test_commit_with_writer_stages_the_acknowledgement_at_the_checkpoint(
    tmp_path,
):
    source, ledger = make_ledger(tmp_path)
    ident = source.source_identity
    stale = incremental.EventCheckpoint(ident, 0)
    ack = incremental.EventAcknowledgement(ident, "g", ("a",), stale)
    ledger.commit_with_writer([make_event("a", 3)], lambda events: None, ack)

    staged = ledger.load_pending_acknowledgement()
    assert staged.checkpoint.sequence == 3
    assert staged.consumer_group == "g"
    text = ledger.pending_acknowledgement_path.read_text(encoding="utf-8")
    assert text == json.dumps(staged.to_dict(), sort_keys=True) + "\n"
    ledger.clear_pending_acknowledgement()
    assert ledger.load_pending_acknowledgement() is None
    ledger.clear_pending_acknowledgement()

    empty = incremental.EventAcknowledgement(ident, "g", (), stale)
    ledger.commit_with_writer([make_event("b", 4)], lambda events: None, empty)
    assert ledger.load_pending_acknowledgement() is None


def test_pending_acknowledgement_loading_errors(tmp_path):
    source, ledger = make_ledger(tmp_path)
    ledger.pending_acknowledgement_path.write_text(
        "{not json", encoding="utf-8"
    )
    with pytest.raises(incremental.IncrementalError) as bad:
        ledger.load_pending_acknowledgement()
    assert bad.value.code == "invalid_pending_acknowledgement"
    assert bad.value.details["path"] == str(ledger.pending_acknowledgement_path)
    assert bad.value.source_uri == source.source_identity

    foreign = incremental.EventAcknowledgement(
        "other", "g", (), incremental.EventCheckpoint("other")
    )
    ledger.pending_acknowledgement_path.write_text(
        json.dumps(foreign.to_dict()), encoding="utf-8"
    )
    with pytest.raises(incremental.IncrementalError) as mismatch:
        ledger.load_pending_acknowledgement()
    assert mismatch.value.code == "acknowledgement_source_mismatch"
    assert mismatch.value.details["ack_source_identity"] == "other"


def make_consumer(tmp_path, events, acknowledge=None, group=" g "):
    source = make_source(events)
    source, ledger = make_ledger(tmp_path, source)
    sink = []
    delivered = []
    consumer = incremental.EventConsumer(
        source,
        ledger,
        group,
        sink.append,
        acknowledge or delivered.append,
    )
    return consumer, ledger, sink, delivered


def test_consumer_validation(tmp_path):
    source = make_source()
    source, ledger = make_ledger(tmp_path, source)
    with pytest.raises(ValueError, match=r"^consumer_group must not be empty$"):
        incremental.EventConsumer(source, ledger, " ", print, print)
    other = incremental.IngestionLedger(tmp_path / "o.json", "other")
    with pytest.raises(incremental.IncrementalError) as caught:
        incremental.EventConsumer(source, other, "g", print, print)
    assert caught.value.code == "source_mismatch"
    assert caught.value.source_uri == "other"
    assert caught.value.details["source_identity"] == source.source_identity
    with pytest.raises(TypeError, match=r"^writer must be callable$"):
        incremental.EventConsumer(source, ledger, "g", None, print)
    with pytest.raises(TypeError, match=r"^acknowledge must be callable$"):
        incremental.EventConsumer(source, ledger, "g", print, None)


def test_consume_commits_and_acknowledges_everything_seen(tmp_path):
    events = [make_event("a", 1), make_event("b", 2)]
    consumer, ledger, sink, delivered = make_consumer(tmp_path, events)
    assert consumer.consumer_group == "g"

    done = consumer.consume()
    assert done.consumer_group == "g"
    assert done.receipt.accepted_event_ids == ("a", "b")
    assert done.acknowledged_event_ids == ("a", "b")
    assert len(sink) == 1
    assert delivered[0].event_ids == ("a", "b")
    assert delivered[0].checkpoint.sequence == 2
    assert ledger.load_pending_acknowledgement() is None

    replay = consumer.consume()
    assert replay.receipt.accepted_event_ids == ()
    assert replay.acknowledged_event_ids == ("a", "b")
    assert len(sink) == 1


def test_a_failed_acknowledgement_is_kept_and_retried(tmp_path):
    attempts = []

    def flaky(ack):
        attempts.append(ack)
        if len(attempts) == 1:
            raise RuntimeError("broker down")

    consumer, ledger = make_consumer(
        tmp_path, [make_event("a", 1)], acknowledge=flaky
    )[:2]
    with pytest.raises(incremental.IncrementalError) as caught:
        consumer.consume()
    error = caught.value
    assert error.code == "acknowledgement_failed"
    assert error.source_uri == consumer.source.source_uri
    assert error.details["consumer_group"] == "g"
    assert error.details["event_ids"] == ["a"]
    assert isinstance(error.__cause__, RuntimeError)
    assert ledger.load_pending_acknowledgement() is not None

    assert consumer.deliver_pending_acknowledgement() == ("a",)
    assert ledger.load_pending_acknowledgement() is None
    assert consumer.deliver_pending_acknowledgement() == ()


def test_a_pending_acknowledgement_of_another_group_is_refused(tmp_path):
    consumer, ledger = make_consumer(tmp_path, [make_event("a", 1)])[:2]
    ack = incremental.EventAcknowledgement(
        consumer.source.source_identity,
        "elsewhere",
        ("a",),
        incremental.EventCheckpoint(consumer.source.source_identity, 1),
    )
    ledger.write_pending_acknowledgement(ack)
    with pytest.raises(incremental.IncrementalError) as caught:
        consumer.deliver_pending_acknowledgement()
    assert caught.value.code == "consumer_group_mismatch"
    assert caught.value.source_uri == consumer.source.source_uri
    assert caught.value.details["pending_group"] == "elsewhere"
