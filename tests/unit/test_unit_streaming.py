"""Behavioural tests for streamed reading, sampling and the batch journal."""

import json

import numpy
import pandas
import pytest

from scmle.contracts import errors
from scmle.data.readers.streaming import archive
from scmle.data.readers.streaming import consumer
from scmle.data.readers.streaming import failure
from scmle.data.readers.streaming import feed
from scmle.data.readers.streaming import journal
from scmle.data.readers.streaming import progress
from scmle.data.readers.streaming import sampling
from scmle.data.readers.streaming import transaction


def csv(tmp_path, rows=5, name="t.csv"):
    path = tmp_path / name
    frame = pandas.DataFrame(
        {
            "a": range(rows),
            "t": [f"2026-01-0{1 + i % 9}T00:00:00Z" for i in range(rows)],
        }
    )
    frame.to_csv(path, index=False)
    return feed.locate(path)


def book(tmp_path, source):
    return journal.Journal(tmp_path / "state.json", source)


class Sink(transaction.Sink):
    def __init__(self, receipt=None):
        self.calls = []
        self.receipt = receipt

    def commit(self, batch):
        self.calls.append(batch.batch_id)
        if self.receipt is not None:
            return self.receipt
        return transaction.Voucher(
            batch.source_identity, batch.batch_id, batch.next_checkpoint, "ok"
        )


def ack(batch, group="g"):
    return transaction.Acknowledgement(
        batch.source_identity, group, batch.batch_id, batch.next_checkpoint
    )


# failure


def test_a_streaming_error_carries_code_and_source():
    error = failure.StreamingError("c", "m", "src", {"k": 1})
    assert error.details == {"code": "c", "source_identity": "src", "k": 1}
    assert (error.code, error.source_identity, str(error)) == ("c", "src", "m")
    assert failure.StreamingError("c", "m", "s").details == {
        "code": "c",
        "source_identity": "s",
    }


# feed


def test_locate_picks_the_reader_by_suffix(tmp_path):
    for name, kind in (
        ("a.CSV", feed.Csv),
        ("a.csv.gz", feed.Csv),
        ("a.jsonl", feed.Ndjson),
        ("a.ndjson.gz", feed.Ndjson),
        ("a.parquet", feed.Chunked),
        ("a.pq", feed.Chunked),
    ):
        path = tmp_path / name
        path.write_bytes(b"")
        assert isinstance(feed.locate(path), kind)
    with pytest.raises(errors.DataError, match="dataset not found"):
        feed.locate(tmp_path / "missing.csv")
    other = tmp_path / "a.xlsx"
    other.write_bytes(b"")
    with pytest.raises(errors.DataError, match=r"cannot stream '\.xlsx'"):
        feed.locate(other)


def test_feeds_report_shape_and_footprint(tmp_path):
    source = csv(tmp_path, 4)
    assert source.shape() == (None, 2)
    assert source.footprint() == source.path.stat().st_size * 4
    parquet = tmp_path / "t.parquet"
    pandas.DataFrame({"a": range(6), "b": range(6)}).to_parquet(
        parquet, row_group_size=2
    )
    chunked = feed.locate(parquet)
    assert chunked.shape() == (6, 2)
    assert chunked.footprint() == 6 * 2 * 8
    assert sum(len(f) for f in chunked.batches(4)) == 6
    lines = tmp_path / "t.jsonl"
    lines.write_text('{"a": 1, "b": 2}\n{"a": 3, "b": 4}\n')
    assert feed.locate(lines).shape() == (None, 2)
    assert [len(f) for f in feed.locate(lines).batches(1)] == [1, 1]
    assert (feed.BATCH_ROWS, feed.CSV_EXPANSION, feed.BYTES_PER_CELL) == (
        50_000,
        4,
        8,
    )


def test_pick_takes_rows_after_a_position():
    frames = [pandas.DataFrame({"a": [0, 1]}), pandas.DataFrame({"a": [2, 3]})]
    selected, more = feed.pick(frames, 0, 2, "s")
    assert selected["a"].tolist() == [0, 1]
    assert more
    selected, more = feed.pick(frames, 2, 2, "s")
    assert selected["a"].tolist() == [2, 3]
    assert not more
    selected, more = feed.pick(frames, 1, 1, "s")
    assert selected["a"].tolist() == [1]
    assert more
    selected, more = feed.pick(frames, 3, 5, "s")
    assert selected["a"].tolist() == [3]
    assert not more
    trailing = [*frames, pandas.DataFrame({"a": []})]
    assert not feed.pick(trailing, 2, 2, "s")[1]


def test_pick_spans_frames_and_skips_empty_ones():
    empty = pandas.DataFrame({"a": []})
    frames = [
        pandas.DataFrame({"a": [0, 1]}),
        empty,
        pandas.DataFrame({"a": [2]}),
        empty,
        pandas.DataFrame({"a": [3, 4]}),
    ]
    selected, more = feed.pick(frames, 1, 3, "s")
    assert selected["a"].tolist() == [1, 2, 3]
    assert more
    selected, more = feed.pick(frames, 1, 4, "s")
    assert selected["a"].tolist() == [1, 2, 3, 4]
    assert not more
    selected, more = feed.pick(frames, 0, 3, "s")
    assert selected["a"].tolist() == [0, 1, 2]
    assert more
    assert not feed.pick([*frames, empty], 2, 3, "s")[1]


def test_pick_refuses_bad_frames_and_the_end_of_the_source():
    with pytest.raises(failure.StreamingError) as raised:
        feed.pick(["x"], 0, 1, "s")
    assert raised.value.code == "invalid_batch"
    with pytest.raises(failure.StreamingError) as raised:
        feed.pick([pandas.DataFrame({"a": [1]})], 1, 1, "s")
    assert raised.value.code == "end_of_source"
    assert raised.value.details["offset"] == 1


def test_a_resumed_read_with_another_batch_size_reports_rows_left(tmp_path):
    source = csv(tmp_path, 6)
    first = source.read_batch(rows=3)
    second = source.read_batch(first.next_checkpoint, rows=2)
    assert second.frame["a"].tolist() == [3, 4]
    assert second.has_more


def test_align_keeps_the_checkpoints_watermark_column(tmp_path):
    snapshot = "a" * 64
    fresh = progress.Checkpoint("s", {"snapshot": snapshot})
    assert feed.align(None, fresh, "t", "s") == "t"
    assert feed.align(fresh, fresh, "t", "s") == "t"
    resumed = progress.Checkpoint("s", {"snapshot": snapshot}, 3, column="t")
    assert feed.align(resumed, resumed, "t", "s") == "t"
    with pytest.raises(failure.StreamingError) as raised:
        feed.align(resumed, resumed, "u", "s")
    assert raised.value.code == "watermark_mismatch"
    assert raised.value.details["requested_watermark_column"] == "u"


def test_read_batch_advances_offset_and_watermark(tmp_path):
    source = csv(tmp_path, 5)
    first = source.read_batch(rows=2, watermark_column="t")
    assert first.frame["a"].tolist() == [0, 1]
    assert first.start_offset == 0
    assert first.has_more
    assert first.next_checkpoint.offset == 2
    assert first.next_checkpoint.watermark == "2026-01-02T00:00:00+00:00"
    assert first.next_checkpoint.column == "t"
    second = source.read_batch(
        first.next_checkpoint, rows=2, watermark_column="t"
    )
    assert second.frame["a"].tolist() == [2, 3]
    assert second.next_checkpoint.watermark == "2026-01-04T00:00:00+00:00"
    last = source.read_batch(
        second.next_checkpoint, rows=2, watermark_column="t"
    )
    assert last.frame["a"].tolist() == [4]
    assert not last.has_more
    with pytest.raises(failure.StreamingError) as raised:
        source.read_batch(last.next_checkpoint, rows=2, watermark_column="t")
    assert raised.value.code == "end_of_source"
    with pytest.raises(errors.InputError, match="rows must be positive"):
        source.read_batch(rows=0)


def test_rebind_rejects_foreign_and_stale_checkpoints(tmp_path):
    source = csv(tmp_path, 3)
    snapshot = source.snapshot_sha256()
    fresh = source.rebind(None, snapshot)
    assert (fresh.source, fresh.offset) == (source.source_identity(), 0)
    foreign = progress.Checkpoint("src_x", {"snapshot": snapshot})
    with pytest.raises(failure.StreamingError) as raised:
        source.rebind(foreign, snapshot)
    assert raised.value.code == "checkpoint_source_mismatch"
    with pytest.raises(failure.StreamingError) as raised:
        source.rebind(fresh, "b" * 64)
    assert raised.value.code == "source_snapshot_changed"
    assert source.rebind(fresh, snapshot) is fresh


def test_snapshot_reports_an_unreadable_file(tmp_path):
    source = csv(tmp_path, 1)
    source.path.unlink()
    with pytest.raises(failure.StreamingError) as raised:
        source.snapshot_sha256()
    assert raised.value.code == "source_unreadable"


def test_read_batch_detects_a_change_during_the_read(tmp_path, monkeypatch):
    source = csv(tmp_path, 3)
    digests = iter(["a" * 64, "b" * 64])
    monkeypatch.setattr(source, "snapshot_sha256", lambda: next(digests))
    with pytest.raises(failure.StreamingError) as raised:
        source.read_batch(rows=1)
    assert raised.value.code == "source_snapshot_changed"
    assert raised.value.details["final_snapshot_sha256"] == "b" * 64


# sampling


def test_row_hash_is_deterministic_and_seeded():
    positions = numpy.arange(1000)
    first = sampling.row_hash(positions, 0)
    assert first.tolist() == sampling.row_hash(positions, 0).tolist()
    assert ((first >= 0) & (first < 1)).all()
    assert first.tolist() != sampling.row_hash(positions, 1).tolist()
    assert 0.45 < first.mean() < 0.55
    assert len(set(first.tolist())) == 1000


def test_allot_locks_rows_below_the_fraction():
    positions = numpy.arange(10_000)
    sides = sampling.allot(positions, 0.2, 3)
    assert set(sides.tolist()) == {0, 1}
    assert 0.18 < sides.mean() < 0.22
    expected = (sampling.row_hash(positions, 3) < 0.2).astype(int)
    assert sides.tolist() == expected.tolist()
    assert sampling.allot(positions, 0.0, 3).sum() == 0
    assert sampling.allot(positions, 1.0, 3).sum() == 10_000


def test_budgets_split_half_the_memory():
    assert sampling.budgets(10, 1.0, 0.2) == (5_000_000, 1_250_000)
    assert sampling.budgets(0, 1e-9, 0.5) == (1_000, 250)
    assert sampling.budgets(1, 0.0008, 0.25) == (37_500, 12_500)


def test_retain_keeps_the_lowest_priority_rows():
    batch = pandas.DataFrame({"x": [1, 2, 3], "priority": [0.9, 0.1, 0.5]})
    kept = sampling.retain(None, batch, 2)
    assert kept["x"].tolist() == [2, 3]
    more = pandas.DataFrame({"x": [4], "priority": [0.05]})
    assert sampling.retain(kept, more, 2)["x"].tolist() == [4, 2]
    assert len(sampling.retain(kept, more, 3)) == 3


def test_sweep_splits_one_pass_into_two_samples(tmp_path):
    source = csv(tmp_path, 200)
    kept, totals = sampling.sweep(source, 1_000, 1_000, 0.3, seed=1)
    assert totals.rows == 200
    assert totals.development_sample + totals.locked_sample == 200
    assert totals.locked == totals.locked_sample
    sides = sampling.allot(numpy.arange(200), 0.3, 1)
    assert totals.locked == int(sides.sum())
    assert "priority" not in kept.columns
    assert set(kept[sampling.SPLIT_COLUMN]) == {0, 1}
    capped, totals = sampling.sweep(source, 10, 5, 0.3, seed=1)
    assert (totals.development_sample, totals.locked_sample) == (10, 5)
    assert len(capped) == 15
    with pytest.raises(errors.DataError, match="too small to split"):
        sampling.sweep(csv(tmp_path, 1, "one.csv"), 10, 10, 0.5, seed=0)


# archive and journal


def test_an_archive_starts_empty_and_round_trips_state(tmp_path):
    source = csv(tmp_path, 3)
    ledger = book(tmp_path, source)
    state = ledger.load()
    assert state.checkpoint.offset == 0
    assert state.checkpoint.digests == {"snapshot": source.snapshot_sha256()}
    batch = ledger.read_batch(rows=2)
    ledger.write_atomically(journal.settle(state, batch))
    reloaded = ledger.load()
    assert reloaded.checkpoint.offset == 2
    assert reloaded.committed == (batch.batch_id,)


def test_an_archive_fails_closed_on_bad_state(tmp_path):
    source = csv(tmp_path, 3)
    ledger = book(tmp_path, source)
    ledger.path.write_text("{bad")
    with pytest.raises(failure.StreamingError) as raised:
        ledger.load()
    assert raised.value.code == "invalid_state"
    other = csv(tmp_path, 3, "other.csv")
    foreign = transaction.Settlement(
        progress.Checkpoint(
            other.source_identity(), {"snapshot": other.snapshot_sha256()}
        )
    )
    ledger.path.write_text(json.dumps(foreign.to_dict()))
    with pytest.raises(failure.StreamingError) as raised:
        ledger.load()
    assert raised.value.code == "checkpoint_source_mismatch"
    with pytest.raises(failure.StreamingError) as raised:
        ledger.write_atomically(foreign)
    assert raised.value.code == "checkpoint_source_mismatch"
    stale = transaction.Settlement(
        progress.Checkpoint(source.source_identity(), {"snapshot": "c" * 64})
    )
    ledger.path.write_text(json.dumps(stale.to_dict()))
    with pytest.raises(failure.StreamingError) as raised:
        ledger.load()
    assert raised.value.code == "source_snapshot_changed"
    with pytest.raises(failure.StreamingError) as raised:
        ledger.write_atomically(stale)
    assert raised.value.code == "source_snapshot_changed"


def test_confine_checks_identity_then_snapshot(tmp_path):
    source = csv(tmp_path, 1)
    archive.confine(
        source,
        source.source_identity(),
        source.snapshot_sha256(),
        "c",
        ("x", "y"),
    )
    with pytest.raises(failure.StreamingError, match="x") as raised:
        archive.confine(source, "src_z", "a" * 64, "c", ("x", "y"))
    assert raised.value.code == "c"
    with pytest.raises(failure.StreamingError, match="y") as raised:
        archive.confine(
            source, source.source_identity(), "a" * 64, "c", ("x", "y")
        )
    assert raised.value.code == "source_snapshot_changed"


def test_staged_acknowledgements_round_trip(tmp_path):
    source = csv(tmp_path, 3)
    ledger = book(tmp_path, source)
    assert ledger.reclaim() is None
    batch = ledger.read_batch(rows=1)
    ledger.earmark(ack(batch))
    assert ledger.reclaim() == ack(batch)
    ledger.retract()
    assert ledger.reclaim() is None
    ledger.retract()
    ledger.pending_acknowledgement_path.write_text("nope")
    with pytest.raises(failure.StreamingError) as raised:
        ledger.reclaim()
    assert raised.value.code == "invalid_acknowledgement"


def test_vouch_and_attest_compare_with_the_batch(tmp_path):
    source = csv(tmp_path, 3)
    ledger = book(tmp_path, source)
    batch = ledger.read_batch(rows=1)
    other = source.read_batch(batch.next_checkpoint, rows=1)
    ledger.vouch(batch, ack(batch))
    with pytest.raises(failure.StreamingError) as raised:
        ledger.vouch(batch, ack(other))
    assert raised.value.code == "acknowledgement_batch_mismatch"
    shifted = transaction.Acknowledgement(
        batch.source_identity, "g", batch.batch_id, other.next_checkpoint
    )
    with pytest.raises(failure.StreamingError) as raised:
        ledger.vouch(batch, shifted)
    assert raised.value.code == "acknowledgement_checkpoint_mismatch"
    voucher = transaction.Voucher(
        batch.source_identity, other.batch_id, batch.next_checkpoint, "r"
    )
    with pytest.raises(failure.StreamingError) as raised:
        ledger.attest(batch, voucher)
    assert raised.value.code == "sink_receipt_batch_mismatch"
    moved = transaction.Voucher(
        batch.source_identity, batch.batch_id, other.next_checkpoint, "r"
    )
    with pytest.raises(failure.StreamingError) as raised:
        ledger.attest(batch, moved)
    assert raised.value.code == "sink_receipt_checkpoint_mismatch"


def test_finalize_commits_once_and_stages_the_acknowledgement(tmp_path):
    source = csv(tmp_path, 3)
    ledger = book(tmp_path, source)
    batch = ledger.read_batch(rows=2)
    written = []
    receipt = ledger.finalize(
        batch, lambda frame, key: written.append(key), ack(batch)
    )
    assert receipt.accepted and not receipt.duplicate
    assert receipt.checkpoint == batch.next_checkpoint
    assert written == [batch.batch_id]
    assert ledger.reclaim().checkpoint == batch.next_checkpoint
    again = ledger.finalize(batch, lambda frame, key: written.append(key))
    assert again.duplicate and not again.accepted
    assert written == [batch.batch_id]
    assert ledger.load().committed == (batch.batch_id,)


def test_finalize_refuses_mismatches(tmp_path):
    source = csv(tmp_path, 4)
    ledger = book(tmp_path, source)
    first = ledger.read_batch(rows=1)
    later = source.read_batch(first.next_checkpoint, rows=1)
    with pytest.raises(failure.StreamingError) as raised:
        ledger.finalize(later, lambda frame, key: None)
    assert raised.value.code == "checkpoint_mismatch"
    with pytest.raises(failure.StreamingError) as raised:
        ledger.finalize(first, lambda frame, key: None, ack(later))
    assert raised.value.code == "acknowledgement_batch_mismatch"
    with pytest.raises(errors.KindError, match="writer"):
        ledger.finalize(first, None)

    def failing(frame, key):
        raise RuntimeError("sink")

    with pytest.raises(RuntimeError, match="sink"):
        ledger.finalize(first, failing)
    assert not ledger.path.exists()


def test_commit_transaction_records_and_replays_evidence(tmp_path):
    # Enforces website/src/pages/docs/data.astro:76.
    source = csv(tmp_path, 3)
    ledger = book(tmp_path, source)
    batch = ledger.read_batch(rows=2)
    sink = Sink()
    receipt = ledger.commit_transaction(batch, sink, ack(batch))
    assert receipt.accepted
    assert receipt.sink_receipt.receipt == "ok"
    retry = ledger.commit_transaction(batch, sink, ack(batch))
    assert retry.duplicate
    assert retry.sink_receipt == receipt.sink_receipt
    assert sink.calls == [batch.batch_id]
    with pytest.raises(failure.StreamingError) as raised:
        ledger.commit_transaction(batch, sink, ack(batch, "other"))
    assert raised.value.code == "acknowledgement_mismatch"
    state = ledger.load()
    assert [c.batch for c in state.commitments] == [batch.batch_id]


def test_commit_transaction_validates_its_parts(tmp_path):
    source = csv(tmp_path, 3)
    ledger = book(tmp_path, source)
    batch = ledger.read_batch(rows=1)
    with pytest.raises(errors.KindError, match="Sink"):
        ledger.commit_transaction(batch, object(), ack(batch))
    with pytest.raises(errors.KindError, match="Acknowledgement"):
        ledger.commit_transaction(batch, Sink(), "ack")
    with pytest.raises(failure.StreamingError) as raised:
        ledger.commit_transaction(batch, Sink(receipt="x"), ack(batch))
    assert raised.value.code == "invalid_sink_receipt"
    ledger.finalize(batch, lambda frame, key: None)
    with pytest.raises(failure.StreamingError) as raised:
        ledger.commit_transaction(batch, Sink(), ack(batch))
    assert raised.value.code == "transaction_evidence_missing"


def test_settle_adds_the_batch_and_sorted_evidence(tmp_path):
    source = csv(tmp_path, 3)
    ledger = book(tmp_path, source)
    state = ledger.load()
    batch = ledger.read_batch(rows=1)
    settled = journal.settle(state, batch)
    assert settled.committed == (batch.batch_id,)
    assert settled.commitments == ()
    assert settled.checkpoint == batch.next_checkpoint


def test_a_consumer_validates_its_configuration(tmp_path):
    ledger = book(tmp_path, csv(tmp_path, 1))
    with pytest.raises(errors.InputError, match="consumer_group"):
        consumer.Consumer(ledger, " ", print, print)
    with pytest.raises(errors.KindError, match="callable"):
        consumer.Consumer(ledger, "g", None, print)
    with pytest.raises(errors.KindError, match="callable"):
        consumer.Consumer(ledger, "g", print, None)


def test_a_consumer_commits_then_acknowledges(tmp_path):
    ledger = book(tmp_path, csv(tmp_path, 3))
    written, delivered = [], []
    reader = consumer.Consumer(
        ledger,
        "g",
        lambda frame, key: written.append(frame["a"].tolist()),
        lambda acknowledgement: delivered.append(acknowledgement.batch),
    )
    first = reader.consume(rows=2)
    assert first.accepted
    assert written == [[0, 1]]
    assert delivered == [first.batch_id]
    second = reader.consume(rows=2)
    assert written == [[0, 1], [2]]
    assert delivered == [first.batch_id, second.batch_id]
    assert ledger.reclaim() is None
    assert reader.handoff() is None


def test_a_failed_acknowledgement_is_retried_later(tmp_path):
    # Enforces website/src/pages/docs/data.astro:17.
    ledger = book(tmp_path, csv(tmp_path, 3))
    state = {"fail": True}
    delivered = []

    def acknowledge(acknowledgement):
        if state["fail"]:
            raise ConnectionError
        delivered.append(acknowledgement.batch)

    reader = consumer.Consumer(ledger, "g", lambda f, k: None, acknowledge)
    with pytest.raises(failure.StreamingError) as raised:
        reader.consume(rows=1)
    assert raised.value.code == "acknowledgement_failed"
    pending = ledger.reclaim()
    assert raised.value.details["batch_id"] == pending.batch
    state["fail"] = False
    assert reader.handoff() == pending.batch
    assert delivered == [pending.batch]


def test_a_consumer_refuses_another_groups_acknowledgement(tmp_path):
    # Enforces website/src/pages/docs/data.astro:17.
    # Enforces website/src/pages/docs/data.astro:75.
    ledger = book(tmp_path, csv(tmp_path, 3))
    batch = ledger.read_batch(rows=1)
    ledger.earmark(ack(batch, "other"))
    reader = consumer.Consumer(ledger, "g", print, print)
    with pytest.raises(failure.StreamingError) as raised:
        reader.handoff()
    assert raised.value.code == "consumer_group_mismatch"
    assert raised.value.details["pending_consumer_group"] == "other"
