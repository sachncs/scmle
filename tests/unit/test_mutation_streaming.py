"""Streaming evidence refusals: exact messages, codes and named sources."""

import dataclasses

import pandas
import pytest

from scmle.contracts import errors
from scmle.data.readers import streaming

OTHER_BATCH = "batch_" + "0" * 64


class Sink(streaming.Sink):
    """A sink that must never be reached."""

    def commit(self, batch):
        """Fails: the ledger refused the batch before sending it."""
        raise AssertionError("the sink was called")


def source(path) -> streaming.Csv:
    pandas.DataFrame({"value": [1, 2, 3, 4]}).to_csv(path, index=False)
    return streaming.Csv(path)


def first(tmp_path):
    feed = source(tmp_path / "events.csv")
    ledger = streaming.Journal(tmp_path / "state.json", feed)
    return feed, ledger, ledger.read_batch(rows=2)


def acknowledge(batch, batch_id=None) -> streaming.Acknowledgement:
    return streaming.Acknowledgement(
        batch.source_identity,
        "consumer",
        batch_id or batch.batch_id,
        batch.next_checkpoint,
    )


def voucher(batch) -> streaming.Voucher:
    return streaming.Voucher(
        batch.source_identity, batch.batch_id, batch.next_checkpoint, "r1"
    )


def test_a_voucher_from_another_source_is_refused(tmp_path) -> None:
    batch = first(tmp_path)[2]
    other = dataclasses.replace(batch.next_checkpoint, source="x")
    with pytest.raises(
        errors.InputError, match=r"^receipt and checkpoint sources differ$"
    ):
        streaming.Voucher(batch.source_identity, batch.batch_id, other, "r1")


def test_a_commitment_for_another_batch_is_refused(tmp_path) -> None:
    batch = first(tmp_path)[2]
    with pytest.raises(
        errors.InputError,
        match=r"^evidence and acknowledgement batches differ$",
    ):
        streaming.Commitment(
            batch.source_identity,
            batch.batch_id,
            voucher(batch),
            acknowledge(batch, OTHER_BATCH),
        )


def test_a_receipt_for_another_batch_is_refused(tmp_path) -> None:
    batch = first(tmp_path)[2]
    with pytest.raises(
        errors.InputError,
        match=r"^acknowledgement does not match the commit$",
    ):
        streaming.Receipt(
            batch.source_identity,
            batch.batch_id,
            accepted=True,
            duplicate=False,
            checkpoint=batch.next_checkpoint,
            acknowledgement=acknowledge(batch, OTHER_BATCH),
        )


def test_a_batch_past_the_checkpoint_is_refused(tmp_path) -> None:
    feed, ledger, batch = first(tmp_path)
    later = feed.read_batch(batch.next_checkpoint, rows=2)
    with pytest.raises(
        streaming.StreamingError,
        match=r"^batch does not start at the durable checkpoint$",
    ) as raised:
        ledger.commit_transaction(later, Sink(), acknowledge(later))
    assert raised.value.code == "checkpoint_mismatch"


def test_state_of_another_source_names_that_source(tmp_path) -> None:
    feed, ledger, batch = first(tmp_path)
    ledger.finalize(batch, lambda frame, key: None)
    other = source(tmp_path / "other.csv")
    with pytest.raises(streaming.StreamingError) as raised:
        streaming.Journal(ledger.path, other).load()
    assert raised.value.details["state_source_identity"] == (
        feed.source_identity()
    )


@pytest.mark.parametrize(
    "change",
    [{"source": "elsewhere"}, {"digests": {"snapshot": "0" * 64}}],
)
def test_rebind_names_this_source(tmp_path, change) -> None:
    feed, batch = first(tmp_path)[::2]
    checkpoint = dataclasses.replace(batch.next_checkpoint, **change)
    with pytest.raises(streaming.StreamingError) as raised:
        feed.rebind(checkpoint, feed.snapshot_sha256())
    assert raised.value.source_identity == feed.source_identity()
    assert raised.value.details["source_identity"] == feed.source_identity()


def test_json_lines_report_the_columns_of_the_first_record(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"a": 1, "b": 2, "c": 3}\n', encoding="utf-8")
    assert streaming.Ndjson(path).shape() == (None, 3)
