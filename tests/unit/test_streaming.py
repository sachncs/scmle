"""Reading and splitting a table that is larger than the memory allowed."""

import dataclasses
import json
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scml.contracts import errors
from scml.data.readers import streaming


def write_parquet(path, rows, columns=6, chunk=100_000, seed=0):
    """Writes a Parquet file chunk by chunk, so the writer never holds it all."""
    rng = np.random.default_rng(seed)
    writer = None
    for start in range(0, rows, chunk):
        size = min(chunk, rows - start)
        table = pa.table(
            {
                **{f"x{i}": rng.normal(size=size) for i in range(columns)},
                "y": rng.integers(0, 2, size),
            }
        )
        writer = writer or pq.ParquetWriter(path, table.schema)
        writer.write_table(table)
    writer.close()


def test_the_split_side_depends_only_on_the_position_and_the_seed():
    positions = np.arange(10_000)

    first = streaming.side_of(positions, 0.2, 3)
    again = streaming.side_of(positions, 0.2, 3)
    other = streaming.side_of(positions, 0.2, 4)

    assert (first == again).all() and (first != other).any()
    assert first.mean() == pytest.approx(0.2, abs=0.02)


def test_the_sample_does_not_depend_on_the_batch_size(tmp_path):
    path = tmp_path / "t.parquet"
    write_parquet(path, 30_000, chunk=7_000)

    class Sized(streaming.ParquetChunks):
        rows_per_batch = 4_000

        def batches(self, rows=streaming.BATCH_ROWS):
            return super().batches(self.rows_per_batch)

    plain = streaming.sample_pass(
        streaming.ParquetChunks(path), 3_000, 800, 0.2, 5
    )[0]
    small = streaming.sample_pass(Sized(path), 3_000, 800, 0.2, 5)[0]

    pd.testing.assert_frame_equal(
        plain.sort_values("x0").reset_index(drop=True),
        small.sort_values("x0").reset_index(drop=True),
    )


def test_locked_rows_never_enter_the_development_sample(tmp_path):
    path = tmp_path / "t.parquet"
    write_parquet(path, 20_000)

    kept, totals = streaming.sample_pass(
        streaming.ParquetChunks(path), 2_000, 500, 0.2, 1
    )

    assert totals.rows == 20_000
    assert totals.locked == pytest.approx(4_000, rel=0.1)
    assert (
        kept[streaming.SPLIT_COLUMN] == streaming.DEVELOPMENT
    ).sum() == 2_000
    assert (kept[streaming.SPLIT_COLUMN] == streaming.LOCKED).sum() == 500
    assert len(kept) == totals.development_sample + totals.locked_sample


def test_a_file_far_larger_than_the_budget_is_read_in_bounded_memory(tmp_path):
    path = tmp_path / "big.parquet"
    write_parquet(path, 4_000_000, columns=7, chunk=250_000)
    script = textwrap.dedent(
        f"""
        import json, sys
        import numpy as np, pandas as pd
        from scml.data.readers import streaming
        def peak():
            if sys.platform == "win32":
                import psutil
                return psutil.Process().memory_info().peak_wset / 1e6
            import resource
            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024
        source = streaming.ParquetChunks(__import__("pathlib").Path({str(path)!r}))
        for batch in source.batches(1000):
            break
        before = peak()
        kept, totals = streaming.sample_pass(source, 20000, 5000, 0.2, 0)
        print(json.dumps({{"before": before, "after": peak(),
                           "rows": totals.rows, "kept": len(kept)}}))
        """
    )

    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=pathlib.Path(__file__).resolve().parents[2],
    )

    report = json.loads(done.stdout.strip().splitlines()[-1])
    whole_file_mb = 4_000_000 * 8 * 8 / 1e6
    assert report["rows"] == 4_000_000 and report["kept"] == 25_000
    assert report["after"] - report["before"] < whole_file_mb / 2


def test_unsupported_formats_and_missing_files_are_refused(tmp_path):
    (tmp_path / "t.xlsx").write_text("x")

    with pytest.raises(errors.DataError, match="cannot stream"):
        streaming.source_for(tmp_path / "t.xlsx")
    with pytest.raises(errors.DataError, match="not found"):
        streaming.source_for(tmp_path / "missing.parquet")


def test_json_lines_is_streamed_in_batches(tmp_path):
    frame = pd.DataFrame({"a": range(12), "y": [0, 1] * 6})
    path = tmp_path / "data.jsonl"
    frame.to_json(path, orient="records", lines=True)
    source = streaming.source_for(path)
    batches = list(source.batches(rows=5))
    assert [len(batch) for batch in batches] == [5, 5, 2]
    pd.testing.assert_frame_equal(pd.concat(batches, ignore_index=True), frame)


def test_sample_sizes_follow_the_memory_budget():
    small = streaming.sample_caps(10, 0.01, 0.2)
    large = streaming.sample_caps(10, 1.0, 0.2)

    assert large[0] > small[0] * 50 and small[0] >= streaming.MIN_SAMPLE_ROWS


def write_stream_csv(path):
    """Write a small timestamped source for durable streaming tests."""
    frame = pd.DataFrame(
        {
            "event_id": ["a", "b", "c", "d"],
            "event_time": [
                "2026-10-20T10:00:00+00:00",
                "2026-10-20T10:01:00+00:00",
                "2026-10-20T10:02:00+00:00",
                "2026-10-20T10:03:00+00:00",
            ],
            "value": [1, 2, 3, 4],
        }
    )
    frame.to_csv(path, index=False)


def test_stream_checkpoint_binds_snapshot_offset_and_watermark(tmp_path):
    path = tmp_path / "events.csv"
    write_stream_csv(path)
    source = streaming.CsvChunks(path)

    first = source.read_batch(rows=2, watermark_column="event_time")
    second = source.read_batch(
        first.next_checkpoint, rows=2, watermark_column="event_time"
    )

    assert first.start_offset == 0
    assert first.next_checkpoint.offset == 2
    assert first.next_checkpoint.watermark_column == "event_time"
    assert first.next_checkpoint.watermark.endswith("10:01:00+00:00")
    assert first.has_more
    assert second.start_offset == 2
    assert second.next_checkpoint.offset == 4
    assert second.next_checkpoint.watermark.endswith("10:03:00+00:00")
    assert not second.has_more

    changed = dataclasses.replace(
        first.next_checkpoint, snapshot_sha256="0" * 64
    )
    with pytest.raises(
        streaming.StreamingError, match="does not match the checkpoint"
    ):
        source.read_batch(changed, rows=2, watermark_column="event_time")

    with pytest.raises(streaming.StreamingError, match="watermark column"):
        source.read_batch(
            first.next_checkpoint, rows=2, watermark_column="value"
        )


def test_stream_ledger_commits_idempotently_and_persists_checkpoint(tmp_path):
    path = tmp_path / "events.csv"
    write_stream_csv(path)
    source = streaming.CsvChunks(path)
    ledger = streaming.StreamingLedger(tmp_path / "state.json", source)
    batch = ledger.read_batch(rows=2, watermark_column="event_time")
    written = []

    first = ledger.commit_with_writer(
        batch, lambda frame, key: written.append((frame.copy(), key))
    )
    second = ledger.commit_with_writer(
        batch, lambda frame, key: written.append((frame.copy(), key))
    )
    reopened = streaming.StreamingLedger(ledger.path, source)

    assert first.accepted and not first.duplicate
    assert second.duplicate and not second.accepted
    assert len(written) == 1
    assert written[0][1] == batch.batch_id
    assert reopened.load().checkpoint.offset == 2
    assert batch.batch_id in reopened.load().committed_batch_ids


def test_stream_consumer_retries_ack_after_restart_without_replaying_writer(
    tmp_path,
):
    path = tmp_path / "events.csv"
    write_stream_csv(path)
    source = streaming.CsvChunks(path)
    ledger_path = tmp_path / "state.json"
    written = []
    acknowledgements = []
    failures = [True]

    def writer(frame, key):
        written.append((frame["event_id"].tolist(), key))

    def flaky_ack(acknowledgement):
        if failures and failures.pop(0):
            raise OSError("transport unavailable")
        acknowledgements.append(acknowledgement)

    first_consumer = streaming.StreamingConsumer(
        streaming.StreamingLedger(ledger_path, source),
        "events-consumer",
        writer,
        flaky_ack,
    )
    with pytest.raises(streaming.StreamingError, match="retry is required"):
        first_consumer.consume(rows=2, watermark_column="event_time")

    assert len(written) == 1
    assert first_consumer.ledger.pending_acknowledgement_path.exists()

    restarted = streaming.StreamingConsumer(
        streaming.StreamingLedger(ledger_path, source),
        "events-consumer",
        writer,
        flaky_ack,
    )
    second = restarted.consume(rows=2, watermark_column="event_time")

    assert second.accepted
    assert len(written) == 2
    assert [item.batch_id for item in acknowledgements] == [
        written[0][1],
        written[1][1],
    ]
    assert not restarted.ledger.pending_acknowledgement_path.exists()


def test_stream_ledger_refuses_a_changed_source_snapshot(tmp_path):
    path = tmp_path / "events.csv"
    write_stream_csv(path)
    source = streaming.CsvChunks(path)
    ledger = streaming.StreamingLedger(tmp_path / "state.json", source)
    batch = ledger.read_batch(rows=2)
    ledger.commit_with_writer(batch, lambda frame, key: None)

    changed = pd.read_csv(path)
    changed.loc[0, "value"] = 99
    changed.to_csv(path, index=False)

    with pytest.raises(
        streaming.StreamingError, match="another source snapshot"
    ) as raised:
        ledger.load()
    assert raised.value.code == "source_snapshot_changed"
