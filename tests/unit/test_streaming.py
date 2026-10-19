"""Reading and splitting a table that is larger than the memory allowed."""

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
