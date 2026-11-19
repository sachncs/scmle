"""Streamed batch input: chunked readers, bounded memory, resumable output."""

import gzip
import hashlib
import json
import pathlib
import tracemalloc
import zipfile

import numpy
import pandas
import pytest

from scmle import cli
from scmle.contracts import errors
from scmle.data.readers import stream
from scmle.data.readers import tabular
from scmle.deploy import parity
from scmle.serving import server

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "deploy" / "local" / "golden"


@pytest.fixture(scope="module")
def golden():
    return parity.goldens(GOLDEN)


@pytest.fixture
def bundle(golden):
    return parity.handwrite(golden.spec)


def rows(count, seed=0):
    rng = numpy.random.default_rng(seed)
    return pandas.DataFrame(
        {
            "age": rng.integers(18, 90, count).astype(float),
            "balance": rng.normal(1500, 800, count).round(2),
            "plan": rng.choice(["basic", "pro"], count),
        }
    )


def joined(chunks):
    return pandas.concat(list(chunks), ignore_index=True)


@pytest.mark.parametrize("size", [1, 2, 3, 7, 50])
def test_csv_chunks_read_the_same_table_as_a_whole_file(tmp_path, size):
    path = tmp_path / "rows.csv"
    rows(23).to_csv(path, index=False)
    pieces = list(tabular.Csv().chunks(path, size))
    assert [len(piece) for piece in pieces[:-1]] == [size] * (len(pieces) - 1)
    pandas.testing.assert_frame_equal(joined(pieces), tabular.Csv().load(path))


def test_compressed_and_semicolon_files_stream(tmp_path):
    frame = rows(9)
    gz = tmp_path / "rows.csv.gz"
    gz.write_bytes(gzip.compress(frame.to_csv(index=False).encode()))
    archive = tmp_path / "rows.csv.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("rows.csv", frame.to_csv(index=False))
    semicolon = tmp_path / "rows_eu.csv"
    semicolon.write_text("a;b\n1,5;x\n2,25;y\n", encoding="utf-8")
    for path in (gz, archive):
        pandas.testing.assert_frame_equal(
            joined(tabular.Csv().chunks(path, 4)), tabular.Csv().load(path)
        )
    assert joined(tabular.Csv().chunks(semicolon, 1))["a"].tolist() == [
        1.5,
        2.25,
    ]


def test_encodings_are_detected_from_the_first_bytes(tmp_path):
    legacy = tmp_path / "legacy.csv"
    legacy.write_bytes("name,n\ncafé,1\nnaïve,2\n".encode("cp1252"))
    wide = tmp_path / "wide.csv"
    wide.write_bytes("name,n\nx,1\n".encode("utf-16"))
    assert stream.detect(legacy.read_bytes()) == "cp1252"
    assert stream.detect(wide.read_bytes()) == "utf-16"
    assert stream.detect("é".encode()[:1]) == "utf-8"
    assert joined(tabular.Csv().chunks(legacy, 1))["name"].tolist() == [
        "café",
        "naïve",
    ]
    assert joined(tabular.Csv().chunks(wide, 5))["n"].tolist() == [1]
    with pytest.raises(errors.InputError, match="NUL bytes"):
        stream.detect(b"a\x00b")


def test_streaming_refuses_what_loading_refuses(tmp_path):
    ragged = tmp_path / "ragged.csv"
    ragged.write_text("a,b\n1,2\n3\n", encoding="utf-8")
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("a,a\n1,2\n", encoding="utf-8")
    ambiguous = tmp_path / "ambiguous.csv"
    ambiguous.write_text("a,b;c\n1,2;3\n", encoding="utf-8")
    with pytest.raises(
        errors.InputError, match=r"ragged\.csv: line 3 has 1 fields"
    ):
        list(tabular.Csv().chunks(ragged, 10))
    with pytest.raises(errors.InputError, match="duplicate column names"):
        list(tabular.Csv().chunks(duplicate, 10))
    with pytest.raises(errors.InputError, match="state the delimiter"):
        list(tabular.Csv().chunks(ambiguous, 10))
    with pytest.raises(errors.InputError, match="positive"):
        list(stream.frames(ragged, 0))


def test_the_loader_streams_every_format_and_checks_like_load(tmp_path):
    frame = rows(10)
    parquet = tmp_path / "rows.parquet"
    frame.to_parquet(parquet)
    lines = tmp_path / "rows.jsonl"
    frame.to_json(lines, orient="records", lines=True)
    feather = tmp_path / "rows.feather"
    frame.to_feather(feather)
    loader = tabular.Loader()
    for path in (parquet, lines, feather):
        pieces = list(loader.stream(path, 4))
        assert [len(piece) for piece in pieces] == [4, 4, 2]
        pandas.testing.assert_frame_equal(joined(pieces), loader.load(path))
    empty = tmp_path / "empty.csv"
    empty.write_text("age,balance,plan\n", encoding="utf-8")
    with pytest.raises(errors.DataError, match=r"empty\.csv contains no rows"):
        list(loader.stream(empty, 4))
    with pytest.raises(errors.DataError, match="dataset not found"):
        list(loader.stream(tmp_path / "absent.csv", 4))
    with pytest.raises(errors.DataError, match="positive"):
        list(loader.stream(parquet, 0))
    broken = tmp_path / "broken.parquet"
    broken.write_bytes(b"not parquet")
    with pytest.raises(errors.DataError, match="failed to read broken"):
        list(loader.stream(broken, 4))


def peak(callable_):
    tracemalloc.start()
    try:
        callable_()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_batch_memory_follows_the_chunk_not_the_file(bundle, tmp_path):
    # A whole-file read grows linearly with the rows; a streamed one stays
    # at one chunk. Four times the rows may cost at most half as much again
    # (allocator noise), while a whole-file read would cost four times.
    small, large = tmp_path / "small.csv", tmp_path / "large.csv"
    rows(5_000).to_csv(small, index=False)
    rows(20_000, seed=1).to_csv(large, index=False)
    adapter = server.Batcher(bundle)
    first = peak(lambda: adapter.score_file(small, tmp_path / "a.csv", 500))
    second = peak(lambda: adapter.score_file(large, tmp_path / "b.csv", 500))
    assert second < 1.5 * first


def test_streamed_output_matches_the_golden_scores(golden, bundle, tmp_path):
    source = tmp_path / "records.csv"
    pandas.DataFrame(golden.records).to_csv(source, index=False)
    target = tmp_path / "scored.csv"
    count = server.Batcher(bundle).score_file(source, target, chunk_rows=2)
    scored = pandas.read_csv(target)
    expected = parity.read_json(GOLDEN / "expected.json")
    report = parity.compare(
        expected,
        scored["prediction"].tolist(),
        scored[["proba_0", "proba_1"]].to_numpy().tolist(),
        golden.same_process,
    )
    assert report.passed, report.mismatches
    manifest = json.loads(
        (tmp_path / "scored.csv.manifest.json").read_text(encoding="utf-8")
    )
    content = target.read_bytes()
    assert count == manifest["rows"] == len(golden.records)
    assert manifest["output_sha256"] == (
        "sha256:" + hashlib.sha256(content).hexdigest()
    )
    assert content.count(b"prediction") == 1
    assert not (tmp_path / "scored.csv.partial").exists()
    assert not (tmp_path / "scored.csv.progress.json").exists()


class CrashError(Exception):
    """Stands in for a crash between two chunks."""


def interrupt_after(adapter, monkeypatch, calls):
    original = adapter.infer
    seen = []

    def infer(frame, remember=True):
        if len(seen) == calls:
            raise CrashError
        seen.append(len(frame))
        return original(frame, remember)

    monkeypatch.setattr(adapter, "infer", infer)
    return seen


def test_an_interrupted_batch_resumes_after_its_last_chunk(
    bundle, tmp_path, monkeypatch
):
    source = tmp_path / "rows.csv"
    rows(23).to_csv(source, index=False)
    whole = tmp_path / "whole.csv"
    server.Batcher(bundle).score_file(source, whole, chunk_rows=5)
    target = tmp_path / "resumed.csv"
    adapter = server.Batcher(bundle)
    interrupt_after(adapter, monkeypatch, 2)
    with pytest.raises(CrashError):
        adapter.score_file(source, target, chunk_rows=5)
    assert not target.exists()
    journal = json.loads(
        (tmp_path / "resumed.csv.progress.json").read_text(encoding="utf-8")
    )
    assert journal["rows"] == 10
    # A crash after the append and before the journal leaves extra bytes.
    with (tmp_path / "resumed.csv.partial").open("ab") as handle:
        handle.write(b"torn,row\n")
    monkeypatch.undo()
    resumed = server.Batcher(bundle)
    scored = interrupt_after(resumed, monkeypatch, 99)
    assert resumed.score_file(source, target, chunk_rows=4) == 23
    assert scored == [2, 4, 4, 3]
    assert target.read_bytes() == whole.read_bytes()
    manifests = [
        json.loads(
            (tmp_path / f"{name}.csv.manifest.json").read_text(encoding="utf-8")
        )
        for name in ("whole", "resumed")
    ]
    assert manifests[0] == manifests[1]


def test_a_journal_for_another_scoring_starts_over(
    bundle, tmp_path, monkeypatch
):
    source = tmp_path / "rows.csv"
    rows(12).to_csv(source, index=False)
    target = tmp_path / "out.csv"
    adapter = server.Batcher(bundle)
    interrupt_after(adapter, monkeypatch, 1)
    with pytest.raises(CrashError):
        adapter.score_file(source, target, chunk_rows=5)
    monkeypatch.undo()
    strict = server.Batcher(bundle, on_unseen="reject")
    scored = interrupt_after(strict, monkeypatch, 99)
    strict.score_file(source, target, chunk_rows=5)
    assert scored == [5, 5, 2]
    monkeypatch.undo()
    again = server.Batcher(bundle)
    interrupt_after(again, monkeypatch, 1)
    with pytest.raises(CrashError):
        again.score_file(source, target, chunk_rows=5)
    monkeypatch.undo()
    fresh = server.Batcher(bundle)
    scored = interrupt_after(fresh, monkeypatch, 99)
    fresh.score_file(source, target, chunk_rows=5, resume=False)
    assert scored == [5, 5, 2]


def test_an_unreadable_journal_starts_over(bundle, tmp_path, caplog):
    source = tmp_path / "rows.csv"
    rows(4).to_csv(source, index=False)
    target = tmp_path / "out.csv"
    (tmp_path / "out.csv.partial").write_bytes(b"x")
    (tmp_path / "out.csv.progress.json").write_text("{", encoding="utf-8")
    assert server.Batcher(bundle).score_file(source, target, 3) == 4
    assert "unreadable batch journal" in caplog.text
    with pytest.raises(errors.DataError, match="dataset not found"):
        server.Batcher(bundle).score_file(tmp_path / "absent.csv", target)


def test_the_batch_command_resumes_from_the_cli(
    golden, bundle, tmp_path, monkeypatch
):
    folder = bundle.save(tmp_path / "bundle")
    source = tmp_path / "rows.csv"
    rows(9).to_csv(source, index=False)
    output = tmp_path / "scored.csv"
    original = server.Batcher.infer
    calls, failures = [], [errors.DataError("disk went away")]

    def infer(self, frame, remember=True):
        calls.append(len(frame))
        if len(calls) == 2 and failures:
            raise failures.pop()
        return original(self, frame, remember)

    monkeypatch.setattr(server.Batcher, "infer", infer)
    arguments = [
        "batch",
        str(folder),
        str(source),
        "--output",
        str(output),
        "--trusted",
        "--chunk-rows",
        "4",
    ]
    assert cli.main(arguments) == cli.EXIT_FAILED
    assert cli.main(arguments) == cli.EXIT_OK
    assert calls == [4, 4, 4, 1]
    assert len(pandas.read_csv(output)) == 9
    calls.clear()
    assert cli.main([*arguments, "--restart"]) == cli.EXIT_OK
    assert calls == [4, 4, 1]
