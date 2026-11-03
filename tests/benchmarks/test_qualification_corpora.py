"""Corpus manifest: schema, hash-verified fetching and offline behaviour."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from benchmarks.qualification import corpora

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = b"id,label\n1,0\n2,1\n"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def entry(**changes):
    base = {
        "id": "toy-corpus",
        "name": "A toy corpus",
        "source_url": None,
        "licence": None,
        "sha256": None,
        "size_bytes": None,
        "ground_truth_file": None,
        "adjudication_source": None,
        "intended_use": "certificate",
        "status": "candidate",
        "notes": "candidate: existence, licence and hash to be verified",
    }
    return base | changes


def verified(url, **changes):
    return entry(
        status="verified",
        source_url=url,
        licence="CC0-1.0",
        sha256=DIGEST,
        size_bytes=len(PAYLOAD),
        ground_truth_file="truth.csv",
        adjudication_source="two human raters",
        **changes,
    )


def source_file(tmp_path, payload=PAYLOAD):
    path = tmp_path / "source.csv"
    path.write_bytes(payload)
    return path.as_uri()


def test_the_committed_manifest_is_valid_and_claims_nothing_unchecked():
    manifest = corpora.load_manifest()
    ids = {c.id for c in manifest}
    assert {
        "titanic-boat-body",
        "human-adjudicated-label-errors",
        "tabarena-v0pt1-tasks",
    } <= ids
    for corpus in manifest:
        if corpus.status == "verified":
            assert corpus.sha256 and corpus.source_url and corpus.licence
        else:
            assert "to be verified" in corpus.notes
    assert all(c.status == "candidate" for c in manifest)


def test_the_tabarena_entry_points_at_the_task_list_in_the_repo():
    tasks = ROOT / "benchmarks" / "sota" / "tasks_tabarena_v0pt1.csv"
    assert tasks.is_file()
    corpus = corpora.find(corpora.load_manifest(), "tabarena-v0pt1-tasks")
    assert "tasks_tabarena_v0pt1.csv" in corpus.notes


@pytest.mark.parametrize(
    "changes",
    [
        {"id": "Bad Id"},
        {"id": ""},
        {"intended_use": "other"},
        {"status": "maybe"},
        {"sha256": "abc"},
        {"sha256": "A" * 64},
        {"source_url": "http://example.com/x.csv"},
        {"source_url": "ftp://example.com/x.csv"},
        {"size_bytes": -1},
        {"notes": ""},
        {"status": "verified"},
        {"status": "verified", "source_url": "https://example.com/x"},
    ],
)
def test_the_schema_rejects_malformed_entries(changes):
    with pytest.raises(corpora.CorpusError):
        corpora.parse_entry(entry(**changes))


def test_the_schema_rejects_unknown_and_missing_fields_and_duplicates():
    extra = entry() | {"extra": 1}
    with pytest.raises(corpora.CorpusError, match="unexpected"):
        corpora.parse_entry(extra)
    short = {k: v for k, v in entry().items() if k != "licence"}
    with pytest.raises(corpora.CorpusError, match="missing"):
        corpora.parse_entry(short)
    with pytest.raises(corpora.CorpusError, match="duplicate"):
        corpora.parse_manifest(
            {"schema_version": 1, "corpora": [entry(), entry()]}
        )
    with pytest.raises(corpora.CorpusError, match="schema_version"):
        corpora.parse_manifest({"schema_version": 2, "corpora": []})


def test_a_candidate_cannot_be_fetched(tmp_path):
    with pytest.raises(corpora.CorpusUnavailableError, match="not verified"):
        corpora.fetch(corpora.parse_entry(entry()), tmp_path)


def test_a_verified_corpus_is_fetched_and_its_hash_checked(tmp_path):
    item = corpora.parse_entry(verified(source_file(tmp_path)))
    root = tmp_path / "cache"
    path = corpora.fetch(item, root, offline=False)
    assert path.read_bytes() == PAYLOAD
    assert corpora.sha256_of(path) == DIGEST
    assert corpora.is_available(item, root)
    assert path == corpora.destination(item, root)


def test_a_present_file_is_not_downloaded_again(tmp_path):
    source = tmp_path / "source.csv"
    source.write_bytes(PAYLOAD)
    item = corpora.parse_entry(verified(source.as_uri()))
    root = tmp_path / "cache"
    corpora.fetch(item, root, offline=False)
    source.unlink()
    assert corpora.fetch(item, root, offline=False).read_bytes() == PAYLOAD


def test_a_hash_mismatch_aborts_and_leaves_nothing_behind(tmp_path):
    item = corpora.parse_entry(
        verified(source_file(tmp_path, b"tampered contents\n"))
    )
    root = tmp_path / "cache"
    with pytest.raises(corpora.CorpusHashMismatchError):
        corpora.fetch(item, root, offline=False)
    assert not corpora.destination(item, root).exists()
    assert list((root / item.id).glob("*")) == []
    assert not corpora.is_available(item, root)


def test_a_corrupted_local_copy_is_replaced_from_the_source(tmp_path):
    item = corpora.parse_entry(verified(source_file(tmp_path)))
    root = tmp_path / "cache"
    target = corpora.destination(item, root)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"corrupted")
    assert not corpora.is_available(item, root)
    assert corpora.fetch(item, root, offline=False).read_bytes() == PAYLOAD


def test_a_download_larger_than_declared_is_refused(tmp_path):
    big = PAYLOAD * 10
    item = corpora.parse_entry(
        verified(source_file(tmp_path, big)) | {"sha256": DIGEST}
    )
    with pytest.raises(corpora.CorpusHashMismatchError, match="larger"):
        corpora.fetch(item, tmp_path / "cache", offline=False)


def test_offline_mode_refuses_to_fetch_but_serves_a_present_file(
    tmp_path, monkeypatch
):
    item = corpora.parse_entry(verified(source_file(tmp_path)))
    root = tmp_path / "cache"
    with pytest.raises(corpora.CorpusUnavailableError, match="offline"):
        corpora.fetch(item, root, offline=True)
    corpora.fetch(item, root, offline=False)
    assert corpora.fetch(item, root, offline=True).exists()
    monkeypatch.setenv("SCMLE_OFFLINE", "1")
    assert corpora.is_offline()
    monkeypatch.setenv("SCMLE_OFFLINE", "0")
    assert not corpora.is_offline()
    monkeypatch.delenv("SCMLE_OFFLINE")
    assert not corpora.is_offline()


def test_the_corpus_root_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SCMLE_CORPORA_DIR", str(tmp_path))
    assert corpora.corpus_root() == tmp_path
    monkeypatch.delenv("SCMLE_CORPORA_DIR")
    assert corpora.corpus_root().name == "corpora"


def corpus_or_skip(corpus_id):
    """Skips with the reason the skip manifest approves."""
    item = corpora.find(corpora.load_manifest(), corpus_id)
    if not corpora.is_available(item, corpora.corpus_root()):
        pytest.skip(f"corpus '{corpus_id}' is not available locally")
    return corpora.destination(item, corpora.corpus_root())


def test_the_titanic_leak_corpus_when_it_is_present_locally():
    path = corpus_or_skip("titanic-boat-body")
    assert path.exists()


def test_the_skip_reason_is_registered_in_the_skip_manifest():
    spec = importlib.util.spec_from_file_location(
        "check_pytest_skips", ROOT / "scripts" / "check_pytest_skips.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rules = module.load_skip_rules()
    reason = "corpus 'titanic-boat-body' is not available locally"
    assert module.classify_reason(reason, rules) == "environment"
    manifest = json.loads(
        (ROOT / "scripts" / "pytest_skip_manifest.json").read_text()
    )
    assert "environment" in manifest["approved_categories"]
