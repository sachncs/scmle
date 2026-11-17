"""Stored hashes are re-pointed only after a proven, rename-only move."""

import hashlib
import importlib.util
import json
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "rebind.py"
OLD = "a" * 64
NEW = "b" * 64


def load():
    """Imports the script as a module, since `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("rebind", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recording(*digests, status=0):
    return {"check": {"status": status, "digests": list(digests)}}


def git(root, *arguments):
    subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True
    )


@pytest.fixture
def tree(tmp_path):
    """A committed toy checkout with one store."""
    files = {
        "scmle/__init__.py": '"""Package."""\n',
        "scmle/core/__init__.py": '"""Core."""\n',
        "scmle/runtime/retry.py": "VALUE = 1\n",
        "scmle/certificates/store.json": json.dumps(
            {"code": OLD, "short": OLD[:16], "other": "c" * 64}
        ),
    }
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(
        tmp_path,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-qm",
        "start",
    )
    return tmp_path


def test_pairing_matches_digests_by_position():
    module = load()
    pairs = module.pairing(recording(OLD, "c" * 64), recording(NEW, "c" * 64))
    assert pairs == {OLD: NEW}


def test_pairing_refuses_recordings_of_another_shape():
    module = load()
    with pytest.raises(SystemExit, match="digests before"):
        module.pairing(recording(OLD), recording(NEW, NEW))
    with pytest.raises(SystemExit, match="two values"):
        module.pairing(recording(OLD, OLD), recording(NEW, "c" * 64))


def test_lookup_maps_truncated_prefixes_and_refuses_ambiguity():
    module = load()
    assert module.lookup(OLD[:16], {OLD: NEW}) == NEW[:16]
    assert module.lookup("d" * 16, {OLD: NEW}) is None
    with pytest.raises(SystemExit, match="several"):
        module.lookup(OLD[:16], {OLD: NEW, OLD[:63] + "f": "c" * 64})


def test_repoint_rewrites_only_matching_values(tree):
    module = load()
    entries = module.repoint(tree, {OLD: NEW}, "step")
    stored = json.loads((tree / "scmle/certificates/store.json").read_text())
    assert stored == {"code": NEW, "short": NEW[:16], "other": "c" * 64}
    assert [entry["new"] for entry in entries] == [NEW, NEW[:16]]


def test_apply_refuses_a_store_that_was_stale_before(tree):
    module = load()
    with pytest.raises(SystemExit, match="stale before"):
        module.apply(tree, recording(OLD, status=1), "step", {}, set())


def test_apply_refuses_a_change_that_is_not_a_rename(tree):
    module = load()
    (tree / "scmle/runtime/retry.py").write_text("VALUE = 2\n")
    moves = {"scmle.runtime.retry": "scmle.low.retry"}
    with pytest.raises(SystemExit, match="--verify fails"):
        module.apply(tree, recording(OLD), "step", moves, set())
    before = (tree / "scmle/certificates/store.json").read_text()
    assert OLD in before


def test_recorder_sees_every_digest(monkeypatch):
    module = load()
    for name in module.CONSTRUCTORS:
        monkeypatch.setattr(hashlib, name, getattr(hashlib, name))
    seen = []
    module.instrument(seen)
    first = hashlib.sha256(b"x").hexdigest()
    second = hashlib.new("sha256", b"y").digest().hex()
    copied = hashlib.blake2b(b"z")
    third = copied.copy().hexdigest()
    assert seen == [first, second, third]
