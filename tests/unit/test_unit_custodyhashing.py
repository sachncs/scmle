"""Behavioural tests of the custody journal and the detector hashes."""

import ast
import hashlib
import json

import pytest

from scmle.assurance.custody import journal
from scmle.assurance.detectors import hashing
from scmle.contracts import errors

# ---- custody journal --------------------------------------------------------


def make_journal(tmp_path):
    return journal.Journal.locate(tmp_path / "runs" / "r1")


def test_the_genesis_is_the_hash_of_the_journal_name():
    assert (
        hashlib.sha256(b"scmle.custody.journal").hexdigest() == journal.GENESIS
    )
    assert journal.FOLDER == "custody"


def test_digest_hashes_canonical_json_regardless_of_key_order():
    first = journal.digest({"kind": "opened", "run": "r"})
    second = journal.digest({"run": "r", "kind": "opened"})
    expected = hashlib.sha256(b'{"kind":"opened","run":"r"}').hexdigest()
    assert first == second == expected


def test_the_journal_lives_beside_the_run_directory(tmp_path):
    book = make_journal(tmp_path)
    assert book.path == tmp_path / "runs" / "custody" / "r1.jsonl"
    assert book.run == "r1"
    assert book.read() == []
    assert book.current() == []


def test_entries_are_chained_from_the_genesis(tmp_path):
    book = make_journal(tmp_path)
    book.start("cfg")
    book.decide("gate", False, "ann", "too risky")
    book.witness()
    entries = book.read()
    assert [e["kind"] for e in entries] == ["started", "decided", "opened"]
    assert entries[0]["previous"] == journal.GENESIS
    assert entries[0]["config"] == "cfg"
    assert entries[1]["previous"] == journal.digest(entries[0])
    assert entries[2]["previous"] == journal.digest(entries[1])
    assert all(e["run"] == "r1" for e in entries)
    assert "digest" not in entries[0]
    assert entries[1]["approved"] is False
    assert (entries[1]["reviewer"], entries[1]["reason"]) == (
        "ann",
        "too risky",
    )
    lines = book.path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[2])["digest"] == journal.digest(entries[2])


def test_an_edited_entry_is_refused(tmp_path):
    book = make_journal(tmp_path)
    book.start("cfg")
    book.witness()
    lines = book.path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["config"] = "other"
    lines[0] = json.dumps(record)
    book.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(errors.BlockedError, match="r1 cannot be resumed"):
        book.read()


def test_a_removed_entry_breaks_the_chain(tmp_path):
    book = make_journal(tmp_path)
    book.start("cfg")
    book.witness()
    book.witness()
    lines = book.path.read_text(encoding="utf-8").splitlines()
    book.path.write_text(lines[0] + "\n" + lines[2] + "\n", encoding="utf-8")
    with pytest.raises(errors.BlockedError, match="was edited"):
        book.read()


def test_current_starts_at_the_latest_start(tmp_path):
    book = make_journal(tmp_path)
    book.witness()
    assert book.current() == []
    book.start("a")
    book.witness()
    book.start("b")
    book.decide("g", True, "r", "ok")
    current = book.current()
    assert [e["kind"] for e in current] == ["started", "decided"]
    assert current[0]["config"] == "b"


def test_rejected_keeps_only_the_latest_verdict_of_each_gate(tmp_path):
    book = make_journal(tmp_path)
    book.start("cfg")
    book.decide("a", False, "ann", "no")
    book.decide("b", False, "bob", "no")
    book.decide("a", True, "cat", "fixed")
    book.decide("c", True, "dan", "fine")
    book.witness()
    assert book.rejected() == {"b": "bob"}


def test_admit_refuses_a_run_without_a_record_or_with_an_opened_test_set(
    tmp_path,
):
    book = make_journal(tmp_path)
    with pytest.raises(
        errors.BlockedError, match="run r1 has no custody record"
    ):
        book.admit()
    book.start("cfg")
    book.decide("g", True, "r", "ok")
    book.admit()
    book.witness()
    with pytest.raises(
        errors.BlockedError, match="test set of run r1 was already opened"
    ):
        book.admit()
    book.start("cfg")
    book.admit()


class Box:
    def __init__(self):
        self.ids = (1, 2)
        self.opened = False
        self.calls = []

    def mint(self):
        return "token"

    def bind(self, binding):
        self.calls.append(("bind", binding))

    def open(self, token, binding=None):
        self.calls.append(("open", token, binding))
        self.opened = True
        return "frame", "labels"

    def close(self):
        self.calls.append(("close",))


def test_the_custodian_records_the_opening_before_releasing(tmp_path):
    book = make_journal(tmp_path)
    book.start("cfg")
    inner = Box()
    guard = journal.Custodian(inner, book)
    assert guard.ids == (1, 2)
    guard.ids = (3,)
    assert inner.ids == (3,)
    assert guard.opened is False
    assert guard.mint() == "token"
    guard.bind("sel")
    assert guard.open("token", "sel") == ("frame", "labels")
    assert guard.opened is True
    guard.opened = False
    assert inner.opened is False
    guard.close()
    assert inner.calls == [
        ("bind", "sel"),
        ("open", "token", "sel"),
        ("close",),
    ]
    assert [e["kind"] for e in book.read()] == ["started", "opened"]


# ---- detector hashing -------------------------------------------------------


def test_documentation_finds_lines_held_only_by_docstrings():
    source = (
        '"""Module."""\n\n\ndef f():\n    """Doc\n    more."""\n    return 1\n'
    )
    assert hashing.documentation(ast.parse(source)) == {1, 5, 6}
    shared = 'def g(): "doc"; return 1\n'
    assert hashing.documentation(ast.parse(shared)) == set()
    plain = "def h():\n    return 'x'\n"
    assert hashing.documentation(ast.parse(plain)) == set()


def test_hash_source_ignores_comments_docstrings_and_blank_lines():
    bare = "def f(x):\n    return x + 1\n"
    noisy = (
        "def f(x):  # a comment\n"
        '    """Adds one."""\n'
        "\n"
        "    # another\n"
        "    return x + 1   \n"
    )
    assert hashing.hash_source(bare) == hashing.hash_source(noisy)
    expected = hashlib.sha256(b"def f(x):\n    return x + 1").hexdigest()
    assert hashing.hash_source(bare) == expected
    assert hashing.hash_source(bare) != hashing.hash_source(
        bare.replace("1", "2")
    )


def test_hash_source_dedents_and_refuses_invalid_python():
    assert hashing.hash_source("    x = 1\n") == hashing.hash_source("x = 1\n")
    with pytest.raises(SyntaxError):
        hashing.hash_source("def (:\n")


def sample_one():
    return 1


def sample_two():
    return 2


def test_code_hash_combines_sources_in_order():
    first = hashing.code_hash(sample_one, sample_two)
    second = hashing.code_hash(sample_two, sample_one)
    assert first != second
    parts = [
        hashing.hash_source("def sample_one():\n    return 1\n"),
        hashing.hash_source("def sample_two():\n    return 2\n"),
    ]
    assert first == hashlib.sha256("\n".join(parts).encode("ascii")).hexdigest()
    with pytest.raises(
        errors.InputError, match="code_hash needs at least one object"
    ):
        hashing.code_hash()


def test_corpus_hash_ignores_order_and_checks_each_part():
    a, b = "a" * 64, "b" * 64
    assert hashing.corpus_hash([a, b]) == hashing.corpus_hash([b, a])
    expected = hashlib.sha256(f"{a}\n{b}".encode("ascii")).hexdigest()
    assert hashing.corpus_hash([b, a]) == expected
    with pytest.raises(
        errors.InputError, match="corpus_hash needs at least one part"
    ):
        hashing.corpus_hash([])
    with pytest.raises(
        errors.InputError, match="corpus part must be a 64-character"
    ):
        hashing.corpus_hash([a, "A" * 64])


@pytest.mark.parametrize(
    "value", ["a" * 63, "a" * 65, "g" * 64, "A" * 64, 7, None]
)
def test_require_digest_refuses_anything_but_a_lowercase_sha256(value):
    with pytest.raises(errors.InputError, match="name must be a 64-character"):
        hashing.require_digest(value, "name")


def test_require_digest_returns_a_valid_digest_unchanged():
    value = "0123456789abcdef" * 4
    assert hashing.require_digest(value, "name") is value
