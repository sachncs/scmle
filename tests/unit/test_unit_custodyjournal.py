"""Unit tests of the custody journal, imported by its full module name."""

import hashlib
import json

import pytest

from scmle.assurance.custody import journal
from scmle.contracts import errors


def make(tmp_path, name="run1"):
    return journal.Journal.locate(tmp_path / "runs" / name)


def test_genesis_is_the_hash_of_the_journal_name():
    assert journal.FOLDER == "custody"
    assert (
        hashlib.sha256(b"scmle.custody.journal").hexdigest() == journal.GENESIS
    )


def test_digest_hashes_canonical_json_regardless_of_key_order():
    first = journal.digest({"kind": "opened", "run": "r"})
    second = journal.digest({"run": "r", "kind": "opened"})
    text = '{"kind":"opened","run":"r"}'
    assert first == second == hashlib.sha256(text.encode()).hexdigest()


def test_locate_puts_the_journal_beside_the_run(tmp_path):
    book = make(tmp_path)
    assert book.path == tmp_path / "runs" / "custody" / "run1.jsonl"
    assert book.run == "run1"


def test_a_missing_journal_reads_as_empty(tmp_path):
    book = make(tmp_path)
    assert book.read() == []
    assert book.current() == []
    assert book.rejected() == {}


def test_entries_are_chained_and_read_back(tmp_path):
    book = make(tmp_path)
    book.start("cfg")
    book.decide("plan", approved=False, reviewer="ann", reason="no")
    book.witness()
    entries = book.read()
    assert [e["kind"] for e in entries] == ["started", "decided", "opened"]
    assert entries[0]["previous"] == journal.GENESIS
    assert entries[0]["config"] == "cfg"
    assert entries[0]["run"] == "run1"
    assert entries[1]["previous"] == journal.digest(entries[0])
    assert entries[2]["previous"] == journal.digest(entries[1])
    assert entries[1]["reason"] == "no"
    assert "digest" not in entries[0]
    lines = book.path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["digest"] == journal.digest(entries[0])


def test_an_edited_entry_is_refused(tmp_path):
    book = make(tmp_path)
    book.start("cfg")
    book.start("cfg2")
    lines = book.path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["config"] = "forged"
    lines[0] = json.dumps(record)
    book.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(errors.BlockedError, match="was edited; run run1"):
        book.read()


def test_a_removed_entry_breaks_the_chain(tmp_path):
    book = make(tmp_path)
    book.start("a")
    book.start("b")
    lines = book.path.read_text(encoding="utf-8").splitlines()
    book.path.write_text(lines[1] + "\n", encoding="utf-8")
    with pytest.raises(errors.BlockedError, match="cannot be resumed"):
        book.read()


def test_current_starts_at_the_latest_start(tmp_path):
    book = make(tmp_path)
    book.start("a")
    book.witness()
    book.start("b")
    book.decide("g", approved=True, reviewer="r", reason="ok")
    current = book.current()
    assert [e["kind"] for e in current] == ["started", "decided"]
    assert current[0]["config"] == "b"


def test_current_is_empty_without_a_start(tmp_path):
    book = make(tmp_path)
    book.witness()
    assert book.current() == []


def test_rejected_keeps_the_latest_verdict_per_gate(tmp_path):
    book = make(tmp_path)
    book.start("a")
    book.decide("plan", approved=False, reviewer="ann", reason="x")
    book.decide("data", approved=False, reviewer="bob", reason="y")
    book.decide("plan", approved=True, reviewer="cat", reason="z")
    book.witness()
    assert book.rejected() == {"data": "bob"}


def test_admit_refuses_a_run_without_a_record(tmp_path):
    book = make(tmp_path)
    with pytest.raises(errors.BlockedError, match="has no custody record"):
        book.admit()


def test_admit_refuses_an_opened_test_set_and_allows_a_shut_one(tmp_path):
    book = make(tmp_path)
    book.start("a")
    book.decide("g", approved=True, reviewer="r", reason="ok")
    book.admit()
    book.witness()
    with pytest.raises(errors.BlockedError, match="was already opened"):
        book.admit()
    book.start("b")
    book.admit()


class Box:
    """A stand-in test set that records what was asked of it."""

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
        return "features", "labels"

    def close(self):
        self.calls.append(("close",))


def test_a_custodian_records_the_opening_before_it_opens(tmp_path):
    book = make(tmp_path)
    book.start("a")
    inner = Box()
    guard = journal.Custodian(inner, book)
    assert guard.ids == (1, 2)
    guard.ids = (3,)
    assert inner.ids == (3,)
    assert guard.opened is False
    assert guard.mint() == "token"
    guard.bind("sel")
    assert guard.open("token", "sel") == ("features", "labels")
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


def test_a_failed_opening_is_still_recorded(tmp_path):
    book = make(tmp_path)

    class Broken(Box):
        def open(self, token, binding=None):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        journal.Custodian(Broken(), book).open("t")
    assert [e["kind"] for e in book.read()] == ["opened"]
