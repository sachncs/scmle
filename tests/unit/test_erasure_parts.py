"""Receipts, journals, retirement marks and source purges of an erasure."""

import json

import pandas
import pytest

from scmle.api import erasure
from scmle.api.erasure import receipts
from scmle.api.erasure import steps
from scmle.artifacts.bundle import tombstone
from scmle.contracts import errors

KEY = "unit-key"


def receipt(**changes):
    fields = {
        "identifier": "forget-0123456789abcdef",
        "kind": "forget",
        "reason": "ticket",
        "request": {"subject": {"digests": ["abc"]}},
        "steps": ({"action": "remove", "target": "/x", "result": {}},),
        "started": "2026-11-19T00:00:00+00:00",
        "finished": "2026-11-19T00:00:01+00:00",
    }
    return receipts.Receipt(**{**fields, **changes})


def test_a_receipt_verifies_only_unchanged_and_under_its_key(tmp_path):
    signed = receipt().sign(KEY)
    path = tmp_path / "r.json"
    signed.write(path)

    read = erasure.Receipt.read(path)

    assert read == signed
    assert read.verify(KEY)
    assert not read.verify("another-key")
    assert not receipt().verify(KEY)
    assert not erasure.Receipt.load(
        {**signed.to_dict(), "reason": "changed"}
    ).verify(KEY)
    assert read.tally() == {"remove": 1}


def test_a_receipt_of_another_format_is_refused(tmp_path):
    with pytest.raises(errors.ArtifactError, match="format 2"):
        erasure.Receipt.load({**receipt().to_dict(), "format": 2})
    (tmp_path / "bad.json").write_text("[1]", encoding="utf-8")
    with pytest.raises(errors.ArtifactError, match="not an erasure receipt"):
        erasure.Receipt.read(tmp_path / "bad.json")


def test_the_journal_survives_a_torn_last_line(tmp_path):
    journal = receipts.Journal(tmp_path / "job")
    journal.record({"steps": []})
    journal.finish(0, {"removed": True})
    with (tmp_path / "job" / receipts.JOURNAL).open("a") as handle:
        handle.write('{"step": 1, "res')

    journal.finish(1, {"removed": False})

    assert journal.done() == {0: {"removed": True}, 1: {"removed": False}}
    assert journal.plan() == {"steps": []}


def test_a_step_round_trips_and_an_unknown_one_is_refused():
    step = steps.Mark("/b", {"content": "c"}, "/r", "churn", 3)

    assert steps.Step.load(step.to_dict()) == step
    with pytest.raises(errors.ArtifactError, match="unknown erasure step"):
        steps.Step.load({"action": "shred", "target": "/x"})


def test_a_retirement_mark_refuses_and_a_damaged_one_still_refuses(tmp_path):
    tombstone.mark(tmp_path, {"reason": "drift", "retired": "2026-11-19"})

    with pytest.raises(tombstone.RetiredError, match=r"2026-11-19.*drift"):
        tombstone.enforce(tmp_path)
    (tmp_path / tombstone.FILE).write_text("{", encoding="utf-8")
    with pytest.raises(tombstone.RetiredError, match="unreadable"):
        tombstone.enforce(tmp_path)
    assert tombstone.read(tmp_path / "missing") is None


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".parquet", ".jsonl"])
def test_a_purge_removes_exactly_the_subject_rows_once(tmp_path, suffix):
    frame = pandas.DataFrame(
        {"id": ["a1", "b2", "a1", "c3"], "x": ["1", "2", "3", "4"]}
    )
    path = tmp_path / f"data{suffix}"
    path.write_bytes(steps.Table.find(path).write(frame))
    purge = steps.Purge(
        str(path), {}, "id", (steps.fingerprint("a1"), steps.fingerprint("z"))
    )
    order = steps.Order("forget-x", "ticket", KEY.encode())

    first = purge.perform(order)
    second = purge.perform(order)

    left = steps.Table.find(path).read(path)
    assert left.astype(str).to_dict("list") == {
        "id": ["b2", "c3"],
        "x": ["2", "4"],
    }
    assert first["rows"] == 2
    assert second == {"rows": 0, "sha256": first["sha256"]}


def test_a_subject_keeps_only_hashes_and_refuses_other_formats(tmp_path):
    path = tmp_path / "people.csv"
    path.write_text("id,y\nP-1,0\nP-2,1\n", encoding="utf-8")

    subject = erasure.Subject(path, "id", ["P-1", "P-1"])

    described = json.dumps(subject.describe())
    assert "P-1" not in described
    assert subject.hashes == (steps.fingerprint("P-1"),)
    with pytest.raises(errors.InputError, match="no column 'name'"):
        erasure.Subject(path, "name", ["P-1"]).identities()
    (tmp_path / "people.xlsx").write_bytes(b"")
    with pytest.raises(errors.InputError, match="cannot rewrite"):
        erasure.Subject(tmp_path / "people.xlsx", "id", ["P-1"])
    with pytest.raises(errors.InputError, match="at least one identifier"):
        erasure.Subject(path, "id", [])
    with pytest.raises(errors.InputError, match="at least one digest"):
        erasure.Dataset([])
