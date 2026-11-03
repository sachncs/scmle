"""The hash-chained open log and named sealed sets."""

import json

import pandas as pd
import pytest

from scmle.contracts import errors
from scmle.data.custody import sealed
from scmle.data.splitting import splitters


def resident(values=(1.0, 2.0, 3.0)):
    return splitters.ResidentTestSet(
        pd.DataFrame({"a": list(values)}),
        pd.Series([0, 1, 0]),
        (10, 11, 12),
    )


def test_log_chains_entries_and_verifies():
    log = sealed.OpenLog()
    first = log.append("C", "sealed", "orchestrator")
    second = log.append("C", "open", "confirmation")
    assert first["prev"] == sealed.GENESIS
    assert second["prev"] == first["hash"]
    log.verify()


def test_log_detects_a_changed_entry(tmp_path):
    path = tmp_path / "custody_log.jsonl"
    log = sealed.OpenLog(path)
    log.append("C", "sealed", "orchestrator")
    log.append("C", "open", "confirmation")
    lines = path.read_text().splitlines()
    record = json.loads(lines[0])
    record["actor"] = "someone else"
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(errors.ArtifactError, match="chain"):
        sealed.OpenLog.verify_file(path)


def test_log_detects_a_removed_entry(tmp_path):
    path = tmp_path / "custody_log.jsonl"
    log = sealed.OpenLog(path)
    for event in ("sealed", "token_issued", "open"):
        log.append("C", event, "orchestrator")
    lines = path.read_text().splitlines()
    del lines[1]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(errors.ArtifactError, match="chain"):
        sealed.OpenLog.verify_file(path)


def test_log_reloads_and_continues_the_chain(tmp_path):
    path = tmp_path / "custody_log.jsonl"
    sealed.OpenLog(path).append("C", "sealed", "orchestrator")
    again = sealed.OpenLog(path)
    entry = again.append("C", "open", "confirmation")
    assert entry["seq"] == 1
    sealed.OpenLog.verify_file(path)


def test_a_set_opens_once_and_every_attempt_is_logged():
    log = sealed.OpenLog()
    held = sealed.SealedSet("C", resident(), log)
    token = held.issue_token()
    rows = held.open(token, actor="confirmation")
    assert list(rows[0]["a"]) == [1.0, 2.0, 3.0]
    with pytest.raises(errors.LeakageError, match="already used"):
        held.open(token, actor="exploration")
    events = [(e["event"], e["actor"]) for e in log.entries]
    assert ("open", "confirmation") in events
    assert ("denied", "exploration") in events
    log.verify()


def test_a_wrong_token_is_denied_and_logged_without_opening():
    log = sealed.OpenLog()
    held = sealed.SealedSet("C", resident(), log)
    with pytest.raises(errors.LeakageError, match="invalid token"):
        held.open("guess", actor="exploration")
    assert held.opened is False
    assert [e["event"] for e in log.entries][-1] == "denied"
    held.open(held.issue_token(), actor="confirmation")


def test_the_log_never_contains_row_values():
    log = sealed.OpenLog()
    held = sealed.SealedSet("C", resident((123456.789, 2.0, 3.0)), log)
    held.open(held.issue_token(), actor="confirmation")
    assert "123456.789" not in json.dumps(log.entries)


def test_the_vault_gives_each_set_its_own_token():
    log = sealed.OpenLog()
    vault = sealed.CustodyVault(log)
    first = vault.adopt("C", resident())
    second = vault.adopt("final_test", resident())
    assert first.issue_token() != second.issue_token()
    with pytest.raises(errors.LeakageError):
        first.open(second.issue_token(), actor="x")
    vault.close()


def test_the_vault_refuses_a_duplicate_set_name():
    vault = sealed.CustodyVault(sealed.OpenLog())
    vault.adopt("C", resident())
    with pytest.raises(ValueError, match="already"):
        vault.adopt("C", resident())
    vault.close()


def test_exploration_view_holds_no_reference_to_a_sealed_set():
    view = sealed.ExplorationView(
        features=pd.DataFrame({"a": [1.0]}),
        target=pd.Series([0]),
        ids=(0,),
        folds=(),
        manifest_sha256="f" * 64,
    )
    assert not hasattr(view, "__dict__")
    assert set(view.__slots__) == {
        "features",
        "target",
        "ids",
        "folds",
        "manifest_sha256",
    }
