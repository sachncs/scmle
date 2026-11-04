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


def committed(process_set):
    log = sealed.OpenLog()
    held = sealed.SealedSet("C", process_set, log)
    held.commit("claims-1", "ledger-1")
    return held, log


def test_a_committed_set_opens_only_for_its_claims_and_ledger():
    held, log = committed(resident())
    token = held.issue_token()
    for claims, ledger in (
        (None, None),
        ("claims-2", "ledger-1"),
        ("claims-1", "ledger-2"),
    ):
        with pytest.raises(errors.LeakageError, match="binding mismatch"):
            held.open(token, "attacker", claims, ledger)
    assert held.opened is False
    held.open(token, "confirmation", "claims-1", "ledger-1")
    assert held.opened is True
    events = [e["event"] for e in log.entries]
    assert events.count("denied") == 3 and events[-1] == "open"
    log.verify()


def test_the_binding_and_both_hashes_are_logged():
    held, log = committed(resident())
    bound = next(e for e in log.entries if e["event"] == "bound")
    assert bound["detail"] == {
        "claims_hash": "claims-1",
        "ledger_hash": "ledger-1",
    }
    assert held.committed is True
    assert held.binding == sealed.binding_digest("claims-1", "ledger-1")


def test_a_set_commits_once_and_not_after_it_opened():
    held, _ = committed(resident())
    with pytest.raises(errors.LeakageError, match="already bound"):
        held.commit("claims-2", "ledger-1")
    other = sealed.SealedSet("F", resident(), sealed.OpenLog())
    other.open(other.issue_token())
    with pytest.raises(errors.LeakageError, match="already bound or used"):
        other.commit("claims-1", "ledger-1")


def test_the_binding_digest_depends_on_both_hashes():
    base = sealed.binding_digest("a", "b")
    assert base != sealed.binding_digest("a", "c")
    assert base != sealed.binding_digest("c", "b")
    assert base != sealed.binding_digest("b", "a")
    assert len(base) == 64


def test_an_unbound_set_still_opens_without_hashes():
    held = sealed.SealedSet("C", resident(), sealed.OpenLog())
    assert held.committed is False
    assert len(held.open(held.issue_token())[0]) == 3


def rechain(path, key=None):
    """What a forger without the log key can do: recompute every hash."""
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    previous = sealed.GENESIS
    out = []
    for index, record in enumerate(lines):
        body = {k: v for k, v in record.items() if k not in ("hash", "mac")}
        body["seq"], body["prev"] = index, previous
        record = {**body, "hash": sealed.entry_hash(body)}
        previous = record["hash"]
        out.append(json.dumps(record))
    path.write_text("\n".join(out) + "\n")


def test_a_keyed_log_carries_a_mac_on_every_entry(tmp_path):
    log = sealed.OpenLog(tmp_path / "log.jsonl", signing_key="k1")
    first = log.append("C", "sealed", "orchestrator")
    assert first["mac"] == sealed.entry_mac(first["hash"], b"k1")
    assert len(first["mac"]) == 64
    sealed.OpenLog.verify_file(tmp_path / "log.jsonl", signing_key="k1")


def test_a_forger_who_recomputes_the_chain_fails_the_mac(tmp_path):
    path = tmp_path / "log.jsonl"
    log = sealed.OpenLog(path, signing_key="k1")
    log.append("C", "sealed", "orchestrator")
    log.append("C", "open", "confirmation")
    lines = path.read_text().splitlines()
    record = json.loads(lines[1])
    record["actor"] = "someone else"
    lines[1] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n")
    rechain(path)
    assert len(sealed.OpenLog(path).entries) == 2  # a checksum is satisfied
    with pytest.raises(errors.ArtifactError, match="MAC"):
        sealed.OpenLog.verify_file(path, signing_key="k1")


def test_the_wrong_key_and_stripped_macs_are_refused(tmp_path):
    path = tmp_path / "log.jsonl"
    sealed.OpenLog(path, signing_key="k1").append("C", "sealed", "o")
    with pytest.raises(errors.ArtifactError, match="MAC"):
        sealed.OpenLog.verify_file(path, signing_key="k2")
    record = json.loads(path.read_text())
    del record["mac"]
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(errors.ArtifactError, match="MAC"):
        sealed.OpenLog.verify_file(path, signing_key="k1")


def test_the_environment_key_signs_when_no_key_is_given(tmp_path, monkeypatch):
    monkeypatch.setenv("SCMLE_SIGNING_KEY", "from-env")
    entry = sealed.OpenLog(tmp_path / "log.jsonl").append("C", "sealed", "o")
    assert entry["mac"] == sealed.entry_mac(entry["hash"], b"from-env")
    monkeypatch.delenv("SCMLE_SIGNING_KEY")
    assert "mac" not in sealed.OpenLog().append("C", "sealed", "o")


def test_head_is_the_newest_hash_and_genesis_when_empty():
    log = sealed.OpenLog()
    assert log.head == sealed.GENESIS
    entry = log.append("C", "sealed", "o")
    assert log.head == entry["hash"]
