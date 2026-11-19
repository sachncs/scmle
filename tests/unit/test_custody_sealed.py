"""The hash-chained open log and named sealed sets."""

import json

import pandas
import pytest

from scmle.assurance.custody import sealed
from scmle.contracts import errors
from scmle.data.splitting.splitters import lockbox


def resident(values=(1.0, 2.0, 3.0)):
    return lockbox.Resident(
        pandas.DataFrame({"a": list(values)}),
        pandas.Series([0, 1, 0]),
        (10, 11, 12),
    )


def test_log_chains_entries_and_verifies():
    # Enforces docs/ARTIFACTS.md:38.
    # Enforces website/src/pages/docs/artifacts.astro:23.
    # Enforces website/src/pages/docs/threat-model.astro:24.
    log = sealed.Journal()
    first = log.append("C", "sealed", "orchestrator")
    second = log.append("C", "open", "confirmation")
    assert first["prev"] == sealed.GENESIS
    assert second["prev"] == first["hash"]
    log.verify()


def test_log_detects_a_changed_entry(tmp_path):
    path = tmp_path / "custody_log.jsonl"
    log = sealed.Journal(path)
    log.append("C", "sealed", "orchestrator")
    log.append("C", "open", "confirmation")
    lines = path.read_text().splitlines()
    record = json.loads(lines[0])
    record["actor"] = "someone else"
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(errors.ArtifactError, match="chain"):
        sealed.Journal(path)


def test_log_detects_a_removed_entry(tmp_path):
    path = tmp_path / "custody_log.jsonl"
    log = sealed.Journal(path)
    for event in ("sealed", "token_issued", "open"):
        log.append("C", event, "orchestrator")
    lines = path.read_text().splitlines()
    del lines[1]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(errors.ArtifactError, match="chain"):
        sealed.Journal(path)


def test_log_reloads_and_continues_the_chain(tmp_path):
    path = tmp_path / "custody_log.jsonl"
    sealed.Journal(path).append("C", "sealed", "orchestrator")
    again = sealed.Journal(path)
    entry = again.append("C", "open", "confirmation")
    assert entry["seq"] == 1
    sealed.Journal(path)


def test_a_set_opens_once_and_every_attempt_is_logged():
    log = sealed.Journal()
    held = sealed.Locker("C", resident(), log)
    held.commit("c", "l")
    token = held.mint()
    rows = held.open(token, "confirmation", claims_hash="c", ledger_hash="l")
    assert list(rows[0]["a"]) == [1.0, 2.0, 3.0]
    with pytest.raises(errors.LeakageError, match="already used"):
        held.open(token, "exploration", claims_hash="c", ledger_hash="l")
    events = [(e["event"], e["actor"]) for e in log.entries]
    assert ("open", "confirmation") in events
    assert ("denied", "exploration") in events
    log.verify()


def test_a_wrong_token_is_denied_and_logged_without_opening():
    # Enforces docs/ARTIFACTS.md:38.
    # Enforces website/src/pages/docs/artifacts.astro:23.
    log = sealed.Journal()
    held = sealed.Locker("C", resident(), log)
    with pytest.raises(errors.LeakageError, match="invalid token"):
        held.open("guess", actor="exploration")
    assert held.opened is False
    assert [e["event"] for e in log.entries][-1] == "denied"
    held.commit("c", "l")
    held.open(held.mint(), "confirmation", claims_hash="c", ledger_hash="l")


def test_the_log_never_contains_row_values():
    # Enforces docs/ARTIFACTS.md:38.
    log = sealed.Journal()
    held = sealed.Locker("C", resident((123456.789, 2.0, 3.0)), log)
    held.commit("c", "l")
    held.open(held.mint(), "confirmation", claims_hash="c", ledger_hash="l")
    assert "123456.789" not in json.dumps(log.entries)


def test_the_vault_gives_each_set_its_own_token():
    log = sealed.Journal()
    vault = sealed.Vault(log)
    first = vault.adopt("C", resident())
    second = vault.adopt("final_test", resident())
    assert first.mint() != second.mint()
    with pytest.raises(errors.LeakageError):
        first.open(second.mint(), actor="x")
    vault.close()


def test_the_vault_refuses_a_duplicate_set_name():
    vault = sealed.Vault(sealed.Journal())
    vault.adopt("C", resident())
    with pytest.raises(ValueError, match="already"):
        vault.adopt("C", resident())
    vault.close()


def test_exploration_view_holds_no_reference_to_a_sealed_set():
    view = sealed.View(
        features=pandas.DataFrame({"a": [1.0]}),
        target=pandas.Series([0]),
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
    log = sealed.Journal()
    held = sealed.Locker("C", process_set, log)
    held.commit("claims-1", "ledger-1")
    return held, log


def test_a_committed_set_opens_only_for_its_claims_and_ledger():
    held, log = committed(resident())
    token = held.mint()
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
    held, ignored = committed(resident())
    with pytest.raises(errors.LeakageError, match="already bound"):
        held.commit("claims-2", "ledger-1")
    other = sealed.Locker("F", resident(), sealed.Journal())
    other.commit("claims-0", "ledger-0")
    other.open(other.mint(), claims_hash="claims-0", ledger_hash="ledger-0")
    with pytest.raises(errors.LeakageError, match="already bound or used"):
        other.commit("claims-1", "ledger-1")


def test_the_binding_digest_depends_on_both_hashes():
    base = sealed.binding_digest("a", "b")
    assert base != sealed.binding_digest("a", "c")
    assert base != sealed.binding_digest("c", "b")
    assert base != sealed.binding_digest("b", "a")
    assert len(base) == 64


def test_an_unbound_set_is_refused():
    # GATE-3: a set that was never committed used to open with the token
    # alone; every release is now bound to what it is released for.
    held = sealed.Locker("C", resident(), sealed.Journal())
    assert held.committed is False
    with pytest.raises(errors.LeakageError, match="bind it"):
        held.open(held.mint())
    assert held.opened is False


def rechain(path, key=None):
    """What a forger without the log key can do: recompute every hash."""
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    previous = sealed.GENESIS
    out = []
    for index, record in enumerate(lines):
        body = {k: v for k, v in record.items() if k not in ("hash", "mac")}
        body["seq"], body["prev"] = index, previous
        rewritten = {**body, "hash": sealed.hashed(body)}
        previous = rewritten["hash"]
        out.append(json.dumps(rewritten))
    path.write_text("\n".join(out) + "\n")


def test_a_keyed_log_carries_a_mac_on_every_entry(tmp_path):
    # Enforces website/src/pages/docs/qualification.astro:80.
    log = sealed.Journal(tmp_path / "log.jsonl", signing_key="k1")
    first = log.append("C", "sealed", "orchestrator")
    assert first["mac"] == sealed.authenticate(first["hash"], b"k1")
    assert len(first["mac"]) == 64
    sealed.Journal(tmp_path / "log.jsonl", signing_key="k1")


def test_a_forger_who_recomputes_the_chain_fails_the_mac(tmp_path):
    path = tmp_path / "log.jsonl"
    log = sealed.Journal(path, signing_key="k1")
    log.append("C", "sealed", "orchestrator")
    log.append("C", "open", "confirmation")
    lines = path.read_text().splitlines()
    record = json.loads(lines[1])
    record["actor"] = "someone else"
    lines[1] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n")
    rechain(path)
    assert len(sealed.Journal(path).entries) == 2  # a checksum is satisfied
    with pytest.raises(errors.ArtifactError, match="MAC"):
        sealed.Journal(path, signing_key="k1")


def test_the_wrong_key_and_stripped_macs_are_refused(tmp_path):
    path = tmp_path / "log.jsonl"
    sealed.Journal(path, signing_key="k1").append("C", "sealed", "o")
    with pytest.raises(errors.ArtifactError, match="MAC"):
        sealed.Journal(path, signing_key="k2")
    record = json.loads(path.read_text())
    del record["mac"]
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(errors.ArtifactError, match="MAC"):
        sealed.Journal(path, signing_key="k1")


def test_the_environment_key_signs_when_no_key_is_given(tmp_path, monkeypatch):
    monkeypatch.setenv("SCMLE_SIGNING_KEY", "from-env")
    entry = sealed.Journal(tmp_path / "log.jsonl").append("C", "sealed", "o")
    assert entry["mac"] == sealed.authenticate(entry["hash"], b"from-env")
    monkeypatch.delenv("SCMLE_SIGNING_KEY")
    assert "mac" not in sealed.Journal().append("C", "sealed", "o")


def test_head_is_the_newest_hash_and_genesis_when_empty():
    log = sealed.Journal()
    assert log.head == sealed.GENESIS
    entry = log.append("C", "sealed", "o")
    assert log.head == entry["hash"]


def test_an_unsigned_log_says_so(caplog):
    with caplog.at_level("WARNING", logger=sealed.__name__):
        sealed.Journal(signing_key=None)
    assert any("not signed" in r.getMessage() for r in caplog.records)


def test_a_signed_log_cannot_be_verified_without_its_key(tmp_path):
    path = tmp_path / "log.jsonl"
    signed = sealed.Journal(path, signing_key="k")
    signed.append("C", "sealed", "orchestrator")
    with pytest.raises(errors.ArtifactError, match="cannot be verified"):
        sealed.Journal(path, signing_key=None)
    assert len(sealed.Journal(path, signing_key="k").entries) == 1
