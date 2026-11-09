"""Tests added to kill surviving mutants in the sealed, vault and merkle code.

See `tests/unit/MUTANTS_custody.md` for the run and the survivors that are
left on purpose. Messages are compared whole (not with a substring) because
a mutant that wraps a message in extra characters still contains it.
"""

from functools import partial
import hashlib
import hmac
import json

import pandas
import pytest

from scmle.contracts import errors
from scmle.data.custody import merkle
from scmle.data.custody import sealed
from scmle.data.splitting import splitters
from scmle.governance import vault


def resident(values=(1.0, 2.0, 3.0)):
    return splitters.ResidentTestSet(
        pandas.DataFrame({"a": list(values)}),
        pandas.Series([0, 1, 0], name="y"),
        (10, 11, 12),
    )


def refusal(call):
    with pytest.raises(errors.LeakageError) as caught:
        call()
    return str(caught.value)


def test_every_lifecycle_event_is_logged_with_its_exact_fields():
    log = sealed.OpenLog()
    held = sealed.SealedSet("C", resident(), log)
    held.commit("claims-1", "ledger-1")
    token = held.issue_token()
    refusal(lambda: held.open("guess", "attacker"))
    held.open(token, "confirmation", "claims-1", "ledger-1")
    rows = [
        (e["set"], e["event"], e["actor"], e["detail"]) for e in log.entries
    ]
    assert rows == [
        ("C", "sealed", "orchestrator", {"rows": 3}),
        (
            "C",
            "bound",
            "orchestrator",
            {"claims_hash": "claims-1", "ledger_hash": "ledger-1"},
        ),
        ("C", "token_issued", "orchestrator", {}),
        (
            "C",
            "denied",
            "attacker",
            {"reason": "test set is locked: invalid token"},
        ),
        ("C", "open", "confirmation", {"rows": 3}),
    ]
    assert held.set_id == "C"


def test_a_denied_open_records_the_reason_of_the_refusal():
    log = sealed.OpenLog()
    held = sealed.SealedSet("F", resident(), log)
    held.commit("c", "l")
    token = held.issue_token()
    refusal(lambda: held.open(token, "x"))
    assert log.entries[-1]["detail"] == {
        "reason": "test set is locked: binding mismatch"
    }


def test_the_binding_digest_is_the_sha256_of_its_canonical_json():
    expected = hashlib.sha256(
        b'{"claims_hash":"a","ledger_hash":"b"}'
    ).hexdigest()
    assert sealed.binding_digest("a", "b") == expected


def test_the_mac_is_a_real_hmac_of_the_entry_hash(tmp_path):
    log = sealed.OpenLog(signing_key="secret")
    entry = log.append("C", "sealed", "o")
    expected = hmac.new(
        b"secret", entry["hash"].encode(), hashlib.sha256
    ).hexdigest()
    assert entry["mac"] == expected
    assert sealed.entry_mac(entry["hash"], b"secret") == expected


def test_entries_are_written_sorted_and_the_folder_is_created(tmp_path):
    path = tmp_path / "deep" / "er" / "log.jsonl"
    log = sealed.OpenLog(path)
    entry = log.append("C", "sealed", "o", {"b": 1, "a": 2})
    line = path.read_text(encoding="utf-8").rstrip("\n")
    assert line == json.dumps(entry, sort_keys=True)
    assert list(json.loads(line)) == sorted(entry)


def forged(path, change):
    """Rewrites the log with one field changed and every hash recomputed."""
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    previous = sealed.GENESIS
    out = []
    for index, record in enumerate(lines):
        body = {k: v for k, v in record.items() if k not in ("hash", "mac")}
        body["seq"], body["prev"] = index, previous
        body.update(change(index))
        record = {**body, "hash": sealed.entry_hash(body)}
        previous = record["hash"]
        out.append(json.dumps(record))
    path.write_text("\n".join(out) + "\n")


@pytest.mark.parametrize("field", ["seq", "prev"])
def test_the_chain_fails_on_either_a_wrong_sequence_or_a_wrong_link(
    tmp_path, field
):
    path = tmp_path / "log.jsonl"
    log = sealed.OpenLog(path)
    log.append("C", "sealed", "o")
    log.append("C", "open", "o")
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    previous = sealed.GENESIS
    out = []
    for index, record in enumerate(lines):
        body = {k: v for k, v in record.items() if k != "hash"}
        body["prev"] = previous
        if index == 1:
            body[field] = 7 if field == "seq" else "f" * 64
        record = {**body, "hash": sealed.entry_hash(body)}
        previous = record["hash"]
        out.append(json.dumps(record))
    path.write_text("\n".join(out) + "\n")
    with pytest.raises(errors.ArtifactError) as caught:
        sealed.OpenLog.verify_file(path)
    assert str(caught.value) == "custody log chain broken at entry 1"


def test_the_mac_error_names_the_entry(tmp_path):
    path = tmp_path / "log.jsonl"
    sealed.OpenLog(path, signing_key="k").append("C", "sealed", "o")
    with pytest.raises(errors.ArtifactError) as caught:
        sealed.OpenLog.verify_file(path, signing_key="other")
    assert str(caught.value) == "custody log MAC invalid at entry 0"


def test_the_vault_seals_in_process_only_when_asked_not_to_use_one():
    frame = pandas.DataFrame({"a": [1.0, 2.0, 3.0]})
    target = pandas.Series([0, 1, 0], name="y")
    protocol = sealed.Vault(sealed.OpenLog(), process=False)
    held = protocol.seal("C", frame, target, (4, 5, 6))
    assert isinstance(held.locked, splitters.ResidentTestSet)
    assert protocol.level == sealed.PROTOCOL_LEVEL
    features, labels = held.open(held.issue_token())
    assert features.equals(frame) and labels.equals(target)
    assert held.ids == (4, 5, 6)
    process = sealed.Vault(sealed.OpenLog(), process=True)
    try:
        real = process.seal("C", frame, target, (4, 5, 6))
        assert isinstance(real.locked, vault.SealedTestSet)
        assert process.level == sealed.VAULT_LEVEL
        features, labels = real.open(real.issue_token())
        assert features.equals(frame) and labels.equals(target)
    finally:
        process.close()


def test_adopting_a_set_names_it_in_the_log_and_refuses_a_repeat():
    log = sealed.OpenLog()
    custody = sealed.Vault(log, process=False)
    custody.adopt("C", resident())
    assert log.entries[0]["set"] == "C"
    with pytest.raises(ValueError) as caught:
        custody.adopt("C", resident())
    assert str(caught.value) == "set C is already sealed"


def test_a_partial_binding_is_not_enough_for_a_committed_set():
    held = sealed.SealedSet("C", resident(), sealed.OpenLog())
    held.commit("c", "l")
    token = held.issue_token()
    for claims, ledger in (("c", None), (None, "l")):
        text = refusal(partial(held.open, token, "x", claims, ledger))
        assert text == "test set is locked: binding mismatch"


def test_the_vault_token_has_32_hex_digits_and_close_ends_the_process():
    locked = vault.SealedTestSet(
        pandas.DataFrame({"a": [1.0]}), pandas.Series([1]), (0,)
    )
    assert len(locked.token) == 32 and int(locked.token, 16) >= 0
    assert locked.process.poll() is None
    locked.close()
    assert locked.process.poll() is not None


def test_vault_refusals_are_whole_messages():
    locked = vault.SealedTestSet(
        pandas.DataFrame({"a": [1.0]}), pandas.Series([1]), (0,)
    )
    try:
        assert refusal(lambda: locked.open("guess")) == (
            "test set is locked: invalid token"
        )
        locked.open(locked.issue_token())
        assert refusal(lambda: locked.open(locked.token)) == (
            "test set was already used for final evaluation"
        )
    finally:
        locked.close()


def test_a_vault_that_is_gone_says_so_in_full():
    locked = vault.SealedTestSet(
        pandas.DataFrame({"a": [1.0]}), pandas.Series([1]), (0,)
    )
    locked.process.kill()
    locked.process.wait()
    locked.channel.close()
    assert refusal(lambda: locked.open(locked.token)) == (
        "the test-set vault is gone"
    )


def test_merkle_pairs_are_adjacent_and_an_odd_leaf_is_carried_up():
    a, b, c, d, e = (merkle.row_hash(i, [("v", i)]) for i in range(5))
    assert merkle.merkle_root([a, b, c, d]) == merkle.combine(
        merkle.combine(a, b), merkle.combine(c, d)
    )
    assert merkle.merkle_root([a, b, c, d, e]) == merkle.combine(
        merkle.combine(merkle.combine(a, b), merkle.combine(c, d)), e
    )


def test_row_hashes_reject_a_misaligned_row_instead_of_truncating():
    frame = pandas.DataFrame({"a": [1.0, 2.0]})
    assert len(merkle.row_hashes(frame, [0, 1])) == 2
    assert merkle.row_hashes(frame, [1])[0] == merkle.row_hash(1, [("a", 2.0)])
