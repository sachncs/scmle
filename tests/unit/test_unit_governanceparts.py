"""Unit tests of the envelope guard and the data-access ledger, by full name."""

import datetime
import hashlib
import types

import pytest

from scmle.contracts import experiments
from scmle.governance import deployment_ledger
from scmle.governance import envelope
from scmle.governance import ledger

KEY = "person-key"
TODAY = datetime.date(2026, 11, 10)


def book():
    return deployment_ledger.DeploymentLedger.create(
        None,
        charter_hash="c" * 64,
        policy={},
        alpha_promo="1/20",
        system_keys=[deployment_ledger.thumbprint("system")],
    )


def signed(key=KEY, **changes):
    fields = {
        "identifier": "e1",
        "limits": {"train_challenger": 1},
        "statement": "retrain once",
        "role": "owner",
        "attestor": "A. Person",
        "date": "2026-11-01",
        "scope": "v1:s",
        "expires": "2026-11-30",
    }
    fields.update(changes)
    return deployment_ledger.Envelope.create(key, **fields)


def guard(keys=None, today=TODAY):
    return envelope.Guard(book(), today=lambda: today, verification_keys=keys)


# ---- envelope --------------------------------------------------------------


def test_the_only_action_is_training_a_challenger():
    assert [a.value for a in envelope.Action] == ["train_challenger"]


def test_a_refusal_names_the_action_and_reason():
    error = envelope.RefusedError("x", "because")
    assert str(error) == "no envelope allows x: because"
    assert (error.action, error.reason) == ("x", "because")


def test_a_grant_is_spent_once_and_recorded_first():
    keeper = guard()
    calls = []
    keeper.on_write = lambda: calls.append(1)
    keeper.grant(signed())
    assert keeper.remaining("e1") == {"train_challenger": 1}
    used = keeper.authorize(envelope.Action.TRAIN_CHALLENGER, "v1:s", {"b": 7})
    assert used == "e1"
    row = keeper.ledger.rows[-1]
    assert row["kind"] == deployment_ledger.ENVELOPE_USED
    assert row["body"] == {
        "envelope_id": "e1",
        "action": "train_challenger",
        "scope": "v1:s",
        "date": "2026-11-10",
        "detail": {"b": 7},
    }
    assert keeper.remaining("e1") == {"train_challenger": 0}
    assert calls == [1, 1]
    with pytest.raises(envelope.RefusedError, match="e1: its uses are spent"):
        keeper.authorize(envelope.Action.TRAIN_CHALLENGER, "v1:s")


def test_without_grants_nothing_is_allowed():
    with pytest.raises(envelope.RefusedError, match="no envelope was granted"):
        guard().authorize(envelope.Action.TRAIN_CHALLENGER, "v1:s")


def test_refusal_reasons_for_scope_dates_and_revocation():
    keeper = guard()
    keeper.grant(signed())
    grant = keeper.ledger.envelopes["e1"]
    assert keeper.refusal("e1", grant, "train_challenger", "v2:s") == (
        "a different scope"
    )
    assert keeper.refusal("e1", grant, "promote", "v1:s") == (
        "does not cover promote"
    )
    assert keeper.refusal("e1", grant, "train_challenger", "v1:s") is None
    keeper.revoke("e1", "retired")
    assert keeper.refusal("e1", grant, "train_challenger", "v1:s") == "revoked"
    assert keeper.remaining("e1") == {}
    assert keeper.remaining("nope") == {}
    assert keeper.ledger.rows[-1]["body"] == {
        "envelope_id": "e1",
        "reason": "retired",
    }


@pytest.mark.parametrize(
    ("day", "valid"),
    [
        (datetime.date(2026, 10, 31), False),
        (datetime.date(2026, 11, 1), True),
        (datetime.date(2026, 11, 30), True),
        (datetime.date(2026, 12, 1), False),
    ],
)
def test_an_envelope_is_valid_only_between_its_dates(day, valid):
    keeper = guard(today=day)
    keeper.grant(signed())
    reason = keeper.invalidity(keeper.ledger.envelopes["e1"])
    assert reason == (None if valid else "not valid today")


def test_verification_keys_check_the_grant_and_every_use():
    keeper = guard(keys={deployment_ledger.thumbprint(KEY): KEY})
    assert keeper.verifies(signed()) is True
    assert keeper.verifies(signed(key="stranger")) is False
    with pytest.raises(envelope.RefusedError, match="does not verify"):
        keeper.grant(signed(key="stranger"))
    keeper.grant(signed())
    keeper.verification_keys = {}
    assert keeper.invalidity(keeper.ledger.envelopes["e1"]) == (
        "its signature does not verify"
    )
    assert guard(keys=None).verifies(signed(key="stranger")) is True


def test_a_never_list_action_is_refused_before_any_envelope(monkeypatch):
    keeper = guard()
    keeper.grant(signed())
    monkeypatch.setattr(
        envelope.never_list, "Forbidden", [envelope.Action.TRAIN_CHALLENGER]
    )
    with pytest.raises(envelope.RefusedError, match="on the never list"):
        keeper.authorize(envelope.Action.TRAIN_CHALLENGER, "v1:s")
    assert keeper.ledger.rows[-1]["kind"] == deployment_ledger.ENVELOPE


def test_the_first_allowing_envelope_is_spent_and_others_are_listed():
    keeper = guard()
    keeper.grant(signed(identifier="a", scope="other"))
    keeper.grant(signed(identifier="b"))
    assert keeper.authorize(envelope.Action.TRAIN_CHALLENGER, "v1:s") == "b"
    with pytest.raises(
        envelope.RefusedError,
        match="a: a different scope; b: its uses are spent",
    ):
        keeper.authorize(envelope.Action.TRAIN_CHALLENGER, "v1:s")


# ---- ledger ----------------------------------------------------------------


def test_distill_counts_and_hashes_sorted_ids():
    expected = hashlib.sha256(b"1\x1f2\x1f3").hexdigest()[:16]
    assert ledger.distill([3, 1, 2]) == {"rows": 3, "digest": expected}
    assert ledger.distill(["2", 1, 3]) == ledger.distill([1, 2, 3])
    assert ledger.distill([])["rows"] == 0


def manifest(
    folds=(((0, 1), (2,)), ((1, 2), (0,))), train=(10, 11, 12), test=(13,)
):
    return types.SimpleNamespace(
        folds=[(list(a), list(b)) for a, b in folds],
        training=list(train),
        test=list(test),
        fingerprint="fp",
        strategy="random",
    )


def trial(seen, status=experiments.Termination.OK):
    return types.SimpleNamespace(observed_access=seen, status=status)


def faithful(split):
    return [
        {"fit": ledger.distill(list(a)), "evaluate": ledger.distill(list(b))}
        for a, b in split.manifest.folds
    ]


def test_observed_access_matching_the_manifest_is_clean():
    split = types.SimpleNamespace(manifest=manifest())
    out = ledger.observed_access(split, [trial(faithful(split))])
    assert out == {
        "folds_checked": 2,
        "folds_mismatched": 0,
        "test_rows_seen": False,
    }


def test_observed_access_counts_mismatches_and_extra_folds():
    split = types.SimpleNamespace(manifest=manifest())
    seen = faithful(split)
    wrong = [seen[1], seen[0], seen[0]]
    out = ledger.observed_access(split, [trial(wrong)])
    assert out == {
        "folds_checked": 3,
        "folds_mismatched": 3,
        "test_rows_seen": True,
    }


def test_overlapping_partitions_count_as_test_rows_seen():
    split = types.SimpleNamespace(manifest=manifest(test=(12,)))
    out = ledger.observed_access(split, [])
    assert out["test_rows_seen"] is True
    assert out["folds_checked"] == 0


def test_itemize_lists_four_operations():
    out = ledger.itemize(manifest(), 5, True, 1)
    assert [o["operation"] for o in out] == [
        "cross_validation",
        "final_refit",
        "threshold_optimization",
        "final_evaluation",
    ]
    assert out[0]["candidates"] == 5
    assert out[0]["fits_per_candidate"] == 2
    assert out[0]["fold_structure"] == [
        {"fit_rows": 2, "validation_rows": 1},
        {"fit_rows": 2, "validation_rows": 1},
    ]
    assert out[1]["rows"] == 3
    assert out[2]["used"] is True
    assert out[3] == {
        "operation": "final_evaluation",
        "rows": 1,
        "partition": "locked test",
        "opened": 1,
    }


def test_compose_verifies_a_clean_run():
    split = types.SimpleNamespace(manifest=manifest())
    failed = trial(
        [{"fit": {}, "evaluate": {}}], experiments.Termination.FAILED
    )
    out = ledger.compose(split, [trial(faithful(split)), failed], 1, False)
    assert out["partitions"] == {
        "development_rows": 3,
        "locked_test_rows": 1,
        "split_fingerprint": "fp",
        "strategy": "random",
    }
    assert out["operations"][0]["candidates"] == 1
    assert all(out["verified"].values())


def test_compose_flags_each_broken_guarantee():
    split = types.SimpleNamespace(
        manifest=manifest(folds=(((0, 5), (1,)),), test=(12,))
    )
    out = ledger.compose(split, [], 2, True)
    verified = out["verified"]
    assert verified["observed_fit_and_scoring_rows_match_the_manifest"] is False
    assert verified["no_observed_access_to_test_rows"] is False
    assert verified["development_and_test_are_disjoint"] is False
    assert verified["development_and_test_cover_all_rows"] is False
    assert verified["every_fold_inside_development"] is False
    assert verified["test_set_opened_exactly_once"] is False
