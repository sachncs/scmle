"""The single confirmation: order, gate, unlock-once and failure policy."""

from fractions import Fraction
import json

import numpy as np
import pandas as pd
import pytest

from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.data.custody import sealed
from scmle.data.ledger import bivariate
from scmle.data.ledger import budget as b
from scmle.data.ledger import confirm as cf
from scmle.data.ledger import controls
from scmle.data.ledger import explore
from scmle.data.ledger import ledger as ledger_module
from scmle.data.ledger import online_fdr
from scmle.data.splitting import splitters

F = Fraction
ALPHA = F(1, 20)
SIGNAL = ("s1", "s2", "s3")


def world(rows, seed, effect=0.6):
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {f"n{i}": rng.normal(size=rows) for i in range(9)}
        | {name: rng.normal(size=rows) for name in SIGNAL}
    )
    target = sum(effect * features[name] for name in SIGNAL) + rng.normal(
        size=rows
    )
    return features, pd.Series(target, name="y")


def passing_control(observed=6):
    return controls.summarize("permuted_target", "strata", observed, (0,) * 20)


def explored(tmp_path=None, effect=0.6, control=True, rule=online_fdr.ELOND):
    x, y = world(600, 1, effect)
    config = online_fdr.RuleConfig(rule, F(1, 10))
    path = None if tmp_path is None else tmp_path / "hypothesis_ledger.jsonl"
    ledger = ledger_module.Ledger.create(
        path,
        charter_hash="c" * 64,
        config=config,
        alpha_confirm=ALPHA,
        seed=3,
    )
    plan = explore.plan_exploration(x, explore.NUMERIC)
    explore.run_exploration(ledger, x, y, plan)
    if control:
        ledger.record_control(passing_control().to_dict())
    return ledger


def source_with(features, target, ids=None):
    log = sealed.OpenLog()
    locked = splitters.ResidentTestSet(
        features, target, tuple(ids or range(len(target)))
    )
    return sealed.SealedSet("C", locked, log)


def c_rows(effect=0.6, seed=2):
    return world(500, seed, effect)


def test_the_full_flow_confirms_true_effects_and_only_those():
    ledger = explored()
    source = source_with(*c_rows())
    result = cf.confirm(ledger, source, alpha_confirm=ALPHA)
    assert result.outcome == cf.CONFIRMED
    confirmed = {c.claim_id for c in result.confirmed}
    names = {ledger.hypotheses[i].spec.column: i for i in ledger.hypotheses}
    assert {names[s] for s in SIGNAL} == confirmed
    assert source.opened
    for check in result.checks:
        assert check.set_id is qualification.SetId.CONFIRMATION
    for claim in result.confirmed:
        assert claim.tier is qualification.Tier.CONFIRMED
        assert claim.adjusted_p_value is not None
        assert any("independent of rows of X" in a for a in claim.assumptions)
        assert claim.check_ids == (f"{claim.claim_id}:X", f"{claim.claim_id}:C")


def test_by_is_the_default_and_the_statistics_are_those_on_c():
    ledger = explored()
    features, target = c_rows()
    result = cf.confirm(
        ledger, source_with(features, target), alpha_confirm=ALPHA
    )
    assert result.rule == "by"
    expected = {}
    for check in result.checks:
        hid = check.check_id.split(":")[0]
        column = ledger.hypotheses[hid].spec.column
        expected[hid] = bivariate.spearman(
            features[column].to_numpy(), target.to_numpy()
        ).p
        assert check.p_value == expected[hid]
    survivors = [
        ledger.hypotheses[i] for i in ledger.survivor_set.hypothesis_ids
    ]
    assert len(survivors) == len(result.claims)


def test_selection_is_frozen_and_logged_before_c_opens():
    ledger = explored()
    source = source_with(*c_rows())
    result = cf.confirm(ledger, source, alpha_confirm=ALPHA)
    events = [e["event"] for e in source.log.entries]
    assert events == [
        "sealed",
        "selection_frozen",
        "token_issued",
        "open",
    ]
    frozen = source.log.entries[1]["detail"]
    assert frozen["survivors_sha256"] == ledger.survivor_set.sha256
    survivors_row = next(r for r in ledger.rows if r["kind"] == "survivors")
    assert frozen["ledger_head"] == survivors_row["hash"]
    assert result.custody_head == source.log.entries[-1]["hash"]
    assert result.ledger_head == ledger.head
    source.log.verify()
    ledger.verify()


def test_a_second_confirmation_raises_and_is_logged():
    ledger = explored()
    source = source_with(*c_rows())
    cf.confirm(ledger, source, alpha_confirm=ALPHA)
    with pytest.raises(errors.LeakageError, match="already used"):
        cf.confirm(ledger, source, alpha_confirm=ALPHA)
    last = source.log.entries[-1]
    assert (last["event"], last["actor"]) == ("denied", cf.ACTOR)
    source.log.verify()


def test_c_itself_refuses_a_second_open_after_the_stage():
    ledger = explored()
    source = source_with(*c_rows())
    cf.confirm(ledger, source, alpha_confirm=ALPHA)
    with pytest.raises(errors.LeakageError):
        source.open(source.issue_token(), actor="exploration")


def test_unrejected_survivors_stay_unverified_with_their_adjusted_p():
    ledger = explored(effect=0.6)
    # C has no effect for any feature: nothing should confirm.
    features, target = c_rows(effect=0.0)
    result = cf.confirm(
        ledger, source_with(features, target), alpha_confirm=ALPHA
    )
    assert result.outcome == cf.CONFIRMED
    assert result.confirmed == ()
    assert all(c.tier is qualification.Tier.UNVERIFIED for c in result.claims)
    assert all(c.adjusted_p_value is not None for c in result.claims)


def test_no_survivors_means_c_is_never_opened():
    x, y = world(300, 1, effect=0.0)
    ledger = ledger_module.Ledger.create(
        None,
        charter_hash="c" * 64,
        config=online_fdr.RuleConfig(online_fdr.ELOND, F(1, 100)),
        alpha_confirm=ALPHA,
        seed=0,
    )
    explore.run_exploration(
        ledger, x, y, explore.plan_exploration(x, explore.NUMERIC)
    )
    source = source_with(*c_rows())
    result = cf.confirm(
        ledger, source, alpha_confirm=ALPHA, require_control=False
    )
    assert result.outcome == cf.NO_SURVIVORS
    assert not source.opened
    assert [e["event"] for e in source.log.entries] == ["sealed"]
    assert result.claims == ()


def test_a_failed_gate_promotes_nothing_and_leaves_c_closed():
    ledger = explored(control=False)
    ledger.record_control(passing_control(observed=0).to_dict())
    source = source_with(*c_rows())
    result = cf.confirm(ledger, source, alpha_confirm=ALPHA)
    assert result.outcome == cf.FLOOR_NOT_EXCEEDED
    assert "do not clearly exceed" in result.reason
    assert not source.opened
    assert result.claims and all(
        c.tier is qualification.Tier.UNVERIFIED for c in result.claims
    )


def test_a_missing_control_blocks_promotion_unless_waived():
    blocked = explored(control=False)
    source = source_with(*c_rows())
    result = cf.confirm(blocked, source, alpha_confirm=ALPHA)
    assert result.outcome == cf.NO_CONTROL and not source.opened
    waived = explored(control=False)
    source = source_with(*c_rows())
    result = cf.confirm(
        waived, source, alpha_confirm=ALPHA, require_control=False
    )
    assert result.outcome == cf.CONFIRMED and source.opened


def test_without_a_confirmation_set_claims_are_unverified():
    ledger = explored()
    result = cf.confirm(ledger, None, alpha_confirm=ALPHA)
    assert result.outcome == cf.NO_CONFIRMATION_SET
    assert result.confirmed == ()
    assert all("no confirmation set" in c.assumptions[0] for c in result.claims)


def test_a_crash_after_the_open_follows_the_unconfirmed_policy():
    ledger = explored()
    source = source_with(*c_rows())

    def broken(target):
        raise RuntimeError("encoder exploded")

    result = cf.confirm(ledger, source, alpha_confirm=ALPHA, encode=broken)
    assert result.outcome == cf.PIPELINE_FAILURE
    assert "RuntimeError" in result.reason and "unconfirmed" in result.reason
    assert source.opened  # C is spent
    assert all(c.tier is qualification.Tier.UNVERIFIED for c in result.claims)
    assert ledger.confirmation["outcome"] == cf.PIPELINE_FAILURE
    with pytest.raises(errors.LeakageError):
        cf.confirm(ledger, source, alpha_confirm=ALPHA)


def test_a_crash_while_opening_is_a_failure_not_an_exception():
    ledger = explored()

    class Broken:
        log = sealed.OpenLog()

        def issue_token(self):
            return "t"

        def open(self, token, actor="x"):
            raise OSError("vault died")

    result = cf.confirm(ledger, Broken(), alpha_confirm=ALPHA)
    assert result.outcome == cf.PIPELINE_FAILURE
    assert "OSError" in result.reason


def test_thresholdout_is_refused_before_anything_happens():
    ledger = explored()
    source = source_with(*c_rows())
    with pytest.raises(errors.ConfigError, match="not implemented"):
        cf.confirm(ledger, source, alpha_confirm=ALPHA, policy=cf.THRESHOLDOUT)
    with pytest.raises(errors.ConfigError, match="policy"):
        cf.confirm(ledger, source, alpha_confirm=ALPHA, policy="retry")
    assert not ledger.frozen and not source.opened


def test_rules_change_the_family_adjustment():
    results = {}
    for rule in (
        b.MultiplicityRule.BH,
        b.MultiplicityRule.BY,
        b.MultiplicityRule.HOLM,
    ):
        ledger = explored()
        results[rule] = cf.confirm(
            ledger, source_with(*c_rows()), alpha_confirm=ALPHA, rule=rule
        )
        assert results[rule].rule == rule.value
    by = {
        c.claim_id: c.adjusted_p_value
        for c in results[b.MultiplicityRule.BY].claims
    }
    bh = {
        c.claim_id: c.adjusted_p_value
        for c in results[b.MultiplicityRule.BH].claims
    }
    assert all(by[i] >= bh[i] for i in by)


def test_claims_file_round_trips_and_carries_the_hashes(tmp_path):
    ledger = explored(tmp_path)
    source = source_with(*c_rows())
    result = cf.confirm(ledger, source, alpha_confirm=ALPHA)
    path = tmp_path / cf.CLAIMS_FILE
    result.write(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["outcome"] == cf.CONFIRMED
    assert record["alpha_confirm"] == "1/20"
    assert record["survivors_sha256"] == ledger.survivor_set.sha256
    assert record["ledger_head"] == ledger.head
    assert record["negative_control"]["exceeds"] is True
    assert record["guarantee"].startswith("FDR <=")
    assert [qualification.Claim.from_dict(c) for c in record["claims"]] == list(
        result.claims
    )
    assert [qualification.Check.from_dict(c) for c in record["checks"]] == list(
        result.checks
    )
    # Tail truncation is caught against the head recorded in the claims file.
    rows = (tmp_path / ledger_module.FILE_NAME).read_text().splitlines()
    (tmp_path / ledger_module.FILE_NAME).write_text("\n".join(rows[:-1]) + "\n")
    with pytest.raises(ledger_module.LedgerError, match="head"):
        ledger_module.Ledger.load(
            tmp_path / ledger_module.FILE_NAME,
            expected_head=record["ledger_head"],
        )


def test_confirmation_with_class_labels_uses_the_exploration_coding():
    y = world(400, 1)[1]
    labels = pd.Series(np.where(y > y.median(), "yes", "no"))
    codes = explore.encode_target(labels, True)
    assert set(codes.unique()) == {0.0, 1.0}
    assert float(codes[labels == "yes"].iloc[0]) == 1.0  # sorted: no < yes
    flipped = explore.encode_target(labels, True, classes=("yes", "no"))
    assert float(flipped[labels == "yes"].iloc[0]) == 0.0
    with pytest.raises(ValueError, match="numeric"):
        explore.encode_target(pd.Series(["a", "b"]), False)
    assert cf.default_encoding(labels).tolist() == codes.tolist()
