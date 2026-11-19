"""Unit tests of the confirmation submodules, each by its full name."""

from fractions import Fraction
import json

import numpy
import pandas
import pytest

from scmle.assurance.custody import sealed
from scmle.assurance.ledger import budget
from scmle.assurance.ledger import controls
from scmle.assurance.ledger import explore
from scmle.assurance.ledger import online_fdr
from scmle.assurance.ledger.confirm import evaluation
from scmle.assurance.ledger.confirm import opening
from scmle.assurance.ledger.confirm import records
import scmle.assurance.ledger.ledger
from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.data.splitting.splitters import lockbox

ALPHA = Fraction(1, 20)


def world(rows, seed, effect=0.8):
    rng = numpy.random.default_rng(seed)
    features = pandas.DataFrame(
        {"n0": rng.normal(size=rows), "n1": rng.normal(size=rows)}
        | {"s1": rng.normal(size=rows), "s2": rng.normal(size=rows)}
    )
    target = effect * (features["s1"] + features["s2"]) + rng.normal(size=rows)
    return features, pandas.Series(target, name="y")


def explored(effect=0.8, control=True, exceeds=True):
    x, y = world(400, 1, effect)
    ledger = scmle.assurance.ledger.ledger.Ledger.create(
        None,
        charter_hash="c" * 64,
        config=online_fdr.Settings(online_fdr.ELOND, Fraction(1, 10)),
        alpha_confirm=ALPHA,
        seed=3,
    )
    explore.explore(ledger, x, y, explore.hypothesize(x, explore.NUMERIC))
    if control:
        observed = 6 if exceeds else 0
        summary = controls.summarize(
            "permuted_target", "strata", observed, (0,) * 20
        )
        ledger.negatives(summary.to_dict())
    return ledger


def locker(features, target):
    log = sealed.Journal()
    locked = lockbox.Resident(features, target, tuple(range(len(target))))
    return sealed.Locker("C", locked, log)


def commitment(claim_id="baseline"):
    return records.Commitment(claim_id, "baseline", "beats the rule", "a" * 64)


def docket(claim_id="baseline", doc="baseline.json", fail=False):
    def run(features, target):
        if fail:
            raise RuntimeError("runner broke")
        return records.Output({claim_id: "pass"}, {doc: {"rows": len(target)}})

    return records.Docket((commitment(claim_id),), run)


# ---- records ---------------------------------------------------------------


def test_the_outcome_codes_have_their_values():
    assert records.POLICIES == ("unconfirmed", "thresholdout")
    assert records.ACTOR == "confirmation"
    assert records.CLAIMS_FILE == "confirmed_claims.json"
    assert (records.CONFIRMED, records.NO_SURVIVORS) == (
        "confirmed",
        "no_survivors",
    )
    assert records.FLOOR_NOT_EXCEEDED == "floor_not_exceeded"
    assert records.NO_CONFIRMATION_SET == "no_confirmation_set"
    assert records.PIPELINE_FAILURE == "pipeline_failure"
    assert records.NO_CONTROL == "no_negative_control"
    assert records.ALL_WITHDRAWN == "all_withdrawn"
    assert (records.NOT_EVALUATED, records.FAILED) == (
        "not_evaluated",
        "failed",
    )


def test_a_withdrawal_with_reasons_must_be_accounted():
    with pytest.raises(errors.InputError, match="must be accounted"):
        records.Withdrawal({"h": "why"}, {}, accounted=False)
    assert records.Withdrawal({}, {}, accounted=False).accounted is False


def test_merge_returns_none_or_the_single_runner():
    assert records.merge() is None
    assert records.merge(None, None) is None
    one = docket()
    assert records.merge(None, one) is one


def test_merge_refuses_repeated_claim_ids():
    with pytest.raises(errors.InputError, match="claim ids must be unique"):
        records.merge(docket("a"), docket("a", "other.json"))


def test_merged_runners_share_one_frame_and_isolate_a_crash():
    merged = records.merge(
        docket("a", "a.json"), docket("b", fail=True), docket("c", "c.json")
    )
    assert [c.identifier for c in merged.registered] == ["a", "b", "c"]
    out = merged.run(pandas.DataFrame(), pandas.Series([1, 2]))
    assert out.statuses == {"a": "pass", "b": "failed", "c": "pass"}
    assert out.documents == {"a.json": {"rows": 2}, "c.json": {"rows": 2}}


def test_merged_runners_refuse_a_shared_document_name():
    merged = records.merge(docket("a", "x.json"), docket("b", "x.json"))
    with pytest.raises(
        errors.InputError, match=r"two claims wrote \['x.json'\]"
    ):
        merged.run(pandas.DataFrame(), pandas.Series([1]))


def make_claim(identifier, tier):
    return qualification.Claim(
        identifier=identifier,
        statement="s",
        checks=(f"{identifier}:X",),
        adjusted_p_value=None,
        effect=0.1,
        tier=tier,
        assumptions=("a",),
    )


def test_a_confirmation_lists_confirmed_claims_and_serializes(tmp_path):
    result = records.Confirmation(
        outcome="confirmed",
        reason="r",
        claims=(
            make_claim("h1", qualification.Tier.UNVERIFIED),
            qualification.Claim(
                identifier="h2",
                statement="s",
                checks=("h2:X", "h2:C"),
                adjusted_p_value=0.01,
                effect=0.2,
                tier=qualification.Tier.CONFIRMED,
                assumptions=("a",),
            ),
        ),
        checks=(),
        rule="by",
        confirm=Fraction(1, 20),
        survivors_sha256="s" * 64,
        ledger_head="l" * 64,
        head=None,
        control=None,
        documents={"d.json": {"x": 1}},
    )
    assert [c.identifier for c in result.confirmed] == ["h2"]
    record = result.to_dict()
    assert record["schema_version"] == 2
    assert record["alpha"] == "1/20"
    assert record["digests"] == {"survivors": "s" * 64}
    assert record["heads"] == {"ledger": "l" * 64, "custody": None}
    assert record["negative_control"] is None
    assert "second" not in record
    assert "registered_claims" not in record
    assert record["attachments"] == {
        "d.json": scmle.assurance.ledger.ledger.digest({"x": 1})
    }
    assert record["note"].startswith("Only claims with tier 'confirmed'")
    path = tmp_path / "deep" / "c.json"
    result.write(path)
    assert json.loads(path.read_text()) == json.loads(json.dumps(record))


# ---- evaluation ------------------------------------------------------------


def test_assumptions_list_the_rule_and_a_withdrawal():
    lines = evaluation.assumptions(budget.Correction.BY, 3)
    assert len(lines) == 5
    assert lines[0].endswith(
        "(rule by, family of 3 survivors fixed before C was opened)"
    )
    more = evaluation.assumptions(budget.Correction.HOLM, 2, withdrawn=1)
    assert len(more) == 6
    assert more[1].startswith("1 survivor(s) were withdrawn")


def test_default_encoding_codes_labels_in_text_order():
    assert evaluation.encode(pandas.Series([3, 1])).tolist() == [
        3.0,
        1.0,
    ]
    codes = evaluation.encode(pandas.Series(["b", None, "a", "b"]))
    assert codes[[0, 2, 3]].tolist() == [1.0, 0.0, 1.0]
    assert numpy.isnan(codes[1])


def test_disclaim_and_verify_need_frozen_survivors():
    ledger = explored()
    with pytest.raises(
        scmle.assurance.ledger.ledger.LedgerError, match="freeze"
    ):
        evaluation.disclaim(ledger, "x")
    with pytest.raises(
        scmle.assurance.ledger.ledger.LedgerError, match="freeze"
    ):
        evaluation.verify(
            ledger,
            world(10, 0),
            budget.Correction.BY,
            ALPHA,
            evaluation.encode,
        )


def test_disclaim_gives_one_unverified_claim_per_survivor():
    ledger = explored()
    survivors = ledger.freeze().hypotheses
    claims = evaluation.disclaim(ledger, "because")
    assert [c.identifier for c in claims] == list(survivors)
    assert {c.tier for c in claims} == {qualification.Tier.UNVERIFIED}
    assert claims[0].assumptions[0] == "not confirmed on C: because"
    spec = ledger.hypotheses[survivors[0]].spec
    assert claims[0].statement == f"{spec.name}: evidence against '{spec.null}'"
    assert evaluation.statement(spec) == claims[0].statement


def test_block_applies_the_gates_in_order():
    assert evaluation.block(explored(), require_control=True) is None
    missing = evaluation.block(explored(control=False), require_control=True)
    assert missing[0] == records.NO_CONTROL
    assert (
        evaluation.block(explored(control=False), require_control=False) is None
    )
    floor = evaluation.block(explored(exceeds=False), require_control=True)
    assert floor[0] == records.FLOOR_NOT_EXCEEDED
    assert floor[1].endswith("; no claim is promoted")
    empty = evaluation.block(explored(effect=0.0), require_control=True)
    if empty is not None and empty[0] == records.NO_SURVIVORS:
        assert empty[2] == ()


def test_stage_two_record_accounts_the_remaining_family():
    ledger = explored()
    survivors = ledger.freeze().hypotheses
    first = survivors[0]
    record = evaluation.account(
        ledger, records.Withdrawal({first: "unstable"}, {"name": "stability"})
    )
    assert record["withdrawn"] == {first: "unstable"}
    assert record["family"] == list(survivors[1:])
    assert record["survivors"] == list(survivors)
    assert record["rule"] == {"name": "stability"}
    assert record["statement"] == records.STAGE_TWO_STATEMENT
    with pytest.raises(
        scmle.assurance.ledger.ledger.LedgerError, match="not frozen survivors"
    ):
        evaluation.account(ledger, records.Withdrawal({"zz": "x"}, {}))


def test_verify_confirms_the_signal_on_fresh_rows():
    ledger = explored()
    survivors = ledger.freeze().hypotheses
    claims, checks = evaluation.verify(
        ledger,
        world(400, 2),
        budget.Correction.BY,
        ALPHA,
        evaluation.encode,
    )
    assert [c.identifier for c in claims] == list(survivors)
    assert len(checks) == len(survivors)
    confirmed = {
        ledger.hypotheses[c.identifier].spec.column
        for c in claims
        if c.tier is qualification.Tier.CONFIRMED
    }
    assert confirmed == {"s1", "s2"}
    for check in checks:
        assert check.subset is qualification.Subset.CONFIRMATION
        passed = check.tier is qualification.Tier.CONFIRMED
        assert check.decision is (
            qualification.Decision.WARN
            if passed
            else qualification.Decision.PASS
        )


def test_verify_leaves_withdrawn_survivors_untested():
    ledger = explored()
    survivors = ledger.freeze().hypotheses
    record = evaluation.account(
        ledger, records.Withdrawal({survivors[0]: "why"}, {})
    )
    claims, checks = evaluation.verify(
        ledger,
        world(400, 2),
        budget.Correction.BY,
        ALPHA,
        evaluation.encode,
        record,
    )
    assert [c.identifier for c in claims] == list(survivors)
    assert claims[0].tier is qualification.Tier.UNVERIFIED
    assert claims[0].assumptions[0].startswith("withdrawn before C was opened")
    assert len(checks) == len(survivors) - 1


# ---- opening ---------------------------------------------------------------


def test_enforce_accepts_both_policies_and_match_needs_the_session():
    opening.enforce("unconfirmed")
    opening.enforce("thresholdout")
    with pytest.raises(errors.ConfigError, match="needs a reuse session"):
        opening.match("thresholdout", None)
    with pytest.raises(errors.ConfigError, match="failure policy must be one"):
        opening.enforce("lenient")
    with pytest.raises(errors.ConfigError, match="failure policy must be one"):
        opening.match("lenient", None)


def test_phrase_and_tip_describe_a_crash_and_a_log():
    sentence = opening.phrase("testing on C", ValueError("bad"))
    assert sentence == (
        "the confirmation failed while testing on C (ValueError: bad); C is "
        "spent and all claims stay unverified (failure policy: unconfirmed)"
    )
    assert opening.tip(None) is None
    assert (
        opening.tip(type("L", (), {"entries": [{"hash": "h1"}, {"hash": 2}]})())
        == "2"
    )


def test_account_without_a_runner_adds_nothing():
    body = {}
    assert opening.account(body, None, None) == (None, {})
    assert body == {}


def test_account_marks_unevaluated_claims():
    body = {}
    registered, documents = opening.account(body, docket(), None)
    assert registered == {
        "baseline": {
            "kind": "baseline",
            "sha256": "a" * 64,
            "status": "not_evaluated",
        }
    }
    assert documents == {}
    assert body["attachments"] == {}
    ran = records.Output({"baseline": "pass"}, {"b.json": {"k": 1}})
    registered, documents = opening.account(body, docket(), ran)
    assert registered["baseline"]["status"] == "pass"
    assert body["attachments"] == {
        "b.json": scmle.assurance.ledger.ledger.digest({"k": 1})
    }


def test_confirm_without_c_records_no_confirmation_set():
    ledger = explored()
    result = opening.confirm(ledger, None, alpha_confirm=ALPHA)
    assert result.outcome == records.NO_CONFIRMATION_SET
    assert result.reason == "no confirmation set exists for this run"
    assert result.head is None
    assert result.checks == ()
    assert ledger.confirmation is not None
    with pytest.raises(errors.LeakageError, match="already used"):
        opening.confirm(ledger, None, alpha_confirm=ALPHA)


def test_confirm_tests_once_and_denies_a_second_attempt():
    ledger = explored()
    source = locker(*world(400, 2))
    result = opening.confirm(ledger, source, alpha_confirm=ALPHA)
    assert result.outcome == records.CONFIRMED
    assert result.reason.endswith("survivors confirmed on C by by at 0.05")
    assert result.head == source.log.entries[-1]["hash"]
    events = [e["event"] for e in source.log.entries]
    assert "selection_frozen" in events
    with pytest.raises(errors.LeakageError):
        opening.confirm(ledger, source, alpha_confirm=ALPHA)
    assert source.log.entries[-1]["event"] == "denied"


def test_a_blocked_family_still_opens_c_for_registered_claims():
    ledger = explored(control=False)
    source = locker(*world(50, 2))
    result = opening.confirm(
        ledger, source, alpha_confirm=ALPHA, claim_runner=docket()
    )
    assert result.outcome == records.NO_CONTROL
    assert result.reason.endswith(
        "; C was opened for the registered claims only"
    )
    assert result.registered["baseline"]["status"] == "pass"
    assert result.documents == {"baseline.json": {"rows": 50}}


def test_a_crashing_runner_marks_its_claims_failed():
    ledger = explored()
    source = locker(*world(400, 2))
    result = opening.confirm(
        ledger, source, alpha_confirm=ALPHA, claim_runner=docket(fail=True)
    )
    assert result.outcome == records.CONFIRMED
    assert result.registered["baseline"]["status"] == "failed"
    assert "claims_failed" in [e["event"] for e in source.log.entries]


def test_a_failing_encoder_gives_a_pipeline_failure():
    ledger = explored()

    def broken(target):
        raise ArithmeticError("nope")

    result = opening.confirm(
        ledger, locker(*world(400, 2)), alpha_confirm=ALPHA, encode=broken
    )
    assert result.outcome == records.PIPELINE_FAILURE
    assert "testing on C (ArithmeticError: nope)" in result.reason
    assert {c.tier for c in result.claims} == {qualification.Tier.UNVERIFIED}


def test_withdrawing_every_survivor_keeps_c_shut():
    ledger = explored()
    survivors = ledger.freeze().hypotheses
    source = locker(*world(50, 2))
    result = opening.confirm(
        ledger,
        source,
        alpha_confirm=ALPHA,
        withdrawal=records.Withdrawal(dict.fromkeys(survivors, "why"), {}),
    )
    assert result.outcome == records.ALL_WITHDRAWN
    assert not source.opened
    assert result.second["family"] == []


def test_preregister_hashes_the_family_and_registered_claims():
    ledger = explored()
    survivors = ledger.freeze()
    log = sealed.Journal()
    plain = opening.preregister(
        log, ledger, survivors, budget.Correction.BY, None, ()
    )
    assert plain == survivors.sha256
    entry = log.entries[-1]
    assert entry["event"] == "selection_frozen"
    assert entry["detail"]["rule"] == "by"
    with_claims = opening.preregister(
        log, ledger, survivors, budget.Correction.BY, None, (commitment(),)
    )
    assert with_claims == scmle.assurance.ledger.ledger.digest(
        {"family": survivors.sha256, "registered": [["baseline", "a" * 64]]}
    )
    assert log.entries[-1]["detail"]["registered_claims"] == ["baseline"]


def test_finish_needs_frozen_survivors():
    ledger = explored()
    with pytest.raises(
        scmle.assurance.ledger.ledger.LedgerError, match="freeze"
    ):
        opening.finish(
            ledger,
            outcome="x",
            reason="r",
            claims=(),
            checks=(),
            rule=budget.Correction.BY,
            alpha=ALPHA,
            custody_head=None,
        )
