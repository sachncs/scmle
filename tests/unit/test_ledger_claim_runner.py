"""Registered claims: registered by hash, evaluated once on the one opening."""

import pytest

from scmle.data.ledger import confirm
from scmle.data.ledger import governed
from scmle.data.ledger import ledger as ledger_module
from scmle.data.ledger import online_fdr
from tests.unit.test_charter_contracts import make_charter
from tests.unit.test_ledger_governed import build
from tests.unit.test_ledger_governed import source_for

SHA = "a" * 64


def runner(calls, status="done", boom=False):
    def run(features, target):
        calls.append((len(features), len(target)))
        if boom:
            raise ValueError("estimator failed")
        return confirm.ClaimRun(
            {"extra": status}, {"extra_report.json": {"n": len(features)}}
        )

    return confirm.ClaimRunner(
        (confirm.RegisteredClaim("extra", "baseline", "a claim", SHA),), run
    )


def go(claim_runner, effect=0.7, source=True, **kwargs):
    features, target = build(rows=500, seed=0, effect=effect)
    cfeatures, ctarget = build(rows=400, seed=9, effect=effect)
    return governed.run_governed(
        charter=make_charter(),
        features=features,
        target=target,
        classification=False,
        source=source_for(cfeatures, ctarget) if source else None,
        run_dir=None,
        claim_runner=claim_runner,
        **kwargs,
    )


def test_the_runner_gets_the_one_opened_frame_and_is_sealed_in_the_ledger():
    calls = []
    result = go(runner(calls))

    assert calls == [(400, 400)]
    confirmation = result.confirmation
    assert confirmation.outcome == confirm.CONFIRMED
    assert confirmation.registered["extra"]["status"] == "done"
    sealed = result.ledger.confirmation
    assert sealed["claims"] == confirmation.registered
    assert set(sealed["attachments"]) == {"extra_report.json"}
    assert result.ledger.claims["extra"]["sha256"] == SHA
    rows = [r["kind"] for r in result.ledger.rows]
    assert rows.index("claim") < rows.index("survivors")
    result.ledger.verify()


def test_the_claim_is_bound_into_the_release_and_c_opens_once():
    features, target = build(rows=500, seed=0, effect=0.7)
    cfeatures, ctarget = build(rows=400, seed=9, effect=0.7)
    source = source_for(cfeatures, ctarget)
    calls = []
    governed.run_governed(
        charter=make_charter(),
        features=features,
        target=target,
        classification=False,
        source=source,
        run_dir=None,
        claim_runner=runner(calls),
    )
    events = [e["event"] for e in source.log.entries]
    assert events.count("open") == 1
    frozen = next(
        e for e in source.log.entries if e["event"] == "selection_frozen"
    )
    assert frozen["detail"]["registered_claims"] == ["extra"]


def test_a_null_world_still_evaluates_the_registered_claim_on_c():
    calls = []
    result = go(runner(calls), effect=0.0)

    assert result.confirmation.outcome in {
        confirm.NO_SURVIVORS,
        confirm.FLOOR_NOT_EXCEEDED,
        confirm.CONFIRMED,
    }
    assert calls == [(400, 400)]
    assert result.confirmation.registered["extra"]["status"] == "done"


def test_no_confirmation_set_means_not_evaluated():
    calls = []
    result = go(runner(calls), source=False)

    assert calls == []
    assert result.confirmation.registered["extra"]["status"] == (
        confirm.NOT_EVALUATED
    )
    assert result.confirmation.documents == {}


def test_a_crashing_runner_is_failed_and_the_family_is_still_confirmed():
    calls = []
    result = go(runner(calls, boom=True))

    assert result.confirmation.outcome == confirm.CONFIRMED
    assert result.confirmation.registered["extra"]["status"] == confirm.FAILED


def new_ledger():
    charter = make_charter()
    return ledger_module.Ledger.create(
        None,
        charter_hash=charter.charter_hash,
        config=online_fdr.RuleConfig("e-lond", 0.1),
        alpha_confirm=charter.error_budget.confirmation.alpha_confirm,
        seed=0,
    )


def test_a_sealed_ledger_takes_no_new_claim():
    result = go(runner([]))
    with pytest.raises(ledger_module.LedgerError):
        result.ledger.register_claim("late", "baseline", "x", SHA)


def test_the_seal_must_state_every_registered_claim():
    ledger = new_ledger()
    ledger.register_claim("c1", "baseline", "s", SHA)
    ledger.freeze()
    with pytest.raises(ledger_module.LedgerError, match="registered claim"):
        ledger.seal({"outcome": "x"})


def test_registering_twice_with_the_same_hash_is_idempotent():
    ledger = new_ledger()
    first = ledger.register_claim("c1", "baseline", "s", SHA)
    assert ledger.register_claim("c1", "baseline", "s", SHA) is first
    with pytest.raises(ledger_module.LedgerError):
        ledger.register_claim("c1", "baseline", "s", "b" * 64)


def test_merged_runners_share_the_one_frame_and_fail_separately():
    calls = []

    def second(name, boom=False):
        def run(features, target):
            calls.append(name)
            if boom:
                raise ValueError("no")
            return confirm.ClaimRun({name: "ok"}, {f"{name}.json": {"k": 1}})

        return confirm.ClaimRunner(
            (confirm.RegisteredClaim(name, "baseline", name, SHA),), run
        )

    merged = confirm.merge_runners(
        None, second("a"), second("b", boom=True), second("c")
    )
    result = go(merged)

    assert calls == ["a", "b", "c"]
    states = {k: v["status"] for k, v in result.confirmation.registered.items()}
    assert states == {"a": "ok", "b": confirm.FAILED, "c": "ok"}
    assert set(result.confirmation.documents) == {"a.json", "c.json"}
    assert confirm.merge_runners(None, None) is None
    with pytest.raises(ValueError, match="unique"):
        confirm.merge_runners(second("a"), second("a"))
