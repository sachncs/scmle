"""Behavioural tests of the per-hypothesis helpers of the confirmation."""

from fractions import Fraction

import numpy
import pandas
import pytest

from scmle.assurance.ledger import explore
from scmle.assurance.ledger import online_fdr
from scmle.assurance.ledger.confirm import evaluation
from scmle.assurance.ledger.confirm import opening
from scmle.assurance.ledger.confirm import records
import scmle.assurance.ledger.ledger
from scmle.contracts import qualification


def world(rows, seed):
    rng = numpy.random.default_rng(seed)
    features = pandas.DataFrame(
        {"n0": rng.normal(size=rows), "s1": rng.normal(size=rows)}
    )
    target = 0.9 * features["s1"] + rng.normal(size=rows)
    return features, pandas.Series(target, name="y")


@pytest.fixture
def ledger():
    x, y = world(400, 1)
    book = scmle.assurance.ledger.ledger.Ledger.create(
        None,
        charter_hash="c" * 64,
        config=online_fdr.Settings(online_fdr.ELOND, Fraction(1, 10)),
        alpha_confirm=Fraction(1, 20),
        seed=3,
    )
    explore.run_exploration(book, x, y, explore.hypothesize(x, explore.NUMERIC))
    book.freeze()
    return book


def test_withdraw_writes_an_unverified_claim_with_the_reason(ledger):
    identity = ledger.survivor_set.hypothesis_ids[0]
    claims = evaluation.withdraw(ledger, {identity: "too fragile"})
    claim = claims[identity]
    spec = ledger.hypotheses[identity].spec
    assert list(claims) == [identity]
    assert claim.identifier == identity
    assert claim.statement == f"{spec.name}: evidence against '{spec.null}'"
    assert claim.checks == (f"{identity}:X",)
    assert claim.adjusted_p_value is None
    assert claim.effect == ledger.results[identity].check.effect
    assert claim.tier is qualification.Tier.UNVERIFIED
    assert claim.assumptions[0] == (
        "withdrawn before C was opened, not tested on C: too fragile"
    )
    assert evaluation.withdraw(ledger, {}) == {}


def test_measure_runs_each_statistic_on_its_column(ledger):
    ids = list(ledger.survivor_set.hypothesis_ids)
    x, y = world(300, 9)
    codes = evaluation.default_encoding(y)
    outcomes = evaluation.measure(ledger, ids, x, codes)
    assert len(outcomes) == len(ids)
    for identity, outcome in zip(ids, outcomes, strict=True):
        assert outcome.n == 300
        spec = ledger.hypotheses[identity].spec
        if spec.column == "s1":
            assert outcome.p < 0.01


def test_examine_builds_a_confirmed_or_unverified_check(ledger):
    identity = ledger.survivor_set.hypothesis_ids[0]
    x, y = world(300, 9)
    [outcome] = evaluation.measure(
        ledger, [identity], x, evaluation.default_encoding(y)
    )
    passed = evaluation.examine(ledger, identity, 2, outcome, True)
    spec = ledger.hypotheses[identity].spec
    assert passed.identifier == f"{identity}:C"
    assert passed.null == spec.null
    assert passed.statistic == outcome.statistic
    assert passed.p_value == outcome.p
    assert passed.effect == outcome.effect
    assert passed.n == outcome.n
    assert passed.interval is None
    assert passed.decision is qualification.Decision.WARN
    assert passed.tier is qualification.Tier.CONFIRMED
    assert passed.subset is qualification.Subset.CONFIRMATION
    assert passed.seeds[0] == 3 and passed.seeds[2] == 2
    failed = evaluation.examine(ledger, identity, 0, outcome, False)
    assert failed.decision is qualification.Decision.PASS
    assert failed.tier is qualification.Tier.UNVERIFIED


class Log:
    def __init__(self):
        self.entries = []

    def append(self, set_id, event, actor, detail=None):
        entry = {
            "set": set_id,
            "event": event,
            "actor": actor,
            "detail": detail,
        }
        entry["hash"] = f"h{len(self.entries)}"
        self.entries.append(entry)
        return entry


def test_deny_writes_a_refusal_only_when_there_is_a_log():
    opening.deny(None)
    log = Log()
    opening.deny(log)
    assert log.entries == [
        {
            "set": "C",
            "event": "denied",
            "actor": "confirmation",
            "detail": {"reason": "confirmation was already performed"},
            "hash": "h0",
        }
    ]
    assert opening.tip(log) == "h0"


def test_evaluate_marks_every_claim_failed_when_the_runner_crashes():
    def broken(features, target):
        raise KeyError("column")

    runner = records.Docket(
        (
            records.Commitment("a", "baseline", "s", "0" * 64),
            records.Commitment("b", "coverage", "s", "1" * 64),
        ),
        broken,
    )
    log = Log()
    out = opening.evaluate(runner, (pandas.DataFrame(), pandas.Series()), log)
    assert dict(out.statuses) == {"a": "failed", "b": "failed"}
    assert dict(out.documents) == {}
    assert log.entries[0]["event"] == "claims_failed"
    assert log.entries[0]["detail"]["reason"] == opening.phrase(
        "evaluating the registered claims", KeyError("column")
    )
    assert (
        opening.evaluate(None, (pandas.DataFrame(), pandas.Series()), log)
        is None
    )


def test_crash_gives_a_pipeline_failure_with_unverified_claims(ledger):
    outcome, reason, claims = opening.crash(
        ledger, "testing on C", ValueError("x")
    )
    assert outcome == "pipeline_failure"
    assert reason == (
        "the confirmation failed while testing on C (ValueError: x); C is "
        "spent and all claims stay unverified (failure policy: unconfirmed)"
    )
    assert len(claims) == len(ledger.survivor_set.hypothesis_ids)
    assert all(c.tier is qualification.Tier.UNVERIFIED for c in claims)
