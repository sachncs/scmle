"""Thresholdout: derived terms, answers, budget, journal and the opening."""

from fractions import Fraction
import json
import math

import numpy
import pandas
import pytest

from scmle.assurance.custody import sealed
from scmle.assurance.ledger import confirm
from scmle.assurance.ledger import controls
from scmle.assurance.ledger import explore
from scmle.assurance.ledger import online_fdr
from scmle.assurance.ledger.confirm import opening
from scmle.assurance.ledger.confirm import thresholdout
import scmle.assurance.ledger.ledger
from scmle.contracts import errors
from scmle.data.splitting.splitters import lockbox

ALPHA = Fraction(1, 20)


def coins(rows, seed, columns=3):
    rng = numpy.random.default_rng(seed)
    features = pandas.DataFrame(
        rng.choice([-1, 1], size=(rows, columns)),
        columns=[f"f{i}" for i in range(columns)],
    )
    return features, pandas.Series(rng.choice([-1, 1], size=rows), name="y")


def linked(rows, seed):
    features = coins(rows, seed)[0]
    return features, pandas.Series(features["f0"] * features["f1"], name="y")


def agree(column):
    return lambda f, y: (1 + f[column].to_numpy() * y.to_numpy()) / 2


def test_the_terms_follow_the_written_derivation():
    terms = thresholdout.Terms(20000, queries=120, budget=20)
    levels = 101
    count = math.log(sum(math.comb(120, k) * levels**k for k in range(21)))
    sampling = math.sqrt((count + math.log(4 * 120 / 0.05)) / 40000)
    threshold = (0.1 - sampling) / (1 + 2 * math.log(40) / 4)
    assert thresholdout.transcripts(120, 20, 0.01) == pytest.approx(
        count, rel=1e-12
    )
    assert terms.sampling == pytest.approx(sampling, rel=1e-12)
    assert terms.threshold == pytest.approx(threshold, rel=1e-12)
    assert terms.noise == pytest.approx(threshold / 4, rel=1e-12)
    record = terms.to_dict()
    assert set(record) == {
        "size", "tolerance", "confidence", "queries", "budget", "step",
        "sampling", "threshold", "noise",
    }  # fmt: skip
    assert thresholdout.Terms.from_dict(record) == terms


def test_the_minimum_is_the_smallest_size_that_fits():
    least = thresholdout.minimum(queries=120, budget=20)
    assert least == 8934
    thresholdout.Terms(least, queries=120, budget=20)
    with pytest.raises(errors.InputError, match="at least 8934 rows"):
        thresholdout.Terms(least - 1, queries=120, budget=20)


@pytest.mark.parametrize(
    "changes",
    [
        {"tolerance": 0.0},
        {"tolerance": 1.0},
        {"confidence": 1.5},
        {"step": 0.0},
        {"queries": 0},
        {"budget": 0},
        {"budget": 101},
        {"size": 2.5},
        {"queries": True},
    ],
)
def test_invalid_terms_are_refused(changes):
    with pytest.raises(errors.InputError):
        thresholdout.Terms(**({"size": 100000} | changes))


def test_a_coarse_step_has_no_minimum():
    with pytest.raises(errors.InputError, match="too coarse"):
        thresholdout.minimum(tolerance=0.1, step=0.5)


def test_from_dict_names_missing_parameters():
    with pytest.raises(errors.InputError, match=r"lack \['step'\]"):
        thresholdout.Terms.from_dict(
            {"size": 1, "tolerance": 0.1, "confidence": 0.05}
            | {"queries": 1, "budget": 1}
        )


def agreeing(f, y):
    return (1 + f["f0"].to_numpy() * f["f1"].to_numpy() * y.to_numpy()) / 2


def test_matching_sets_answer_from_train_and_a_gap_spends_budget():
    train = linked(20000, 1)
    guard = thresholdout.Thresholdout(train, train)
    answer = guard.query(agreeing)
    assert answer == thresholdout.Answer(0, 1.0, "train", 10)
    other = thresholdout.Thresholdout(train, (train[0], -train[1]))
    answer = other.query(agreeing)
    assert answer.origin == "holdout" and answer.left == 9
    assert answer.value == round(answer.value / 0.01) * 0.01
    assert answer.value <= 0.02
    assert (other.asked, other.spent, other.left) == (1, 1, 9)


def test_the_budget_and_the_query_count_are_enforced():
    train = linked(20000, 1)
    terms = thresholdout.Terms(20000, queries=3, budget=1)
    guard = thresholdout.Thresholdout(train, (train[0], -train[1]), terms)
    guard.query(agreeing)
    with pytest.raises(errors.LeakageError, match="budget of 1"):
        guard.query(agree("f1"))
    same = thresholdout.Thresholdout(train, train, terms)
    for column in ("f0", "f1", "f2"):
        same.query(agree(column))
    with pytest.raises(errors.LeakageError, match="its 3 answers"):
        same.query(agree("f0"))


@pytest.mark.parametrize(
    ("query", "message"),
    [
        (lambda f, y: numpy.full(len(y), 1.5), r"lie in \[0, 1\]"),
        (lambda f, y: numpy.full(len(y), numpy.nan), "finite"),
        (lambda f, y: numpy.zeros(3), "score each of the 20000 train"),
    ],
)
def test_invalid_scores_are_refused_and_spend_nothing(query, message):
    train = coins(20000, 1)
    guard = thresholdout.Thresholdout(train, train)
    with pytest.raises(errors.InputError, match=message):
        guard.query(query)
    assert (guard.asked, guard.spent, guard.answers) == (0, 0, [])


def test_the_terms_must_fit_the_holdout():
    rows = coins(20000, 1)
    with pytest.raises(errors.InputError, match="terms are for 30000 rows"):
        thresholdout.Thresholdout(rows, rows, thresholdout.Terms(30000))
    small = coins(500, 2)
    with pytest.raises(errors.InputError, match="needs at least"):
        thresholdout.Thresholdout(small, small)


def test_the_journal_resumes_with_the_budget_spent(tmp_path):
    journal = tmp_path / "reuse.json"
    train = linked(20000, 1)
    holdout = (train[0], -train[1])
    first = thresholdout.Thresholdout(train, holdout, journal=journal)
    given = [first.query(agreeing), first.query(agreeing)]
    stored = json.loads(journal.read_text(encoding="utf-8"))
    assert stored["schema_version"] == 1
    assert (stored["asked"], stored["spent"]) == (2, 2)
    again = thresholdout.Thresholdout(train, holdout, journal=journal)
    assert (again.asked, again.spent, again.answers) == (2, 2, given)
    assert again.query(agreeing) == first.query(agreeing)
    assert again.spent == 3


def test_a_journal_for_other_rows_or_a_broken_one_is_refused(tmp_path):
    journal = tmp_path / "reuse.json"
    train = coins(20000, 1)
    thresholdout.Thresholdout(train, train, journal=journal).query(agree("f0"))
    with pytest.raises(errors.LeakageError, match="other rows or terms"):
        thresholdout.Thresholdout(train, coins(20000, 3), journal=journal)
    journal.write_text("{", encoding="utf-8")
    with pytest.raises(errors.LeakageError, match="unreadable"):
        thresholdout.Thresholdout(train, train, journal=journal)


def test_a_failed_write_spends_nothing(tmp_path, monkeypatch):
    train = linked(20000, 1)
    guard = thresholdout.Thresholdout(
        train, (train[0], -train[1]), journal=tmp_path / "reuse.json"
    )
    state = guard.generator.bit_generator.state

    def refuse(path, content):
        raise OSError("disk full")

    monkeypatch.setattr(thresholdout.durability, "write_atomically", refuse)
    with pytest.raises(OSError, match="disk full"):
        guard.query(agreeing)
    assert (guard.asked, guard.spent, guard.answers) == (0, 0, [])
    assert guard.generator.bit_generator.state == state


def world(rows, seed):
    rng = numpy.random.default_rng(seed)
    features = pandas.DataFrame(
        {f"n{i}": rng.normal(size=rows) for i in range(4)}
    )
    target = features["n0"] * 0.6 + rng.normal(size=rows)
    return features, pandas.Series(target, name="y")


def explored():
    x, y = world(600, 1)
    ledger = scmle.assurance.ledger.ledger.Ledger.create(
        None,
        charter_hash="c" * 64,
        config=online_fdr.Settings(online_fdr.ELOND, Fraction(1, 10)),
        alpha_confirm=ALPHA,
        seed=3,
    )
    explore.explore(ledger, x, y, explore.hypothesize(x, explore.NUMERIC))
    summary = controls.summarize("permuted_target", "strata", 6, (0,) * 20)
    ledger.negatives(summary.to_dict())
    return ledger, x, y


def locker(features, target):
    resident = lockbox.Resident(features, target, tuple(range(len(target))))
    return sealed.Locker("C", resident, sealed.Journal())


def above(guard):
    answer = guard.query(
        lambda f, y: (f["n0"].to_numpy() * y.to_numpy() > 0).astype(float)
    )
    return {"agreement": answer.value, "origin": answer.origin}


def test_the_policy_reuses_c_on_the_single_opening():
    ledger, x, y = explored()
    source = locker(*world(20000, 2))
    session = thresholdout.Session((x, y), above)
    result = confirm.confirm(
        ledger,
        source,
        alpha_confirm=ALPHA,
        policy=confirm.THRESHOLDOUT,
        reuse=session,
    )
    assert result.outcome == confirm.CONFIRMED
    assert result.registered["thresholdout"]["status"] == "answered"
    assert result.registered["thresholdout"]["kind"] == "reuse"
    document = result.documents["thresholdout.json"]
    assert document["status"] == "answered"
    assert document["asked"] == 1 and len(document["answers"]) == 1
    assert document["findings"]["agreement"] == document["answers"][0]["value"]
    assert "state" not in document
    assert document["terms"]["size"] == 20000
    events = [e["event"] for e in source.log.entries]
    assert events.count("open") == 1


def test_a_session_that_runs_into_the_budget_is_exhausted():
    ledger, x, y = explored()
    source = locker(*world(20000, 2))

    def greedy(guard):
        while True:
            guard.query(lambda f, y: numpy.zeros(len(y)))

    session = thresholdout.Session((x, y), greedy, queries=3, budget=1)
    result = confirm.confirm(
        ledger,
        source,
        alpha_confirm=ALPHA,
        policy=confirm.THRESHOLDOUT,
        reuse=session,
    )
    document = result.documents["thresholdout.json"]
    assert result.registered["thresholdout"]["status"] == "exhausted"
    assert document["asked"] == 3
    assert "its 3 answers" in document["findings"]["stopped"]


def test_a_holdout_too_small_fails_the_claim_not_the_run():
    ledger, x, y = explored()
    source = locker(*world(300, 2))
    session = thresholdout.Session((x, y), above)
    result = confirm.confirm(
        ledger,
        source,
        alpha_confirm=ALPHA,
        policy=confirm.THRESHOLDOUT,
        reuse=session,
    )
    assert result.outcome == confirm.CONFIRMED
    assert result.registered["thresholdout"]["status"] == "failed"
    assert [e["event"] for e in source.log.entries][-1] == "claims_failed"


def test_the_policy_and_the_session_come_together():
    ledger, x, y = explored()
    source = locker(*world(20000, 2))
    with pytest.raises(errors.ConfigError, match="needs a reuse session"):
        confirm.confirm(
            ledger, source, alpha_confirm=ALPHA, policy=confirm.THRESHOLDOUT
        )
    with pytest.raises(errors.ConfigError, match="needs the 'thresholdout'"):
        confirm.confirm(
            ledger,
            source,
            alpha_confirm=ALPHA,
            reuse=thresholdout.Session((x, y), above),
        )
    assert not ledger.frozen and not source.opened
    opening.enforce(confirm.THRESHOLDOUT)


def test_a_crash_names_the_policy():
    failure = RuntimeError("boom")
    assert opening.phrase("testing on C", failure) == (
        "the confirmation failed while testing on C (RuntimeError: boom); "
        "C is spent and all claims stay unverified (failure policy: "
        "unconfirmed)"
    )
    assert opening.phrase("testing on C", failure, "thresholdout") == (
        "the confirmation failed while testing on C (RuntimeError: boom); "
        "C is spent for the family and all its claims stay unverified; the "
        "reuse session's answers, if any, are attached (failure policy: "
        "thresholdout)"
    )
