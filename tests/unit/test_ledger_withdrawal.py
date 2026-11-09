"""The second stage: an exploration-only withdrawal before C opens."""

import json

import numpy
import pandas
import pytest
from scipy import stats

from scmle.data.ledger import confirm
from scmle.data.ledger import governed
from scmle.data.ledger import ledger
from tests.unit.test_charter_contracts import make_charter
from tests.unit.test_ledger_governed import build
from tests.unit.test_ledger_governed import source_for

WORLD = {"rows": 500, "seed": 0, "effect": 0.7}


def withdraw_columns(*columns, seen=None, log=None):
    """A rule that withdraws the survivors on `columns`, reading `X` only."""

    def stage(survivors):
        if seen is not None:
            seen["events_before"] = list(log.entries) if log else []
        reasons = {
            h.hypothesis_id: "fragile on X"
            for h in survivors
            if h.spec.column in columns
        }
        return confirm.Withdrawal(
            reasons, {"name": "test-rule", "columns": list(columns)}
        )

    return stage


def run(second_stage, tmp_path=None, source=None, **world):
    values = {**WORLD, **world}
    features, target = build(**values)
    cfeatures, ctarget = build(rows=400, seed=9, effect=values["effect"])
    source = source or source_for(cfeatures, ctarget)
    return governed.run_governed(
        charter=make_charter(),
        features=features,
        target=target,
        classification=False,
        source=source,
        run_dir=tmp_path,
        second_stage=second_stage,
    )


def columns_of(run_, claims):
    return {run_.ledger.hypotheses[c.claim_id].spec.column for c in claims}


def test_a_withdrawn_survivor_is_not_tested_on_c_and_is_unverified(tmp_path):
    result = run(withdraw_columns("f1"), tmp_path)

    confirmation = result.confirmation
    assert confirmation.outcome == confirm.CONFIRMED
    assert columns_of(result, confirmation.confirmed) == {"f0"}
    tested = {c.check_id.split(":")[0] for c in confirmation.checks}
    withdrawn = next(
        h.hypothesis_id
        for h in result.ledger.survivors()
        if h.spec.column == "f1"
    )
    assert withdrawn not in tested
    claim = next(c for c in confirmation.claims if c.claim_id == withdrawn)
    assert claim.tier.value == "unverified" and claim.adjusted_p_value is None
    assert "withdrawn before C was opened" in claim.assumptions[0]
    assert any(
        "withdrawn before C" in a for a in confirmation.confirmed[0].assumptions
    )
    assert result.summary()["withdrawn"] == 1


def test_the_accounting_is_in_the_claims_file_and_the_sealed_ledger(tmp_path):
    result = run(withdraw_columns("f1"), tmp_path)

    document = json.loads((tmp_path / confirm.CLAIMS_FILE).read_text())
    stage_two = document["stage_two"]
    assert stage_two["statement"] == confirm.STAGE_TWO_STATEMENT
    assert stage_two["rule"] == {"name": "test-rule", "columns": ["f1"]}
    assert len(stage_two["survivors"]) == 2 and len(stage_two["family"]) == 1
    assert set(stage_two["withdrawn"]) == set(stage_two["survivors"]) - set(
        stage_two["family"]
    )
    reloaded = ledger.Ledger.load(
        tmp_path / ledger.FILE_NAME, expected_head=result.ledger.head
    )
    assert reloaded.confirmation["stage_two"] == stage_two
    assert reloaded.survivor_set.sha256 == stage_two["survivors_sha256"]
    assert reloaded.survivor_set.hypothesis_ids == tuple(stage_two["survivors"])


def test_the_family_hash_is_logged_before_c_opens_and_x_only_ran_first():
    cfeatures, ctarget = build(rows=400, seed=9, effect=0.7)
    source = source_for(cfeatures, ctarget)
    seen = {}
    result = run(
        withdraw_columns("f1", seen=seen, log=source.log), source=source
    )

    # the rule ran before C was selected for, issued a token or opened
    assert {e["event"] for e in seen["events_before"]} == {"sealed"}
    events = [e["event"] for e in source.log.entries]
    assert events.index("selection_frozen") < events.index("open")
    frozen = next(
        e for e in source.log.entries if e["event"] == "selection_frozen"
    )
    stage_two = result.confirmation.stage_two
    assert frozen["detail"]["family_sha256"] == stage_two["family_sha256"]
    assert frozen["detail"]["withdrawn"] == 1


def test_withdrawing_everything_leaves_c_unopened():
    cfeatures, ctarget = build(rows=400, seed=9, effect=0.7)
    source = source_for(cfeatures, ctarget)
    result = run(withdraw_columns("f0", "f1"), source=source)

    assert result.confirmation.outcome == confirm.ALL_WITHDRAWN
    assert result.confirmation.confirmed == ()
    assert len(result.confirmation.claims) == 2
    assert not [e for e in source.log.entries if e["event"] == "open"]


def test_a_withdrawal_naming_a_non_survivor_is_refused():
    def stage(survivors):
        return confirm.Withdrawal({"h999999": "invented"}, {"name": "bad"})

    with pytest.raises(ledger.LedgerError, match="not frozen survivors"):
        run(stage)


def test_without_a_withdrawal_nothing_changes(tmp_path):
    result = run(None, tmp_path)
    document = json.loads((tmp_path / confirm.CLAIMS_FILE).read_text())
    assert "stage_two" not in document
    assert result.confirmation.stage_two is None
    assert result.summary()["withdrawn"] is None
    assert columns_of(result, result.confirmation.confirmed) == {"f0", "f1"}


def test_an_empty_withdrawal_keeps_every_survivor():
    result = run(withdraw_columns())
    assert columns_of(result, result.confirmation.confirmed) == {"f0", "f1"}
    assert result.confirmation.stage_two["withdrawn"] == {}


def weak_world(seed):
    rng = numpy.random.default_rng(seed)

    def draw(rows):
        frame = pandas.DataFrame(
            {f"f{i}": rng.normal(size=rows) for i in range(10)}
        )
        signal = 0.35 * (frame["f0"] + frame["f1"] + frame["f2"])
        return frame, pandas.Series(signal + rng.normal(size=rows))

    return draw(400), draw(300)


def below_median_rule(features, target):
    """Withdraws survivors with a small exploration effect (reads X only)."""

    def stage(survivors):
        effects = {
            h.hypothesis_id: abs(
                stats.spearmanr(features[h.spec.column], target)[0]
            )
            for h in survivors
        }
        cut = float(numpy.median(list(effects.values()))) if effects else 0.0
        reasons = {
            i: "small effect on X" for i, v in effects.items() if v < cut
        }
        return confirm.Withdrawal(reasons, {"name": "below-median"})

    return stage


def test_the_false_claim_rate_stays_below_alpha_with_an_x_only_withdrawal():
    """Simulated check of the claim in `confirm.STAGE_TWO_STATEMENT`.

    Features f0-f2 are real and f3-f9 are null. The rule reads `X` only and
    is tied to the signal. The mean false discovery proportion of the
    confirmed claims must stay below alpha_confirm = 0.05 with and without
    the rule, and the rule must really withdraw something in this world.
    """
    reps = 40
    proportions = {"with": [], "without": []}
    withdrawn = 0
    for seed in range(reps):
        (features, target), (cfeatures, ctarget) = weak_world(seed)
        for name in proportions:
            result = governed.run_governed(
                charter=make_charter(),
                features=features,
                target=target,
                classification=False,
                source=source_for(cfeatures, ctarget),
                run_dir=None,
                second_stage=(
                    below_median_rule(features, target)
                    if name == "with"
                    else None
                ),
            )
            claims = result.confirmation.confirmed
            false = sum(
                columns_of(result, [c]).pop() not in {"f0", "f1", "f2"}
                for c in claims
            )
            proportions[name].append(false / max(len(claims), 1))
            if name == "with" and result.confirmation.stage_two:
                withdrawn += len(result.confirmation.stage_two["withdrawn"])
    assert withdrawn > 0
    for name, values in proportions.items():
        assert sum(values) / reps <= 0.05, name
