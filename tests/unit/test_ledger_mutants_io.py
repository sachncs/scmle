"""Pins for the ledger file, the confirmation record and the exploration plan.

Like `test_ledger_mutants`, each test exists because a mutation of the code
survived the broader tests.
"""

from fractions import Fraction
import json
import re

import numpy as np
import pandas as pd
import pytest

from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.custody import sealed
from scmle.data.ledger import bivariate
from scmle.data.ledger import budget as b
from scmle.data.ledger import confirm as cf
from scmle.data.ledger import controls
from scmle.data.ledger import explore
from scmle.data.ledger import governed
from scmle.data.ledger import ledger as ledger_module
from scmle.data.ledger import online_fdr
from scmle.data.splitting import splitters
from tests.unit.test_charter_contracts import make_charter

F = Fraction
TEN_PERCENT = F(1, 10)


def raises(error, message):
    return pytest.raises(error, match=f"^{re.escape(message)}$")


def fresh(path=None, alpha=TEN_PERCENT, rule=online_fdr.ELOND):
    return ledger_module.Ledger.create(
        path,
        charter_hash="c" * 64,
        config=online_fdr.RuleConfig(rule, alpha),
        alpha_confirm=F(1, 20),
        seed=7,
    )


def signal_data(rows=300, seed=0):
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {"s": rng.normal(size=rows), "n": rng.normal(size=rows)}
    )
    return features, pd.Series(0.9 * features["s"] + rng.normal(size=rows))


def spec(column, name=None, direction=bivariate.TWO_SIDED):
    return ledger_module.HypothesisSpec.create(
        name=name or f"{column}~y",
        family="f",
        statistic="spearman",
        null="no association",
        column=column,
        direction=direction,
    )


# --- ledger -----------------------------------------------------------------


def test_files_are_written_with_sorted_keys_and_nested_folders(tmp_path):
    path = tmp_path / "deep" / "er" / ledger_module.FILE_NAME
    ledger = fresh(path)
    features, target = signal_data()
    ledger.test(ledger.register(spec("s")), features, target)
    for line in path.read_text(encoding="utf-8").splitlines():
        keys = list(json.loads(line))
        assert keys == sorted(keys)
        assert line == json.dumps(json.loads(line), sort_keys=True)


def test_open_creates_missing_folders_and_passes_the_horizon(tmp_path):
    charter = make_charter()
    ledger = ledger_module.Ledger.open(tmp_path / "a" / "b", charter, horizon=4)
    assert ledger.config.horizon == 4
    again = ledger_module.Ledger.open(tmp_path / "a" / "b", charter, horizon=4)
    assert again.head == ledger.head
    with raises(
        ledger_module.LedgerError, "ledger was created with a different rule"
    ):
        ledger_module.Ledger.open(tmp_path / "a" / "b", charter, horizon=5)


def test_verify_file_checks_the_recorded_head(tmp_path):
    path = tmp_path / ledger_module.FILE_NAME
    ledger = fresh(path)
    assert (
        ledger_module.Ledger.verify_file(path, ledger.head).head == ledger.head
    )
    with raises(
        ledger_module.LedgerError,
        "ledger head does not match the recorded head: rows were removed "
        "from the end or the file was replaced",
    ):
        ledger_module.Ledger.verify_file(path, "0" * 64)


def rechained(rows):
    previous = ledger_module.GENESIS
    out = []
    for index, row in enumerate(rows):
        row["seq"] = index
        row["prev"] = previous
        row["hash"] = ledger_module.digest(
            {k: row[k] for k in ("seq", "kind", "body", "prev")}
        )
        previous = row["hash"]
        out.append(row)
    return out


def write_rows(path, rows):
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )


def explored_file(tmp_path, p_value=1e-6):
    path = tmp_path / ledger_module.FILE_NAME
    ledger = fresh(path)
    _features, _target = signal_data()
    handle = ledger.register(spec("n"))
    ledger.record(handle, bivariate.Outcome(p_value, 0.1, 0.1, 100))
    return path


def test_a_wrong_sequence_number_alone_breaks_the_chain(tmp_path):
    path = explored_file(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["seq"] = 5
    rows[1]["hash"] = ledger_module.digest(
        {k: rows[1][k] for k in ("seq", "kind", "body", "prev")}
    )
    write_rows(path, rows)
    with raises(ledger_module.LedgerError, "ledger chain broken at row 1"):
        ledger_module.Ledger.load(path)


def stored_e_with_error(tmp_path, p_value, relative_error):
    path = explored_file(tmp_path, p_value)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = next(r for r in rows if r["kind"] == "result")
    expected = online_fdr.calibrate(p_value)
    shifted = expected + relative_error * max(1.0, expected)
    result["body"]["evidence"]["e"] = shifted
    result["body"]["e_value"] = shifted
    write_rows(path, rechained(rows))
    return path


@pytest.mark.parametrize("p_value", [1e-6, 0.5])
def test_the_calibration_tolerance_is_one_part_in_a_trillion(tmp_path, p_value):
    inside = stored_e_with_error(tmp_path / "in", p_value, 0.5e-12)
    ledger_module.Ledger.load(inside)
    outside = stored_e_with_error(tmp_path / "out", p_value, 1.5e-12)
    with raises(
        ledger_module.LedgerError, "stored e-value is not the calibrated p"
    ):
        ledger_module.Ledger.load(outside)


def test_a_result_row_without_a_reason_loads_with_an_empty_one(tmp_path):
    path = explored_file(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    del next(r for r in rows if r["kind"] == "result")["body"]["reason"]
    write_rows(path, rechained(rows))
    loaded = ledger_module.Ledger.load(path)
    assert next(iter(loaded.results.values())).reason == ""


def test_a_hypothesis_without_a_column_reads_the_first_one():
    features, target = signal_data()
    ledger = fresh()
    plain = ledger_module.HypothesisSpec.create(
        name="first", family="f", statistic="spearman", null="x"
    )
    result = ledger.test(ledger.register(plain), features, target)
    expected = bivariate.spearman(features["s"].to_numpy(), target.to_numpy())
    assert result.check.p_value == expected.p


def test_the_direction_and_seed_path_reach_the_check():
    features, target = signal_data()
    ledger = fresh()
    up = ledger.test(
        ledger.register(spec("n", "up", bivariate.GREATER)), features, target
    )
    down = ledger.test(
        ledger.register(spec("n", "down", bivariate.LESS)), features, target
    )
    both = bivariate.spearman(features["n"].to_numpy(), target.to_numpy())
    assert (
        up.check.p_value
        == bivariate.spearman(
            features["n"].to_numpy(), target.to_numpy(), bivariate.GREATER
        ).p
    )
    assert up.check.p_value + down.check.p_value == pytest.approx(1.0)
    assert up.check.p_value != both.p
    assert up.check.seed_path == randomness.seed_path(7, "ledger.spearman", 0)
    assert down.check.seed_path == randomness.seed_path(7, "ledger.spearman", 1)


# --- confirmation record ----------------------------------------------------


def source_for(features, target):
    log = sealed.OpenLog()
    locked = splitters.ResidentTestSet(
        features, target, tuple(range(len(target)))
    )
    return sealed.SealedSet("C", locked, log)


def explored_ledger(alpha=TEN_PERCENT, control_observed=5):
    features, target = signal_data(600, 1)
    ledger = fresh(None, alpha)
    explore.run_exploration(
        ledger,
        features,
        target,
        explore.plan_exploration(features, explore.NUMERIC),
    )
    ledger.record_control(
        controls.summarize(
            "permuted_target", "strata", control_observed, (0,) * 20
        ).to_dict()
    )
    return ledger


def test_the_claims_file_is_indented_sorted_json_in_a_new_folder(tmp_path):
    ledger = explored_ledger()
    features, target = signal_data(400, 2)
    result = cf.confirm(
        ledger, source_for(features, target), alpha_confirm=F(1, 20)
    )
    path = tmp_path / "new" / "folder" / cf.CLAIMS_FILE
    result.write(path)
    assert path.read_text(encoding="utf-8") == (
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    )


def test_assumptions_are_stated_in_full():
    texts = cf.assumptions(b.MultiplicityRule.BY, 3)
    assert texts == (
        "FDR <= alpha * m0 / m under arbitrary dependence (rule by, family "
        "of 3 survivors fixed before C was opened)",
        "rows of C are independent of rows of X (grouped split by entity, "
        "embargo for temporal data)",
        "C was opened once and its rows were used only by this confirmation",
        "each statistic is valid under its null on C: Spearman and Kruskal "
        "assume independent rows",
        "exploration FDR is a triage budget, not part of this guarantee",
    )


def test_policy_messages_are_exact():
    with raises(
        errors.ConfigError,
        "the Thresholdout reusable holdout is not implemented; use the "
        "'unconfirmed' failure policy",
    ):
        cf.check_policy(cf.THRESHOLDOUT)
    with raises(
        errors.ConfigError,
        "failure policy must be one of ('unconfirmed', 'thresholdout')",
    ):
        cf.check_policy("retry")
    cf.check_policy(cf.UNCONFIRMED)


def test_unconfirmed_claims_keep_the_exploration_effect_and_say_why():
    ledger = explored_ledger(control_observed=0)
    result = cf.confirm(ledger, None, alpha_confirm=F(1, 20))
    assert result.outcome == cf.FLOOR_NOT_EXCEEDED
    claim = result.claims[0]
    expected = ledger.results[claim.claim_id].check.effect
    assert claim.effect == expected and claim.effect != 0.0
    assert claim.assumptions[0].startswith("not confirmed on C: findings do")
    assert claim.check_ids == (f"{claim.claim_id}:X",)
    assert claim.adjusted_p_value is None


def test_reasons_and_event_names_are_exact():
    # Confirmed: reason sentence, custody events, check decisions and seeds.
    ledger = explored_ledger()
    features, target = signal_data(400, 2)
    source = source_for(features, target)
    result = cf.confirm(ledger, source, alpha_confirm=F(1, 20))
    assert result.reason == "1 of 1 survivors confirmed on C by by at 0.05"
    frozen = source.log.entries[1]
    assert (frozen["set"], frozen["event"], frozen["actor"]) == (
        "C",
        "selection_frozen",
        cf.ACTOR,
    )
    assert frozen["detail"]["rule"] == "by"
    assert frozen["detail"]["survivors"] == 1
    check = result.checks[0]
    assert check.decision is qualification.Decision.WARN
    assert check.tier is qualification.Tier.CONFIRMED
    assert check.seed_path == randomness.seed_path(7, "confirm.spearman", 0)
    # A repeat is denied on set C with a reason.
    with pytest.raises(errors.LeakageError):
        cf.confirm(ledger, source, alpha_confirm=F(1, 20))
    denied = source.log.entries[-1]
    assert (denied["set"], denied["event"]) == ("C", "denied")
    assert denied["detail"] == {"reason": "confirmation was already performed"}


def test_non_rejected_survivors_pass_and_stay_unverified():
    ledger = explored_ledger()
    features, target = signal_data(400, 2)
    features["s"] = np.random.default_rng(9).normal(size=400)  # no effect on C
    result = cf.confirm(
        ledger, source_for(features, target), alpha_confirm=F(1, 20)
    )
    check = result.checks[0]
    assert check.decision is qualification.Decision.PASS
    assert check.tier is qualification.Tier.UNVERIFIED
    assert result.reason == "0 of 1 survivors confirmed on C by by at 0.05"


def test_no_survivors_reason_and_crash_reasons():
    features, target = signal_data(300, 3)
    quiet = fresh(None, F(1, 100))
    explore.run_exploration(
        quiet,
        features.assign(s=np.random.default_rng(0).normal(size=300)),
        target,
        explore.plan_exploration(features, explore.NUMERIC),
    )
    result = cf.confirm(
        quiet,
        source_for(*signal_data(300, 4)),
        alpha_confirm=F(1, 20),
        require_control=False,
    )
    assert result.reason == (
        "no hypothesis survived exploration; C was not opened"
    )
    ledger = explored_ledger()
    result = cf.confirm(
        ledger,
        source_for(*signal_data(300, 4)),
        alpha_confirm=F(1, 20),
        encode=lambda y: 1 / 0,
    )
    assert result.reason.startswith(
        "the confirmation failed while testing on C (ZeroDivisionError: "
    )

    class Dies:
        log = sealed.OpenLog()

        def issue_token(self):
            return "t"

        def open(self, token, actor="x"):
            raise OSError("gone")

    result = cf.confirm(explored_ledger(), Dies(), alpha_confirm=F(1, 20))
    assert result.reason.startswith(
        "the confirmation failed while opening C (OSError: gone)"
    )


def test_a_custody_log_without_entries_gives_no_head():
    class Appender:
        def append(self, set_id, event, actor, detail=None):
            return {}

    ledger = explored_ledger()
    features, target = signal_data(400, 2)
    result = cf.confirm(
        ledger,
        source_for(features, target),
        alpha_confirm=F(1, 20),
        custody_log=Appender(),
    )
    assert result.custody_head is None
    assert cf.head_of(object()) is None


def test_hypotheses_without_a_column_or_with_a_direction_run_on_c():
    features, target = signal_data(600, 1)
    ledger = fresh(None, F(1, 10))
    plain = ledger_module.HypothesisSpec.create(
        name="first", family="f", statistic="spearman", null="x"
    )
    up = spec("s", "up", bivariate.GREATER)
    for s in (plain, up):
        ledger.test(ledger.register(s), features, target)
    ledger.record_control(
        controls.summarize("permuted_target", "strata", 5, (0,) * 20).to_dict()
    )
    cfeatures, ctarget = signal_data(400, 2)
    result = cf.confirm(
        ledger, source_for(cfeatures, ctarget), alpha_confirm=F(1, 20)
    )
    expected = bivariate.spearman(cfeatures["s"].to_numpy(), ctarget.to_numpy())
    expected_up = bivariate.spearman(
        cfeatures["s"].to_numpy(), ctarget.to_numpy(), bivariate.GREATER
    )
    assert [c.p_value for c in result.checks] == [expected.p, expected_up.p]
    assert expected_up.p != expected.p


def test_class_codes_sort_by_text_like_in_exploration():
    labels = pd.Series([2, 10] * 5, dtype=object)
    codes = cf.default_encoding(labels)
    assert list(codes[:2]) == [1.0, 0.0]  # "10" sorts before "2"
    assert list(explore.encode_target(labels, True)[:2]) == [1.0, 0.0]
    assert list(explore.encode_target(labels, True, classes=(2, 10))[:2]) == [
        0.0,
        1.0,
    ]


# --- exploration plan and governed run ---------------------------------------


def test_the_plan_keeps_going_after_a_skipped_column_and_names_the_null():
    frame = pd.DataFrame(
        {
            "entity": [1.0, 2.0, 3.0, 4.0],
            "text": ["a", "b", "c", "d"],
            "empty": [np.nan] * 4,
            "one": [1.0, np.nan, np.nan, np.nan],
            "x": [0.1, 0.4, 0.2, 0.9],
            "y": [3.0, 1.0, 2.0, 0.0],
        }
    )
    plan = explore.plan_exploration(frame, explore.NUMERIC, ("entity",))
    assert [s.column for s in plan.specs] == ["x", "y"]
    assert dict(plan.skipped) == {
        "entity": "structure column, not a candidate",
        "text": "not numeric: descriptive only",
        "empty": "constant or entirely missing",
        "one": "constant or entirely missing",
    }
    assert (
        plan.specs[0].null
        == "x is not monotonically associated with the target"
    )
    assert plan.specs[0].family == "bivariate" and plan.target_kind == "numeric"
    kruskal_plan = explore.plan_exploration(
        frame, explore.MULTICLASS, ("entity",)
    )
    assert kruskal_plan.specs[0].statistic == "kruskal"
    assert (
        kruskal_plan.specs[0].null
        == "x has the same distribution in every class"
    )


def test_target_kinds():
    assert explore.target_kind(pd.Series([0, 1, 1]), True) == explore.BINARY
    assert explore.target_kind(pd.Series([0, 1, 2]), True) == explore.MULTICLASS
    assert explore.target_kind(pd.Series([0, 1, 2]), False) == explore.NUMERIC


def build(rows=500, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({f"f{i}": rng.normal(size=rows) for i in range(6)})
    return frame, pd.Series(
        0.8 * (frame["f0"] + frame["f1"]) + rng.normal(size=rows)
    )


def test_a_governed_run_reports_exact_counts_and_artifacts(tmp_path):
    features, target = build()
    cfeatures, ctarget = build(400, 9)
    run = governed.run_governed(
        charter=make_charter(),
        features=features,
        target=target,
        classification=False,
        source=source_for(cfeatures, ctarget),
        run_dir=tmp_path,
        exclude=("f5",),
        rule=online_fdr.LORD_PLUS_PLUS,
        confirmation_rule=b.MultiplicityRule.BH,
    )
    summary = run.summary()
    assert summary["rule"] == "lord++" and run.confirmation.rule == "bh"
    assert run.ledger.config.rule == "lord++"
    assert [s.column for s in run.plan.specs] == ["f0", "f1", "f2", "f3", "f4"]
    assert summary["hypotheses"] == 5 and summary["tested"] == 5
    assert summary["rejected_on_x"] == 2 == run.control.observed
    assert summary["skipped_features"] == 1
    assert summary["control_scope"] == "global"
    text = (tmp_path / governed.CONTROL_FILE).read_text(encoding="utf-8")
    assert text == (
        json.dumps(run.control.to_dict(), indent=2, sort_keys=True) + "\n"
    )


def test_strata_labels_edge_cases():
    frame = pd.DataFrame({"t": [5.0, 1.0, 3.0, 2.0, 6.0, 4.0]})
    two = governed.strata_labels(frame, None, "t", blocks=2)
    assert sorted(set(two)) == [0, 1]
    assert list(np.bincount(two)) == [3, 3]
    one = governed.strata_labels(frame, None, "t", blocks=1)
    assert set(one) == {0}
    assert governed.strata_labels(frame, None, "absent") is None
    messy = pd.DataFrame({"t": ["2020", "oops", "2022", "2021"]})
    labels = governed.strata_labels(messy, None, "t", blocks=2)
    assert len(labels) == 4


# --- last round ---------------------------------------------------------------


def two_signal_ledger():
    rng = np.random.default_rng(5)
    features = pd.DataFrame(
        {"a": rng.normal(size=600), "b": rng.normal(size=600)}
    )
    target = pd.Series(features["a"] + features["b"] + rng.normal(size=600))
    ledger = fresh(None, F(1, 10))
    explore.run_exploration(
        ledger,
        features,
        target,
        explore.plan_exploration(features, explore.NUMERIC),
    )
    ledger.record_control(
        controls.summarize("permuted_target", "strata", 5, (0,) * 20).to_dict()
    )
    return ledger


def test_unconfirmed_outcomes_carry_empty_checks_the_level_and_reasons():
    for kwargs, outcome, reason in (
        (
            {"require_control": True},
            cf.NO_CONFIRMATION_SET,
            "no confirmation set exists for this run",
        ),
    ):
        ledger = two_signal_ledger()
        result = cf.confirm(ledger, None, alpha_confirm=F(1, 20), **kwargs)
        assert result.outcome == outcome and result.reason == reason
        assert result.checks == () and result.alpha_confirm == F(1, 20)
        assert result.custody_head is None
    bare = fresh(None, F(1, 10))
    features, target = signal_data()
    explore.run_exploration(
        bare,
        features,
        target,
        explore.plan_exploration(features, explore.NUMERIC),
    )
    result = cf.confirm(bare, None, alpha_confirm=F(1, 20))
    assert result.outcome == cf.NO_CONTROL
    assert (
        result.reason
        == "no negative control was recorded; no claim is promoted"
    )
    assert result.checks == () and result.alpha_confirm == F(1, 20)
    failed = explored_ledger(control_observed=0)
    result = cf.confirm(failed, None, alpha_confirm=F(1, 20))
    assert result.checks == () and result.alpha_confirm == F(1, 20)
    assert result.custody_head is None


def test_a_crash_records_the_custody_head_and_the_level():
    ledger = two_signal_ledger()
    features, target = signal_data(300, 4)
    source = source_for(features, target)
    result = cf.confirm(
        ledger, source, alpha_confirm=F(1, 20), encode=lambda y: 1 / 0
    )
    assert result.custody_head == source.log.entries[-1]["hash"]
    assert result.checks == () and result.alpha_confirm == F(1, 20)

    class Dies:
        log = sealed.OpenLog()

        def issue_token(self):
            return "t"

        def open(self, token, actor="x"):
            raise OSError("gone")

    dying = Dies()
    result = cf.confirm(two_signal_ledger(), dying, alpha_confirm=F(1, 20))
    assert result.custody_head == dying.log.entries[-1]["hash"]
    assert result.checks == () and result.alpha_confirm == F(1, 20)


def test_every_check_on_c_has_its_own_seed_index():
    ledger = two_signal_ledger()
    rng = np.random.default_rng(8)
    cfeatures = pd.DataFrame(
        {"a": rng.normal(size=400), "b": rng.normal(size=400)}
    )
    ctarget = pd.Series(cfeatures["a"] + cfeatures["b"] + rng.normal(size=400))
    result = cf.confirm(
        ledger, source_for(cfeatures, ctarget), alpha_confirm=F(1, 20)
    )
    assert [c.seed_path for c in result.checks] == [
        randomness.seed_path(7, "confirm.spearman", 0),
        randomness.seed_path(7, "confirm.spearman", 1),
    ]


def test_a_known_id_with_another_specification_is_not_registered():
    ledger = fresh()
    ledger.register(spec("s"))
    impostor = ledger_module.Hypothesis("h000001", spec("n", "other"))
    with raises(
        ledger_module.LedgerError,
        "hypothesis h000001 is not registered: registration must precede "
        "evaluation",
    ):
        ledger.test(impostor, *signal_data())


def test_notes_reach_the_stored_reason_and_survive_a_reload(tmp_path):
    path = tmp_path / ledger_module.FILE_NAME
    ledger = fresh(path)
    features, target = signal_data()
    features["flat"] = 1.0
    result = ledger.test(ledger.register(spec("flat")), features, target)
    assert result.reason == "a variable is constant"
    assert result.check.p_value == 1.0
    again = ledger_module.Ledger.load(path)
    assert next(iter(again.results.values())).reason == "a variable is constant"


def test_an_in_memory_governed_ledger_records_the_charter():
    features, target = build()
    charter = make_charter()
    run = governed.run_governed(
        charter=charter,
        features=features,
        target=target,
        classification=False,
        source=None,
        run_dir=None,
    )
    assert run.ledger.header["charter_hash"] == charter.charter_hash
    assert run.ledger.header["alpha_confirm"] == str(
        charter.error_budget.confirmation.alpha_confirm
    )
    assert run.ledger.header["seed"] == charter.seed


def test_a_governed_multiclass_run_uses_kruskal_for_every_hypothesis():
    rng = np.random.default_rng(4)
    features = pd.DataFrame(
        {"a": rng.normal(size=600), "b": rng.normal(size=600)}
    )
    target = pd.Series(
        pd.cut(features["a"], [-9, -0.5, 0.5, 9], labels=["lo", "mid", "hi"])
    ).astype(str)
    run = governed.run_governed(
        charter=make_charter(),
        features=features,
        target=target,
        classification=True,
        source=None,
        run_dir=None,
    )
    assert {s.statistic for s in run.plan.specs} == {"kruskal"}
    assert run.plan.target_kind == explore.MULTICLASS


def test_missing_values_in_y_are_dropped_too():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y = [2.0, None, 5.0, 3.0, 8.0, 9.0]
    assert bivariate.spearman(x, y).n == 5
