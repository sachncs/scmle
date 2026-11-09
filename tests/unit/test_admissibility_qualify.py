"""Wiring of the admissibility tests and data detectors onto the exploration set."""

import numpy as np
import pandas as pd
import pytest

from scmle.contracts import qualification
from scmle.data.admissibility import dag as dag_module
from scmle.data.admissibility import qualify
from scmle.data.admissibility import screen
from scmle.data.charter import causal as causal_module
from scmle.data.detectors import certificates
from scmle.data.ledger import ledger as ledger_module
from scmle.data.ledger import online_fdr
from tests.unit.test_charter_contracts import make_charter


def leaky(rows=400, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    target = pd.Series(
        (frame.a + rng.normal(size=rows) > 0).astype(int), name="y"
    )
    frame["copy"] = target.to_numpy() * 1.0
    return frame, target


def test_a_missing_store_is_descriptive_and_says_so(tmp_path):
    loaded = qualify.load_store(qualify.F5_STORE, tmp_path)

    assert loaded.store is None
    assert "descriptive" in loaded.note


def test_an_unreadable_store_is_descriptive_and_says_so(tmp_path):
    (tmp_path / qualify.F5_STORE).write_text("not json")

    loaded = qualify.load_store(qualify.F5_STORE, tmp_path)

    assert loaded.store is None
    assert "unreadable" in loaded.note


def test_the_shipped_stores_are_found_from_a_checkout():
    assert qualify.load_store(qualify.F5_STORE).store is not None
    assert screen.shipped_store() is not None


def test_without_certificates_nothing_is_quarantined(tmp_path):
    features, target = leaky()

    found = qualify.run(
        features,
        target,
        kind="predictive",
        classification=True,
        seed=0,
        directory=tmp_path,
    )

    assert found.leakage_gate is certificates.GateStatus.DESCRIPTIVE
    assert "copy" in found.suspects
    assert found.quarantined == {}
    assert found.table is not None
    assert not found.table.names(qualify.rules.Admission.QUARANTINE)
    assert found.f5_store is None
    assert any("unusable" in note for note in found.notes)
    assert all(
        s is certificates.GateStatus.DESCRIPTIVE
        for by_defect in found.detectors.status.values()
        for s in by_defect.values()
    )


def test_the_charter_run_uses_the_same_screen_as_a_run_without_one():
    features, target = leaky()

    found = qualify.run(
        features, target, kind="predictive", classification=True, seed=0
    )
    direct = screen.screen_table(
        features.assign(y=target),
        "y",
        "classification",
        list(features.columns),
        0,
    )

    assert found.leakage is not None
    assert found.leakage.to_dict() == direct.to_dict()
    assert [f.feature for f in found.leakage.suspects] == [
        f.feature for f in direct.suspects
    ]


def test_a_certified_suspect_is_quarantined_and_ledgered():
    features, target = leaky()

    found = qualify.run(
        features, target, kind="predictive", classification=True, seed=0
    )

    assert found.leakage_gate is certificates.GateStatus.GATING
    assert set(found.quarantined) == {"copy"}
    assert "leakage suspicion" in found.quarantined["copy"]
    sources = {source for source, check in found.checks}
    assert sources == {screen.DETECTOR_ID}
    by_id = {check.check_id: check for source, check in found.checks}
    assert by_id["leakage.screen.skill:copy"].decision is (
        qualification.Decision.QUARANTINE
    )
    assert by_id["leakage.screen.skill:a"].decision is (
        qualification.Decision.PASS
    )


def test_an_uncertified_mechanism_only_warns(monkeypatch):
    features, target = leaky()
    monkeypatch.setattr(
        screen,
        "gate_statuses",
        lambda *args, **kwargs: dict.fromkeys(
            screen.MECHANISM_DEFECTS, certificates.GateStatus.DESCRIPTIVE
        ),
    )

    found = qualify.run(
        features, target, kind="predictive", classification=True, seed=0
    )

    assert found.suspects == {"copy": True}
    assert found.quarantined == {}
    copy = next(
        c for source, c in found.checks if c.check_id.endswith("skill:copy")
    )
    assert copy.decision is qualification.Decision.WARN


def test_a_regression_target_is_never_gated():
    features, target = leaky()
    target = target.astype(float) + np.arange(len(target)) * 1e-3

    found = qualify.run(
        features, target, kind="predictive", classification=False, seed=0
    )

    assert found.leakage_gate is certificates.GateStatus.DESCRIPTIVE
    assert found.quarantined == {}


def test_a_causal_estimand_without_a_graph_skips_admissibility():
    features, target = leaky(rows=120)

    found = qualify.run(
        features, target, kind="causal", classification=True, seed=0
    )

    assert found.table is None and found.leakage is None
    assert found.checks == ()
    assert any("graph" in note for note in found.notes)


def causal_world(rows=500, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    t = (rng.uniform(size=rows) < 1 / (1 + np.exp(-x1))).astype(float)
    after = t + rng.normal(size=rows)  # a descendant of the treatment
    y = pd.Series(2 * t + x1 + rng.normal(size=rows), name="y")
    return pd.DataFrame({"x1": x1, "t": t, "after": after}), y


def spec(assumptions):
    return causal_module.CausalSpec(
        treatment="t",
        outcome="y",
        estimand="ATE",
        assumptions=assumptions,
        strategies=("backdoor_aipw",),
    )


def graph():
    return dag_module.Dag(
        ("after", "t", "x1", "y"),
        (("t", "after"), ("t", "y"), ("x1", "t"), ("x1", "y")),
        (),
        "test",
        "none",
    )


def test_a_causal_estimand_with_a_graph_applies_the_graph_rules():
    features, target = causal_world()

    found = qualify.run(
        features,
        target,
        kind="causal",
        classification=False,
        seed=0,
        causal=spec(("unconfoundedness_given:x1", "positivity")),
        dag=graph(),
    )

    assert found.table is not None and found.table.kind == "causal"
    assert set(found.quarantined) == {"after"}
    assert "descendant of the treatment" in found.quarantined["after"]
    assert found.dag["dag_hash"] == graph().digest()
    assert found.dag["adjustment_members_refused"] == {}
    assert found.dag["falsification"]["ran"] is True
    assert not any("graph" in note for note in found.notes)


def test_a_refused_adjustment_member_is_reported_and_called_ineligible():
    features, target = causal_world()

    found = qualify.run(
        features,
        target,
        kind="causal",
        classification=False,
        seed=0,
        causal=spec(("unconfoundedness_given:x1,after", "positivity")),
        dag=graph(),
    )

    assert set(found.dag["adjustment_members_refused"]) == {"after"}
    assert any("backdoor_aipw is ineligible" in n for n in found.notes)


def test_a_graph_that_names_missing_columns_cannot_be_tested_but_runs():
    features, target = causal_world()
    wide = dag_module.Dag(
        ("t", "u", "x1", "y"),
        (("t", "y"), ("u", "y"), ("x1", "t")),
        (),
        "test",
        "none",
    )

    found = qualify.run(
        features,
        target,
        kind="causal",
        classification=False,
        seed=0,
        causal=spec(("unconfoundedness_given:x1", "positivity")),
        dag=wide,
    )

    assert found.dag["falsification"]["ran"] is False
    assert any("could not be tested" in n for n in found.notes)


def test_many_candidates_without_a_declared_set_say_the_claim_is_attested_at_best():
    rng = np.random.default_rng(1)
    columns = {f"c{i}": rng.normal(size=300) for i in range(17)}
    columns["t"] = rng.integers(0, 2, 300).astype(float)
    features = pd.DataFrame(columns)
    target = pd.Series(rng.normal(size=300), name="y")
    nodes = tuple(sorted([*columns, "y"]))
    edges = tuple(sorted([("t", "y"), *[(f"c{i}", "y") for i in range(17)]]))

    found = qualify.run(
        features,
        target,
        kind="causal",
        classification=False,
        seed=0,
        causal=spec(("positivity",)),
        dag=dag_module.Dag(nodes, edges, (), "test", "none"),
    )

    assert any("attested at best" in n for n in found.notes)


def test_structure_columns_are_not_candidates():
    features, target = leaky(rows=120)
    features["g"] = np.arange(len(target)) % 6

    found = qualify.run(
        features,
        target,
        kind="predictive",
        classification=True,
        seed=0,
        structure=("g",),
        group_column="g",
    )

    assert found.table is not None
    assert "g" not in [d.name for d in found.table.decisions]


def make_ledger(path=None):
    charter = make_charter()
    return ledger_module.Ledger.create(
        path,
        charter_hash=charter.charter_hash,
        config=online_fdr.RuleConfig(
            online_fdr.ELOND, charter.error_budget.exploration.initial
        ),
        alpha_confirm=charter.error_budget.confirmation.alpha_confirm,
        seed=0,
    )


def sample_check(set_id=qualification.SetId.EXPLORATION):
    return qualification.Check(
        check_id="leakage.skill:copy",
        null_hypothesis="no skill",
        statistic=0.9,
        p_value=0.01,
        interval=None,
        effect=0.9,
        n=100,
        decision=qualification.Decision.WARN,
        set_id=set_id,
        seed_path=(0, "leakage", 0),
        tier=qualification.Tier.UNVERIFIED,
    )


def test_a_diagnostic_spends_no_wealth_and_survives_reload(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = make_ledger(path)
    before = ledger.wealth

    ledger.record_diagnostic(sample_check(), "f4-leakage-skill")
    ledger.freeze()

    assert ledger.wealth == before
    assert not ledger.hypotheses
    reloaded = ledger_module.Ledger.load(path)
    assert reloaded.diagnostics == [sample_check()]
    assert reloaded.head == ledger.head


def test_a_diagnostic_cannot_follow_the_freeze():
    ledger = make_ledger()
    ledger.freeze()

    with pytest.raises(ledger_module.LedgerError, match="frozen"):
        ledger.record_diagnostic(sample_check(), "f4-leakage-skill")


def test_a_diagnostic_that_read_c_is_refused():
    ledger = make_ledger()
    check = sample_check(qualification.SetId.CONFIRMATION)

    with pytest.raises(ledger_module.LedgerError, match="exploration"):
        ledger.record_diagnostic(check, "f4-leakage-skill")
