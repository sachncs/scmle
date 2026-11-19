import json

import pytest
import scipy.stats

from benchmarks.qualification import f11_certify
from benchmarks.qualification import f11_generators
from scmle.assurance.detectors import certificates


def test_the_committed_f11_certificates_are_current():
    report = f11_certify.check()
    assert report["ok"], report["problems"]


def test_every_gate_has_a_runner_and_a_code_hash_source():
    assert (
        set(f11_certify.GATES)
        == set(f11_certify.RUNNERS)
        == set(f11_certify.sources())
    )
    assert len(f11_certify.current_digests()) == len(f11_certify.GATES)


def test_a_changed_gate_makes_its_certificate_stale():
    document = json.loads(f11_certify.STORE_PATH.read_text(encoding="utf-8"))
    document["gates"]["B1"]["code_hash"] = "0" * 64
    assert any("B1: stale" in p for p in f11_certify.problems(document))
    del document["gates"]["A1"]
    assert any(
        "A1: no certificate" in p for p in f11_certify.problems(document)
    )


def test_a_missing_file_is_a_problem(monkeypatch, tmp_path):
    monkeypatch.setattr(f11_certify, "STORE_PATH", tmp_path / "none.json")
    assert f11_certify.problems() == ["f11_certificates.json is missing"]
    assert f11_certify.descriptive_only() == list(f11_certify.GATES)


def test_stored_floors_are_recomputed_from_counts():
    document = json.loads(f11_certify.STORE_PATH.read_text(encoding="utf-8"))
    gates = document["gates"]
    b1 = gates["B1"]
    assert b1["rate"] == b1["ever_promote"] / b1["replicates"]
    assert b1["passed"] == (
        b1["rate"] <= b1["bound"]
        and b1["class_agreement"]["agree"] == b1["class_agreement"]["runs"]
    )
    for key in ("B2", "B4", "B7"):
        entry = gates[key]
        if entry.get("not_run"):
            continue
        hits = next(
            entry[name]
            for name in ("promotions", "false_rollbacks", "false_promotions")
            if name in entry
        )
        low, high = certificates.bound(hits, entry["replicates"])
        assert entry["interval"] == pytest.approx([low, high])
        assert entry["passed"] == (hits / entry["replicates"] <= entry["bound"])


def test_the_pre_registered_floors_are_in_the_docstring():
    text = f11_certify.__doc__
    for gate in ("A1 ", "A3 ", "A5 ", "A7 ", "A8 ", "B1 ", "B6 ", "B9 "):
        assert gate in text
    assert "not retuned" in text


def test_certification_and_development_seeds_do_not_overlap():
    largest = max(runs for ignored, runs in f11_certify.RUNNERS.values())
    assert largest <= f11_certify.DEV_SEED_BASE


def test_generators_have_the_stated_truth():
    import numpy

    assert f11_generators.target_value() == pytest.approx(0.425)
    ts = numpy.array([0.1, 0.3])
    nb = f11_generators.binormal_nb(0.2, 1.5, 0.8, ts)
    tpr = 1 - scipy.stats.norm.cdf(0.8 - 1.5)
    fpr = 1 - scipy.stats.norm.cdf(0.8)
    expected = 0.2 * tpr - 0.8 * fpr * (ts / (1 - ts))
    assert nb == pytest.approx(expected)
    rng = numpy.random.default_rng(0)
    score, y, groups = f11_generators.fairness_world(rng, 400_000, 0.1)
    acts = score >= 0.0
    for g, rate in ((0, 0.7), (1, 0.6)):
        mask = (groups == g) & (y == 1)
        assert acts[mask].mean() == pytest.approx(rate, abs=0.01)


def test_the_a4_null_worlds_sit_on_the_decision_boundary():
    import numpy

    grid = numpy.linspace(*f11_certify.RANGE, 21)
    for share, precision, rest in f11_certify.BOUNDARY.values():
        truth = f11_generators.acted_differences(share, precision, rest, grid)
        assert truth.min() == pytest.approx(0.0, abs=1e-12)
        assert share * precision + (1 - share) * rest == pytest.approx(
            f11_certify.PREVALENCE
        )
    independent = f11_generators.acted_differences(0.2, 0.2, 0.2, grid)
    assert independent.min() < -0.05


def test_the_acted_world_has_the_stated_rates():
    import numpy

    rng = numpy.random.default_rng(0)
    y, acts = f11_generators.acted_world(rng, 400_000, 0.4, 0.4, 0.02)
    assert acts.mean() == pytest.approx(0.4, abs=0.005)
    assert y[acts == 1].mean() == pytest.approx(0.4, abs=0.005)
    assert y[acts == 0].mean() == pytest.approx(0.02, abs=0.002)


def test_gate_a4_decides_each_boundary_setting_with_the_exact_bound():
    from benchmarks import acceptance

    variables = acceptance.load()
    rule = f11_certify.FALSE_USEFUL
    alpha = variables["alpha"].value
    result = f11_certify.gate_a4(3, f11_certify.DEV_SEED_BASE)
    assert set(result["null"]) == {
        f"{name}/{n}" for name in f11_certify.BOUNDARY for n in (2000, 20000)
    }
    for entry in result["null"].values():
        assert entry["runs"] == 3
        assert entry["meets_floor"] == rule.passes(
            entry["useful"], entry["runs"], alpha
        )
    # three worlds are below the 34 a ceiling of 0.0845 needs to pass at all
    assert result["minimum"] == rule.minimum(alpha) == 34
    assert result["passed"] is False


def test_hook_agreement_counts_only_deciding_worlds():
    def scenario(deciding, agree):
        return {"worlds": 100, "deciding": deciding, "agree": agree}

    silent = {k: scenario(0, 0) for k in ("a", "b", "c", "d")}
    verdict = f11_certify.agreement(silent)
    assert verdict["deciding"] == 0 and verdict["passed"] is False
    assert verdict["minimum"] == 59
    enough = {"a": scenario(40, 40), "b": scenario(30, 30)}
    assert f11_certify.agreement(enough)["passed"] is True
    one_off = {"a": scenario(40, 40), "b": scenario(30, 29)}
    assert f11_certify.agreement(one_off)["passed"] is False
    few = {"a": scenario(30, 30)}
    assert f11_certify.agreement(few)["passed"] is False


def test_stored_a4_and_b9_verdicts_follow_from_their_counts():
    document = json.loads(f11_certify.STORE_PATH.read_text(encoding="utf-8"))
    a4 = document["gates"]["A4"]
    from benchmarks import acceptance

    variables = acceptance.load()
    alpha = variables["alpha"].value
    rule = f11_certify.FALSE_USEFUL
    for entry in a4["null"].values():
        assert entry["meets_floor"] == rule.passes(
            entry["useful"], entry["runs"], alpha
        )
    assert a4["passed"] == all(e["meets_floor"] for e in a4["null"].values())
    b9 = document["gates"]["B9"]
    for entry in b9["through_hook"].values():
        assert entry["agree"] <= entry["deciding"] <= entry["valid"]
        assert entry["hook_decisions"] <= entry["deciding"]
    verdict = f11_certify.agreement(b9["through_hook"])
    assert b9["hook_agreement"] == verdict
    runs = b9["replicates"]
    promote = acceptance.sized(f11_certify.ALPHA_PROMOTE)
    rollback = acceptance.sized(f11_certify.ALPHA_ROLLBACK)
    assert b9["alpha"] == alpha
    for name in ("no_effect", "below_delta_min"):
        hits = b9[name]["false_promotions"]
        assert b9[name]["upper"] == promote.limit(hits, runs, alpha)
    hits = b9["false_rollback"]["rollbacks"]
    assert b9["false_rollback"]["upper"] == rollback.limit(hits, runs, alpha)
    assert b9["passed"] == (
        verdict["passed"]
        and promote.passes(b9["no_effect"]["false_promotions"], runs, alpha)
        and promote.passes(
            b9["below_delta_min"]["false_promotions"], runs, alpha
        )
        and rollback.passes(hits, runs, alpha)
    )


def test_the_hook_is_compared_in_worlds_where_the_e_process_acts():
    document = json.loads(f11_certify.STORE_PATH.read_text(encoding="utf-8"))
    through = document["gates"]["B9"]["through_hook"]
    assert through["clear_gain"]["deciding"] > 0
    assert through["clear_loss"]["deciding"] > 0


def test_the_decision_package_has_no_custody_or_vault_imports():
    assert f11_certify.forbidden_imports() == []


def test_hillstrom_gates_report_not_run_without_the_file(monkeypatch):
    monkeypatch.delenv(f11_certify.HILLSTROM_ENV, raising=False)
    for runner in (
        f11_certify.gate_a6,
        f11_certify.gate_b7,
        f11_certify.gate_b8,
    ):
        result = runner(2)
        assert result["passed"] is False and "not_run" in result


def test_a_wrong_hash_is_not_accepted(monkeypatch, tmp_path):
    path = tmp_path / "hillstrom.csv"
    path.write_text("segment\nNo E-Mail\n")
    monkeypatch.setenv(f11_certify.HILLSTROM_ENV, str(path))
    assert f11_certify.load_hillstrom() is None


def test_check_f11_is_wired_into_the_harness(capsys):
    from benchmarks.qualification import __main__

    assert __main__.main(["--check", "f11"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
