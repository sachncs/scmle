import json

import pytest
import scipy.stats

from benchmarks.qualification import f11_certify
from benchmarks.qualification import f11_generators
from scmle.data.detectors import certificates


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
        low, high = certificates.wilson_interval(hits, entry["replicates"])
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
