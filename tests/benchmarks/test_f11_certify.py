import json

import pytest

from benchmarks.qualification import f11_certify as cert
from benchmarks.qualification import f11_generators as worlds
from scmle.data.detectors import certificates


def test_the_committed_f11_certificates_are_current():
    report = cert.check()
    assert report["ok"], report["problems"]


def test_every_gate_has_a_runner_and_a_code_hash_source():
    assert set(cert.GATES) == set(cert.RUNNERS) == set(cert.sources())
    assert len(cert.current_digests()) == len(cert.GATES)


def test_a_changed_gate_makes_its_certificate_stale():
    document = json.loads(cert.STORE_PATH.read_text(encoding="utf-8"))
    document["gates"]["B1"]["code_hash"] = "0" * 64
    assert any("B1: stale" in p for p in cert.problems(document))
    del document["gates"]["A1"]
    assert any("A1: no certificate" in p for p in cert.problems(document))


def test_a_missing_file_is_a_problem(monkeypatch, tmp_path):
    monkeypatch.setattr(cert, "STORE_PATH", tmp_path / "none.json")
    assert cert.problems() == ["f11_certificates.json is missing"]
    assert cert.descriptive_only() == list(cert.GATES)


def test_stored_floors_are_recomputed_from_counts():
    document = json.loads(cert.STORE_PATH.read_text(encoding="utf-8"))
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
    text = cert.__doc__
    for gate in ("A1 ", "A3 ", "A5 ", "A7 ", "A8 ", "B1 ", "B6 ", "B9 "):
        assert gate in text
    assert "not retuned" in text


def test_certification_and_development_seeds_do_not_overlap():
    largest = max(runs for ignored, runs in cert.RUNNERS.values())
    assert largest <= cert.DEV_SEED_BASE


def test_generators_have_the_stated_truth():
    import numpy as np

    assert worlds.target_value() == pytest.approx(0.425)
    ts = np.array([0.1, 0.3])
    nb = worlds.binormal_nb(0.2, 1.5, 0.8, ts)
    tpr = 1 - worlds.Z.cdf(0.8 - 1.5)
    fpr = 1 - worlds.Z.cdf(0.8)
    expected = 0.2 * tpr - 0.8 * fpr * (ts / (1 - ts))
    assert nb == pytest.approx(expected)
    rng = np.random.default_rng(0)
    score, y, groups = worlds.fairness_world(rng, 400_000, 0.1)
    acts = score >= 0.0
    for g, rate in ((0, 0.7), (1, 0.6)):
        mask = (groups == g) & (y == 1)
        assert acts[mask].mean() == pytest.approx(rate, abs=0.01)


def test_the_decision_package_has_no_custody_or_vault_imports():
    assert cert.forbidden_imports() == []


def test_hillstrom_gates_report_not_run_without_the_file(monkeypatch):
    monkeypatch.delenv(cert.HILLSTROM_ENV, raising=False)
    for runner in (cert.gate_a6, cert.gate_b7, cert.gate_b8):
        result = runner(2)
        assert result["passed"] is False and "not_run" in result


def test_a_wrong_hash_is_not_accepted(monkeypatch, tmp_path):
    path = tmp_path / "hillstrom.csv"
    path.write_text("segment\nNo E-Mail\n")
    monkeypatch.setenv(cert.HILLSTROM_ENV, str(path))
    assert cert.load_hillstrom() is None


def test_check_f11_is_wired_into_the_harness(capsys):
    from benchmarks.qualification import __main__ as harness

    assert harness.main(["--check", "f11"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
