"""Data-detector certification harness and the committed detector certificates."""

import pytest

from benchmarks.qualification import __main__
from benchmarks.qualification import f5
from benchmarks.qualification import standins
from scmle.data.detectors import certificates
from scmle.data.detectors import duplicates
from scmle.data.detectors import registry


@pytest.fixture(scope="module")
def committed():
    return certificates.Store.load(f5.CERT_PATH)


def test_policy_is_the_pre_declared_one():
    assert f5.POLICY.recall_floor == 0.80
    assert f5.POLICY.false_alarm_ceiling == 0.05
    assert f5.POLICY.confidence == 0.95
    assert set(f5.ROW_SEEDS).isdisjoint(range(100, 200))


def test_committed_certificates_are_current_and_consistent(committed):
    assert f5.problems(committed) == []
    assert f5.audit_problems(committed) == []
    assert {c.detector_id for c in committed.certificates} == set(
        registry.SPECS
    )
    for item in committed.certificates:
        types = {o.certificate.defect_type for o in item.outcomes}
        assert types == set(registry.SPECS[item.detector_id].defect_types)


def test_committed_certificates_meet_the_planted_count_requirement(committed):
    for row in f5.summary(committed):
        assert row["planted"] >= 200, row


def test_gating_lists_only_passing_outcomes(committed):
    registered = {(g[0], g[2]) for g in f5.gating(committed)}
    for row in f5.summary(committed):
        key = (row["detector"], row["defect_type"])
        assert (key in registered) is bool(row["gates"])


def test_a_changed_detector_makes_its_certificate_stale(committed, monkeypatch):
    monkeypatch.setattr(registry.Identity, "code_hash", lambda self: "f" * 64)
    assert len(f5.problems(committed)) == len(registry.SPECS)
    assert f5.gating(committed) == []


def test_certify_all_is_deterministic_and_certifies_row_detectors():
    first = f5.certify_all(seeds=(0,), only=["f5-sentinel", "f5-duplicates"])
    second = f5.certify_all(seeds=(0,), only=["f5-sentinel", "f5-duplicates"])
    assert first.certificates == second.certificates
    assert {c.detector_id for c in first.certificates} == {
        "f5-sentinel",
        "f5-duplicates",
    }
    summary = {(r["detector"], r["defect_type"]): r for r in f5.summary(first)}
    assert summary[("f5-sentinel", "sentinel")]["recall_lower"] > 0.9
    assert summary[("f5-duplicates", "duplicate_exact")]["gates"]


def test_missingness_certificate_counts_replicates():
    frame = standins.demo_frame()
    certificate = f5.certify_missingness(frame, seeds=(0, 1, 2))
    assert certificate.detector_id == "f5-missingness"
    for outcome in certificate.outcomes:
        assert outcome.certificate.planted == 3
        assert outcome.certificate.clean_rows == 3
        assert outcome.certificate.detected == 3
    assert certificate.policy == f5.POLICY


def test_renamed_plan_changes_only_the_certificate_key():
    plan = f5.row_plans()["f5-label-noise"][0]
    injection = plan.plant(standins.demo_frame(), 0)
    assert plan.defect_type == injection.defect_type == "label_flip_symmetric"


def test_cut_selection_and_the_shipped_default():
    chosen = f5.select_cut(standins.demo_frame())
    assert chosen == 0.1
    assert pytest.approx(chosen / 10) == duplicates.DEFAULT_CUT


def test_check_command_passes_on_committed_files(capsys):
    assert __main__.check_certificates() == 0
    assert '"f5_problems": []' in capsys.readouterr().out
