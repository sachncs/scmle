"""F4 certificates: the stored file is current, honest and reproducible."""

import json

import pandas as pd
import pytest

from benchmarks.qualification import f4_certify as f4
from scmle.data.admissibility import asof
from scmle.data.admissibility import leakage
from scmle.data.admissibility import reconciliation
from scmle.data.detectors import certificates


def stored():
    return certificates.CertificateStore.load(f4.STORE_PATH)


def test_the_stored_certificates_are_for_the_current_code():
    store = stored()
    digests = f4.current_digests()
    for detector, digest in digests.items():
        state = store.state(detector, digest)
        assert state is certificates.CertificateState.CURRENT, detector


def test_every_required_defect_type_has_an_outcome_on_file():
    store = stored()
    digests = f4.current_digests()
    for detector, defects in f4.required_defects().items():
        (cert,) = store.current(detector, digests[detector])
        assert {o.certificate.defect_type for o in cert.outcomes} == set(
            defects
        )


def test_stale_and_missing_certificates_are_reported():
    empty = certificates.CertificateStore()
    found = f4.problems(empty)
    assert found and all(item.endswith(": missing") for item in found)
    store = stored()
    forged = {
        d: (
            "0" * 64,
            next(iter(defects)),
        )
        for d, defects in f4.required_defects().items()
    }
    report = certificates.audit(store, forged)
    assert set(report.stale) == set(forged)


def test_planned_sizes_follow_the_phase_floors():
    assert f4.NULL_REPLICATES >= 1000
    assert f4.PLANT_REPLICATES >= 200
    assert f4.LEAK_POLICY.recall_floor == 0.9
    assert f4.LEAK_POLICY.false_alarm_ceiling == 0.05


def test_measurements_agree_with_the_certificate():
    document = json.loads(f4.MEASUREMENTS_PATH.read_text(encoding="utf-8"))
    store = stored()
    (cert,) = store.current(leakage.DETECTOR_ID, leakage.code_digest())
    for outcome in cert.outcomes:
        c = outcome.certificate
        assert c.false_alarms == document["suspect_false_alarms"]
        assert c.detected == document["detected"][c.defect_type]
        assert c.clean_rows == document["null_replicates"]
        assert c.planted == document["plant_replicates_per_type"]
    assert (
        document["settings"]["skill_ceiling"] == leakage.CERTIFIED.skill_ceiling
    )
    assert document["alpha"] == leakage.CERTIFIED.alpha


def test_the_99_percent_band_is_the_normal_band():
    low, high = f4.band_99(0.05, 1000)
    assert low == pytest.approx(0.05 - 2.5758293035489004 * 0.0068920243760451)
    assert high == pytest.approx(0.05 + 2.5758293035489004 * 0.0068920243760451)


def test_plant_leak_produces_the_four_defect_columns():
    frame = f4.breast_cancer().head(120)
    target = frame[f4.TARGET]
    copy = f4.plant_leak(frame, "leak_copy", 0)
    assert (copy == target).all()
    noisy = f4.plant_leak(frame, "leak_noisy_copy", 0)
    assert 0 <= (noisy != target).sum() <= 12
    threshold = f4.plant_leak(frame, "leak_threshold", 0)
    assert ((threshold > 0.5).astype(int) == target).mean() > 0.9
    missing = f4.plant_leak(frame, "leak_missing_for_class", 0)
    assert missing.isna().any() and missing.notna().any()
    assert (missing.isna() == (target == target.iloc[0])).all()
    assert set(leakage.LEAK_DEFECTS) == {
        "leak_copy",
        "leak_noisy_copy",
        "leak_threshold",
        "leak_missing_for_class",
    }


def test_a_planted_leak_replicate_is_detected_and_a_null_one_is_not():
    suspect, p = f4.plant_replicate("leak_copy", 1)
    assert suspect and p == pytest.approx(1 / 200)
    suspect, p = f4.null_replicate(1)
    assert not suspect
    assert f4.replicate("leak_copy", 1) == f4.plant_replicate("leak_copy", 1)
    assert f4.replicate("", 1) == f4.null_replicate(1)


def test_asof_world_is_clean_and_overwrites_are_found_exactly():
    rows, log = f4.asof_world(0)
    detect = f4.asof_detector(log)
    assert not detect(rows).any()
    plan = f4.overwrite_plan(rows)
    planted = plan.plant(rows, 0)
    flags = detect(planted.frame)
    assert flags.sum() == planted.row_mask.sum() > 0
    assert (flags == planted.row_mask).all()


def test_gap_world_is_clean_and_unexplained_gaps_are_found_exactly():
    frame = f4.gap_frame(0)
    assert (frame["explained"] != 0).sum() > 0
    assert not f4.gap_detector(frame).any()
    planted = f4.gap_plan().plant(frame, 0)
    flags = f4.gap_detector(planted.frame)
    assert flags.sum() == planted.row_mask.sum() > 0
    assert (flags == planted.row_mask).all()


def test_the_cheap_certificates_reproduce_and_pass():
    first = f4.certify_asof()
    again = f4.certify_asof()
    assert first == again
    for cert in (first, f4.certify_reconciliation()):
        (outcome,) = cert.outcomes
        assert outcome.passed
        assert outcome.certificate.false_alarms == 0
        assert outcome.certificate.recall == 1.0
        assert outcome.recall_interval[0] >= 0.9


def test_detector_code_hashes_cover_the_whole_module():
    assert f4.current_digests()[asof.DETECTOR_ID] == certificates.code_hash(
        asof
    )
    assert f4.current_digests()[reconciliation.DETECTOR_ID] == (
        certificates.code_hash(reconciliation)
    )
    assert isinstance(f4.asof_world(0)[0], pd.DataFrame)


def test_the_measured_verdicts_are_the_ones_the_gates_rely_on():
    store = stored()
    digests = f4.current_digests()
    (cert,) = store.current(leakage.DETECTOR_ID, digests[leakage.DETECTOR_ID])
    verdicts = {o.certificate.defect_type: o.passed for o in cert.outcomes}
    # Measured, not tuned: a 2 percent noisy copy sits at the 0.95 ceiling
    # and its recall lower bound (0.66) misses the 0.9 floor.
    assert verdicts == {
        "leak_copy": True,
        "leak_noisy_copy": False,
        "leak_threshold": True,
        "leak_missing_for_class": True,
    }
    # So the skill test stays descriptive: gating needs every leak type.
    assert (
        leakage.gate_status(store, leakage.CERTIFIED)
        is certificates.GateStatus.DESCRIPTIVE
    )
    for detector in (asof.DETECTOR_ID, reconciliation.DETECTOR_ID):
        (other,) = store.current(detector, digests[detector])
        assert all(o.passed for o in other.outcomes)
    assert f4.problems(store) == []
    assert f4.audit_labels(store)["uncertified"] == [
        "f4-leakage-skill/leak_noisy_copy"
    ]
