"""F6 certificates: the stored file is current, honest and reproducible."""

import json

import numpy as np
import pytest
from scipy import stats

from benchmarks.qualification import f6_certify as f6
from scmle.data.detectors import certificates
from scmle.data.robustness import shift
from scmle.data.robustness import stability


def stored():
    return certificates.CertificateStore.load(f6.STORE_PATH)


def measurements():
    return json.loads(f6.MEASUREMENTS_PATH.read_text(encoding="utf-8"))


def test_the_stored_certificates_are_for_the_current_code():
    store = stored()
    for detector, digest in f6.current_digests().items():
        state = store.state(detector, digest)
        assert state is certificates.CertificateState.CURRENT, detector
    assert f6.problems(store) == []


def test_every_required_defect_type_has_an_outcome_on_file():
    store = stored()
    digests = f6.current_digests()
    for detector, defects in f6.required_defects().items():
        (cert,) = store.current(detector, digests[detector])
        assert {o.certificate.defect_type for o in cert.outcomes} == set(
            defects
        )


def test_missing_and_stale_certificates_are_reported():
    found = f6.problems(certificates.CertificateStore())
    assert found and all(item.endswith(": missing") for item in found)
    forged = {
        detector: ("0" * 64, defects[0])
        for detector, defects in f6.required_defects().items()
    }
    assert set(certificates.audit(stored(), forged).stale) == set(forged)


def test_the_floors_were_fixed_before_measuring():
    assert (
        certificates.CertificationPolicy(0.80, 0.075, 0.95) == f6.SHIFT_POLICY
    )
    assert (
        certificates.CertificationPolicy(0.80, 0.10, 0.95)
        == f6.STABILITY_POLICY
    )
    assert f6.SHIFT_NULL >= 1000 and f6.SHIFT_ROWS == 2000
    assert f6.PLANTED_AUC == 0.60


def test_development_and_certification_seeds_do_not_overlap():
    largest = max(
        f6.SHIFT_NULL, f6.SHIFT_PLANTED, f6.STABLE_REPS, f6.PLANT_REPS
    )
    assert largest < f6.DEV_SEED_BASE
    assert measurements()["development_seed_base"] == f6.DEV_SEED_BASE
    assert measurements()[shift.C2ST_ID]["seed_offset"] == 0


def test_measurements_agree_with_the_certificates():
    document = measurements()
    store = stored()
    (cert,) = store.current(shift.C2ST_ID, shift.code_digest())
    (outcome,) = cert.outcomes
    row = document[shift.C2ST_ID]
    assert outcome.certificate.false_alarms == row["decision_false_alarms"]
    assert outcome.certificate.detected == round(
        row["power"] * row["planted_replicates"]
    )
    (cert,) = store.current(stability.DETECTOR_ID, stability.code_digest())
    by_type = {o.certificate.defect_type: o.certificate for o in cert.outcomes}
    for kind, counts in document[stability.DETECTOR_ID]["by_defect"].items():
        assert by_type[kind].detected == counts["detected"]
        assert by_type[kind].planted == counts["planted"]
    clean = document[stability.DETECTOR_ID]
    assert by_type[stability.DEFECT_NULL].false_alarms == (
        clean["genuine_effects"] - clean["genuine_labelled_stable"]
    )


def test_the_size_by_p_value_alone_is_in_the_99_percent_band():
    nulls = {shift.C2ST_ID: f6.SHIFT_NULL, shift.MMD_ID: f6.MMD_NULL}
    for test, count in nulls.items():
        row = measurements()[test]
        assert row["size_in_band"] is True
        assert row["null_replicates"] == count
        assert row["band_99"] == pytest.approx(
            list(f6.band_99(shift.ALPHA, count))
        )


def test_an_uncertified_check_is_descriptive_not_an_error():
    store = stored()
    labels = f6.audit_labels(store)
    assert labels["missing"] == [] and labels["stale"] == []
    for item in labels["uncertified"]:
        detector, defect = item.split("/")
        status = store.status(detector, f6.current_digests()[detector], defect)
        assert status is certificates.GateStatus.DESCRIPTIVE


def test_the_planted_shift_has_the_stated_bayes_auc():
    delta = f6.shift_delta()
    assert stats.norm.cdf(delta / np.sqrt(2.0)) == pytest.approx(0.60)
    first, second = f6.shift_samples(True, 0)
    assert second["x0"].mean() - first["x0"].mean() == pytest.approx(
        delta, abs=0.1
    )
    null_first, null_second = f6.shift_samples(False, 0)
    assert abs(null_second["x0"].mean() - null_first["x0"].mean()) < 0.15


def test_the_semi_synthetic_worlds_plant_what_they_say():
    clean_x, clean_y = f6.stability_world(f6.CLEAN, 0)
    seen = np.isfinite(clean_x)
    assert 0.05 < 1 - seen.mean() < 0.3
    assert stats.spearmanr(clean_x[seen], clean_y[seen]).statistic > 0.3
    null_x, null_y = f6.stability_world(stability.DEFECT_NULL, 0)
    assert (
        abs(stats.spearmanr(null_x, null_y, nan_policy="omit").statistic) < 0.2
    )
    half = f6.STABILITY_ROWS // 2
    rev_x, rev_y = f6.stability_world(stability.DEFECT_REVERSAL, 0)
    first = np.isfinite(rev_x[:half])
    second = np.isfinite(rev_x[half:])
    assert (
        stats.spearmanr(rev_x[:half][first], rev_y[:half][first]).statistic
        > 0.2
    )
    assert (
        stats.spearmanr(rev_x[half:][second], rev_y[half:][second]).statistic
        < -0.2
    )
    out_x = f6.stability_world(stability.DEFECT_OUTLIER, 0)[0]
    assert (np.abs(out_x[np.isfinite(out_x)]) >= f6.EXTREME_SIGMAS).sum() > 20
    with pytest.raises(ValueError, match="unknown kind"):
        f6.stability_world("other", 0)


def test_one_replicate_of_each_kind_is_labelled_as_planted():
    assert f6.stability_replicate(f6.CLEAN, f6.DEV_SEED_BASE) == "stable"
    for kind in (*f6.FRAGILE_KINDS, stability.DEFECT_NULL):
        assert f6.stability_replicate(kind, f6.DEV_SEED_BASE) == "fragile"


def test_a_shift_replicate_is_reproducible_and_rejects_a_large_plant():
    again = f6.shift_replicate(shift.MMD_ID, True, f6.DEV_SEED_BASE)
    assert again == f6.shift_replicate(shift.MMD_ID, True, f6.DEV_SEED_BASE)
    assert again[0] is True
