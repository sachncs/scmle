"""Robustness certificates: the stored file is current, honest and reproducible."""

import json

import numpy
import pytest
from scipy import stats

from benchmarks import acceptance
from benchmarks.qualification import f6_certify
from scmle.assurance.detectors import certificates
from scmle.assurance.robustness import shift
from scmle.assurance.robustness import stability


def stored():
    return certificates.Store.load(f6_certify.STORE_PATH)


def measurements():
    return json.loads(f6_certify.MEASUREMENTS_PATH.read_text(encoding="utf-8"))


def test_the_stored_certificates_are_for_the_current_code():
    store = stored()
    for detector, digest in f6_certify.current_digests().items():
        state = store.state(detector, digest)
        assert state is certificates.Freshness.CURRENT, detector
    assert f6_certify.problems(store) == []


def test_every_required_defect_type_has_an_outcome_on_file():
    store = stored()
    digests = f6_certify.current_digests()
    for detector, defects in f6_certify.required_defects().items():
        (cert,) = store.current(detector, digests[detector])
        assert {o.certificate.defect for o in cert.outcomes} == set(defects)


def test_missing_and_stale_certificates_are_reported():
    found = f6_certify.problems(certificates.Store())
    assert found and all(item.endswith(": missing") for item in found)
    forged = {
        detector: ("0" * 64, defects[0])
        for detector, defects in f6_certify.required_defects().items()
    }
    assert set(certificates.audit(stored(), forged).stale) == set(forged)


def test_the_floors_were_fixed_before_measuring():
    variables = acceptance.load()
    alpha = variables["alpha"].value
    shift_policy = certificates.Standard(
        variables["shift_recall_min"].value,
        acceptance.sized(shift.ALPHA, variables).bound,
        1 - 2 * alpha,
        "exact",
    )
    assert shift_policy == f6_certify.SHIFT_POLICY
    assert shift_policy.recall == 0.80
    assert shift_policy.alarms == pytest.approx(0.05 * 1.69)
    stability_policy = certificates.Standard(
        variables["stability_recall_min"].value,
        variables["stability_false_alarm_max"].value,
        1 - 2 * alpha,
        "exact",
    )
    assert stability_policy == f6_certify.STABILITY_POLICY
    assert (stability_policy.recall, stability_policy.alarms) == (0.80, 0.10)
    assert f6_certify.SHIFT_NULL == 1000 and f6_certify.SHIFT_ROWS == 2000
    assert f6_certify.PLANTED_AUC == 0.60


def test_both_shift_tests_are_decided_on_the_stated_null_count():
    store = stored()
    digests = f6_certify.current_digests()
    for test in (shift.C2ST_ID, shift.MMD_ID):
        (cert,) = store.current(test, digests[test])
        assert cert.policy == f6_certify.SHIFT_POLICY
        (outcome,) = cert.outcomes
        assert outcome.certificate.clean == f6_certify.SHIFT_NULL
        upper = stats.beta.ppf(
            1 - acceptance.load()["alpha"].value,
            outcome.certificate.alarms + 1,
            f6_certify.SHIFT_NULL - outcome.certificate.alarms,
        )
        assert outcome.alarms[1] == pytest.approx(upper)
    (cert,) = store.current(
        stability.DETECTOR_ID, digests[stability.DETECTOR_ID]
    )
    assert cert.policy == f6_certify.STABILITY_POLICY


def test_a_certificate_under_another_policy_is_reported():
    store = stored()
    (cert,) = store.current(shift.MMD_ID, shift.fingerprint())
    old = certificates.Standard(0.80, 0.075, 0.95)
    store.add(
        certificates.certify(
            cert.detector,
            cert.digests["code"],
            cert.digests["corpus"],
            cert.seeds,
            [o.certificate for o in cert.outcomes],
            old,
        )
    )
    assert f6_certify.problems(store) == [
        f"{shift.MMD_ID}: measured under another policy or null count"
    ]


def test_development_and_certification_seeds_do_not_overlap():
    largest = max(
        f6_certify.SHIFT_NULL,
        f6_certify.SHIFT_PLANTED,
        f6_certify.STABLE_REPS,
        f6_certify.PLANT_REPS,
    )
    assert largest < f6_certify.DEV_SEED_BASE
    assert measurements()["development_seed_base"] == f6_certify.DEV_SEED_BASE
    assert measurements()[shift.C2ST_ID]["seed_offset"] == 0


def test_measurements_agree_with_the_certificates():
    document = measurements()
    store = stored()
    (cert,) = store.current(shift.C2ST_ID, shift.fingerprint())
    (outcome,) = cert.outcomes
    row = document[shift.C2ST_ID]
    assert outcome.certificate.alarms == row["decision_false_alarms"]
    assert outcome.certificate.detected == round(
        row["power"] * row["planted_replicates"]
    )
    (cert,) = store.current(stability.DETECTOR_ID, stability.fingerprint())
    by_type = {o.certificate.defect: o.certificate for o in cert.outcomes}
    for kind, counts in document[stability.DETECTOR_ID]["by_defect"].items():
        assert by_type[kind].detected == counts["detected"]
        assert by_type[kind].planted == counts["planted"]
    clean = document[stability.DETECTOR_ID]
    assert by_type[stability.DEFECT_NULL].alarms == (
        clean["genuine_effects"] - clean["genuine_labelled_stable"]
    )


def test_the_size_by_p_value_alone_is_in_the_99_percent_band():
    for test in (shift.C2ST_ID, shift.MMD_ID):
        count = f6_certify.SHIFT_NULL
        row = measurements()[test]
        assert row["size_in_band"] is True
        assert row["null_replicates"] == count
        assert row["band_99"] == pytest.approx(
            list(f6_certify.band_99(shift.ALPHA, count))
        )


def test_an_uncertified_check_is_descriptive_not_an_error():
    store = stored()
    labels = f6_certify.audit_labels(store)
    assert labels["missing"] == [] and labels["stale"] == []
    for item in labels["uncertified"]:
        detector, defect = item.split("/")
        status = store.status(
            detector, f6_certify.current_digests()[detector], defect
        )
        assert status is certificates.Authority.DESCRIPTIVE


def test_the_planted_shift_has_the_stated_bayes_auc():
    delta = f6_certify.shift_delta()
    assert stats.norm.cdf(delta / numpy.sqrt(2.0)) == pytest.approx(0.60)
    first, second = f6_certify.shift_samples(True, 0)
    assert second["x0"].mean() - first["x0"].mean() == pytest.approx(
        delta, abs=0.1
    )
    null_first, null_second = f6_certify.shift_samples(False, 0)
    assert abs(null_second["x0"].mean() - null_first["x0"].mean()) < 0.15


def test_the_semi_synthetic_worlds_plant_what_they_say():
    clean_x, clean_y = f6_certify.stability_world(f6_certify.CLEAN, 0)
    seen = numpy.isfinite(clean_x)
    assert 0.05 < 1 - seen.mean() < 0.3
    assert stats.spearmanr(clean_x[seen], clean_y[seen]).statistic > 0.3
    null_x, null_y = f6_certify.stability_world(stability.DEFECT_NULL, 0)
    assert (
        abs(stats.spearmanr(null_x, null_y, nan_policy="omit").statistic) < 0.2
    )
    half = f6_certify.STABILITY_ROWS // 2
    rev_x, rev_y = f6_certify.stability_world(stability.DEFECT_REVERSAL, 0)
    first = numpy.isfinite(rev_x[:half])
    second = numpy.isfinite(rev_x[half:])
    assert (
        stats.spearmanr(rev_x[:half][first], rev_y[:half][first]).statistic
        > 0.2
    )
    assert (
        stats.spearmanr(rev_x[half:][second], rev_y[half:][second]).statistic
        < -0.2
    )
    out_x = f6_certify.stability_world(stability.DEFECT_OUTLIER, 0)[0]
    assert (
        numpy.abs(out_x[numpy.isfinite(out_x)]) >= f6_certify.EXTREME_SIGMAS
    ).sum() > 20
    with pytest.raises(ValueError, match="unknown kind"):
        f6_certify.stability_world("other", 0)


def test_one_replicate_of_each_kind_is_labelled_as_planted():
    assert (
        f6_certify.stability_replicate(
            f6_certify.CLEAN, f6_certify.DEV_SEED_BASE
        )
        == "stable"
    )
    for kind in (*f6_certify.FRAGILE_KINDS, stability.DEFECT_NULL):
        assert (
            f6_certify.stability_replicate(kind, f6_certify.DEV_SEED_BASE)
            == "fragile"
        )


def test_a_shift_replicate_is_reproducible_and_rejects_a_large_plant():
    again = f6_certify.shift_replicate(
        shift.MMD_ID, True, f6_certify.DEV_SEED_BASE
    )
    assert again == f6_certify.shift_replicate(
        shift.MMD_ID, True, f6_certify.DEV_SEED_BASE
    )
    assert again[0] is True
