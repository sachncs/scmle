"""Screen certificates: the stored file is current, honest and complete."""

import json

import pytest

from benchmarks.qualification import f4_certify
from benchmarks.qualification import screen_certify
from scmle.data.admissibility import leakage
from scmle.data.admissibility import screen
from scmle.data.detectors import certificates


def stored():
    return certificates.CertificateStore.load(screen_certify.STORE_PATH)


def test_the_shipped_store_is_inside_the_package():
    assert screen_certify.STORE_PATH == screen.STORE_PATH
    assert screen_certify.STORE_PATH.parent.name == "admissibility"
    assert screen_certify.STORE_PATH.parent.parent.parent.name == "scmle"


def test_the_stored_certificate_is_for_the_current_code():
    state = stored().state(screen.DETECTOR_ID, screen.code_digest())
    assert state is certificates.CertificateState.CURRENT
    assert screen_certify.problems(stored()) == []


def test_every_leak_type_of_every_mechanism_has_an_outcome_on_file():
    (cert,) = stored().current(screen.DETECTOR_ID, screen.code_digest())
    types = {o.certificate.defect_type for o in cert.outcomes}
    wanted = {d for ds in screen.MECHANISM_DEFECTS.values() for d in ds}
    assert types == wanted == set(screen_certify.DEFECTS)


def test_stale_and_missing_certificates_are_reported():
    empty = certificates.CertificateStore()
    assert all(p.endswith(": missing") for p in screen_certify.problems(empty))
    assert screen_certify.problems(empty)
    digest = screen.code_digest()
    assert digest != leakage.code_digest()


def test_planned_sizes_follow_the_phase_floors():
    assert screen_certify.NULL_REPLICATES >= 1000
    assert screen_certify.PLANT_REPLICATES >= 200
    assert screen_certify.POLICY is f4_certify.LEAK_POLICY
    assert screen_certify.POLICY.recall_floor == 0.9
    assert screen_certify.POLICY.false_alarm_ceiling == 0.05


def test_measurements_agree_with_the_certificate():
    document = json.loads(
        screen_certify.MEASUREMENTS_PATH.read_text(encoding="utf-8")
    )
    (cert,) = stored().current(screen.DETECTOR_ID, screen.code_digest())
    for outcome in cert.outcomes:
        c = outcome.certificate
        assert c.false_alarms == document["suspect_false_alarms"]
        assert c.detected == document["recall"][c.defect_type]["detected"]
        assert c.clean_rows == document["null_replicates"]
        assert c.planted == document["plant_replicates_per_type"]
    assert document["settings"]["skill_ceiling"] == (
        leakage.CERTIFIED.skill_ceiling
    )
    assert document["rule"] == "benjamini-yekutieli"
    assert document["alpha"] == leakage.CERTIFIED.alpha


def test_the_global_null_size_is_inside_its_band_and_nothing_is_hidden():
    document = json.loads(
        screen_certify.MEASUREMENTS_PATH.read_text(encoding="utf-8")
    )
    assert document["significant_rate_within_alpha"]
    assert document["significant_rate"] <= document["band_99"][1]
    assert document["suspect_false_alarm_wilson_95"][1] <= 0.05
    assert document["old_rule_false_alarms"] >= 0


def test_a_clean_world_and_a_planted_world_behave_as_certified():
    clean = screen_certify.replicate("", 0)
    assert (
        clean["suspect"] is False
        and clean["family"] > screen_certify.TABLE_COLUMNS
    )
    for defect in screen_certify.DEFECTS:
        assert screen_certify.replicate(defect, 0)["found"] is True


def test_the_rare_record_plant_is_recorded_for_one_class_only():
    frame = f4_certify.breast_cancer()
    column = screen_certify.plant_column(frame, "leak_rare_record", 0)
    recorded = frame.loc[column.notna(), f4_certify.TARGET]
    assert recorded.nunique() == 1
    assert 0.05 < column.notna().mean() < 0.15


def test_the_old_rule_is_scored_on_the_same_plants():
    assert screen_certify.replicate("leak_copy", 0)["old"] is True
    assert screen_certify.replicate("leak_noisy_copy", 0)["old"] is False


def test_unknown_defect_names_are_refused():
    with pytest.raises(KeyError):
        screen_certify.plant_column(
            f4_certify.breast_cancer(), "leak_unknown", 0
        )
