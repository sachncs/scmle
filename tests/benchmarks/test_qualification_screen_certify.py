"""Screen certificates: the stored file is current, honest and complete."""

import json

import pytest

from benchmarks.qualification import f4_certify
from benchmarks.qualification import screen_certify
from scmle.assurance.admissibility import leakage
from scmle.assurance.admissibility import screen
import scmle.assurance.admissibility.screen
from scmle.assurance.detectors import certificates


def stored():
    return certificates.Store.load(
        scmle.assurance.admissibility.screen.STORE_PATH
    )


def test_the_shipped_store_is_inside_the_package():
    assert scmle.assurance.admissibility.screen.STORE_PATH == screen.STORE_PATH
    assert (
        scmle.assurance.admissibility.screen.STORE_PATH.parent.name
        == "certificates"
    )
    assert (
        scmle.assurance.admissibility.screen.STORE_PATH.parent.parent.name
        == "scmle"
    )


def test_the_stored_certificate_is_for_the_current_code():
    state = stored().state(screen.DETECTOR_ID, screen.code_digest())
    assert state is certificates.Freshness.CURRENT
    assert screen_certify.problems(stored()) == []


@pytest.mark.parametrize("kind", screen_certify.KINDS)
def test_every_leak_type_of_every_mechanism_has_an_outcome_on_file(kind):
    (cert,) = stored().current(screen.DETECTORS[kind], screen.code_digest())
    types = {o.certificate.defect for o in cert.outcomes}
    wanted = {d for ds in screen.MECHANISM_DEFECTS.values() for d in ds}
    assert types == wanted == set(screen_certify.DEFECTS)


@pytest.mark.parametrize("kind", screen_certify.KINDS)
def test_every_kind_of_target_has_a_current_published_certificate(kind):
    task, classes = {
        "binary": ("classification", 2),
        "multiclass": ("classification", 10),
        "regression": ("regression", 0),
    }[kind]
    assert screen.family(task, classes) == kind
    assert screen.evidence(task, classes) == (
        f"published certificate for {kind} targets: current"
    )
    assert screen_certify.uncertified(stored()) == []


def test_more_classes_than_measured_have_no_published_certificate():
    document = json.loads(
        screen_certify.MEASUREMENTS_PATH.read_text(encoding="utf-8")
    )
    measured = document["multiclass"]["classes"]
    assert measured == screen.MAX_CERTIFIED_CLASSES
    assert screen.family("classification", measured + 1) is None
    assert "no published certificate" in screen.evidence(
        "classification", measured + 1
    )


@pytest.mark.parametrize("kind", screen_certify.KINDS)
def test_measurements_agree_with_the_certificate(kind):
    document = json.loads(
        screen_certify.MEASUREMENTS_PATH.read_text(encoding="utf-8")
    )[kind]
    (cert,) = stored().current(screen.DETECTORS[kind], screen.code_digest())
    for outcome in cert.outcomes:
        c = outcome.certificate
        assert c.alarms == document["suspect_false_alarms"]
        assert c.detected == document["recall"][c.defect]["detected"]
        assert c.clean == document["clean_worlds"]
        assert c.planted == document["plant_replicates_per_type"]
    assert document["settings"]["skill_ceiling"] == (
        leakage.CERTIFIED.skill_ceiling
    )
    assert document["rule"] == "benjamini-yekutieli"
    assert document["alpha"] == leakage.CERTIFIED.alpha


@pytest.mark.parametrize("kind", screen_certify.KINDS)
def test_the_global_null_size_is_inside_its_band_and_nothing_is_hidden(kind):
    document = json.loads(
        screen_certify.MEASUREMENTS_PATH.read_text(encoding="utf-8")
    )[kind]
    assert document["significant_rate_within_alpha"]
    assert document["significant_rate"] <= document["band_99"][1]
    assert document["suspect_false_alarm_wilson_95"][1] <= 0.05
    assert document["old_rule_false_alarms"] >= 0


@pytest.mark.parametrize("kind", ["binary", "regression"])
def test_a_clean_world_and_a_planted_world_behave_as_certified(kind):
    clean = screen_certify.replicate("", 0, kind)
    assert (
        clean["suspect"] is False
        and clean["family"] > screen_certify.TABLE_COLUMNS
    )
    for defect in screen_certify.DEFECTS:
        assert screen_certify.replicate(defect, 0, kind)["found"] is True


def test_the_multiclass_table_has_ten_classes_and_no_constant_column():
    frame = screen_certify.digits()
    assert frame[f4_certify.TARGET].nunique() == 10
    assert (frame.drop(columns=f4_certify.TARGET).std() > 0).all()


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


def finding(feature, suspect=True):
    return screen.Finding(
        feature,
        screen.SKILL,
        0.99,
        0.001,
        0.01,
        suspect,
        0.97 if suspect else None,
        suspect,
        True,
        certificates.Authority.DESCRIPTIVE,
    )


def test_a_suspect_legitimate_column_in_a_planted_world_is_a_false_alarm():
    # The planted column is the only leak in a planted world; any other
    # suspect is a real column flagged, a false alarm (audit of CERT-1).
    found = (finding("leak_copy"), finding("worst perimeter"))
    assert screen_certify.alarmed(found, "leak_copy") is True
    assert screen_certify.alarmed(found[:1], "leak_copy") is False
    quiet = (finding("leak_copy"), finding("worst area", suspect=False))
    assert screen_certify.alarmed(quiet, "leak_copy") is False
    # A null world has no planted column: every suspect is a false alarm.
    assert screen_certify.alarmed(found[1:], "") is True


def test_every_world_reports_whether_a_legitimate_column_was_flagged():
    for defect in ("", "grouped", "leak_copy"):
        assert isinstance(screen_certify.replicate(defect, 0)["alarm"], bool)


@pytest.mark.parametrize("kind", screen_certify.KINDS)
def test_false_alarms_are_counted_over_every_world(kind):
    document = json.loads(
        screen_certify.MEASUREMENTS_PATH.read_text(encoding="utf-8")
    )[kind]
    worlds = (
        document["null_replicates"]
        + document["grouped_null_replicates"]
        + document["plant_replicates_per_type"] * len(screen_certify.DEFECTS)
    )
    assert document["clean_worlds"] == worlds
    planted = sum(document["planted_world_false_alarms"].values())
    assert document["suspect_false_alarms"] == (
        document["null_false_alarms"]
        + document["grouped_suspect_false_alarms"]
        + planted
    )
