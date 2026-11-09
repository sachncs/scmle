"""The ingestion certificates and measurements are current and honestly labelled."""

import json

from benchmarks.qualification import ingest_certify
from benchmarks.qualification import ingest_worlds
from scmle.data.detectors import certificates


def test_certificate_file_is_current_and_passing():
    store = certificates.CertificateStore.load(ingest_certify.STORE_PATH)
    assert ingest_certify.problems(store) == []
    assert ingest_certify.audit_labels(store)["uncertified"] == []
    ids = {c.detector_id for c in store.certificates}
    assert ids == {ingest_certify.TORN_ID, "ingest-reconcile"}


def test_measurements_match_the_code_and_label_unreferenced_gates():
    document = json.loads(ingest_certify.MEASUREMENTS_PATH.read_text())
    assert ingest_certify.measurement_problems(document) == []
    for name in ("unit_verdict", "joins", "entities"):
        section = document[name]
        assert section["external_referent"] is None
        assert section["gate"] == "descriptive"
    assert document["development_seed_base"] == ingest_worlds.DEV_SEED_BASE
    assert document["point_in_time"]["view_leaked_rows"] == 0
    assert document["point_in_time"]["baseline_leaked_rows"] > 0


def test_seed_ranges_do_not_overlap():
    certified = max(
        ingest_certify.TORN_RUNS,
        ingest_certify.RECONCILE_RUNS,
        ingest_certify.LATE_WORLDS,
        ingest_certify.DIFFERENTIAL_CASES,
        ingest_certify.METAMORPHIC_CASES,
    )
    assert certified <= ingest_worlds.DEV_SEED_BASE


def test_live_check_passes():
    report = ingest_certify.check()
    assert report["ok"], report["problems"]
    assert report["floors_missed"] == []
    assert set(report["gates_without_external_referent"]) == {
        "unit_verdict",
        "joins",
        "entities",
    }


def test_development_look_runs_on_development_seeds():
    look = ingest_certify.development_look()
    assert look["late"] == [0] * 6
    assert all(item[2] == item[3] for item in look["unit"])
