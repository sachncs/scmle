import json

from benchmarks.qualification import __main__ as harness
from benchmarks.qualification import framing_certify as cert
from scmle.data.detectors import certificates
from scmle.data.framing import baseline


def test_committed_framing_files_are_current():
    report = cert.check()
    assert report["ok"], report["problems"]
    assert "gold corpus" not in report["problems"]


def test_measured_floors_are_recomputed_from_counts():
    document = cert.load_document()
    assert document is not None
    for row in document["size"].values():
        assert row["rate"] == row["rejections"] / row["runs"]
        assert row["meets_bound"] == (
            row["rate"] <= cert.size_bound(row["runs"])
        )
    assert document["min_clusters_measured"] == baseline.MIN_CLUSTERS
    labels = document["selective_labels"]
    low = certificates.wilson_interval(labels["detected"], labels["planted"])[0]
    high = certificates.wilson_interval(
        labels["false_alarms"], labels["clean"]
    )[1]
    assert low >= cert.LABEL_POLICY.recall_floor
    assert high <= cert.LABEL_POLICY.false_alarm_ceiling
    assert document["schema_fuzz"]["accepted"] == 0


def test_certification_and_development_seeds_do_not_overlap():
    assert cert.DEV_SEED_BASE > cert.RUNS
    assert cert.DEV_SEED_BASE > cert.FEASIBILITY_RUNS


def test_selective_label_certificate_is_stale_when_code_changes():
    store = certificates.CertificateStore.load(cert.STORE_PATH)
    state = store.state(baseline.DETECTOR_ID, cert.code_digest())
    assert state is certificates.CertificateState.CURRENT
    assert store.state(baseline.DETECTOR_ID, "0" * 64) is (
        certificates.CertificateState.STALE
    )


def test_check_framing_is_wired_into_the_harness(capsys):
    assert harness.main(["--check", "framing"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]


def test_problems_flag_an_inconsistent_min_clusters():
    store = certificates.CertificateStore.load(cert.STORE_PATH)
    document = cert.load_document()
    document = {**document, "min_clusters_measured": 50}
    assert any("MIN_CLUSTERS" in p for p in cert.problems(store, document))
    assert cert.problems(store, None)
