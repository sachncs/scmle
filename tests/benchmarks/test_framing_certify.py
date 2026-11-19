import json

import pytest

from benchmarks import sources
from benchmarks.qualification import __main__
from benchmarks.qualification import framing_certify
from scmle.assurance.detectors import certificates
from scmle.assurance.framing import baseline


def test_committed_framing_files_are_current():
    report = framing_certify.check()
    assert report["ok"], report["problems"]
    assert "gold corpus" not in report["problems"]


def test_measured_floors_are_recomputed_from_counts():
    document = framing_certify.load_document()
    assert document is not None
    for row in document["size"].values():
        assert row["rate"] == row["rejections"] / row["runs"]
        assert row["meets_bound"] == framing_certify.SIZE.passes(
            row["rejections"], row["runs"], framing_certify.RISK
        )
    assert document["min_clusters_measured"] == baseline.MIN_CLUSTERS
    labels = document["selective_labels"]
    policy = framing_certify.LABEL_POLICY
    low = certificates.interval(
        labels["detected"], labels["planted"], policy.confidence, policy.method
    )[0]
    high = certificates.interval(
        labels["false_alarms"],
        labels["clean"],
        policy.confidence,
        policy.method,
    )[1]
    assert low >= framing_certify.LABEL_POLICY.recall
    assert high <= framing_certify.LABEL_POLICY.alarms
    assert document["schema_fuzz"]["accepted"] == 0


def test_certification_and_development_seeds_do_not_overlap():
    assert framing_certify.DEV_SEED_BASE > framing_certify.RUNS
    assert framing_certify.DEV_SEED_BASE > framing_certify.FEASIBILITY_RUNS


def test_selective_label_certificate_is_stale_when_code_changes():
    store = certificates.Store.load(framing_certify.STORE_PATH)
    state = store.state(baseline.DETECTOR_ID, framing_certify.code_digest())
    assert state is certificates.Freshness.CURRENT
    assert store.state(baseline.DETECTOR_ID, "0" * 64) is (
        certificates.Freshness.STALE
    )


def test_check_framing_is_wired_into_the_harness(capsys):
    assert __main__.main(["--check", "framing"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]


def test_problems_flag_an_inconsistent_min_clusters():
    store = certificates.Store.load(framing_certify.STORE_PATH)
    document = framing_certify.load_document()
    document = {**document, "min_clusters_measured": 50}
    assert any(
        "MIN_CLUSTERS" in p for p in framing_certify.problems(store, document)
    )
    assert framing_certify.problems(store, None)


def test_measurements_without_the_current_code_hash_are_stale():
    store = certificates.Store.load(framing_certify.STORE_PATH)
    document = framing_certify.load_document()
    assert document["code"] == framing_certify.fingerprint()
    for stale in (
        {**document, "code": "0" * 64},
        {k: v for k, v in document.items() if k != "code"},
    ):
        assert "framing_measurements.json: measured other code" in (
            framing_certify.problems(store, stale)
        )


@pytest.mark.parametrize(
    "module",
    [
        "scmle.assurance.framing.feasibility",
        "scmle.assurance.framing.draft",
        "scmle.assurance.charter.feasibility",
        "scmle.assurance.charter.contracts",
    ],
)
def test_a_change_to_the_product_code_it_runs_makes_check_stale(
    monkeypatch, tmp_path, module
):
    original = sources.locate(module)
    copy = tmp_path / "copy.py"
    copy.write_text(original.read_text() + "\nEDITED = 1\n")
    real = sources.locate

    def edited(target, root=sources.ROOT):
        return copy if target == module else real(target, root)

    monkeypatch.setattr(sources, "locate", edited)
    report = framing_certify.check()
    assert not report["ok"]
    assert (
        "framing_measurements.json: measured other code" in (report["problems"])
    )
