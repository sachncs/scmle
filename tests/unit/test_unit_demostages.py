"""The demo stages that train: prepare, package, compare and the drill."""

import json
import os
import pathlib

import pytest

from scmle.artifacts import bundle
from scmle.deploy import provenance
from scmle.governance import registry
from scmle.rehearsal.demo import fixture
from scmle.rehearsal.demo import journal
from scmle.rehearsal.demo import stages
from scmle.serving.server import web

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "deploy" / "local" / "data"


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("demo")
    log = journal.Transcript()
    paths = stages.prepare(workdir, DATA, log)
    return workdir, paths, log


def test_prepare_trains_registers_and_packages(prepared):
    workdir, paths, log = prepared
    assert paths == {
        "service": workdir / "service",
        "golden_service": workdir / "golden-service",
    }
    assert [s.name for s in log.steps] == [
        "train and evaluate",
        "register and package",
    ]
    assert all(step.ok for step in log.steps)
    evaluation = json.loads((workdir / "evaluation.json").read_text())
    assert sorted(evaluation) == ["bad_release", "good"]
    assert evaluation["good"]["roc_auc"] > fixture.MINIMUM_ROC_AUC
    packaged = log.steps[1].detail
    assert packaged["versions"] == [1, 2]
    assert packaged["champion"] == 1
    assert packaged["bundle_digest"] == provenance.bundle_digest(
        workdir / "service" / "model"
    )
    assert "provenance.json" in packaged["files"]
    assert (workdir / "golden-service" / "serve.py").is_file()


def test_compare_matches_online_and_batch_with_the_bundle(prepared):
    workdir, paths, ignored = prepared

    loaded = bundle.Bundle.load(
        paths["service"] / "model", signing_key=fixture.DEMO_SIGNING_KEY
    )
    adapter = web.HttpServing(loaded)
    adapter.start()
    try:
        checks = stages.compare(
            adapter.address, paths["service"] / "model", DATA, workdir
        )
    finally:
        adapter.stop()
    assert checks["online_matches_bundle"] is True
    assert checks["batch_matches_bundle"] is True
    assert checks["max_probability_difference"] < fixture.PROBABILITY_TOLERANCE
    assert checks["same_digest"] is True
    assert checks["digest_in_response"] == checks["digest_in_batch_manifest"]


def test_the_rollback_drill_restores_the_good_version(prepared):
    workdir = prepared[0]
    before = os.environ.get("SCMLE_SIGNING_KEY")
    drill = stages.rollback(workdir, DATA)
    assert os.environ.get("SCMLE_SIGNING_KEY") == before
    assert drill["champion_before_rollback"] == fixture.PROMOTED_VERSION
    assert drill["rolled_back"] is True
    assert drill["restored_version"] == 1
    assert drill["champion_after_rollback"] == 1
    assert drill["serves_restored_model"] is True
    assert drill["audit_has_rollback_check"] is True
    store = registry.ModelRegistryStore(workdir / "registry")
    assert store.resolve(fixture.MODEL).version == 1
