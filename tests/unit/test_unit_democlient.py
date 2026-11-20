"""Behaviour of the demo's transcript, fixture, client and command line."""

import json
import pathlib
import re
import subprocess
import sys
import urllib.request

import pandas
import pytest

import scmle.artifacts.bundle
from scmle.contracts import errors
from scmle.deploy import mocks
from scmle.deploy import packaging
from scmle.deploy import parity
from scmle.deploy import provenance
from scmle.governance import registry
from scmle.rehearsal.demo import client
from scmle.rehearsal.demo import command
from scmle.rehearsal.demo import fixture
from scmle.rehearsal.demo import journal
from scmle.rehearsal.demo import stages
from scmle.serving.server import web

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "deploy" / "local" / "golden"
DATA = ROOT / "deploy" / "local" / "data"
ROW = {"age": 52.0, "balance": 2300.0, "plan": "pro"}


@pytest.fixture(scope="module")
def golden():
    return parity.handwrite(parity.goldens(GOLDEN).spec)


@pytest.fixture(scope="module")
def running(golden, tmp_path_factory):
    folder = tmp_path_factory.mktemp("bundle") / "b"
    golden.save(folder)
    loaded = scmle.artifacts.bundle.Bundle.load(folder, trusted=True)
    adapter = web.Service(loaded)
    adapter.start()
    yield adapter
    adapter.stop()


@pytest.fixture(scope="module")
def service(golden, tmp_path_factory):
    folder = tmp_path_factory.mktemp("service") / "service"
    return packaging.scaffold(
        golden, folder, signing_key=fixture.DEMO_SIGNING_KEY
    )


@pytest.fixture
def depot(tmp_path):
    made = mocks.Depot(tmp_path / "store")
    made.start()
    yield made
    made.stop()


@pytest.fixture
def sink(tmp_path):
    made = mocks.Collector(tmp_path / "metrics.jsonl")
    made.start()
    yield made
    made.stop()


# ---- journal ---------------------------------------------------------------


def test_an_empty_transcript_has_not_passed():
    log = journal.Transcript()
    assert log.passed is False
    assert log.to_dict() == {"passed": False, "steps": []}


def test_a_transcript_records_and_prints_each_step(capsys):
    log = journal.Transcript()
    log.record("first", 0.0, True, rows=3)
    output = capsys.readouterr().out
    assert re.match(r"\[ok\] first \(\d+(\.\d+)?s\)\n      rows: 3\n", output)
    assert log.passed is True
    log.record("second", 0.0, False)
    assert "[FAIL] second" in capsys.readouterr().out
    assert log.passed is False
    record = log.to_dict()
    assert [s["name"] for s in record["steps"]] == ["first", "second"]
    assert record["steps"][0]["detail"] == {"rows": 3}
    assert record["steps"][1]["ok"] is False


# ---- fixture ---------------------------------------------------------------


def test_fixture_constants_are_the_published_demo_values():
    assert fixture.DEMO_SIGNING_KEY.endswith("published-fixture")
    assert fixture.MODEL == "churn"
    assert fixture.TARGET == "churned"
    assert fixture.MINIMUM_ROC_AUC == 0.8
    assert fixture.EXPECTED_SAMPLES == 4
    assert fixture.PROMOTED_VERSION == 2
    assert fixture.FAST_OPTIONS["seed"] == 7
    assert fixture.FAST_OPTIONS["search"].top_models == 0


def test_now_is_an_iso_utc_second():
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", fixture.now())


def test_sanction_approves_one_version_with_the_demo_key():
    record = registry.Release(
        model="churn",
        version=4,
        registered_at=fixture.now(),
        metrics={},
        config={},
        run_id="r",
        path="churn/v4",
        content_id="a" * 64,
        signed=True,
    )
    made = fixture.sanction(record, "fine", "report.json")
    assert (made.model, made.version, made.content) == ("churn", 4, "a" * 64)
    assert made.reviewer == "demo-reviewer"
    assert made.decision is registry.Verdict.APPROVED
    assert made.evidence == ("report.json",)
    made.validate("churn", 4, "a" * 64, signing_key=fixture.DEMO_SIGNING_KEY)


# ---- client ----------------------------------------------------------------


def test_the_authenticator_adds_a_token_only_to_listed_services():
    handler = client.Authenticator("tok", ("http://a/",))
    inside = urllib.request.Request("http://a/x")
    outside = urllib.request.Request("http://b/x")
    assert handler.http_request(inside) is inside
    assert inside.get_header("Authorization") == "Bearer tok"
    handler.http_request(outside)
    assert outside.get_header("Authorization") is None
    assert client.Authenticator.handler_order == 100


def test_authenticate_installs_a_token_only_with_a_secret():
    assert client.authenticate({}, ("http://a/",)) is False
    try:
        assert client.authenticate(
            {"SCMLE_AUTH_HS256_SECRET": "s" * 32}, ("http://a/",)
        )
    finally:
        urllib.request.install_opener(None)


def test_retrieve_and_post_return_errors_as_statuses(running):
    status, body = client.retrieve(f"{running.address}/readyz")
    assert status == 200
    assert body["ready"] is True
    assert client.retrieve(f"{running.address}/nope")[0] == 404
    status, body = client.post(f"{running.address}/predict", {"records": [ROW]})
    assert (status, body["predictions"]) == (200, [1])
    assert client.post(f"{running.address}/predict", {"records": []})[0] == 400


def test_settle_reports_readiness_or_a_dead_process(running):
    assert client.settle(running.address, timeout=5.0) >= 0.0
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    assert client.settle(running.address, done, timeout=5.0) is None
    assert (
        client.settle(f"http://127.0.0.1:{client.unused()}", timeout=0.3)
        is None
    )


def test_halt_stops_a_process_and_ignores_none():
    client.halt(None)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stderr=subprocess.PIPE,
    )
    client.halt(child)
    assert child.returncode is not None
    assert child.stderr.closed


def test_unused_finds_a_bindable_port():
    port = client.unused()
    assert 0 < port < 65536


def test_sync_and_download_round_trip_a_folder(depot, tmp_path):
    folder = tmp_path / "up"
    (folder / "deep").mkdir(parents=True)
    (folder / "a.txt").write_bytes(b"alpha")
    (folder / "deep" / "b.bin").write_bytes(b"beta")
    etags = client.sync(depot.address, folder, "models/x")
    assert sorted(etags) == ["a.txt", "deep/b.bin"]
    assert etags["a.txt"] == provenance.sha256_digest(b"alpha")
    back = tmp_path / "down"
    client.download(depot.address, "models/x", etags, back)
    assert (back / "deep" / "b.bin").read_bytes() == b"beta"
    with pytest.raises(errors.ArtifactError):
        client.download(
            depot.address, "models/x", {"a.txt": "sha256:" + "0" * 64}, back
        )


def test_serialize_gives_string_keys():
    frame = pandas.DataFrame({1: [2], "b": ["x"]})
    assert client.serialize(frame) == [{"1": 2, "b": "x"}]


def test_footprint_reads_a_resident_size():
    assert client.footprint(None) is None
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    try:
        size = client.footprint(child)
        assert size is None or size > 0
    finally:
        child.kill()
        child.wait()


def test_measure_reports_latency_percentiles(running):
    found = client.measure(running.address, [ROW], 5)
    assert found["requests"] == 5.0
    assert sorted(found) == [
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "requests",
    ]
    assert 0 < found["p50_ms"] <= found["p95_ms"] <= found["p99_ms"]


def test_measure_refuses_a_failed_request(running):
    with pytest.raises(
        errors.ScmleError, match="measurement request 0 failed: 422"
    ):
        client.measure(running.address, [{"age": 1.0}], 3)


def test_push_sends_typed_samples(sink):
    accepted = client.push(
        sink.address,
        {
            "latency_x": (1.0, "ms"),
            "memory_y": (2.0, "MiB"),
            "drift_z": (0.0, "bool"),
        },
        "v9",
    )
    assert accepted == 3
    kinds = [s["kind"] for s in sink.samples()]
    assert kinds == ["latency", "memory", "drift"]
    assert sink.samples()[0]["labels"] == {
        "model": "churn",
        "model_version": "v9",
    }


def test_unready_raises_the_tail_of_stderr():
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('x' * 900 + 'END')",
        ],
        stderr=subprocess.PIPE,
    )
    child.wait()
    with pytest.raises(errors.ArtifactError) as caught:
        client.unready(child)
    assert str(caught.value).endswith("END")
    assert len(str(caught.value)) == 800
    child.stderr.close()


# ---- stages that need no training ------------------------------------------


def test_mirror_round_trips_a_service_folder(service, depot, tmp_path):
    log = journal.Transcript()
    fetched = stages.mirror(log, depot.address, service, tmp_path)
    assert fetched == tmp_path / "fetched-service"
    assert provenance.digest(fetched / "model") == provenance.digest(
        service / "model"
    )
    step = log.steps[0]
    assert step.name == "object store round trip"
    assert step.ok is True
    assert step.detail["objects"] > 0
    stages.mirror(log, depot.address, service, tmp_path)
    assert log.steps[1].ok is True


def test_serve_score_and_instrument_a_service(service, sink, tmp_path):
    log = journal.Transcript()
    processes = []
    try:
        base, process, startup = stages.serve(log, service, processes)
        assert processes == [process]
        assert startup > 0
        assert log.steps[0].ok is True
        assert log.steps[0].detail["artifact_digest"].startswith("sha256:")
        stages.score(log, base, service, DATA, tmp_path)
        scored = log.steps[1]
        assert scored.name == "predict (online and batch)"
        assert scored.detail["online_matches_bundle"] is True
        assert scored.detail["batch_matches_bundle"] is True
        assert scored.detail["same_digest"] is True
        assert scored.ok is True
        stages.instrument(
            log,
            base=base,
            process=process,
            data=DATA,
            metrics_url=sink.address,
            rolled_back=True,
            startup=startup,
        )
        measured = log.steps[2]
        assert measured.ok is True
        assert measured.detail["samples_in_sink"] == 4
        assert sink.samples()[-1]["value"] == 1.0
    finally:
        for item in processes:
            client.halt(item)


def test_reject_sees_damaged_copies_never_ready(service, tmp_path):
    log = journal.Transcript()
    stages.reject(log, service, tmp_path)
    step = log.steps[0]
    assert step.ok is True
    for label in ("corrupted", "unsigned"):
        assert step.detail[label]["livez"] == 200
        assert step.detail[label]["readyz"] == 503
        assert step.detail[label]["predict"] == 503
        assert step.detail[label]["reason"]


def test_corroborate_checks_golden_parity(service):
    log = journal.Transcript()
    processes = []
    try:
        stages.corroborate(log, service, processes)
    finally:
        for item in processes:
            client.halt(item)
    assert len(processes) == 1
    assert log.steps[0].name == "golden-record parity"
    assert log.steps[0].ok is True


# ---- command ---------------------------------------------------------------


class Fake:
    """Stands in for the stages that train, so the run's wiring shows."""

    def __init__(self, drill):
        self.calls = []
        self.drill = drill

    def patch(self, monkeypatch, tmp_path):
        def prepare(workdir, data, transcript):
            self.calls.append("prepare")
            transcript.record("train and evaluate", 0.0, ok=True)
            return {"service": tmp_path / "s", "golden_service": tmp_path / "g"}

        def mirror(transcript, store_url, service, workdir):
            self.calls.append(("mirror", store_url.startswith("http://")))
            return tmp_path / "f"

        def serve(transcript, fetched, processes):
            self.calls.append("serve")
            return "http://x", None, 1.0

        def note(name):
            def step(*args, **kwargs):
                self.calls.append(name)

            return step

        monkeypatch.setattr(stages, "prepare", prepare)
        monkeypatch.setattr(stages, "mirror", mirror)
        monkeypatch.setattr(stages, "serve", serve)
        for name in ("score", "corroborate", "reject", "instrument"):
            monkeypatch.setattr(stages, name, note(name))
        monkeypatch.setattr(
            stages, "rollback", lambda workdir, data: self.drill
        )


GOOD_DRILL = {
    "rolled_back": True,
    "champion_before_rollback": 2,
    "champion_after_rollback": 1,
    "serves_restored_model": True,
    "audit_has_rollback_check": True,
}


def test_run_wires_every_stage_and_records_the_drill(monkeypatch, tmp_path):
    fake = Fake(GOOD_DRILL)
    fake.patch(monkeypatch, tmp_path)
    log = command.run(tmp_path)
    assert fake.calls == [
        "prepare",
        ("mirror", True),
        "serve",
        "score",
        "corroborate",
        "reject",
        "instrument",
    ]
    assert [s.name for s in log.steps] == [
        "train and evaluate",
        "monitor and rollback drill",
    ]
    assert log.passed is True


@pytest.mark.parametrize(
    "change",
    [
        {"rolled_back": False},
        {"champion_before_rollback": 1},
        {"champion_after_rollback": 2},
        {"serves_restored_model": False},
        {"audit_has_rollback_check": False},
    ],
)
def test_a_drill_that_misses_any_check_fails(monkeypatch, tmp_path, change):
    Fake({**GOOD_DRILL, **change}).patch(monkeypatch, tmp_path)
    assert command.run(tmp_path).steps[-1].ok is False


def test_a_stage_that_raises_stops_the_run_in_the_transcript(
    monkeypatch, tmp_path
):
    def broken(*args, **kwargs):
        raise OSError("disk")

    monkeypatch.setattr(stages, "prepare", broken)
    log = command.run(tmp_path, store_url="http://s", metrics_url="http://m")
    assert [s.name for s in log.steps] == ["demo stopped"]
    assert log.steps[0].detail["error"] == "OSError('disk')"
    assert log.passed is False


def test_main_writes_the_transcript_and_returns_its_verdict(
    monkeypatch, tmp_path, capsys
):
    def prepare(workdir, data, transcript):
        transcript.record("prepared", 0.0, ok=True)

    monkeypatch.setattr(stages, "prepare", prepare)
    assert command.main(["prepare", "--workdir", str(tmp_path / "w")]) == 0
    written = json.loads(
        (tmp_path / "w" / "transcript-prepare.json").read_text()
    )
    assert written["passed"] is True
    assert "DEMO PASSED" in capsys.readouterr().out
    monkeypatch.setattr(command, "run", lambda *a: journal.Transcript())
    assert command.main(["run", "--workdir", str(tmp_path / "r")]) == 1
    assert "DEMO FAILED" in capsys.readouterr().out
    seen = []

    def verify(api, golden_api, workdir, data, metrics_url):
        seen.append((api, golden_api, metrics_url))
        return journal.Transcript()

    monkeypatch.setattr(command, "verify", verify)
    assert (
        command.main(
            [
                "verify",
                "--workdir",
                str(tmp_path / "v"),
                "--api-url",
                "http://a",
                "--golden-api-url",
                "http://g",
                "--metrics-url",
                "http://m",
            ]
        )
        == 1
    )
    assert seen == [("http://a", "http://g", "http://m")]


def test_verify_stops_when_the_services_are_not_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "settle", lambda base: None)
    log = command.verify("http://a", "http://g", tmp_path, DATA, "http://m")
    assert [(s.name, s.ok) for s in log.steps] == [("services ready", False)]
