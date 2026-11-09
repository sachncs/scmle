"""The serving contract: schemas, readiness, limits, policies, parity."""

import json
import pathlib
import threading
import urllib.error
import urllib.request

import pandas
import pytest

from scmle import cli
import scmle.artifacts.bundle
from scmle.contracts import errors
import scmle.contracts.serving
from scmle.deploy import provenance
from scmle.inference import guard
from scmle.inference import parity
from scmle.inference import serving

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "deploy" / "local" / "golden"
SHA = r"^sha256:[0-9a-f]{64}$"


@pytest.fixture(scope="module")
def golden():
    return parity.load_golden(GOLDEN)


@pytest.fixture(scope="module")
def expected():
    return parity.read_json(GOLDEN / "expected.json")


@pytest.fixture(scope="module")
def bundle(golden, tmp_path_factory):
    folder = tmp_path_factory.mktemp("golden")
    parity.reference_bundle(golden.spec).save(folder / "bundle")
    return scmle.artifacts.bundle.Bundle.load(folder / "bundle", trusted=True)


def call_full(url, payload=None, raw=None, headers=None):
    body = raw if raw is not None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        body if (payload is not None or raw is not None) else None,
        {"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return (
                response.status,
                json.loads(response.read()),
                response.headers,
            )
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read()), error.headers


def call(*args, **kwargs):
    return call_full(*args, **kwargs)[:2]


def wait_until_settled(adapter, timeout):
    """Joins the background loader and reports whether the service is ready."""
    adapter.loading.join(timeout)
    return adapter.readiness.ready


@pytest.fixture
def running(bundle):
    adapter = serving.HttpServing(bundle)
    adapter.start()
    yield adapter
    adapter.stop()


def test_the_reference_matches_the_closed_form_in_process(
    golden, expected, bundle
):
    frame = pandas.DataFrame(golden.records)
    report = parity.compare(
        expected,
        bundle.predict(frame).tolist(),
        bundle.predict_proba(frame).tolist(),
        golden.same_process,
    )
    assert report.passed, report.mismatches
    assert report.max_probability_error < 1e-12


def test_the_committed_expected_outputs_are_the_closed_form(golden, expected):
    recomputed = parity.expected_outputs(golden.spec, golden.records)
    assert recomputed["labels"] == expected["labels"]
    assert recomputed["probability_of_class_1"] == pytest.approx(
        expected["probability_of_class_1"], abs=0, rel=0
    )


def test_online_serving_matches_the_golden_records(golden, expected, running):
    status, body, headers = call_full(
        f"{running.address}/predict", {"records": golden.records}
    )
    assert status == 200
    report = parity.compare(
        expected,
        body["predictions"],
        body["probabilities"],
        golden.same_process,
    )
    assert report.passed, report.mismatches
    assert headers["X-Request-ID"] == body["request_id"]


def test_batch_serving_shares_the_online_path(
    golden, expected, running, bundle, tmp_path
):
    source = tmp_path / "records.csv"
    pandas.DataFrame(golden.records).to_csv(source, index=False)
    destination = tmp_path / "scored.csv"
    batch = serving.BatchServing(bundle)
    assert batch.score_file(source, destination, chunk_rows=3) == len(
        golden.records
    )
    scored = pandas.read_csv(destination)
    online = call(f"{running.address}/predict", {"records": golden.records})[1]
    assert scored["prediction"].tolist() == online["predictions"]
    assert scored["proba_1"].tolist() == pytest.approx(
        [row[1] for row in online["probabilities"]], abs=1e-12
    )
    report = parity.compare(
        expected,
        scored["prediction"].tolist(),
        scored[["proba_0", "proba_1"]].to_numpy().tolist(),
        golden.same_process,
    )
    assert report.passed, report.mismatches
    manifest = json.loads(
        (tmp_path / "scored.csv.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifact_digest"] == online["artifact_digest"]
    assert manifest["model_version"] == online["model_version"]
    assert manifest["rows"] == len(golden.records)
    assert manifest["rows_with_nulls"] == 3
    assert manifest["rows_with_unseen_categories"] == 1
    assert manifest["output_sha256"] == provenance.file_digest(destination)


def test_every_response_carries_version_and_artifact_digest(
    golden, bundle, running
):
    status, body = call(
        f"{running.address}/predict", {"records": golden.records[:2]}
    )
    assert status == 200
    for key in scmle.contracts.serving.RESPONSE_KEYS:
        assert key in body, key
    assert body["schema_version"] == "1"
    assert body["model_version"] == "golden-1"
    assert body["record_count"] == 2 and len(body["rows"]) == 2
    assert body["artifact_digest"] == running.artifact_digest
    assert body["artifact_digest"].startswith("sha256:")
    status, health = call(f"{running.address}/health")
    assert health["artifact_digest"] == body["artifact_digest"]
    status, error = call(f"{running.address}/predict", {"records": []})
    assert status == 400
    for key in ("schema_version", "request_id", "error", "category"):
        assert key in error


def test_the_artifact_digest_names_the_saved_manifest(bundle, tmp_path):
    folder = bundle.save(tmp_path / "saved")
    loaded = scmle.artifacts.bundle.Bundle.load(folder, trusted=True)
    assert provenance.bundle_digest(folder) == provenance.manifest_digest(
        loaded.info.to_dict()
    )
    assert serving.HttpServing(loaded).artifact_digest == (
        provenance.bundle_digest(folder)
    )


def test_the_versioned_path_and_unknown_versions(golden, running):
    status, body = call(
        f"{running.address}/v1/predict", {"records": golden.records[:1]}
    )
    assert status == 200 and body["schema_version"] == "1"
    status, body = call(
        f"{running.address}/predict",
        {"schema_version": "2", "records": golden.records[:1]},
    )
    assert status == 409 and body["category"] == "schema_change"
    assert body["details"]["supported"] == ["1"]
    status, body = call(
        f"{running.address}/predict",
        {"records": golden.records[:1], "surprise": 1},
    )
    assert status == 400 and "unknown request fields" in body["error"]


def test_nulls_unseen_categories_and_extra_columns_are_reported_per_row(
    golden, running
):
    status, body = call(
        f"{running.address}/predict", {"records": golden.records}
    )
    assert status == 200
    rows = body["rows"]
    assert rows[3]["missing"] == ["age"] and rows[4]["missing"] == ["balance"]
    assert rows[5]["missing"] == ["plan"] and rows[5]["unseen"] == {}
    assert rows[6]["unseen"] == {"plan": "enterprise"}
    assert rows[0] == {"missing": [], "unseen": {}}
    assert body["diagnostics"]["ignored_columns"] == ["note"]
    assert body["diagnostics"]["unseen_categories"] == {"plan": ["enterprise"]}
    assert body["schema_valid"] is True


def test_policies_can_reject_nulls_and_unseen_categories(golden, bundle):
    strict = serving.HttpServing(bundle, on_null="reject", on_unseen="reject")
    strict.start()
    try:
        clean = golden.records[:3]
        assert call(f"{strict.address}/predict", {"records": clean})[0] == 200
        status, body = call(
            f"{strict.address}/predict", {"records": [golden.records[3]]}
        )
        assert status == 422 and body["category"] == "data_policy"
        assert body["details"]["records"] == [0]
        status, body = call(
            f"{strict.address}/predict", {"records": [golden.records[6]]}
        )
        assert status == 422 and "unseen" in body["error"]
    finally:
        strict.stop()
    with pytest.raises(errors.ConfigError, match="on_null"):
        serving.HttpServing(bundle, on_null="guess")


def test_schema_changes_are_explicit(golden, running):
    base = running.address
    status, body = call(
        f"{base}/predict", {"records": [{"age": 1.0, "plan": "pro"}]}
    )
    assert status == 422 and body["category"] == "schema"
    assert body["details"]["missing"] == ["balance"]
    mixed = [golden.records[0], {"age": 1.0, "plan": "pro"}]
    status, body = call(f"{base}/predict", {"records": mixed})
    assert status == 422 and body["details"]["records"] == [1]
    status, body = call(
        f"{base}/predict",
        {
            "records": [{"age": "old", "balance": 1.0, "plan": "pro"}],
        },
    )
    assert status == 422 and "non-numeric" in body["error"]
    served = call(f"{base}/schema")[1]["schema_digest"]
    status, body = call(
        f"{base}/predict",
        {"records": golden.records[:1], "expected_schema_digest": served},
    )
    assert status == 200
    status, body = call(
        f"{base}/predict",
        {
            "records": golden.records[:1],
            "expected_schema_digest": "sha256:" + "0" * 64,
        },
    )
    assert status == 409 and body["category"] == "schema_change"
    assert body["details"]["served"] == served


def test_the_schema_endpoint_publishes_schemas_and_limits(running):
    status, body = call(f"{running.address}/schema")
    assert status == 200
    assert body["request_schema"]["$id"] == "scmle.serving.request.v1"
    assert body["response_schema"]["$id"] == "scmle.serving.response.v1"
    assert body["limits"]["max_concurrent"] == 8
    assert body["limits"]["on_null"] == "impute"
    assert body["schema"]["target"] == "churned"


def test_size_and_record_limits(golden, bundle):
    adapter = serving.HttpServing(bundle, max_request_bytes=200, max_records=2)
    adapter.start()
    try:
        url = f"{adapter.address}/predict"
        assert call(url, {"records": golden.records[:1]})[0] == 200
        assert call(url, {"records": golden.records[:3]})[0] == 413
        big = {"records": golden.records[:1], "pad": "x" * 500}
        assert call(url, big)[0] == 413
    finally:
        adapter.stop()


def test_request_ids_are_echoed_when_safe_and_refused_when_not(running):
    status, body, headers = call_full(
        f"{running.address}/livez", headers={"X-Request-ID": "trace-1"}
    )
    assert headers["X-Request-ID"] == "trace-1"
    status, body, headers = call_full(
        f"{running.address}/livez", headers={"X-Request-ID": "bad id!"}
    )
    assert status == 400 and body["category"] == "request"


class SlowGate:
    """Holds the scoring call until released."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()


def test_concurrency_limit_answers_429_while_slots_are_busy(
    golden, bundle, monkeypatch
):
    adapter = serving.HttpServing(bundle, max_concurrent=1)
    gate = SlowGate()
    original = adapter.respond

    def slow(records):
        gate.entered.set()
        gate.release.wait(20)
        return original(records)

    monkeypatch.setattr(adapter, "respond", slow)
    adapter.start()
    results = []
    try:
        url = f"{adapter.address}/predict"
        first = threading.Thread(
            target=lambda: results.append(
                call(url, {"records": golden.records[:1]})[0]
            )
        )
        first.start()
        assert gate.entered.wait(20)
        status, body, headers = call_full(url, {"records": golden.records[:1]})
        assert status == 429 and body["category"] == "overloaded"
        assert headers["Retry-After"] == "1"
        assert call(f"{adapter.address}/livez")[0] == 200
        gate.release.set()
        first.join(20)
        assert results == [200]
        assert call(url, {"records": golden.records[:1]})[0] == 200
    finally:
        gate.release.set()
        adapter.stop()


def test_a_scoring_call_past_its_deadline_answers_504(
    golden, bundle, monkeypatch
):
    adapter = serving.HttpServing(bundle, scoring_timeout_seconds=0.2)
    gate = SlowGate()
    monkeypatch.setattr(
        adapter, "respond", lambda records: gate.release.wait(5) and {}
    )
    adapter.start()
    try:
        status, body = call(
            f"{adapter.address}/predict", {"records": golden.records[:1]}
        )
        assert status == 504 and body["category"] == "timeout"
    finally:
        gate.release.set()
        adapter.stop()


def test_readiness_follows_verify_then_load_then_warm_up(golden, bundle):
    order = []

    def loader(readiness):
        readiness.begin(guard.Phase.VERIFYING)
        order.append("verify")
        readiness.complete(guard.Phase.VERIFYING)
        readiness.begin(guard.Phase.LOADING)
        order.append("load")
        readiness.complete(guard.Phase.LOADING)
        return bundle

    adapter = serving.HttpServing(loader=loader)
    adapter.start()
    try:
        assert wait_until_settled(adapter, 20)
        status, body = call(f"{adapter.address}/readyz")
        assert status == 200 and body["ready"] is True
        assert set(body["completed"]) == {"verifying", "loading", "warming"}
        assert body["notes"]["warming"] == "synthetic record from the schema"
        assert body["artifact_digest"] == adapter.artifact_digest
        assert order == ["verify", "load"]
        assert (
            call(f"{adapter.address}/predict", {"records": golden.records[:1]})[
                0
            ]
            == 200
        )
    finally:
        adapter.stop()


def test_a_failed_verification_keeps_the_service_alive_but_never_ready(
    golden,
):
    def loader(readiness):
        readiness.begin(guard.Phase.VERIFYING)
        raise errors.ArtifactError("bundle signature is invalid")

    adapter = serving.HttpServing(loader=loader)
    adapter.start()
    try:
        assert not wait_until_settled(adapter, 20)
        assert call(f"{adapter.address}/livez")[0] == 200
        status, body, headers = call_full(f"{adapter.address}/readyz")
        assert status == 503 and body["ready"] is False
        assert body["phase"] == "failed"
        assert "signature is invalid" in body["reason"]
        assert "model_version" not in body
        assert headers["Retry-After"] == "1"
        status, body, headers = call_full(
            f"{adapter.address}/predict", {"records": golden.records[:1]}
        )
        assert status == 503 and body["category"] == "not_ready"
        assert call(f"{adapter.address}/schema")[0] == 503
        assert call(f"{adapter.address}/health")[0] == 503
    finally:
        adapter.stop()


def test_not_ready_while_the_loader_is_still_running(bundle):
    release = threading.Event()

    def loader(readiness):
        readiness.begin(guard.Phase.VERIFYING)
        release.wait(20)
        readiness.complete(guard.Phase.VERIFYING)
        readiness.begin(guard.Phase.LOADING)
        readiness.complete(guard.Phase.LOADING)
        return bundle

    adapter = serving.HttpServing(loader=loader)
    adapter.start()
    try:
        status, body = call(f"{adapter.address}/readyz")
        assert status == 503 and body["phase"] == "verifying"
        release.set()
        assert wait_until_settled(adapter, 20)
        assert call(f"{adapter.address}/readyz")[0] == 200
    finally:
        release.set()
        adapter.stop()


def test_a_failing_warm_up_is_not_ready(bundle):
    adapter = serving.HttpServing(
        bundle, warmup_records=[{"age": "not a number"}]
    )
    adapter.start()
    try:
        status, body = call(f"{adapter.address}/readyz")
        assert status == 503 and "warm-up failed" in body["reason"]
    finally:
        adapter.stop()


def test_readiness_steps_cannot_be_skipped_or_reordered():
    readiness = guard.Readiness()
    with pytest.raises(errors.ScmleError, match="cannot begin loading"):
        readiness.begin(guard.Phase.LOADING)
    readiness.begin(guard.Phase.VERIFYING)
    with pytest.raises(errors.ScmleError, match="cannot begin warming"):
        readiness.begin(guard.Phase.WARMING)
    readiness.complete(guard.Phase.VERIFYING)
    assert not readiness.ready
    readiness.fail("broken")
    assert readiness.snapshot()["reason"] == "broken"
    with pytest.raises(errors.ScmleError):
        readiness.begin(guard.Phase.LOADING)


def test_the_loader_and_bundle_arguments_are_exclusive(bundle):
    with pytest.raises(errors.ConfigError, match="exactly one"):
        serving.HttpServing()
    with pytest.raises(errors.ConfigError, match="exactly one"):
        serving.HttpServing(bundle, loader=lambda readiness: bundle)
    with pytest.raises(errors.ConfigError, match="positive"):
        guard.AdmissionControl(0, 1.0)


def test_the_batch_command_scores_a_file_and_writes_a_manifest(
    golden, expected, bundle, tmp_path
):
    folder = bundle.save(tmp_path / "bundle", signing_key="batch-key")
    source = tmp_path / "records.csv"
    pandas.DataFrame(golden.records).to_csv(source, index=False)
    output = tmp_path / "scored.csv"

    refused = cli.main(
        ["batch", str(folder), str(source), "--output", str(output)]
    )
    assert refused == cli.EXIT_FAILED and not output.exists()

    code = cli.main(
        [
            "batch",
            str(folder),
            str(source),
            "--output",
            str(output),
            "--trusted",
        ]
    )
    assert code == cli.EXIT_OK
    scored = pandas.read_csv(output)
    report = parity.compare(
        expected,
        scored["prediction"].tolist(),
        scored[["proba_0", "proba_1"]].to_numpy().tolist(),
        golden.same_process,
    )
    assert report.passed, report.mismatches
    strict = cli.main(
        [
            "batch",
            str(folder),
            str(source),
            "--output",
            str(tmp_path / "strict.csv"),
            "--trusted",
            "--on-null",
            "reject",
        ]
    )
    assert strict == cli.EXIT_FAILED


def test_the_parity_script_passes_a_running_service_and_fails_a_wrong_one(
    running, bundle, golden, monkeypatch, capsys
):
    from scripts import check_serving_parity

    code = check_serving_parity.main(
        ["--url", running.address, "--directory", str(GOLDEN)]
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and report["passed"] is True
    assert report["artifact_digest"] == running.artifact_digest

    original = running.respond

    def shifted(records):
        response = original(records)
        response["probabilities"][0][1] += 0.01
        return response

    monkeypatch.setattr(running, "respond", shifted)
    code = check_serving_parity.main(
        ["--url", running.address, "--directory", str(GOLDEN)]
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 1 and "record 0" in report["mismatches"][0]


def test_the_golden_expected_file_is_current():
    from scripts import make_serving_golden

    assert (
        make_serving_golden.main(["--check", "--directory", str(GOLDEN)]) == 0
    )


def test_the_demo_override_is_explicit_and_named():
    from scmle.deploy import verify

    demo = next(iter(verify.PUBLISHED_KEYS))
    verify.refuse_published_key(demo, True, allowed=True)


def serve_json(answer):
    import http.server

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            data = json.dumps(answer).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_a_remote_answer_without_predictions_is_reported_as_failed():
    server = serve_json({"model_version": "v1"})
    try:
        report = parity.check_remote(
            f"http://127.0.0.1:{server.server_port}", GOLDEN
        )
    finally:
        server.shutdown()
        server.server_close()
    assert report["passed"] is False
    assert any("predictions" in m for m in report["mismatches"])


def test_a_timeout_raised_by_the_work_itself_is_not_a_deadline_error():
    control = guard.AdmissionControl(1, 5.0)

    def work():
        raise TimeoutError("upstream lookup timed out")

    try:
        with pytest.raises(TimeoutError) as caught:
            control.run(work)
        assert not isinstance(caught.value, guard.DeadlineError)
    finally:
        control.close()
