"""The serving contract: schemas, readiness, limits, policies, parity."""

import json
import pathlib
import threading
import time
import urllib.error
import urllib.request

import pandas
import pytest

from scmle import cli
import scmle.artifacts.bundle
from scmle.contracts import errors
import scmle.contracts.serving
from scmle.deploy import parity
from scmle.deploy import provenance
from scmle.inference import guard
import scmle.serving.server

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "deploy" / "local" / "golden"
SHA = r"^sha256:[0-9a-f]{64}$"


@pytest.fixture(scope="module")
def golden():
    return parity.goldens(GOLDEN)


@pytest.fixture(scope="module")
def expected():
    return parity.read_json(GOLDEN / "expected.json")


@pytest.fixture(scope="module")
def bundle(golden, tmp_path_factory):
    folder = tmp_path_factory.mktemp("golden")
    parity.handwrite(golden.spec).save(folder / "bundle")
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
    adapter = scmle.serving.server.HttpServing(bundle)
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
    recomputed = parity.expect(golden.spec, golden.records)
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
    batch = scmle.serving.server.Batcher(bundle)
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
    assert scmle.serving.server.HttpServing(loaded).artifact_digest == (
        provenance.bundle_digest(folder)
    )


def test_the_versioned_path_and_unknown_versions(golden, running):
    # Enforces docs/SERVICE_SLO.md:48.
    # Enforces docs/SERVICE_SLO.md:36.
    # Enforces website/src/pages/docs/service.astro:53.
    # Enforces website/src/pages/docs/service.astro:61.
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
    # Enforces docs/SERVICE_SLO.md:56.
    # Enforces docs/SERVICE_SLO.md:38.
    # Enforces website/src/pages/docs/service.astro:14.
    # Enforces website/src/pages/docs/service.astro:53.
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
    # Enforces docs/SERVICE_SLO.md:56.
    # Enforces website/src/pages/docs/service.astro:14.
    strict = scmle.serving.server.HttpServing(
        bundle, on_null="reject", on_unseen="reject"
    )
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
        scmle.serving.server.HttpServing(bundle, on_null="guess")


def test_schema_changes_are_explicit(golden, running):
    # Enforces docs/SERVICE_SLO.md:38.
    # Enforces docs/WHEN_THINGS_GO_WRONG.md:530.
    # Enforces website/src/learn/when-things-go-wrong.html:221.
    # Enforces website/src/pages/docs/service.astro:53.
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
    adapter = scmle.serving.server.HttpServing(
        bundle, max_request_bytes=200, max_records=2
    )
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
    adapter = scmle.serving.server.HttpServing(bundle, max_concurrent=1)
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
    # Enforces docs/SERVICE_SLO.md:71.
    # Enforces website/src/pages/docs/service.astro:26.
    adapter = scmle.serving.server.HttpServing(
        bundle, scoring_timeout_seconds=0.2
    )
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


def settle(adapter, seconds=10.0):
    """Waits until no worker is busy; returns whether that happened."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if adapter.admission.state()["busy"] == 0:
            return True
        time.sleep(0.02)
    return False


def test_a_hung_call_leaves_the_replica_serving_on_its_other_workers(
    golden, bundle, monkeypatch
):
    # SRV-1: a timed-out call kept its worker and the lock while readiness
    # stayed true, so traffic kept arriving and queued behind it. C7: the
    # fix then refused every request while one call hung, so a call that
    # never returned wedged the replica: not ready, never restarted.
    # Enforces website/src/pages/docs/service.astro:26.
    adapter = scmle.serving.server.HttpServing(
        bundle, max_concurrent=2, scoring_timeout_seconds=0.2
    )
    gate = SlowGate()
    original = adapter.respond

    def hung(records):
        if records[0].get("note") == "hang":
            gate.release.wait(20)
        return original(records)

    monkeypatch.setattr(adapter, "respond", hung)
    adapter.start()
    url = f"{adapter.address}/predict"
    stalled = {"records": [{**golden.records[0], "note": "hang"}]}
    plain = {"records": golden.records[:1]}
    try:
        assert call(url, stalled)[0] == 504
        status, body = call(f"{adapter.address}/readyz")
        assert status == 200 and body["workers"]["stuck"] == 1
        started = time.monotonic()
        assert [call(url, plain)[0] for ignored in range(5)] == [200] * 5
        assert time.monotonic() - started < 2
        assert call(f"{adapter.address}/livez")[0] == 200
        assert call(url, stalled)[0] == 504
        status, body = call(f"{adapter.address}/readyz")
        assert status == 503
        assert body["reason"] == "2 timed-out call(s) hold every worker"
        status, body = call(url, plain)
        assert status == 503 and body["category"] == "not_ready"
        gate.release.set()
        assert settle(adapter)
        assert call(f"{adapter.address}/readyz")[0] == 200
        assert call(url, plain)[0] == 200
        assert adapter.admission.state()["misses"] == 0
    finally:
        gate.release.set()
        adapter.stop()


def test_a_replica_no_call_can_run_on_stops_being_alive(
    golden, bundle, monkeypatch
):
    adapter = scmle.serving.server.HttpServing(
        bundle, max_concurrent=1, scoring_timeout_seconds=0.1
    )
    adapter.capacity.patience = 0.3
    gate = SlowGate()
    original = adapter.respond

    def hung(records):
        gate.release.wait(20)
        return original(records)

    monkeypatch.setattr(adapter, "respond", hung)
    adapter.start()
    try:
        assert (
            call(f"{adapter.address}/predict", {"records": golden.records[:1]})[
                0
            ]
            == 504
        )
        assert call(f"{adapter.address}/livez")[0] == 200
        time.sleep(0.4)
        status, body = call(f"{adapter.address}/livez")
        assert status == 503 and body["status"] == "wedged"
        assert body["misses"] == 1
    finally:
        gate.release.set()
        adapter.stop()


def test_consecutive_timeouts_fail_liveness_and_never_queue():
    # Enforces docs/SERVICE_SLO.md:71.
    # Enforces website/src/pages/docs/service.astro:26.
    control = guard.Admission(2, 0.05, strikes=2)
    release = threading.Event()
    try:
        for ignored in range(2):
            with pytest.raises(guard.DeadlineError):
                control.run(lambda: release.wait(20))
        assert not control.alive and not control.clear
        assert control.state() == {"busy": 2, "stuck": 2, "misses": 2}
        with pytest.raises(guard.OverloadedError, match="workers are busy"):
            control.run(lambda: 1)
        release.set()
        deadline = time.monotonic() + 10
        while not control.clear and time.monotonic() < deadline:
            time.sleep(0.02)
        assert control.clear
        assert control.run(lambda: 1) == 1
        assert control.alive and control.state()["misses"] == 0
    finally:
        release.set()
        control.close()


def test_an_abandoned_call_knows_before_it_mutates_anything():
    # SRV-2: work reported as failed still ran and mutated series history.
    control = guard.Admission(1, 0.05)
    release = threading.Event()
    seen = []

    def work():
        release.wait(20)
        seen.append(guard.abandoned())

    try:
        with pytest.raises(guard.DeadlineError):
            control.run(work)
        release.set()
        deadline = time.monotonic() + 10
        while not seen and time.monotonic() < deadline:
            time.sleep(0.02)
        assert seen == [True]
        assert control.run(guard.abandoned) is False
        assert guard.abandoned() is False
    finally:
        release.set()
        control.close()


def test_readiness_follows_verify_then_load_then_warm_up(golden, bundle):
    # Enforces docs/SERVICE_SLO.md:15.
    # Enforces website/src/pages/docs/service.astro:7.
    order = []

    def loader(readiness):
        readiness.begin(guard.Phase.VERIFYING)
        order.append("verify")
        readiness.complete(guard.Phase.VERIFYING)
        readiness.begin(guard.Phase.LOADING)
        order.append("load")
        readiness.complete(guard.Phase.LOADING)
        return bundle

    adapter = scmle.serving.server.HttpServing(loader=loader)
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
    # Enforces docs/SERVICE_SLO.md:15.
    # Enforces docs/SERVICE_SLO.md:14.
    # Enforces docs/GLOSSARY.md:629.
    # Enforces docs/WHEN_THINGS_GO_WRONG.md:505.
    # Enforces website/src/learn/glossary.html:314.
    # Enforces website/src/learn/when-things-go-wrong.html:211.
    # Enforces website/src/pages/docs/service.astro:6.
    # Enforces website/src/pages/docs/service.astro:7.
    def loader(readiness):
        readiness.begin(guard.Phase.VERIFYING)
        raise errors.ArtifactError("bundle signature is invalid")

    adapter = scmle.serving.server.HttpServing(loader=loader)
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
    # Enforces docs/SERVICE_SLO.md:15.
    # Enforces website/src/pages/docs/service.astro:7.
    release = threading.Event()

    def loader(readiness):
        readiness.begin(guard.Phase.VERIFYING)
        release.wait(20)
        readiness.complete(guard.Phase.VERIFYING)
        readiness.begin(guard.Phase.LOADING)
        readiness.complete(guard.Phase.LOADING)
        return bundle

    adapter = scmle.serving.server.HttpServing(loader=loader)
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
    adapter = scmle.serving.server.HttpServing(
        bundle, warmup_records=[{"age": "not a number"}]
    )
    adapter.start()
    try:
        status, body = call(f"{adapter.address}/readyz")
        assert status == 503 and "warm-up failed" in body["reason"]
    finally:
        adapter.stop()


def test_readiness_steps_cannot_be_skipped_or_reordered():
    # Enforces docs/SERVICE_SLO.md:15.
    # Enforces website/src/pages/docs/service.astro:7.
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


def test_a_loader_input_error_shows_its_type_not_its_message():
    # ERR-20 turned the loader's bare ValueErrors into InputError, a
    # ScmleError; readiness then published their messages (paths, values).
    # Only a framework failure's category and message are shown; an input
    # error keeps the type-name-only form a ValueError had.
    def loader(readiness):
        readiness.begin(guard.Phase.VERIFYING)
        raise errors.InputError("key file /secret/path is empty")

    adapter = scmle.serving.server.HttpServing(loader=loader)
    adapter.start()
    try:
        assert not wait_until_settled(adapter, 20)
        status, body = call(f"{adapter.address}/readyz")
        assert status == 503
        assert "/secret/path" not in body["reason"]
        assert "InputError" in body["reason"]
    finally:
        adapter.stop()


def test_the_loader_and_bundle_arguments_are_exclusive(bundle):
    with pytest.raises(errors.ConfigError, match="exactly one"):
        scmle.serving.server.HttpServing()
    with pytest.raises(errors.ConfigError, match="exactly one"):
        scmle.serving.server.HttpServing(
            bundle, loader=lambda readiness: bundle
        )
    with pytest.raises(errors.ConfigError, match="positive"):
        guard.Admission(0, 1.0)


def test_the_batch_command_scores_a_file_and_writes_a_manifest(
    golden, expected, bundle, tmp_path
):
    folder = bundle.save(tmp_path / "bundle")
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
    verify.refuse(demo, True, allowed=True)


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
        report = parity.crosscheck(
            f"http://127.0.0.1:{server.server_port}", GOLDEN
        )
    finally:
        server.shutdown()
        server.server_close()
    assert report["passed"] is False
    assert any("predictions" in m for m in report["mismatches"])


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "ftp://example.com/", "127.0.0.1:8000"]
)
def test_a_remote_check_refuses_a_url_that_is_not_http_or_https(url):
    with pytest.raises(ValueError, match="http or https"):
        parity.crosscheck(url, GOLDEN)


def test_a_timeout_raised_by_the_work_itself_is_not_a_deadline_error():
    control = guard.Admission(1, 5.0)

    def work():
        raise TimeoutError("upstream lookup timed out")

    try:
        with pytest.raises(TimeoutError) as caught:
            control.run(work)
        assert not isinstance(caught.value, guard.DeadlineError)
    finally:
        control.close()


def test_a_read_only_model_scores_beside_a_hung_call(
    golden, bundle, monkeypatch
):
    # SRV-1: one global lock serialized every scoring call, so a hung call
    # blocked every worker although a model without series is read-only.
    assert not bundle.schema.series
    adapter = scmle.serving.server.HttpServing(bundle)
    gate = SlowGate()
    original = bundle.score

    def score(frame, *args, **kwargs):
        if threading.current_thread().name == "hung":
            gate.entered.set()
            gate.release.wait(20)
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(bundle, "score", score)
    frame = pandas.DataFrame(golden.records[:1])
    hung = threading.Thread(target=adapter.infer, args=(frame,), name="hung")
    hung.start()
    try:
        assert gate.entered.wait(10)
        done = []
        other = threading.Thread(
            target=lambda: done.append(adapter.infer(frame))
        )
        other.start()
        other.join(10)
        assert done and done[0]["record_count"] == 1
    finally:
        gate.release.set()
        hung.join(10)
