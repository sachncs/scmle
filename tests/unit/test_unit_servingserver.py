"""Behaviour of the serving contract, batch scoring and the HTTP adapter."""

import http.client
import json
import pathlib

import pandas
import pytest

import scmle.artifacts.bundle
from scmle.contracts import errors
from scmle.contracts import serving
from scmle.deploy import parity
from scmle.deploy import provenance
from scmle.inference import guard
from scmle.serving.server import batch
from scmle.serving.server import contract
from scmle.serving.server import handler
from scmle.serving.server import web

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "deploy" / "local" / "golden"
ROW = {"age": 52.0, "balance": 2300.0, "plan": "pro"}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    folder = tmp_path_factory.mktemp("golden")
    parity.handwrite(parity.goldens(GOLDEN).spec).save(folder / "bundle")
    return scmle.artifacts.bundle.Bundle.load(folder / "bundle", trusted=True)


@pytest.fixture
def scorer(bundle):
    return batch.Batcher(bundle)


class Monitor:
    def __init__(self, broken=False):
        self.broken = broken
        self.requests = []
        self.frames = []

    def tally(self, at, route, status, seconds):
        if self.broken:
            raise RuntimeError("monitor down")
        self.requests.append((route, status))

    def sense(self, at, frame):
        if self.broken:
            raise RuntimeError("monitor down")
        self.frames.append(len(frame))
        return len(frame)


@pytest.fixture
def running(bundle):
    monitor = Monitor()
    adapter = web.HttpServing(bundle, monitor=monitor, max_request_bytes=4096)
    adapter.start()
    yield adapter
    adapter.stop()


def call(adapter, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection(
        adapter.host, adapter.httpd.server_address[1], timeout=10
    )
    sent = {"Content-Type": "application/json", **(headers or {})}
    connection.request(method, path, body, sent)
    response = connection.getresponse()
    payload = json.loads(response.read() or b"{}")
    found = dict(response.getheaders())
    connection.close()
    return response.status, payload, found


def post(adapter, payload, path="/predict", headers=None):
    return call(adapter, "POST", path, json.dumps(payload).encode(), headers)


# ---- contract --------------------------------------------------------------


def test_contract_limits_are_the_documented_policy():
    assert contract.MAX_REQUEST_BYTES == 16 * 1024 * 1024
    assert contract.MAX_REQUEST_RECORDS == 10_000
    assert contract.MAX_CONCURRENT_REQUESTS == 8
    assert contract.SCORING_TIMEOUT_SECONDS == 30.0
    assert contract.BATCH_CHUNK_ROWS == 10_000


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"on_null": "drop"}, "on_null must be in"),
        ({"on_unseen": "x"}, "on_unseen"),
    ],
)
def test_a_server_refuses_unknown_policies(bundle, changes, message):
    with pytest.raises(errors.ConfigError, match=message):
        batch.Batcher(bundle, **changes)


def test_a_server_with_a_bundle_has_verified_and_loaded(scorer):
    snapshot = scorer.readiness.snapshot()
    assert snapshot["ready"] is False
    assert snapshot["notes"]["verifying"] == "bundle provided by the caller"
    assert snapshot["notes"]["loading"] == "bundle provided by the caller"
    assert scorer.on_null == "impute"
    assert scorer.on_unseen == "report"


def test_a_server_without_a_bundle_is_not_loaded():
    empty = batch.Batcher(None)
    with pytest.raises(guard.UnreadyError, match="the model is not loaded"):
        assert empty.bundle is None
    assert empty.readiness.snapshot()["completed"] == {}


def test_digests_name_the_manifest_and_the_schema(scorer, bundle):
    assert scorer.artifact_digest == provenance.manifest_digest(
        bundle.info.to_dict()
    )
    assert scorer.schema_digest == serving.schema_digest(bundle.schema.anchor())
    assert scorer.schema_digest.startswith("sha256:")


def test_synthesise_uses_zero_and_the_first_known_category(scorer):
    assert scorer.synthesise() == {"age": 0.0, "balance": 0.0, "plan": "basic"}


def test_warm_up_with_a_synthetic_record_makes_the_server_ready(scorer):
    scorer.warm()
    snapshot = scorer.readiness.snapshot()
    assert snapshot["ready"] is True
    assert snapshot["notes"]["warming"] == "synthetic record from the schema"


def test_warm_up_with_explicit_records_notes_them(bundle):
    server = batch.Batcher(bundle, warmup_records=[ROW])
    server.warm()
    assert server.readiness.snapshot()["notes"]["warming"] == (
        "explicit warm-up records"
    )


def test_a_failing_warm_up_fails_readiness_with_the_error_type(bundle):
    server = batch.Batcher(bundle, warmup_records=[{"nothing": object()}])
    server.warm()
    snapshot = server.readiness.snapshot()
    assert snapshot["ready"] is False
    assert snapshot["reason"].startswith("warm-up failed: ")


def test_respond_refuses_no_records(scorer):
    with pytest.raises(errors.InputError, match="records must not be empty"):
        scorer.respond([])


def test_respond_names_missing_features_and_their_records(scorer):
    records = [ROW, {"age": 1.0}, {"age": 1.0, "balance": 2.0}]
    with pytest.raises(errors.SchemaError) as caught:
        scorer.respond(records)
    assert "['balance', 'plan']" in str(caught.value)
    assert caught.value.details == {
        "missing": ["balance", "plan"],
        "records": [1, 2],
    }


def test_respond_scores_and_reports_each_row(scorer, bundle):
    response = scorer.respond([ROW, {**ROW, "plan": "gold", "age": None}])
    assert response["predictions"][0] == 1
    assert response["schema_version"] == serving.SCHEMA_VERSION
    assert response["artifact_digest"] == scorer.artifact_digest
    assert response["schema_digest"] == scorer.schema_digest
    assert response["record_count"] == 2
    assert len(response["rows"]) == 2
    assert not response["rows"][0]["missing"]
    assert response["rows"][1]["missing"]
    assert response["rows"][1]["unseen"]


def test_policies_reject_nulls_and_unseen_categories(bundle):
    rows = [
        {"missing": [], "unseen": []},
        {"missing": ["age"], "unseen": []},
        {"missing": [], "unseen": ["plan"]},
    ]
    nulls = batch.Batcher(bundle, on_null="reject")
    with pytest.raises(serving.PolicyError, match="null features") as caught:
        nulls.enforce(rows)
    assert caught.value.details == {"records": [1]}
    unseen = batch.Batcher(bundle, on_unseen="reject")
    with pytest.raises(
        serving.PolicyError, match="unseen categories"
    ) as caught:
        unseen.enforce(rows)
    assert caught.value.details == {"records": [2]}
    batch.Batcher(bundle).enforce(rows)


def test_policy_details_list_at_most_twenty_records(bundle):
    rows = [{"missing": ["age"], "unseen": []}] * 25
    with pytest.raises(serving.PolicyError) as caught:
        batch.Batcher(bundle, on_null="reject").enforce(rows)
    assert caught.value.details["records"] == list(range(20))


# ---- batch -----------------------------------------------------------------


def test_score_file_writes_predictions_and_a_manifest(scorer, tmp_path):
    source = tmp_path / "in.csv"
    rows = [ROW, {**ROW, "plan": "gold"}, {**ROW, "age": None}]
    pandas.DataFrame(rows).to_csv(source, index=False)
    target = tmp_path / "out.csv"
    assert scorer.score_file(source, target, chunk_rows=2) == 3
    output = pandas.read_csv(target)
    assert list(output.columns) == ["prediction", "proba_0", "proba_1"]
    assert len(output) == 3
    manifest = json.loads((tmp_path / "out.csv.manifest.json").read_text())
    content = target.read_bytes()
    assert manifest == {
        "schema_version": serving.SCHEMA_VERSION,
        "model_version": "golden-1",
        "artifact_digest": scorer.artifact_digest,
        "schema_digest": scorer.schema_digest,
        "rows": 3,
        "output_sha256": provenance.sha256_digest(content),
        "rows_with_nulls": 1,
        "rows_with_unseen_categories": 1,
    }


def test_chunking_does_not_change_the_scores(scorer, tmp_path):
    source = tmp_path / "in.csv"
    pandas.DataFrame(parity.goldens(GOLDEN).records).to_csv(source, index=False)
    scorer.score_file(source, tmp_path / "one.csv", chunk_rows=1)
    scorer.score_file(source, tmp_path / "all.csv")
    assert (tmp_path / "one.csv").read_bytes() == (
        tmp_path / "all.csv"
    ).read_bytes()


def test_score_file_refuses_a_non_positive_chunk(scorer, tmp_path):
    with pytest.raises(errors.ConfigError, match="chunk_rows must be positive"):
        scorer.score_file(
            tmp_path / "in.csv", tmp_path / "out.csv", chunk_rows=0
        )


def test_a_batcher_starts_and_stops_without_effect(scorer):
    assert scorer.start() is None
    assert scorer.stop() is None


# ---- web -------------------------------------------------------------------


def test_an_adapter_needs_exactly_one_of_bundle_and_loader(bundle):
    with pytest.raises(errors.ConfigError, match="exactly one"):
        web.HttpServing()
    with pytest.raises(errors.ConfigError, match="exactly one"):
        web.HttpServing(bundle, loader=lambda readiness: bundle)


@pytest.mark.parametrize(
    "changes",
    [
        {"max_request_bytes": 0},
        {"max_records": 0},
        {"request_timeout_seconds": 0},
    ],
)
def test_an_adapter_needs_positive_bounds(bundle, changes):
    with pytest.raises(errors.InputError, match="must be positive"):
        web.HttpServing(bundle, **changes)


def test_an_adapter_on_a_public_address_needs_a_gateway(bundle):
    with pytest.raises(errors.ConfigError):
        web.HttpServing(bundle, host="0.0.0.0")


def test_an_adapter_reports_its_limits_and_has_no_address_yet(bundle):
    adapter = web.HttpServing(
        bundle, max_records=5, max_concurrent=2, scoring_timeout_seconds=3.0
    )
    assert adapter.limits() == {
        "max_request_bytes": contract.MAX_REQUEST_BYTES,
        "max_records": 5,
        "max_concurrent": 2,
        "scoring_timeout_seconds": 3.0,
        "request_timeout_seconds": 10.0,
        "on_null": "impute",
        "on_unseen": "report",
    }
    with pytest.raises(errors.ScmleError, match="server is not running"):
        assert adapter.address is None
    assert adapter.identity()["model_version"] == "golden-1"
    adapter.admission.close()


def test_identity_is_empty_before_loading(bundle):
    adapter = web.HttpServing(loader=lambda readiness: bundle)
    assert adapter.identity() == {}
    adapter.admission.close()


def test_boot_without_a_loader_is_an_error(bundle):
    adapter = web.HttpServing(bundle)
    with pytest.raises(errors.ScmleError, match="no loader to run"):
        adapter.boot()
    adapter.admission.close()


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (errors.ConfigError("bad key"), "config: bad key"),
        (errors.InputError("/secret/path"), "InputError"),
        (OSError("disk"), "OSError"),
    ],
)
def test_a_failed_loader_keeps_only_a_safe_reason(error, reason):
    def loader(readiness):
        raise error

    adapter = web.HttpServing(loader=loader)
    adapter.boot()
    snapshot = adapter.readiness.snapshot()
    assert snapshot["ready"] is False
    assert snapshot["reason"] == reason
    adapter.admission.close()


def test_a_loader_that_succeeds_warms_up(bundle):
    def loader(readiness):
        readiness.begin(guard.Phase.VERIFYING)
        readiness.complete(guard.Phase.VERIFYING, "ok")
        readiness.begin(guard.Phase.LOADING)
        readiness.complete(guard.Phase.LOADING, "ok")
        return bundle

    adapter = web.HttpServing(loader=loader)
    adapter.boot()
    assert adapter.readiness.snapshot()["ready"] is True
    assert adapter.identity()["model_version"] == "golden-1"
    adapter.admission.close()


def test_a_monitor_failure_never_escapes(bundle):
    adapter = web.HttpServing(bundle, monitor=Monitor(broken=True))
    adapter.observe("/predict", 200, 0.1)
    adapter.feed([ROW])
    web.HttpServing(bundle).feed([ROW])
    adapter.admission.close()


def test_score_refuses_an_unexpected_schema(bundle):
    adapter = web.HttpServing(bundle)
    request = serving.parse(
        {"records": [ROW], "expected_schema_digest": "sha256:" + "0" * 64}, 10
    )
    with pytest.raises(serving.MismatchError) as caught:
        adapter.score(request)
    assert caught.value.details["served"] == adapter.schema_digest
    exact = serving.parse(
        {"records": [ROW], "expected_schema_digest": adapter.schema_digest}, 10
    )
    assert adapter.score(exact)["record_count"] == 1
    adapter.admission.close()


def test_a_running_server_answers_its_probes(running):
    assert running.address.startswith("http://127.0.0.1:")
    status, body, headers = call(running, "GET", "/livez")
    assert status == 200
    assert body["status"] == "alive"
    assert headers["Server"].startswith("scmle-reference")
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Request-ID"] == body["request_id"]
    status, body, ignored = call(running, "GET", "/readyz")
    assert status == 200
    assert body["ready"] is True
    assert body["model_version"] == "golden-1"
    status, body, ignored = call(running, "GET", "/health")
    assert (status, body["status"], body["model_version"]) == (
        200,
        "ok",
        "golden-1",
    )
    status, body, ignored = call(running, "GET", "/schema")
    assert status == 200
    assert body["schema_digest"] == running.schema_digest
    assert body["limits"]["max_request_bytes"] == 4096
    status, body, ignored = call(running, "GET", "/nowhere")
    assert (status, body["error"], body["category"]) == (
        404,
        "not found",
        "request",
    )


def test_a_running_server_scores_and_tells_the_monitor(running):
    status, body, ignored = post(
        running, {"records": [ROW]}, headers={"X-Request-ID": "abc-1"}
    )
    assert status == 200
    assert body["predictions"] == [1]
    assert body["request_id"] == "abc-1"
    status, body, ignored = post(running, {"records": [ROW]}, "/v1/predict")
    assert status == 200
    assert running.monitor.frames == [1, 1]
    assert ("/predict", 200) in running.monitor.requests


def test_a_server_refuses_a_bad_request_id(running):
    status, body, ignored = call(
        running, "GET", "/livez", headers={"X-Request-ID": "bad id!"}
    )
    assert status == 400
    assert "X-Request-ID must be 1-128 ASCII characters" in body["error"]
    assert body["request_id"] != "bad id!"


@pytest.mark.parametrize(
    ("headers", "body", "status", "message"),
    [
        ({"Content-Type": "text/plain"}, b"{}", 415, "Content-Type must be"),
        (
            {"Content-Length": "x"},
            b"",
            400,
            "Content-Length must be an integer",
        ),
        ({"Content-Length": "5000"}, b"", 413, "exceeds 4096 bytes"),
        ({}, b"{not json", 400, "invalid JSON request"),
        ({}, b'{"records": [], "x": 1}', 400, "malformed request"),
        ({}, b'{"records": [{"age": 1}]}', 422, "missing required features"),
    ],
)
def test_a_server_maps_request_failures_to_statuses(
    running, headers, body, status, message
):
    answered, payload, ignored = call(
        running, "POST", "/predict", body, headers
    )
    assert answered == status
    assert message in payload["error"]


def test_a_server_needs_a_content_length(running):
    connection = http.client.HTTPConnection(
        running.host, running.httpd.server_address[1], timeout=10
    )
    connection.putrequest("POST", "/predict")
    connection.putheader("Content-Type", "application/json")
    connection.endheaders()
    response = connection.getresponse()
    payload = json.loads(response.read())
    connection.close()
    assert response.status == 411
    assert payload["error"] == "Content-Length is required"


def test_a_server_answers_409_for_another_schema(running):
    status, body, ignored = post(
        running,
        {"records": [ROW], "expected_schema_digest": "sha256:" + "0" * 64},
    )
    assert status == 409
    assert body["details"]["served"] == running.schema_digest


def test_a_post_to_an_unknown_path_is_not_found(running):
    status = post(running, {"records": [ROW]}, "/score")[0]
    assert status == 404


def test_a_server_that_is_not_ready_answers_503(bundle):
    def loader(readiness):
        raise errors.ConfigError("no")

    adapter = web.HttpServing(loader=loader)
    adapter.start()
    adapter.loading.join(5)
    try:
        status, body, headers = call(adapter, "GET", "/readyz")
        assert status == 503
        assert headers["Retry-After"] == "1"
        status, body, ignored = call(adapter, "GET", "/health")
        assert (status, body["category"]) == (503, "not_ready")
        status, body, ignored = post(adapter, {"records": [ROW]})
        assert (status, body["error"]) == (503, "service is not ready")
    finally:
        adapter.stop()


def test_starting_twice_is_refused(running):
    with pytest.raises(errors.ScmleError, match="already running"):
        running.start()


def test_the_listener_serves_on_daemon_threads():
    assert web.Listener.daemon_threads is True
    assert web.Listener.allow_reuse_address is True
    assert issubclass(handler.Responder, handler.Exchange)
