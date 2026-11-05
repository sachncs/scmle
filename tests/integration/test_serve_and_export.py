"""The serve and export-onnx commands, and the server's refusals."""

import importlib.util
import json
import threading
import types
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest
import support

from scmle import cli
from scmle.artifacts import bundle as bundle_module
from scmle.contracts import data as data_contracts
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.curation import preprocessing
from scmle.inference import serving
from scmle.models import registry as registry_module


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    frame["y"] = (frame.a + frame.b > 0).astype(int)
    root = tmp_path_factory.mktemp("served")
    result = support.make_project(
        frame, "y", time_budget=300, output_dir=root / "runs"
    ).run()
    result.bundle.save(root / "bundle")
    frame.drop(columns="y").to_csv(root / "rows.csv", index=False)
    return root, result


def request(url, body, content_type):
    request = urllib.request.Request(url, body, {"Content-Type": content_type})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def post(url, body):
    return request(url, json.dumps(body).encode(), "application/json")


def test_the_server_rejects_schema_violations_with_422_and_a_reason(saved):
    root = saved[0]
    model = bundle_module.ModelBundle.load(root / "bundle", trusted=True)
    adapter = serving.HttpServing(model)
    adapter.start()
    try:
        status, body = post(
            f"{adapter.address}/predict", {"records": [{"a": 1.0}]}
        )
    finally:
        adapter.stop()

    assert status == 422 and "missing" in body["details"]


def test_the_server_rejects_empty_prediction_requests(saved):
    model = bundle_module.ModelBundle.load(saved[0] / "bundle", trusted=True)
    adapter = serving.HttpServing(model)
    adapter.start()
    try:
        status, body = post(f"{adapter.address}/predict", {"records": []})
    finally:
        adapter.stop()

    assert status == 400
    assert body["error"] == "malformed request: records must not be empty"
    assert body["category"] == "request"
    assert body["schema_version"] == "1" and body["request_id"]


def test_the_server_refuses_oversized_bodies_with_413(saved):
    model = bundle_module.ModelBundle.load(saved[0] / "bundle", trusted=True)
    adapter = serving.HttpServing(model, max_request_bytes=10)
    adapter.start()
    try:
        status, body = post(
            f"{adapter.address}/predict", {"records": [{"a": 1.0, "b": 2.0}]}
        )
    finally:
        adapter.stop()

    assert status == 413 and body["category"] == "request"


def test_the_server_accepts_instance_request_bounds(saved):
    model = bundle_module.ModelBundle.load(saved[0] / "bundle", trusted=True)
    adapter = serving.HttpServing(
        model, max_request_bytes=10_000, max_records=1
    )
    adapter.start()
    try:
        status, body = post(
            f"{adapter.address}/predict",
            {"records": [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]},
        )
    finally:
        adapter.stop()

    assert status == 413 and body["category"] == "request"


def test_the_server_refuses_too_many_records_with_413(saved):
    model = bundle_module.ModelBundle.load(saved[0] / "bundle", trusted=True)
    adapter = serving.HttpServing(model, max_records=1)
    adapter.start()
    try:
        status, body = post(
            f"{adapter.address}/predict",
            {"records": [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]},
        )
    finally:
        adapter.stop()

    assert status == 413 and body["category"] == "request"


def test_the_server_rejects_invalid_request_bounds(saved):
    model = bundle_module.ModelBundle.load(saved[0] / "bundle", trusted=True)

    with pytest.raises(ValueError, match="request bounds"):
        serving.HttpServing(model, max_request_bytes=0)

    with pytest.raises(ValueError, match="request bounds"):
        serving.HttpServing(model, max_records=0)

    with pytest.raises(ValueError, match="timeout"):
        serving.HttpServing(model, request_timeout_seconds=0)


def test_the_server_classifies_malformed_and_internal_requests(
    saved, monkeypatch
):
    """The HTTP boundary emits stable classes for malformed failures."""
    model = bundle_module.ModelBundle.load(saved[0] / "bundle", trusted=True)
    adapter = serving.HttpServing(model)
    adapter.start()

    def fail(records):
        raise RuntimeError("model failed")

    try:
        status, body = request(
            f"{adapter.address}/predict", b"{}", "text/plain"
        )
        assert status == 415 and body["category"] == "request"

        status, body = request(
            f"{adapter.address}/predict", b"not-json", "application/json"
        )
        assert status == 400 and body["category"] == "request"

        monkeypatch.setattr(adapter, "respond", fail)
        status, body = post(
            f"{adapter.address}/predict", {"records": [{"a": 1.0, "b": 2.0}]}
        )
        assert status == 500
        assert body["error"] == "internal serving failure"
        assert body["category"] == "internal"
        assert "model failed" not in json.dumps(body)
    finally:
        adapter.stop()


def test_health_reports_the_model_version_and_checksum(saved):
    root = saved[0]
    model = bundle_module.ModelBundle.load(root / "bundle", trusted=True)
    adapter = serving.HttpServing(model)
    adapter.start()
    try:
        with urllib.request.urlopen(f"{adapter.address}/health") as response:
            health = json.loads(response.read())
            headers = response.headers
    finally:
        adapter.stop()

    assert health["model_version"] == model.info.model_version
    assert health["model_sha256"] == model.info.files[bundle_module.MODEL_FILE]
    assert len(health["model_sha256"]) == 64
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Server"] == "scmle-reference"


def test_the_server_rejects_starting_twice(saved):
    """The serving lifecycle cannot replace a running socket silently."""
    model = bundle_module.ModelBundle.load(saved[0] / "bundle", trusted=True)
    adapter = serving.HttpServing(model)
    adapter.start()
    try:
        with pytest.raises(errors.ScmleError, match="already running"):
            adapter.start()
    finally:
        adapter.stop()


def test_an_unsigned_bundle_is_not_served_unless_trusted(saved):
    root = saved[0]

    with pytest.raises(errors.ArtifactError, match="unauthenticated"):
        cli.ServeCommand().execute(
            cli.build_parser().parse_args(["serve", str(root / "bundle")])
        )


def test_the_serve_command_starts_a_server_for_a_trusted_bundle(
    saved, monkeypatch
):
    root = saved[0]
    started = threading.Event()
    seen = {}

    class Recorder:
        def wait(self):
            seen["address"] = adapters[-1].address
            started.set()
            raise KeyboardInterrupt

    adapters = []
    real = serving.HttpServing

    def build(*args, **kwargs):
        adapters.append(real(*args, **kwargs))
        return adapters[-1]

    monkeypatch.setattr(cli.serving, "HttpServing", build)
    monkeypatch.setattr(cli, "threading", types.SimpleNamespace(Event=Recorder))

    code = cli.main(["serve", str(root / "bundle"), "--trusted", "--port", "0"])

    assert code == 0 and started.is_set()
    assert seen["address"].startswith("http://127.0.0.1:")


@pytest.fixture(scope="module")
def exportable(tmp_path_factory):
    """A saved bundle of a model family the ONNX converter supports."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    target = (frame.a + frame.b > 0).astype(int)
    plan = data_contracts.FeaturePlan(("a", "b"), ())
    fitted = preprocessing.build_pipeline(
        plan,
        registry_module.ModelRegistry.with_builtins().get(
            "logistic_regression"
        ),
        tasks.TaskType.BINARY,
        0,
    ).fit(frame, target)
    bundle = bundle_module.ModelBundle.create(
        fitted, plan, frame, "y", "binary_classification", "roc_auc", "lr"
    )
    root = tmp_path_factory.mktemp("exportable")
    bundle.save(root / "bundle")
    frame.to_csv(root / "rows.csv", index=False)
    return root


needs_onnx = pytest.mark.skipif(
    importlib.util.find_spec("skl2onnx") is None
    or importlib.util.find_spec("onnxruntime") is None,
    reason="the onnx extra is not installed",
)


@needs_onnx
def test_export_onnx_command_writes_a_graph_that_matches_the_pipeline(
    exportable, tmp_path
):
    output = tmp_path / "m.onnx"

    code = cli.main(
        [
            "export-onnx",
            str(exportable / "bundle"),
            "--trusted",
            "--output",
            str(output),
            "--check",
            str(exportable / "rows.csv"),
        ]
    )

    assert code == cli.EXIT_OK and output.stat().st_size > 0


@needs_onnx
def test_export_onnx_command_fails_cleanly_for_an_unsupported_model(
    saved, tmp_path, caplog
):
    root = saved[0]
    unsupported = bundle_module.ModelBundle.load(root / "bundle", trusted=True)
    unsupported.pipeline = object()
    unsupported.save(tmp_path / "bundle")

    code = cli.main(
        [
            "export-onnx",
            str(tmp_path / "bundle"),
            "--trusted",
            "--output",
            str(tmp_path / "m.onnx"),
        ]
    )

    assert code == cli.EXIT_FAILED
    assert "not supported" in caplog.text
    assert not (tmp_path / "m.onnx").exists()
