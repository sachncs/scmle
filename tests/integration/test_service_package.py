"""Any model, including ones ONNX cannot express, ships as a service."""

import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy
import pandas
import pytest

from scmle import cli
from scmle.artifacts import bundle as bundle_module
from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.curation import preprocessing
from scmle.models import registry

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def boosted(tmp_path_factory):
    """A bundle of the model family the ONNX converter refuses."""
    rng = numpy.random.default_rng(0)
    frame = pandas.DataFrame(
        {"a": rng.normal(size=400), "b": rng.normal(size=400)}
    )
    target = (frame.a + frame.b > 0).astype(int)
    plan = data.FeaturePlan(("a", "b"), ())
    fitted = preprocessing.build_pipeline(
        plan,
        registry.ModelRegistry.with_builtins().get("hist_gradient_boosting"),
        tasks.TaskType.BINARY,
        0,
    ).fit(frame, target)
    bundle = bundle_module.ModelBundle.create(
        fitted, plan, frame, "y", "binary_classification", "roc_auc", "boost"
    )
    return bundle, frame, tmp_path_factory.mktemp("service")


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def start(folder, port, environment):
    env = {
        **os.environ,
        "PORT": str(port),
        "HOST": "127.0.0.1",
        "PYTHONPATH": str(ROOT),
        **environment,
    }
    return subprocess.Popen(
        [sys.executable, str(folder / "serve.py")],
        cwd=folder,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def wait_for(port, process):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health"):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def test_the_folder_holds_everything_needed_to_run_and_rebuild(boosted):
    bundle, root = boosted[0], boosted[2]

    folder = bundle.export_service(root / "complete")

    for name in (
        "model",
        "serve.py",
        "Dockerfile",
        "requirements.txt",
        "environment.lock",
        "README.md",
    ):
        assert (folder / name).exists()
    pins = (folder / "requirements.txt").read_text()
    assert f"scmle=={bundle.info.scmle_version}" in pins
    assert f"scikit-learn=={bundle.info.sklearn}" in pins
    assert "python:" in (folder / "Dockerfile").read_text()
    with pytest.raises(errors.ArtifactError, match="already holds"):
        bundle.export_service(folder)


def test_onnx_refuses_the_boosted_model_and_points_to_the_service(boosted):
    bundle, root = boosted[0], boosted[2]

    with pytest.raises(errors.ArtifactError, match="export_service"):
        bundle.export_onnx(root / "m.onnx")


def test_the_generated_server_scores_rows_like_the_bundle(boosted):
    bundle, frame, root = boosted
    folder = bundle.export_service(root / "running")
    port = free_port()
    process = start(folder, port, {"SCMLE_TRUSTED": "1"})
    try:
        assert wait_for(port, process), process.stderr.read().decode()
        rows = frame.head(20).to_dict("records")
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/predict",
            json.dumps({"records": rows}).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            body = json.loads(response.read())
    finally:
        process.terminate()
        process.wait(timeout=10)

    assert body["predictions"] == bundle.predict(frame.head(20)).tolist()


def fetch(port, path):
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}{path}", timeout=5
        ) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def wait_for_state(port, process, phases):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        assert process.poll() is None, process.stderr.read().decode()
        try:
            status, body = fetch(port, "/readyz")
        except OSError:
            time.sleep(0.3)
            continue
        if body.get("phase") in phases:
            return status, body
        time.sleep(0.3)
    raise AssertionError("service never reached " + str(phases))


def test_the_generated_server_stays_unready_for_an_unsigned_bundle(boosted):
    bundle, root = boosted[0], boosted[2]
    folder = bundle.export_service(root / "untrusted")
    port = free_port()
    process = start(folder, port, {})
    try:
        status, body = wait_for_state(port, process, {"failed"})
        assert status == 503 and "unauthenticated" in body["reason"]
        assert fetch(port, "/livez")[0] == 200
        assert fetch(port, "/health")[0] == 503
    finally:
        process.terminate()
        process.wait(timeout=10)
        process.stderr.close()


def test_production_serves_only_a_signed_verified_folder(boosted):
    bundle, root = boosted[0], boosted[2]
    key = "package-test-key"
    folder = bundle.export_service(root / "production", signing_key=key)
    production = {"SCMLE_ENV": "production", "SCMLE_SIGNING_KEY": key}
    port = free_port()
    process = start(folder, port, production)
    try:
        status, body = wait_for_state(port, process, {"ready", "failed"})
        assert status == 200, body
        assert body["model_version"] == bundle.info.model_version
        assert body["artifact_digest"].startswith("sha256:")
        assert "provenance=True" in body["notes"]["verifying"]
    finally:
        process.terminate()
        process.wait(timeout=10)
        process.stderr.close()

    port = free_port()
    process = start(folder, port, {**production, "SCMLE_TRUSTED": "1"})
    try:
        status, body = wait_for_state(port, process, {"ready", "failed"})
        assert status == 503 and "SCMLE_TRUSTED=1 is refused" in body["reason"]
    finally:
        process.terminate()
        process.wait(timeout=10)
        process.stderr.close()

    corrupted = bundle.export_service(root / "corrupted", signing_key=key)
    with (corrupted / "model" / "pipeline.joblib").open("ab") as handle:
        handle.write(b"x")
    port = free_port()
    process = start(corrupted, port, production)
    try:
        status, body = wait_for_state(port, process, {"ready", "failed"})
        assert status == 503 and "checksum mismatch" in body["reason"]
    finally:
        process.terminate()
        process.wait(timeout=10)
        process.stderr.close()


def test_the_package_command_writes_the_folder(boosted, tmp_path):
    bundle = boosted[0]
    bundle.save(tmp_path / "bundle")

    code = cli.main(
        [
            "package",
            str(tmp_path / "bundle"),
            "--trusted",
            "--output",
            str(tmp_path / "svc"),
        ]
    )

    assert code == cli.EXIT_OK and (tmp_path / "svc" / "serve.py").exists()
