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

import numpy as np
import pandas as pd
import pytest

from scml import cli
from scml.artifacts import bundle as bundle_module
from scml.contracts import data as data_contracts
from scml.contracts import errors
from scml.contracts import tasks
from scml.data.curation import preprocessing
from scml.models import registry as registry_module

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def boosted(tmp_path_factory):
    """A bundle of the model family the ONNX converter refuses."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    target = (frame.a + frame.b > 0).astype(int)
    plan = data_contracts.FeaturePlan(("a", "b"), ())
    fitted = preprocessing.build_pipeline(
        plan,
        registry_module.ModelRegistry.with_builtins().get(
            "hist_gradient_boosting"
        ),
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
    assert f"super-charge-ml=={bundle.info.scml_version}" in pins
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
    process = start(folder, port, {"SCML_TRUSTED": "1"})
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


def test_the_generated_server_refuses_an_unsigned_bundle_by_default(boosted):
    bundle, root = boosted[0], boosted[2]
    folder = bundle.export_service(root / "untrusted")
    process = start(folder, free_port(), {})

    code = process.wait(timeout=60)

    assert code != 0
    assert "unauthenticated" in process.stderr.read().decode()


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
