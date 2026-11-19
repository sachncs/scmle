"""The scmle image runs a real job through the container runner.

`SCMLE_IMAGE` names an image built from the repository's Dockerfile
(`python -m scmle.deploy.packaging image`); CI builds it in the release
workflow and runs this module against it.
"""

import json
import os
import shutil
import subprocess

import numpy
import pandas
import pytest

from scmle.runners import base
from scmle.runners import local
from scmle.runtime import retry
from scmle.storage import objectstore

IMAGE = os.environ.get("SCMLE_IMAGE", "")
pytestmark = pytest.mark.skipif(
    not IMAGE or shutil.which("docker") is None,
    reason="SCMLE_IMAGE names no built image",
)
OPTIONS = {
    "time_budget": 20,
    "search": {"top_models": 0},
    "models": {"exclude_families": ["boosting", "deep_learning"]},
}


def docker(*arguments):
    return subprocess.run(
        ["docker", *arguments], capture_output=True, text=True, check=False
    )


def frame():
    rng = numpy.random.default_rng(0)
    data = pandas.DataFrame(
        {"a": rng.normal(size=200), "b": rng.normal(size=200)}
    )
    noise = rng.normal(scale=0.5, size=200)
    data["y"] = ((data.a + 0.5 * data.b + noise) > 0).astype(int)
    return data


def spec(run_id, **changes):
    values = {
        "run_id": run_id,
        "name": "churn",
        "project": {
            "data": "data/train.csv",
            "target": "y",
            "options": OPTIONS,
        },
        "identity": "arn:train",
        "resources": base.Resources(cpu=1, memory_mb=2048),
        "retries": retry.Retries(2, 0.1, 1.0, 0.1),
        "timeout_seconds": 600.0,
    }
    values.update(changes)
    return base.Job(**values)


@pytest.fixture
def store(tmp_path):
    depot = objectstore.LocalObjectStore(tmp_path / "store")
    depot.put("data/train.csv", frame().to_csv(index=False).encode())
    return depot


def test_the_image_runs_scmle_as_a_non_root_user():
    found = docker(
        "run", "--rm", "--read-only", IMAGE, "sh", "-c", "id -u && scmle --help"
    )
    assert found.returncode == 0, found.stderr
    user, usage = found.stdout.split("\n", 1)
    assert user == "10001"
    assert "usage" in usage.lower()


def test_a_project_runs_inside_the_image(store):
    runner = local.Container(
        store, serving_identity="arn:serve", image=IMAGE, poll_seconds=0.5
    )
    manifest = runner.run(spec("image-1"))
    log = store.get("runs/image-1/logs/1.txt")[0].decode()
    assert manifest.state == "succeeded", log[-2000:]
    assert [attempt.state for attempt in manifest.attempts] == ["succeeded"]
    assert manifest.runner == "container"
    result = json.loads(store.get("runs/image-1/result.json")[0])
    assert result["resumed"] is False and result["selected_model"]
    assert (
        docker(
            "ps", "--all", "--quiet", "--filter", "name=scmle-container-image-1"
        ).stdout
        == ""
    )


def test_a_stopped_container_is_cancelled_and_removed(store):
    options = {**OPTIONS, "time_budget": 600}
    project = {"data": "data/train.csv", "target": "y", "options": options}
    runner = local.Container(
        store,
        serving_identity="arn:serve",
        image=IMAGE,
        poll_seconds=0.5,
        grace=5,
    )
    manifest = runner.run(spec("image-2", project=project, timeout_seconds=8.0))
    assert manifest.state == "timed_out"
    assert [attempt.state for attempt in manifest.attempts] == ["cancelled"]
    assert (
        docker(
            "ps", "--all", "--quiet", "--filter", "name=scmle-container-image-2"
        ).stdout
        == ""
    )
