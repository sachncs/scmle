"""Guarantees of docs/PORTABLE_TRAINING.md that no other test enforced."""

import json
import pathlib
import subprocess

import pytest

from scmle.contracts import errors
from scmle.deploy import base_image
from scmle.runners import base
from scmle.runners import entrypoint
from scmle.runners import local
from scmle.runners import roles
from scmle.storage import objectstore
from scmle.storage import tenancy
from tests import support_runners

ROOT = pathlib.Path(__file__).resolve().parents[2]
SERVE = "scmle-serving"


def test_check_separation_refuses_each_overlap_on_its_own():
    """Enforces docs/PORTABLE_TRAINING.md:42.

    Also enforces website/src/pages/docs/portable-training.astro:29.
    """
    roles.check_separation(roles.TRAINING_ROLE, roles.SERVING_ROLE)
    cases = (
        (
            roles.Role("training", (), ("evidence/",)),
            roles.Role("serving", (), ("evidence/x/",)),
            "both write",
        ),
        (
            roles.Role("training", (), ("bundles/",)),
            roles.Role("serving", (), ("audit/",)),
            "released bundles",
        ),
        (
            roles.Role("training", (), ("artifacts/",)),
            roles.Role("serving", (), ("runs/",)),
            "training records",
        ),
    )
    for training, serving, message in cases:
        with pytest.raises(errors.ConfigError, match=message):
            roles.check_separation(training, serving)


def runners(kind=base.Runner):
    """Every concrete runner class, recursively.

    Args:
        kind: The class whose subclasses are listed.

    Returns:
        The subclasses of `kind` and theirs.
    """
    found = []
    for child in kind.__subclasses__():
        found += [child, *runners(child)]
    return found


def test_workloads_get_a_training_store_and_no_runner_runs_as_serving(
    tmp_path,
):
    """Enforces website/src/pages/docs/portable-training.astro:29."""
    store = objectstore.LocalObjectStore(tmp_path / "store")
    training = roles.RoleStore(store, roles.TRAINING_ROLE)
    training.put("runs/r/result.json", b"{}")
    with pytest.raises(roles.RoleDeniedError):
        training.put("bundles/model/v2", b"x")
    assert not store.exists("bundles/model/v2")
    found = runners()
    assert {local.Runner, local.Process} <= set(found)
    # Every runner admits a job through the one base `run`, which refuses
    # the serving identity before anything is stored.
    assert all("run" not in vars(kind) for kind in found)
    spec = base.Job(
        run_id="bad",
        name="churn",
        project={"data": "data/t.csv", "target": "y"},
        identity=SERVE,
        resources=base.Resources(cpu=1.0, memory_mb=1024),
    )
    for runner in (
        local.Runner(store, lambda context: None, serving_identity=SERVE),
        local.Process(store, serving_identity=SERVE, directory=tmp_path),
    ):
        with pytest.raises(errors.ConfigError, match="serving identity"):
            runner.run(spec)
    assert not store.exists("runs/bad/spec.json")


def test_the_entrypoint_opens_the_tenant_store_checks_the_digest_and_runs(
    tmp_path,
):
    """Enforces website/src/pages/docs/portable-training.astro:33."""
    shared = objectstore.LocalObjectStore(tmp_path / "store")
    confined = tenancy.Tenancy(shared, "acme")
    clock = support_runners.FakeClock()
    runner = local.Runner(
        confined,
        lambda context: None,
        serving_identity=SERVE,
        clock=base.Clock(clock.now, clock.sleep),
    )
    spec = base.Job(
        run_id="entry",
        name="churn",
        project={"data": "data/t.csv", "target": "y"},
        identity="scmle-training",
        resources=base.Resources(cpu=1.0, memory_mb=1024),
    )
    runner.register(spec)
    environ = {
        "SCMLE_RUN_ID": "entry",
        "SCMLE_STORE_URI": (tmp_path / "store").as_uri(),
        "SCMLE_TENANT": "acme",
        "SCMLE_SPEC_DIGEST": spec.digest(),
        "SCMLE_WORKLOAD": "tests.support_runners:current",
    }

    def work(context):
        context.finish({"metrics": {"ok": 1}})

    support_runners.CURRENT[0] = work
    assert entrypoint.main(environ) == base.Exit.SUCCEEDED
    stored = confined.get("runs/entry/result.json")[0]
    assert json.loads(stored) == {"metrics": {"ok": 1}}
    assert not shared.exists("runs/entry/result.json")
    wrong = {**environ, "SCMLE_SPEC_DIGEST": "0" * 64}
    assert entrypoint.main(wrong) == base.Exit.FAILED
    outside = {**environ, "SCMLE_TENANT": "globex"}
    assert entrypoint.main(outside) == base.Exit.FAILED


def stage(dockerfile, name):
    """The shell text of the `RUN case` step that checks a build argument.

    Args:
        dockerfile: The Dockerfile's text.
        name: The argument the step checks, such as `BASE_IMAGE`.

    Returns:
        The step's shell, with line continuations joined.
    """
    joined = dockerfile.replace("\\\n", " ")
    for line in joined.splitlines():
        if line.startswith("RUN case") and f"${{{name}}}" in line:
            return line.removeprefix("RUN ")
    raise AssertionError(f"no RUN case step checks {name}")


@pytest.mark.parametrize(
    ("image", "accepted"),
    [
        (None, True),
        ("python:3.12-slim@" + base_image.PLACEHOLDER, False),
        ("python:3.12-slim", False),
    ],
)
def test_the_training_image_is_pinned_hashed_and_non_root(image, accepted):
    """Enforces docs/PORTABLE_TRAINING.md:79-83."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    pinned = next(
        line.split("=", 1)[1]
        for line in dockerfile.splitlines()
        if line.startswith("ARG BASE_IMAGE=")
    )
    name, digest = pinned.split("@")
    assert base_image.PINNED[name] == digest != base_image.PLACEHOLDER
    check = subprocess.run(
        ["sh", "-c", stage(dockerfile, "BASE_IMAGE")],
        env={"BASE_IMAGE": image or pinned, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        check=False,
    )
    assert (check.returncode == 0) is accepted
    assert "COPY deploy/requirements.lock" in dockerfile
    install = " ".join(dockerfile.replace("\\\n", " ").split())
    assert (
        "pip install --no-cache-dir --require-hashes --no-deps "
        "-r /tmp/install/requirements.lock -r /tmp/install/wheel.lock"
    ) in install
    assert "\nUSER 10001:10001\n" in dockerfile
