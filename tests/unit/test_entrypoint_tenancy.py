"""The entrypoint opens the store confined as the job's environment says.

A runner started with `SCMLE_TENANT` set registers its specs under
`tenants/<tenant>/` and its jobs inherit the variable; the entrypoint reads
the spec through the same confinement, decided by the environment it is
given (not by the process's own), so runner and job agree on where the spec
lives.
"""

import json

import pytest

from scmle.runners import base
from scmle.runners import entrypoint
from scmle.runtime import retry
from scmle.storage import objectstore
from scmle.storage import tenancy
from tests import support_runners

TENANT = "acme"


def job(run: str) -> base.Job:
    """A minimal job spec.

    Args:
        run: The run id.

    Returns:
        The job.
    """
    return base.Job(
        identifier=run,
        name="churn",
        project={"data": "data/t.csv", "target": "y"},
        identity="scmle-training",
        resources=base.Resources(cpu=1.0, memory_mb=1024),
        retries=retry.Retries(3, 10.0, 2.0, 60.0),
    )


def store(depot: objectstore.Depot, spec: base.Job) -> None:
    """Writes the spec where a runner on `depot` registers it.

    Args:
        depot: The runner's store, confined or not.
        spec: The job.
    """
    body = json.dumps(spec.to_dict(), sort_keys=True).encode("utf-8")
    depot.put(f"runs/{spec.identifier}/spec.json", body)


def finish(context: object) -> None:
    """A workload that records a result.

    Args:
        context: The job context.
    """
    context.finish({"metrics": {"ok": 1}})  # type: ignore[attr-defined]


def launch(environ: dict[str, str]) -> int:
    """Runs the entrypoint as a job process would, opening the store itself.

    Args:
        environ: The job's environment.

    Returns:
        The exit status.
    """
    support_runners.CURRENT[0] = finish
    return entrypoint.main(
        {**environ, "SCMLE_WORKLOAD": "tests.support_runners:current"}
    )


def test_mount_confines_to_the_tenant_of_the_given_environment(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(tenancy.VARIABLE, raising=False)
    confined = entrypoint.mount(tmp_path.as_uri(), {tenancy.VARIABLE: TENANT})
    assert isinstance(confined, tenancy.Tenancy)
    assert confined.tenant == TENANT
    shared = entrypoint.mount(tmp_path.as_uri(), {})
    assert isinstance(shared, objectstore.Disk)


def test_mount_follows_the_given_environment_not_the_process(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(tenancy.VARIABLE, "other")
    assert isinstance(entrypoint.mount(tmp_path.as_uri(), {}), objectstore.Disk)


def test_mount_refuses_an_invalid_tenant(tmp_path):
    with pytest.raises(tenancy.TenantError):
        entrypoint.mount(tmp_path.as_uri(), {tenancy.VARIABLE: "../x"})


def test_a_tenant_job_reads_the_spec_its_confined_runner_registered(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(tenancy.VARIABLE, raising=False)
    shared = objectstore.Disk(tmp_path)
    runner = tenancy.confine(shared, {tenancy.VARIABLE: TENANT})
    spec = job("tenant-run")
    store(runner, spec)
    environ = {
        "SCMLE_RUN_ID": spec.identifier,
        "SCMLE_STORE_URI": tmp_path.as_uri(),
        "SCMLE_SPEC_DIGEST": spec.digest(),
        tenancy.VARIABLE: TENANT,
    }
    assert launch(environ) == base.Exit.SUCCEEDED
    assert json.loads(runner.get("runs/tenant-run/result.json")[0]) == {
        "metrics": {"ok": 1}
    }
    assert not shared.exists("runs/tenant-run/result.json")


def test_a_tenant_job_does_not_read_a_spec_outside_its_namespace(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(tenancy.VARIABLE, raising=False)
    shared = objectstore.Disk(tmp_path)
    spec = job("shared-run")
    store(shared, spec)
    environ = {
        "SCMLE_RUN_ID": spec.identifier,
        "SCMLE_STORE_URI": tmp_path.as_uri(),
        "SCMLE_SPEC_DIGEST": spec.digest(),
    }
    assert launch(environ) == base.Exit.SUCCEEDED
    assert launch({**environ, tenancy.VARIABLE: TENANT}) == base.Exit.FAILED
