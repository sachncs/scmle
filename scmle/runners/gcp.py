"""GCP job adapter (Cloud Run jobs).

A Cloud Run job fixes its service account, image and resources in the job
definition; an execution may override the command, environment, task count
and timeout. The adapter therefore checks, before it starts anything, that
the spec asks for the job's own service account (never the serving one) and
for no more than the job's resources, and it overrides only what an
execution may override. It talks to a small client protocol returning plain
mappings; `run_v2_client` wraps `google.cloud.run_v2` into it.

It was written against the documented shapes and exercised against scripted
responses in the unit tests. `run_v2_client` and the real service have not
been run: no GCP account was used.

Mapping of an execution: still running or queued is running or pending; a
succeeded task is succeeded; a cancelled one is cancelled; a failed one is
interrupted when its condition names a preemption or SIGTERM, a retryable
failure when it names quota or capacity, and a plain failure otherwise (an
out-of-memory kill included, because repeating the request repeats it).
"""

from collections.abc import Callable
from collections.abc import Mapping
import importlib
from types import ModuleType
from typing import Any
from typing import Protocol

from scmle.contracts import errors
from scmle.data.store import objectstore
from scmle.runners import base
from scmle.runners import roles

INTERRUPTION_WORDS = ("preempt", "sigterm", "instance was terminated")
CAPACITY_WORDS = ("quota", "capacity", "unavailable", "resource exhausted")


class CloudRunJobsClient(Protocol):
    """What the adapter needs from Cloud Run jobs, as plain mappings."""

    def run_job(self, name: str, overrides: Mapping[str, Any]) -> str:
        """Starts an execution; returns its resource name."""

    def get_execution(self, name: str) -> Mapping[str, Any]:
        """Returns task counts, `completion_time` and `conditions`."""

    def cancel_execution(self, name: str) -> None:
        """Cancels an execution."""


class GcpJobRunner(base.Runner):
    """Runs the job as one Cloud Run job execution per attempt."""

    kind = "gcp-job"

    def __init__(
        self,
        client: CloudRunJobsClient,
        store: objectstore.ObjectStore,
        *,
        job_name: str,
        job_service_account: str,
        job_cpu: float,
        job_memory_mb: int,
        store_uri: str,
        serving_identity: str,
        **options: Any,
    ) -> None:
        """Initializes the adapter.

        Args:
            client: The jobs client.
            store: The durable store the job and the runner share.
            job_name: Full resource name of the job
                (`projects/P/locations/L/jobs/J`).
            job_service_account: The service account the job runs as.
            job_cpu: vCPUs the job definition gives each task.
            job_memory_mb: Memory the job definition gives each task.
            store_uri: Where the job finds the store (`gs://bucket/prefix`).
            serving_identity: The serving service account (refused).
            **options: `clock` and `poll_seconds` (see `Runner`).

        Raises:
            ConfigError: If the job runs as the serving identity.
        """
        super().__init__(store, serving_identity=serving_identity, **options)
        roles.check_distinct(job_service_account, serving_identity)
        self.client = client
        self.job_name = job_name
        self.job_service_account = job_service_account
        self.job_cpu = job_cpu
        self.job_memory_mb = job_memory_mb
        self.store_uri = store_uri

    def overrides(self, spec: base.JobSpec, attempt: int) -> dict[str, Any]:
        """The execution overrides for an attempt."""
        environment = {
            **spec.environment,
            "SCMLE_RUN_ID": spec.run_id,
            "SCMLE_ATTEMPT": str(attempt),
            "SCMLE_STORE_URI": self.store_uri,
            "SCMLE_SPEC_DIGEST": spec.digest(),
        }
        return {
            "container_overrides": [
                {
                    "args": ["-m", "scmle.runners.entrypoint"],
                    "env": [
                        {"name": k, "value": v}
                        for k, v in sorted(environment.items())
                    ],
                }
            ],
            "task_count": 1,
            "timeout": f"{int(spec.timeout_seconds)}s",
        }

    def submit(self, spec: base.JobSpec, attempt: int) -> base.JobHandle:
        """Starts an execution.

        Raises:
            ConfigError: If the spec's identity or resources do not match
                the job definition.
        """
        if spec.identity != self.job_service_account:
            raise errors.ConfigError(
                "a Cloud Run job runs as its own service account; the spec "
                "names a different identity"
            )
        asked = spec.resources
        if (
            asked.cpu > self.job_cpu
            or asked.memory_mb > self.job_memory_mb
            or asked.gpu
        ):
            raise errors.ConfigError(
                "the spec asks for more than the job definition provides"
            )
        name = self.client.run_job(self.job_name, self.overrides(spec, attempt))
        return base.JobHandle(name, attempt)

    def poll(self, handle: base.JobHandle) -> base.JobStatus:
        """Maps the execution to a job status."""
        execution = self.client.get_execution(handle.job_id)
        done = execution.get("completion_time")
        if execution.get("succeeded_count", 0) >= 1:
            return base.JobStatus(base.JobState.SUCCEEDED)
        if execution.get("cancelled_count", 0) >= 1:
            return base.JobStatus(base.JobState.CANCELLED, "cancelled")
        if execution.get("failed_count", 0) >= 1 or (
            done and not execution.get("running_count")
        ):
            return self.map_failure(execution)
        if execution.get("running_count", 0) >= 1:
            return base.JobStatus(base.JobState.RUNNING)
        return base.JobStatus(base.JobState.PENDING)

    def map_failure(self, execution: Mapping[str, Any]) -> base.JobStatus:
        """Classifies a failed execution from its conditions."""
        text = " ".join(
            f"{c.get('reason', '')} {c.get('message', '')}"
            for c in execution.get("conditions") or []
        ).strip()
        lowered = text.lower()
        if any(w in lowered for w in INTERRUPTION_WORDS):
            return base.JobStatus(base.JobState.INTERRUPTED, text)
        retryable = any(w in lowered for w in CAPACITY_WORDS)
        return base.JobStatus(base.JobState.FAILED, text, retryable)

    def cancel(self, handle: base.JobHandle, reason: str) -> None:
        """Cancels the execution."""
        self.client.cancel_execution(handle.job_id)


class RunV2Client:
    """`CloudRunJobsClient` over `google.cloud.run_v2` (not run live)."""

    def __init__(self, jobs: Any, executions: Any) -> None:
        """Wraps a `JobsClient` and an `ExecutionsClient`."""
        self.jobs = jobs
        self.executions = executions

    def run_job(self, name: str, overrides: Mapping[str, Any]) -> str:
        """Starts an execution and returns its name."""
        operation = self.jobs.run_job(
            request={"name": name, "overrides": dict(overrides)}
        )
        return str(operation.metadata.name)

    def get_execution(self, name: str) -> Mapping[str, Any]:
        """Reads counts and conditions from an `Execution`."""
        execution = self.executions.get_execution(name=name)
        return {
            "running_count": execution.running_count,
            "succeeded_count": execution.succeeded_count,
            "failed_count": execution.failed_count,
            "cancelled_count": execution.cancelled_count,
            "completion_time": execution.completion_time or None,
            "conditions": [
                {"reason": c.reason, "message": c.message}
                for c in execution.conditions
            ],
        }

    def cancel_execution(self, name: str) -> None:
        """Cancels an execution."""
        self.executions.cancel_execution(request={"name": name})


def run_v2_client(
    importer: Callable[[str], ModuleType] | None = None, **arguments: Any
) -> RunV2Client:
    """A Cloud Run jobs client built from `google-cloud-run`.

    Raises:
        ConfigError: If the library is not installed.
    """
    try:
        module = (importer or importlib.import_module)("google.cloud.run_v2")
    except ImportError as error:
        raise errors.ConfigError("google-cloud-run is not installed") from error
    return RunV2Client(
        module.JobsClient(**arguments), module.ExecutionsClient(**arguments)
    )
