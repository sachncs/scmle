"""AWS container-job adapter (ECS tasks on Fargate or EC2).

The adapter talks to an ECS client through `RunTask`, `DescribeTasks` and
`StopTask`; boto3's `ecs` client satisfies the protocol. It was written
against the documented shapes of those calls and exercised against scripted
responses of that shape in the unit tests. It has not run against an AWS
account.

The training identity goes to the task as ``overrides.taskRoleArn`` (the
role the container's code uses), never the serving role, and the run id is
the idempotency token (`clientToken`) so a repeated submission of the same
attempt does not start a second task.

Mapping of a stopped task:

* exit code 0: succeeded;
* `SpotInterruption`, or a stop reason naming a reclaimed or terminated
  host, or exit code 143 (SIGTERM): interrupted, so the run resumes from its
  checkpoints;
* `UserInitiated`: cancelled;
* `TaskFailedToStart` with a capacity or throttling reason: failed and
  retryable;
* any other exit code, including 137 (killed, usually out of memory): failed
  and not retryable, because repeating the same request repeats the failure.
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

SIGTERM_EXIT = 143
INTERRUPTION_WORDS = ("spot", "reclaim", "terminated", "draining")
CAPACITY_WORDS = ("capacity", "throttl", "resource", "unable to place")


class EcsClient(Protocol):
    """The part of the boto3 ECS client the adapter calls."""

    def run_task(self, **kwargs: Any) -> Any:
        """Starts tasks."""

    def describe_tasks(self, **kwargs: Any) -> Any:
        """Describes tasks."""

    def stop_task(self, **kwargs: Any) -> Any:
        """Stops a task."""


class AwsContainerJobRunner(base.Runner):
    """Runs the job as one ECS task per attempt."""

    kind = "aws-container-job"

    def __init__(
        self,
        client: EcsClient,
        store: objectstore.ObjectStore,
        *,
        cluster: str,
        task_definition: str,
        container: str,
        subnets: list[str],
        security_groups: list[str],
        store_uri: str,
        serving_identity: str,
        capacity_provider: str | None = None,
        **options: Any,
    ) -> None:
        """Initializes the adapter.

        Args:
            client: An ECS client.
            store: The durable store the job and the runner share (checked
                from this side; the job reaches it through `store_uri`).
            cluster: ECS cluster name or ARN.
            task_definition: Task definition family or ARN (the image is
                pinned by digest there, see the deployment contract).
            container: Name of the container to override.
            subnets: Subnets for the task's network interface.
            security_groups: Security groups.
            store_uri: Where the job finds the store (`s3://bucket/prefix`).
            serving_identity: The serving role ARN (refused for jobs).
            capacity_provider: Capacity provider (for example a Spot one);
                Fargate on-demand when omitted.
            **options: `clock` and `poll_seconds` (see `Runner`).
        """
        super().__init__(store, serving_identity=serving_identity, **options)
        self.client = client
        self.cluster = cluster
        self.task_definition = task_definition
        self.container = container
        self.subnets = subnets
        self.security_groups = security_groups
        self.store_uri = store_uri
        self.capacity_provider = capacity_provider
        self.placement_failure = ""

    def request(self, spec: base.JobSpec, attempt: int) -> dict[str, Any]:
        """The `RunTask` request for an attempt (also used by the tests)."""
        environment = {
            **spec.environment,
            "SCMLE_RUN_ID": spec.run_id,
            "SCMLE_ATTEMPT": str(attempt),
            "SCMLE_STORE_URI": self.store_uri,
            "SCMLE_SPEC_DIGEST": spec.digest(),
        }
        override: dict[str, Any] = {
            "name": self.container,
            "command": ["python", "-m", "scmle.runners.entrypoint"],
            "environment": [
                {"name": k, "value": v} for k, v in sorted(environment.items())
            ],
            "cpu": int(spec.resources.cpu * 1024),
            "memory": spec.resources.memory_mb,
        }
        if spec.resources.gpu:
            override["resourceRequirements"] = [
                {"type": "GPU", "value": str(spec.resources.gpu)}
            ]
        request: dict[str, Any] = {
            "cluster": self.cluster,
            "taskDefinition": self.task_definition,
            "count": 1,
            "startedBy": spec.run_id[:36],
            "clientToken": f"{spec.run_id}-{attempt}"[:64],
            "networkConfiguration": {
                "awsvpcConfiguration": {
                    "subnets": self.subnets,
                    "securityGroups": self.security_groups,
                    "assignPublicIp": "DISABLED",
                }
            },
            "overrides": {
                "taskRoleArn": spec.identity,
                "containerOverrides": [override],
            },
            "tags": [
                {"key": "scmle-run-id", "value": spec.run_id},
                {"key": "scmle-attempt", "value": str(attempt)},
            ],
        }
        if self.capacity_provider:
            request["capacityProviderStrategy"] = [
                {"capacityProvider": self.capacity_provider, "weight": 1}
            ]
        else:
            request["launchType"] = "FARGATE"
        return request

    def submit(self, spec: base.JobSpec, attempt: int) -> base.JobHandle:
        """Starts the task.

        Raises:
            ScmleError: If ECS reports a failure to place the task (a
                retryable capacity failure is returned as a failed attempt
                by `poll`, not here, only if a task was created).
        """
        reply = self.client.run_task(**self.request(spec, attempt))
        tasks = reply.get("tasks") or []
        if not tasks:
            failures = reply.get("failures") or []
            reason = "; ".join(str(f.get("reason", "")) for f in failures)
            self.placement_failure = reason or "no task was started"
            return base.JobHandle("failed-to-place", attempt)
        self.placement_failure = ""
        return base.JobHandle(str(tasks[0]["taskArn"]), attempt)

    def poll(self, handle: base.JobHandle) -> base.JobStatus:
        """Maps `DescribeTasks` to a job status."""
        if handle.job_id == "failed-to-place":
            retryable = any(
                w in self.placement_failure.lower() for w in CAPACITY_WORDS
            )
            return base.JobStatus(
                base.JobState.FAILED, self.placement_failure, retryable
            )
        reply = self.client.describe_tasks(
            cluster=self.cluster, tasks=[handle.job_id]
        )
        tasks = reply.get("tasks") or []
        if not tasks:
            return base.JobStatus(base.JobState.FAILED, "task not found")
        return self.map_task(tasks[0])

    def map_task(self, task: Mapping[str, Any]) -> base.JobStatus:
        """Maps one described task."""
        last = str(task.get("lastStatus", ""))
        if last in ("PROVISIONING", "PENDING", "ACTIVATING"):
            return base.JobStatus(base.JobState.PENDING)
        if last != "STOPPED":
            return base.JobStatus(base.JobState.RUNNING)
        stop_code = str(task.get("stopCode", ""))
        reason = str(task.get("stoppedReason", ""))
        containers = task.get("containers") or []
        exit_code = next(
            (c.get("exitCode") for c in containers if "exitCode" in c), None
        )
        text = f"{stop_code}: {reason}".strip(": ")
        lowered = reason.lower()
        if stop_code == "UserInitiated":
            return base.JobStatus(base.JobState.CANCELLED, text)
        if (
            stop_code == "SpotInterruption"
            or exit_code == SIGTERM_EXIT
            or (
                stop_code != "EssentialContainerExited"
                and any(w in lowered for w in INTERRUPTION_WORDS)
            )
        ):
            return base.JobStatus(base.JobState.INTERRUPTED, text)
        if exit_code == 0:
            return base.JobStatus(base.JobState.SUCCEEDED, text)
        if stop_code == "TaskFailedToStart":
            retryable = any(w in lowered for w in CAPACITY_WORDS)
            return base.JobStatus(base.JobState.FAILED, text, retryable)
        return base.JobStatus(
            base.JobState.FAILED, f"{text} (exit {exit_code})".strip()
        )

    def cancel(self, handle: base.JobHandle, reason: str) -> None:
        """Stops the task."""
        if handle.job_id != "failed-to-place":
            self.client.stop_task(
                cluster=self.cluster,
                task=handle.job_id,
                reason=f"scmle: {reason}"[:250],
            )


def ecs_client(
    importer: Callable[[str], ModuleType] | None = None, **arguments: Any
) -> EcsClient:
    """A boto3 ECS client, imported only when asked for.

    Raises:
        ConfigError: If boto3 is not installed.
    """
    try:
        boto3 = (importer or importlib.import_module)("boto3")
    except ImportError as error:
        raise errors.ConfigError("boto3 is not installed") from error
    client: EcsClient = boto3.client("ecs", **arguments)
    return client
