"""Reads and drives an ECS service through the documented API shapes.

`observe_services` turns the answers of ``DescribeServices``,
``DescribeTasks`` and ``DescribeTaskDefinition`` into an `Observation`.
``DescribeServices`` names the primary deployment's task definition (the
revision, ``family:number``) and its rollout state; it does not carry the
image digest. The digest comes from the running tasks
(``containers[].imageDigest``, which ECS resolves) and, when no task of that
revision runs yet, from the pinned image of the task definition.

`EcsControlPlane` drives a service with ``RegisterTaskDefinition`` and
``UpdateService``. ECS rolling deployments move all traffic to the new
revision, so the plane refuses a split. It was written against the documented
shapes and exercised only against recorded and scripted responses of those
shapes; it has not called AWS.
"""

from collections.abc import Mapping
from typing import Any
from typing import Protocol

from scmle.contracts import errors
from scmle.contracts import events
from scmle.deploy import contract
from scmle.deploy import control
from scmle.deploy import observation as observation_module

REGISTER_FIELDS = (
    "family",
    "taskRoleArn",
    "executionRoleArn",
    "networkMode",
    "containerDefinitions",
    "volumes",
    "placementConstraints",
    "requiresCompatibilities",
    "cpu",
    "memory",
    "runtimePlatform",
    "tags",
)


class EcsClient(Protocol):
    """The part of the boto3 ECS client the plane calls."""

    def describe_services(self, **kwargs: Any) -> Any:
        """Describes services."""

    def list_tasks(self, **kwargs: Any) -> Any:
        """Lists tasks."""

    def describe_tasks(self, **kwargs: Any) -> Any:
        """Describes tasks."""

    def describe_task_definition(self, **kwargs: Any) -> Any:
        """Describes a task definition."""

    def register_task_definition(self, **kwargs: Any) -> Any:
        """Registers a task definition."""

    def update_service(self, **kwargs: Any) -> Any:
        """Updates a service."""


def revision_of(arn: str) -> str:
    """``family:number`` from a task definition ARN (or the same text)."""
    return arn.rsplit("/", 1)[-1]


def image_digest(image: str) -> str | None:
    """The digest an image reference is pinned to, or None."""
    if "@" not in image:
        return None
    digest = image.rsplit("@", 1)[1]
    return digest if contract.DIGEST_PATTERN.fullmatch(digest) else None


def task_digests(tasks: Mapping[str, Any] | None) -> dict[str, set[str]]:
    """Revision to the digests its tasks report running."""
    found: dict[str, set[str]] = {}
    for task in (tasks or {}).get("tasks", []):
        revision = revision_of(task["taskDefinitionArn"])
        for container in task.get("containers", []):
            digest = container.get("imageDigest") or image_digest(
                container.get("image", "")
            )
            if digest:
                found.setdefault(revision, set()).add(digest)
    return found


def percentages(deployments: list[Mapping[str, Any]]) -> list[tuple[str, int]]:
    """Traffic share of each deployment by running tasks (primary first)."""
    ordered = sorted(
        deployments, key=lambda d: 0 if d["status"] == "PRIMARY" else 1
    )
    running = [int(d.get("runningCount", 0)) for d in ordered]
    total = sum(running)
    if total == 0:
        shares = [100] + [0] * (len(ordered) - 1)
    else:
        shares = [count * 100 // total for count in running]
        shares[0] += 100 - sum(shares)
    return [
        (revision_of(d["taskDefinition"]), share)
        for d, share in zip(ordered, shares, strict=True)
    ]


def observe_services(
    services: Mapping[str, Any],
    tasks: Mapping[str, Any] | None = None,
    task_definition: Mapping[str, Any] | None = None,
    observed_at: str | None = None,
    service: str | None = None,
) -> observation_module.Observation:
    """Builds an `Observation` from recorded or live ECS answers.

    Args:
        services: The ``DescribeServices`` response.
        tasks: The ``DescribeTasks`` response for the service's tasks.
        task_definition: The ``DescribeTaskDefinition`` response of the
            primary revision (a fallback for the digest).
        observed_at: UTC time of the read.
        service: Service name to pick when several are returned.

    Raises:
        ArtifactError: If the service, its primary deployment or the primary
            revision's digest is not in the answers, or the tasks of one
            revision run different digests.
    """
    found = [
        s
        for s in services.get("services", [])
        if service is None or s.get("serviceName") == service
    ]
    if not found:
        raise errors.ArtifactError(
            "ECS reported no such service",
            {"failures": services.get("failures", [])},
        )
    described = found[0]
    deployments = list(described.get("deployments", []))
    primary = next((d for d in deployments if d["status"] == "PRIMARY"), None)
    if primary is None:
        raise errors.ArtifactError("the service has no primary deployment")
    revision = revision_of(primary["taskDefinition"])
    running = task_digests(tasks)
    for name, digests in running.items():
        if len(digests) != 1:
            raise errors.ArtifactError(
                f"the tasks of {name} run different image digests"
            )
    known: dict[str, str] = {n: next(iter(d)) for n, d in running.items()}
    if task_definition is not None:
        body = task_definition.get("taskDefinition", task_definition)
        name = revision_of(body.get("taskDefinitionArn", ""))
        pinned = image_digest(body["containerDefinitions"][0]["image"])
        if pinned and name not in known:
            known[name] = pinned
    if revision not in known:
        raise errors.ArtifactError(
            f"the image digest of {revision} is not reported"
        )
    shares = percentages(deployments)
    settled = (
        described.get("status") == "ACTIVE"
        and primary.get("rolloutState", "COMPLETED") == "COMPLETED"
        and int(primary.get("runningCount", 0))
        == int(primary.get("desiredCount", 0))
        and int(primary.get("pendingCount", 0)) == 0
        and len(deployments) == 1
    )
    return observation_module.Observation(
        provider="aws",
        service=str(described.get("serviceName", "")),
        revision=revision,
        image_digest=known[revision],
        ready=settled,
        traffic=tuple(
            observation_module.Traffic(name, share) for name, share in shares
        ),
        observed_at=observed_at or events.utc_timestamp(),
        revisions=tuple(sorted(known.items())),
    )


class EcsControlPlane(control.ControlPlane):
    """Drives one ECS service."""

    provider = "aws"

    def __init__(self, client: EcsClient, cluster: str, service: str) -> None:
        """Binds the plane to a cluster and service."""
        self.client = client
        self.cluster = cluster
        self.service = service

    def describe(self) -> Mapping[str, Any]:
        """The service description."""
        return self.client.describe_services(  # type: ignore[no-any-return]
            cluster=self.cluster, services=[self.service]
        )

    def observe(self) -> observation_module.Observation:
        """Reads the service, its running tasks and its primary revision."""
        services = self.describe()
        listed = self.client.list_tasks(
            cluster=self.cluster,
            serviceName=self.service,
            desiredStatus="RUNNING",
        )
        arns = list(listed.get("taskArns", []))
        tasks = (
            self.client.describe_tasks(cluster=self.cluster, tasks=arns)
            if arns
            else None
        )
        primary = next(
            (
                d
                for s in services.get("services", [])
                for d in s.get("deployments", [])
                if d["status"] == "PRIMARY"
            ),
            None,
        )
        definition = (
            None
            if primary is None
            else self.client.describe_task_definition(
                taskDefinition=primary["taskDefinition"]
            )
        )
        return observe_services(
            services, tasks, definition, events.utc_timestamp(), self.service
        )

    def deploy(self, image_digest: str) -> str:
        """Registers a revision of the current definition with the image."""
        contract.require_digest(image_digest, "image_digest")
        services = self.describe()
        current = services["services"][0]["taskDefinition"]
        body = self.client.describe_task_definition(taskDefinition=current)[
            "taskDefinition"
        ]
        request = {k: body[k] for k in REGISTER_FIELDS if k in body}
        containers = [dict(c) for c in request["containerDefinitions"]]
        image = containers[0]["image"]
        repository = image.split("@", 1)[0]
        last = repository.rsplit("/", 1)[-1]
        if ":" in last:
            repository = repository[: -len(last.split(":", 1)[1]) - 1]
        containers[0]["image"] = f"{repository}@{image_digest}"
        request["containerDefinitions"] = containers
        registered = self.client.register_task_definition(**request)
        arn = registered["taskDefinition"]["taskDefinitionArn"]
        self.client.update_service(
            cluster=self.cluster, service=self.service, taskDefinition=arn
        )
        return revision_of(arn)

    def set_traffic(self, allocation: Mapping[str, int]) -> None:
        """Points the service at one revision; ECS cannot split traffic."""
        shares = {k: v for k, v in allocation.items() if v}
        if len(shares) != 1 or sum(shares.values()) != 100:
            raise control.Unsupported(
                "ECS rolling deployments send all traffic to one revision"
            )
        self.client.update_service(
            cluster=self.cluster,
            service=self.service,
            taskDefinition=next(iter(shares)),
        )
