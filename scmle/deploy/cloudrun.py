"""Reads and drives a Cloud Run service through the documented API shapes.

`observe_service` turns a Cloud Run Admin API v2 ``Service`` and the
``Revision`` resources it names into an `Observation`: the revision with the
most traffic, the image digest it runs (the revision's pinned container
image), whether the service's terminal condition succeeded, and the split in
``trafficStatuses`` (what is actually served, not what was asked for).

`CloudRunControlPlane` deploys a revision with no traffic and moves traffic
between revisions by patching ``traffic``. It was written against the
documented shapes and exercised only against recorded and scripted responses
of those shapes; it has not called Google Cloud.
"""

from collections.abc import Mapping
import copy
from typing import Any
from typing import Protocol

from scmle.contracts import errors
from scmle.contracts import events
from scmle.deploy import contract
from scmle.deploy import control
from scmle.deploy import observation as observation_module

SUCCEEDED = "CONDITION_SUCCEEDED"
REVISION_TARGET = "TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION"
LATEST_TARGET = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"


class CloudRunClient(Protocol):
    """The part of the Cloud Run Admin API the plane calls (REST shapes)."""

    def get_service(self, name: str) -> Mapping[str, Any]:
        """Returns the service resource."""

    def get_revision(self, name: str) -> Mapping[str, Any]:
        """Returns a revision resource."""

    def update_service(
        self, name: str, body: Mapping[str, Any], update_mask: str
    ) -> None:
        """Patches fields of the service."""


def short(name: str) -> str:
    """The last path segment of a resource name."""
    return name.rsplit("/", 1)[-1]


def revision_image_digest(revision: Mapping[str, Any]) -> str | None:
    """The digest the revision's first container is pinned to, or None."""
    containers = revision.get("containers") or []
    if not containers:
        return None
    image = str(containers[0].get("image", ""))
    if "@" not in image:
        return None
    digest = image.rsplit("@", 1)[1]
    return digest if contract.DIGEST_PATTERN.fullmatch(digest) else None


def served(service: Mapping[str, Any]) -> list[tuple[str, int]]:
    """Revision and percent of what is served (statuses, else the request)."""
    entries = service.get("trafficStatuses") or service.get("traffic") or []
    latest = short(str(service.get("latestReadyRevision", "")))
    shares: dict[str, int] = {}
    for entry in entries:
        kind = entry.get("type", REVISION_TARGET)
        name = short(str(entry.get("revision", "")))
        if kind == LATEST_TARGET or not name:
            name = latest
        shares[name] = shares.get(name, 0) + int(entry.get("percent", 0))
    return sorted(shares.items(), key=lambda item: (-item[1], item[0]))


def ready(resource: Mapping[str, Any]) -> bool:
    """Whether a service or revision reports its Ready condition succeeded."""
    terminal = resource.get("terminalCondition")
    if terminal is not None:
        return bool(terminal.get("state") == SUCCEEDED)
    return any(
        c.get("type") == "Ready" and c.get("state") == SUCCEEDED
        for c in resource.get("conditions", [])
    )


def observe_service(
    service: Mapping[str, Any],
    revisions: Mapping[str, Mapping[str, Any]],
    observed_at: str | None = None,
) -> observation_module.Observation:
    """Builds an `Observation` from recorded or live Cloud Run answers.

    Args:
        service: The ``Service`` resource.
        revisions: Revision resources by short or full name.
        observed_at: UTC time of the read.

    Raises:
        ArtifactError: If no traffic is reported or the primary revision is
            missing or not pinned by digest.
    """
    shares = served(service)
    if not shares or sum(p for ignored, p in shares) != 100:
        raise errors.ArtifactError("the service reports no complete traffic")
    by_name = {short(k): v for k, v in revisions.items()}
    known: dict[str, str] = {}
    ready_all = ready(service)
    for name, percent in shares:
        resource = by_name.get(name)
        if resource is None:
            raise errors.ArtifactError(f"revision {name} was not read")
        digest = revision_image_digest(resource)
        if digest is None:
            if percent:
                raise errors.ArtifactError(
                    f"revision {name} is not pinned by digest"
                )
            continue
        known[name] = digest
        ready_all = ready_all and (not percent or ready(resource))
    for name, resource in by_name.items():
        digest = revision_image_digest(resource)
        if digest and name not in known:
            known[name] = digest
    primary = shares[0][0]
    return observation_module.Observation(
        provider="gcp",
        service=short(str(service.get("name", ""))),
        revision=primary,
        image_digest=known[primary],
        ready=ready_all,
        traffic=tuple(observation_module.Traffic(n, p) for n, p in shares if p),
        observed_at=observed_at or events.utc_timestamp(),
        revisions=tuple(sorted(known.items())),
    )


class CloudRunControlPlane(control.ControlPlane):
    """Drives one Cloud Run service."""

    provider = "gcp"

    def __init__(self, client: CloudRunClient, service: str) -> None:
        """Binds the plane to ``projects/P/locations/L/services/S``."""
        self.client = client
        self.service = service

    def observe(self) -> observation_module.Observation:
        """Reads the service and every revision it serves or created."""
        service = self.client.get_service(self.service)
        names = {n for n, ignored in served(service)}
        for key in ("latestCreatedRevision", "latestReadyRevision"):
            if service.get(key):
                names.add(short(str(service[key])))
        revisions = {
            n: self.client.get_revision(f"{self.service}/revisions/{n}")
            for n in names
        }
        return observe_service(service, revisions, events.utc_timestamp())

    def current_traffic(
        self, service: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """The served split as explicit revision targets."""
        return [
            {"type": REVISION_TARGET, "revision": name, "percent": percent}
            for name, percent in served(service)
            if percent
        ]

    def deploy(self, image_digest: str) -> str:
        """Creates a revision of the image; traffic stays where it is."""
        contract.require_digest(image_digest, "image_digest")
        service = self.client.get_service(self.service)
        template = copy.deepcopy(dict(service["template"]))
        containers = template["containers"]
        repository = str(containers[0]["image"]).split("@", 1)[0]
        last = repository.rsplit("/", 1)[-1]
        if ":" in last:
            repository = repository[: -len(last.split(":", 1)[1]) - 1]
        containers[0]["image"] = f"{repository}@{image_digest}"
        self.client.update_service(
            self.service,
            {"template": template, "traffic": self.current_traffic(service)},
            "template,traffic",
        )
        created = self.client.get_service(self.service)
        return short(str(created["latestCreatedRevision"]))

    def set_traffic(self, allocation: Mapping[str, int]) -> None:
        """Requests the split by patching ``traffic``."""
        if sum(allocation.values()) != 100:
            raise errors.ConfigError("traffic shares must sum to 100")
        body = {
            "traffic": [
                {"type": REVISION_TARGET, "revision": name, "percent": pct}
                for name, pct in allocation.items()
                if pct
            ]
        }
        self.client.update_service(self.service, body, "traffic")
