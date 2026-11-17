"""The deployment spec: limits, probes and the whole validated spec."""

from collections.abc import Mapping
import dataclasses
import hashlib
import json
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.contracts import errors
from scmle.contracts import records
from scmle.deploy.contract import references

SPEC_FORMAT = 2  # 1: multi-word keys; 2: the vocabulary of phase 5
SECRET_WORDS = ("SECRET", "PASSWORD", "TOKEN", "CREDENTIAL", "PRIVATE", "KEY")
PORT_LIMIT = 65536  # exclusive upper bound of a TCP port number
PRODUCTION = "production"
TRUSTED_VARIABLE = "SCMLE_TRUSTED"
ENVIRONMENT_VARIABLE = "SCMLE_ENV"


@dataclasses.dataclass(frozen=True, slots=True)
class Quota:
    """An amount of CPU and memory, as Kubernetes `requests` and `limits`.

    Attributes:
        cpu: CPUs.
        memory: Memory, in megabytes.
    """

    __module__ = "scmle.deploy.contract"  # the path pinned signatures name

    cpu: float
    memory: int


@dataclasses.dataclass(frozen=True, slots=True)
class Resources:
    """Compute limits and the isolation the container runs with.

    Attributes:
        requests: CPUs and memory (megabytes) requested.
        limits: CPUs and memory (megabytes) allowed; each at least its
            request.
        scratch: Size of the writable scratch space, in megabytes.
        immutable: Whether the root filesystem is read-only (must be True).
        rootless: Whether the container runs as a non-root user (must be
            True).
        concurrency: Requests served at the same time.
    """

    __module__ = "scmle.deploy.contract"  # the path pinned signatures name

    requests: Quota
    limits: Quota
    scratch: int = 64
    immutable: bool = True
    rootless: bool = True
    concurrency: int = 8

    def __post_init__(self) -> None:
        """Validates that limits dominate requests and isolation holds.

        Raises:
            ConfigError: If a limit is smaller than its request, or the root is
                writable or the container runs as root. Non-positive values are
                refused with the same error.
        """
        for side in ("requests", "limits"):
            quota = getattr(self, side)
            references.require_positive(quota.cpu, f"resources.{side}.cpu")
            references.require_positive(
                quota.memory, f"resources.{side}.memory"
            )
        references.require_positive(self.scratch, "resources.scratch")
        references.require_positive(self.concurrency, "resources.concurrency")
        if self.limits.cpu < self.requests.cpu:
            raise errors.ConfigError("limits.cpu must be at least requests.cpu")
        if self.limits.memory < self.requests.memory:
            raise errors.ConfigError(
                "limits.memory must be at least requests.memory"
            )
        if not (self.immutable and self.rootless):
            raise errors.ConfigError(
                "a deployable service runs non-root on a read-only root"
            )

    def to_dict(self) -> dict[str, Any]:
        """Builds a JSON-compatible record.

        Returns:
            A dict of all fields, with `requests` and `limits` as objects.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the limits from a record.

        Args:
            record: A record produced by `to_dict`.

        Returns:
            The limits.
        """
        values = dict(record)
        values["requests"] = Quota(**values["requests"])
        values["limits"] = Quota(**values["limits"])
        return cls(**values)


@dataclasses.dataclass(frozen=True, slots=True)
class Probes:
    """Liveness and readiness probes and their timing.

    Attributes:
        liveness: URL path of the liveness probe.
        readiness: URL path of the readiness probe; differs from the
            liveness path.
        period: Seconds between probes.
        timeout: Seconds a probe may take; at most the period.
        threshold: Consecutive failures before the service is judged
            unhealthy.
        grace: Seconds allowed for the service to start.
    """

    __module__ = "scmle.deploy.contract"  # the path pinned signatures name

    liveness: str = "/livez"
    readiness: str = "/readyz"
    period: float = 10.0
    timeout: float = 2.0
    threshold: int = 3
    grace: float = 60.0

    def __post_init__(self) -> None:
        """Validates distinct paths and positive timings.

        Raises:
            ConfigError: If a path does not start with `/`, the two paths are
                the same, or the probe timeout exceeds the period. Non-positive
                timings are refused with the same error.
        """
        for name in ("liveness", "readiness"):
            if not getattr(self, name).startswith("/"):
                raise errors.ConfigError(f"health.{name} must start with '/'")
        if self.liveness == self.readiness:
            raise errors.ConfigError(
                "liveness and readiness must be different endpoints"
            )
        references.require_positive(self.period, "health.period")
        references.require_positive(self.timeout, "health.timeout")
        references.require_positive(self.threshold, "health.threshold")
        references.require_positive(self.grace, "health.grace")
        if self.timeout > self.period:
            raise errors.ConfigError("probe timeout exceeds the probe period")

    def to_dict(self) -> dict[str, Any]:
        """Builds a JSON-compatible record.

        Returns:
            A dict of all fields.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the probes from a record.

        Args:
            record: A record produced by `to_dict`.

        Returns:
            The probes.
        """
        return cls(**record)


BUNDLE = records.Migration(renames={"model_version": "version"})
"""Version 1 to 2 of a spec's bundle reference."""
RESOURCES = records.nest(
    {
        "scratch_mb": "scratch",
        "read_only_root": "immutable",
        "run_as_non_root": "rootless",
        "max_concurrent_requests": "concurrency",
    },
    groups={
        "requests": {"cpu_request": "cpu", "memory_request_mb": "memory"},
        "limits": {"cpu_limit": "cpu", "memory_limit_mb": "memory"},
    },
)
"""Version 1 to 2 of a spec's resources (the Kubernetes shape)."""
PROBES = records.Migration(
    renames={
        "liveness_path": "liveness",
        "readiness_path": "readiness",
        "period_seconds": "period",
        "timeout_seconds": "timeout",
        "failure_threshold": "threshold",
        "startup_grace_seconds": "grace",
    }
)
"""Version 1 to 2 of a spec's probes."""


def canonize(payload: Mapping[str, Any]) -> bytes:
    """Serializes a payload deterministically.

    Args:
        payload: A JSON-serializable mapping.

    Returns:
        Compact, key-sorted, ASCII-only JSON encoded as UTF-8; the bytes used
        for digests.
    """
    return json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def nonsecret(config: Mapping[str, str]) -> None:
    """Checks that every config entry is a plain, non-secret variable.

    Args:
        config: The environment variables of a deployment.

    Raises:
        ConfigError: If a key is not an upper-case variable name, a value is
            not text, a name looks like a secret or a value holds key
            material.
    """
    for key, value in config.items():
        if references.VARIABLE_PATTERN.fullmatch(key) is None:
            raise errors.ConfigError(f"config key {key!r} is not valid")
        if not isinstance(value, str):
            raise errors.ConfigError(f"config {key} must be text")
        if any(word in key.upper() for word in SECRET_WORDS):
            raise errors.ConfigError(
                f"config {key} looks like a secret: declare it in "
                "`secrets` as a reference"
            )
        if references.PRIVATE_KEY_MARKER in value:
            raise errors.ConfigError(f"config {key} holds key material")


def harden(spec: "Deployment") -> None:
    """Applies the production-only rules to a deployment.

    Args:
        spec: The deployment to check.

    Raises:
        ConfigError: If the spec targets production (by environment or by
            config) and is marked trusted or has an unsigned bundle.
    """
    production = (
        spec.environment is references.Environment.PRODUCTION
        or spec.config.get(ENVIRONMENT_VARIABLE) == PRODUCTION
    )
    if production and spec.config.get(TRUSTED_VARIABLE) == "1":
        raise errors.ConfigError(
            f"{TRUSTED_VARIABLE}=1 is refused in production"
        )
    if production and not spec.bundle.signed:
        raise errors.ConfigError("production requires a signed bundle")


@dataclasses.dataclass(frozen=True, slots=True)
class Deployment:
    """Everything a provider needs to run one version of one service.

    Attributes:
        version: Version label of the deployment.
        environment: Environment the spec targets.
        image: The container image to run.
        bundle: The model bundle to serve.
        resources: Compute limits and isolation.
        health: Liveness and readiness probes.
        config: Plain (non-secret) environment variables, sorted by name.
        secrets: References to the secrets the service may read.
        port: Port the service listens on.
        migrations: The spec's format steps, oldest first; the format is
            stored under `format`.
    """

    __module__ = "scmle.deploy.contract"  # the path pinned signatures name

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            {},
            parts={"bundle": BUNDLE, "resources": RESOURCES, "health": PROBES},
        ),
    )

    version: str
    environment: references.Environment
    image: references.ImageRef
    bundle: references.BundleRef
    resources: Resources
    health: Probes = dataclasses.field(default_factory=Probes)
    config: Mapping[str, str] = dataclasses.field(default_factory=dict)
    secrets: tuple[references.SecretRef, ...] = ()
    port: int = 8000

    def __post_init__(self) -> None:
        """Validates the whole spec, including cross-field rules.

        Raises:
            ConfigError: If the version or port is invalid; a config key is not
                an upper-case variable, its value is not text, its name looks
                like a secret or its value holds key material; secret names
                repeat or collide with config names; or, in production (the
                environment is production, or the config says so), the
                service is marked trusted or the bundle is unsigned. An
                unknown environment is rejected with a `ValueError`.
        """
        if references.VERSION_PATTERN.fullmatch(str(self.version)) is None:
            raise errors.ConfigError("version is not valid")
        object.__setattr__(
            self, "environment", references.Environment(self.environment)
        )
        object.__setattr__(self, "config", dict(sorted(self.config.items())))
        object.__setattr__(self, "secrets", tuple(self.secrets))
        if not (0 < self.port < PORT_LIMIT):
            raise errors.ConfigError("port must be between 1 and 65535")
        nonsecret(self.config)
        names = [secret.name for secret in self.secrets]
        if len(set(names)) != len(names):
            raise errors.ConfigError("secret names must be unique")
        clash = sorted(set(names) & set(self.config))
        if clash:
            raise errors.ConfigError(
                f"names are both config and secret: {', '.join(clash)}"
            )
        harden(self)

    def to_dict(self) -> dict[str, Any]:
        """Builds a deterministic JSON-compatible record.

        Returns:
            A dict that includes the spec format number; secrets are sorted by
            name.
        """
        return {
            "format": SPEC_FORMAT,
            "version": self.version,
            "environment": self.environment.value,
            "image": self.image.to_dict(),
            "bundle": self.bundle.to_dict(),
            "resources": self.resources.to_dict(),
            "health": self.health.to_dict(),
            "config": dict(self.config),
            "secrets": [
                item.to_dict()
                for item in sorted(self.secrets, key=lambda s: s.name)
            ],
            "port": self.port,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a spec; refuses other formats.

        Args:
            record: A record produced by `to_dict`.

        Returns:
            The spec.

        Raises:
            ConfigError: If the record's format is newer than `SPEC_FORMAT`;
                format 1 is migrated. A format that is not a positive integer
                raises `InputError`.
        """
        values = dict(record)
        stored = records.detect("deployment spec", values, "format")
        if stored > SPEC_FORMAT:
            raise errors.ConfigError(
                f"unsupported deployment spec format {stored}"
            )
        values = records.upgrade(cls.migrations, values, stored)
        return cls(
            version=values["version"],
            environment=references.Environment(values["environment"]),
            image=references.ImageRef.from_dict(values["image"]),
            bundle=references.BundleRef.from_dict(values["bundle"]),
            resources=Resources.from_dict(values["resources"]),
            health=Probes.from_dict(values["health"]),
            config=dict(values["config"]),
            secrets=tuple(
                references.SecretRef.from_dict(s) for s in values["secrets"]
            ),
            port=values["port"],
        )

    def digest(self) -> str:
        """Computes the digest of the canonical spec.

        The digest names the spec across versions, so it is anchored at
        format 1: it is taken over `to_dict()` moved back to format 1, and a
        spec keeps the digest it had before the keys were renamed.

        Returns:
            `sha256:` followed by the hex SHA-256 of the canonical JSON of
            the spec at format 1.
        """
        layout = self.to_dict()
        del layout["format"]
        layout = records.downgrade(self.migrations, layout)
        layout["format"] = records.ORIGIN
        return "sha256:" + hashlib.sha256(canonize(layout)).hexdigest()
