"""The deployment spec: limits, probes and the whole validated spec."""

from collections.abc import Mapping
import dataclasses
import hashlib
import json
from typing import Any
from typing import Self

from scmle.contracts import errors
from scmle.deploy.contract import references

SPEC_FORMAT = 1
SECRET_WORDS = ("SECRET", "PASSWORD", "TOKEN", "CREDENTIAL", "PRIVATE", "KEY")
PORT_LIMIT = 65536  # exclusive upper bound of a TCP port number
PRODUCTION = "production"
TRUSTED_VARIABLE = "SCMLE_TRUSTED"
ENVIRONMENT_VARIABLE = "SCMLE_ENV"


@dataclasses.dataclass(frozen=True, slots=True)
class Resources:
    """Compute limits and the isolation the container runs with.

    Attributes:
        cpu_request: CPUs requested.
        cpu_limit: CPUs allowed; at least `cpu_request`.
        memory_request_mb: Memory requested, in megabytes.
        memory_limit_mb: Memory allowed, in megabytes; at least the request.
        scratch_mb: Size of the writable scratch space, in megabytes.
        read_only_root: Whether the root filesystem is read-only (must be True).
        run_as_non_root: Whether the container runs as a non-root user (must be
            True).
        max_concurrent_requests: Requests served at the same time.
    """

    __module__ = "scmle.deploy.contract"  # the path pinned signatures name

    cpu_request: float
    cpu_limit: float
    memory_request_mb: int
    memory_limit_mb: int
    scratch_mb: int = 64
    read_only_root: bool = True
    run_as_non_root: bool = True
    max_concurrent_requests: int = 8

    def __post_init__(self) -> None:
        """Validates that limits dominate requests and isolation holds.

        Raises:
            ConfigError: If a limit is smaller than its request, or the root is
                writable or the container runs as root. Non-positive values are
                refused with the same error.
        """
        references.require_positive(self.cpu_request, "resources.cpu_request")
        references.require_positive(self.cpu_limit, "resources.cpu_limit")
        references.require_positive(
            self.memory_request_mb, "resources.memory_request_mb"
        )
        references.require_positive(
            self.memory_limit_mb, "resources.memory_limit_mb"
        )
        references.require_positive(self.scratch_mb, "resources.scratch_mb")
        references.require_positive(
            self.max_concurrent_requests, "resources.max_concurrent_requests"
        )
        if self.cpu_limit < self.cpu_request:
            raise errors.ConfigError("cpu_limit must be at least cpu_request")
        if self.memory_limit_mb < self.memory_request_mb:
            raise errors.ConfigError(
                "memory_limit_mb must be at least memory_request_mb"
            )
        if not (self.read_only_root and self.run_as_non_root):
            raise errors.ConfigError(
                "a deployable service runs non-root on a read-only root"
            )

    def to_dict(self) -> dict[str, Any]:
        """Builds a JSON-compatible record.

        Returns:
            A dict of all fields.
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
        return cls(**record)


@dataclasses.dataclass(frozen=True, slots=True)
class Probes:
    """Liveness and readiness probes and their timing.

    Attributes:
        liveness_path: URL path of the liveness probe.
        readiness_path: URL path of the readiness probe; differs from the
            liveness path.
        period_seconds: Seconds between probes.
        timeout_seconds: Seconds a probe may take; at most the period.
        failure_threshold: Consecutive failures before the service is judged
            unhealthy.
        startup_grace_seconds: Seconds allowed for the service to start.
    """

    __module__ = "scmle.deploy.contract"  # the path pinned signatures name

    liveness_path: str = "/livez"
    readiness_path: str = "/readyz"
    period_seconds: float = 10.0
    timeout_seconds: float = 2.0
    failure_threshold: int = 3
    startup_grace_seconds: float = 60.0

    def __post_init__(self) -> None:
        """Validates distinct paths and positive timings.

        Raises:
            ConfigError: If a path does not start with `/`, the two paths are
                the same, or the probe timeout exceeds the period. Non-positive
                timings are refused with the same error.
        """
        for name in ("liveness_path", "readiness_path"):
            if not getattr(self, name).startswith("/"):
                raise errors.ConfigError(f"health.{name} must start with '/'")
        if self.liveness_path == self.readiness_path:
            raise errors.ConfigError(
                "liveness and readiness must be different endpoints"
            )
        references.require_positive(
            self.period_seconds, "health.period_seconds"
        )
        references.require_positive(
            self.timeout_seconds, "health.timeout_seconds"
        )
        references.require_positive(
            self.failure_threshold, "health.failure_threshold"
        )
        references.require_positive(
            self.startup_grace_seconds, "health.startup_grace_seconds"
        )
        if self.timeout_seconds > self.period_seconds:
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
    """

    __module__ = "scmle.deploy.contract"  # the path pinned signatures name

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
            ConfigError: If the record's format is not `SPEC_FORMAT`.
        """
        if record.get("format") != SPEC_FORMAT:
            raise errors.ConfigError(
                f"unsupported deployment spec format {record.get('format')}"
            )
        return cls(
            version=record["version"],
            environment=references.Environment(record["environment"]),
            image=references.ImageRef.from_dict(record["image"]),
            bundle=references.BundleRef.from_dict(record["bundle"]),
            resources=Resources.from_dict(record["resources"]),
            health=Probes.from_dict(record["health"]),
            config=dict(record["config"]),
            secrets=tuple(
                references.SecretRef.from_dict(s) for s in record["secrets"]
            ),
            port=record["port"],
        )

    def digest(self) -> str:
        """Computes the digest of the canonical spec.

        Returns:
            `sha256:` followed by the hex SHA-256 of the canonical JSON of
            `to_dict()`.
        """
        return "sha256:" + hashlib.sha256(canonize(self.to_dict())).hexdigest()
