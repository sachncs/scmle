"""The deployment contract every provider adapter must satisfy.

One typed specification says what runs (an image by digest, a model bundle by
digest), how it is configured, which secrets it may read (by reference, never
by value), how much it may use and how it proves it is alive and ready. An
adapter for a platform (local, AWS, GCP) turns that spec into infrastructure
calls and reports what is actually running; `conformance_problems` compares
the two. Adapters differ only in those calls. Nothing here talks to a cloud.
"""

import abc
from collections.abc import Mapping
import dataclasses
import enum
import hashlib
import json
import math
import re
from typing import Any
from typing import Self

from scmle.contracts import errors

SPEC_FORMAT = 1
DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}")
VARIABLE_PATTERN = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
VERSION_PATTERN = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,63}")
SECRET_WORDS = ("SECRET", "PASSWORD", "TOKEN", "CREDENTIAL", "PRIVATE", "KEY")
PRIVATE_KEY_MARKER = "-----BEGIN"
PRODUCTION = "production"
TRUSTED_VARIABLE = "SCMLE_TRUSTED"
ENVIRONMENT_VARIABLE = "SCMLE_ENV"


class Architecture(enum.StrEnum):
    """CPU architectures an image may be built for."""

    AMD64 = "amd64"
    ARM64 = "arm64"


class SecretSource(enum.StrEnum):
    """Where a secret is read from at run time."""

    ENVIRONMENT = "env"
    FILE = "file"
    STORE = "store"


class DeploymentEnvironment(enum.StrEnum):
    """Named environments a spec may target."""

    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


def require_digest(value: str, field_name: str) -> str:
    """Returns `value` if it is a `sha256:<64 hex>` digest."""
    if not isinstance(value, str) or DIGEST_PATTERN.fullmatch(value) is None:
        raise errors.ConfigError(
            f"{field_name} must be 'sha256:' followed by 64 lowercase hex "
            "characters (a tag is not a digest)"
        )
    return value


def require_name(value: str, field_name: str) -> str:
    """Returns `value` if it is a plain identifier."""
    if not isinstance(value, str) or NAME_PATTERN.fullmatch(value) is None:
        raise errors.ConfigError(f"{field_name} is not a valid name: {value!r}")
    return value


def require_positive(value: float, field_name: str) -> float:
    """Returns `value` if it is a positive finite number."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise errors.ConfigError(f"{field_name} must be positive and finite")
    return value


def looks_secret(name: str) -> bool:
    """Whether an environment variable name suggests a secret value."""
    return any(word in name.upper() for word in SECRET_WORDS)


@dataclasses.dataclass(frozen=True, slots=True)
class ImageRef:
    """A container image pinned by digest, with the platforms it covers."""

    repository: str
    digest: str
    architectures: tuple[Architecture, ...] = (
        Architecture.AMD64,
        Architecture.ARM64,
    )

    def __post_init__(self) -> None:
        """Validates the reference."""
        require_name(self.repository, "image.repository")
        require_digest(self.digest, "image.digest")
        converted = tuple(Architecture(item) for item in self.architectures)
        object.__setattr__(self, "architectures", converted)
        if not converted or len(set(converted)) != len(converted):
            raise errors.ConfigError(
                "image.architectures must be non-empty and unique"
            )

    @property
    def reference(self) -> str:
        """The pull reference `repository@digest`."""
        return f"{self.repository}@{self.digest}"

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible record."""
        return {
            "repository": self.repository,
            "digest": self.digest,
            "architectures": sorted(item.value for item in self.architectures),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the reference from a record."""
        return cls(
            repository=record["repository"],
            digest=record["digest"],
            architectures=tuple(
                Architecture(item) for item in record["architectures"]
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class BundleRef:
    """A model bundle pinned by the digest of its manifest."""

    model: str
    model_version: str
    digest: str
    signed: bool

    def __post_init__(self) -> None:
        """Validates the reference."""
        require_name(self.model, "bundle.model")
        if VERSION_PATTERN.fullmatch(str(self.model_version)) is None:
            raise errors.ConfigError("bundle.model_version is not valid")
        require_digest(self.digest, "bundle.digest")
        if not isinstance(self.signed, bool):
            raise errors.ConfigError("bundle.signed must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible record."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the reference from a record."""
        return cls(
            model=record["model"],
            model_version=record["model_version"],
            digest=record["digest"],
            signed=record["signed"],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class SecretRef:
    """Names a secret and where to read it; never holds the value.

    `name` is the environment variable the service sees. `locator` is an
    environment variable name (`env`), an absolute path (`file`) or an opaque
    identifier in a secret store (`store`).
    """

    name: str
    source: SecretSource
    locator: str
    required: bool = True

    def __post_init__(self) -> None:
        """Validates the reference and refuses anything that looks inline."""
        if VARIABLE_PATTERN.fullmatch(self.name) is None:
            raise errors.ConfigError(
                f"secret name must be an upper-case variable: {self.name!r}"
            )
        object.__setattr__(self, "source", SecretSource(self.source))
        text = self.locator
        if (
            not isinstance(text, str)
            or not text.strip()
            or text != text.strip()
            or "\n" in text
            or "=" in text
            or PRIVATE_KEY_MARKER in text
        ):
            raise errors.ConfigError(
                f"secret {self.name} locator must be a plain reference"
            )
        if (
            self.source is SecretSource.ENVIRONMENT
            and VARIABLE_PATTERN.fullmatch(text) is None
        ):
            raise errors.ConfigError(
                f"secret {self.name} env locator must be a variable name"
            )
        if self.source is SecretSource.FILE and not text.startswith("/"):
            raise errors.ConfigError(
                f"secret {self.name} file locator must be an absolute path"
            )
        if self.source is SecretSource.STORE:
            require_name(text, f"secret {self.name} locator")

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible record."""
        return {
            "name": self.name,
            "source": self.source.value,
            "locator": self.locator,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the reference from a record."""
        return cls(
            name=record["name"],
            source=SecretSource(record["source"]),
            locator=record["locator"],
            required=record.get("required", True),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Resources:
    """Compute limits and the isolation the container runs with."""

    cpu_request: float
    cpu_limit: float
    memory_request_mb: int
    memory_limit_mb: int
    scratch_mb: int = 64
    read_only_root: bool = True
    run_as_non_root: bool = True
    max_concurrent_requests: int = 8

    def __post_init__(self) -> None:
        """Validates that limits dominate requests and isolation holds."""
        require_positive(self.cpu_request, "resources.cpu_request")
        require_positive(self.cpu_limit, "resources.cpu_limit")
        require_positive(self.memory_request_mb, "resources.memory_request_mb")
        require_positive(self.memory_limit_mb, "resources.memory_limit_mb")
        require_positive(self.scratch_mb, "resources.scratch_mb")
        require_positive(
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
        """Returns a JSON-compatible record."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the limits from a record."""
        return cls(**record)


@dataclasses.dataclass(frozen=True, slots=True)
class HealthSpec:
    """Liveness and readiness probes and their timing."""

    liveness_path: str = "/livez"
    readiness_path: str = "/readyz"
    period_seconds: float = 10.0
    timeout_seconds: float = 2.0
    failure_threshold: int = 3
    startup_grace_seconds: float = 60.0

    def __post_init__(self) -> None:
        """Validates distinct paths and positive timings."""
        for name in ("liveness_path", "readiness_path"):
            if not getattr(self, name).startswith("/"):
                raise errors.ConfigError(f"health.{name} must start with '/'")
        if self.liveness_path == self.readiness_path:
            raise errors.ConfigError(
                "liveness and readiness must be different endpoints"
            )
        require_positive(self.period_seconds, "health.period_seconds")
        require_positive(self.timeout_seconds, "health.timeout_seconds")
        require_positive(self.failure_threshold, "health.failure_threshold")
        require_positive(
            self.startup_grace_seconds, "health.startup_grace_seconds"
        )
        if self.timeout_seconds > self.period_seconds:
            raise errors.ConfigError("probe timeout exceeds the probe period")

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible record."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the probes from a record."""
        return cls(**record)


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    """Returns the deterministic JSON bytes used for digests."""
    return json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


@dataclasses.dataclass(frozen=True, slots=True)
class DeploymentSpec:
    """Everything a provider needs to run one version of one service."""

    version: str
    environment: DeploymentEnvironment
    image: ImageRef
    bundle: BundleRef
    resources: Resources
    health: HealthSpec = dataclasses.field(default_factory=HealthSpec)
    config: Mapping[str, str] = dataclasses.field(default_factory=dict)
    secrets: tuple[SecretRef, ...] = ()
    port: int = 8000

    def __post_init__(self) -> None:
        """Validates the whole spec, including cross-field rules."""
        if VERSION_PATTERN.fullmatch(str(self.version)) is None:
            raise errors.ConfigError("version is not valid")
        object.__setattr__(
            self, "environment", DeploymentEnvironment(self.environment)
        )
        object.__setattr__(self, "config", dict(sorted(self.config.items())))
        object.__setattr__(self, "secrets", tuple(self.secrets))
        if not (0 < self.port < 65536):
            raise errors.ConfigError("port must be between 1 and 65535")
        for key, value in self.config.items():
            if VARIABLE_PATTERN.fullmatch(key) is None:
                raise errors.ConfigError(f"config key {key!r} is not valid")
            if not isinstance(value, str):
                raise errors.ConfigError(f"config {key} must be text")
            if looks_secret(key):
                raise errors.ConfigError(
                    f"config {key} looks like a secret: declare it in "
                    "`secrets` as a reference"
                )
            if PRIVATE_KEY_MARKER in value:
                raise errors.ConfigError(f"config {key} holds key material")
        names = [secret.name for secret in self.secrets]
        if len(set(names)) != len(names):
            raise errors.ConfigError("secret names must be unique")
        clash = sorted(set(names) & set(self.config))
        if clash:
            raise errors.ConfigError(
                f"names are both config and secret: {', '.join(clash)}"
            )
        production = (
            self.environment is DeploymentEnvironment.PRODUCTION
            or self.config.get(ENVIRONMENT_VARIABLE) == PRODUCTION
        )
        if production and self.config.get(TRUSTED_VARIABLE) == "1":
            raise errors.ConfigError(
                f"{TRUSTED_VARIABLE}=1 is refused in production"
            )
        if production and not self.bundle.signed:
            raise errors.ConfigError("production requires a signed bundle")

    def to_dict(self) -> dict[str, Any]:
        """Returns a deterministic JSON-compatible record."""
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
        """Rebuilds a spec; refuses other formats."""
        if record.get("format") != SPEC_FORMAT:
            raise errors.ConfigError(
                f"unsupported deployment spec format {record.get('format')}"
            )
        return cls(
            version=record["version"],
            environment=DeploymentEnvironment(record["environment"]),
            image=ImageRef.from_dict(record["image"]),
            bundle=BundleRef.from_dict(record["bundle"]),
            resources=Resources.from_dict(record["resources"]),
            health=HealthSpec.from_dict(record["health"]),
            config=dict(record["config"]),
            secrets=tuple(SecretRef.from_dict(s) for s in record["secrets"]),
            port=record["port"],
        )

    def digest(self) -> str:
        """The `sha256:` digest of the canonical spec."""
        return (
            "sha256:"
            + hashlib.sha256(canonical_json(self.to_dict())).hexdigest()
        )


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeReport:
    """What a platform says is running, in the spec's own terms."""

    spec_digest: str
    image_digest: str
    bundle_digest: str
    model_version: str
    ready: bool
    secret_names: tuple[str, ...] = ()
    read_only_root: bool = True
    run_as_non_root: bool = True
    memory_limit_mb: int | None = None
    cpu_limit: float | None = None

    def __post_init__(self) -> None:
        """Validates the digests the report carries."""
        require_digest(self.spec_digest, "report.spec_digest")
        require_digest(self.image_digest, "report.image_digest")
        require_digest(self.bundle_digest, "report.bundle_digest")


def conformance_problems(
    spec: DeploymentSpec, report: RuntimeReport
) -> list[str]:
    """Lists every way the running service differs from the spec.

    An empty list means the platform runs exactly what the spec says. The
    comparison is on digests and limits, never on tags or names.
    """
    problems = []
    if report.spec_digest != spec.digest():
        problems.append("runtime was started from a different spec")
    if report.image_digest != spec.image.digest:
        problems.append(
            f"image digest {report.image_digest} != {spec.image.digest}"
        )
    if report.bundle_digest != spec.bundle.digest:
        problems.append(
            f"bundle digest {report.bundle_digest} != {spec.bundle.digest}"
        )
    if report.model_version != spec.bundle.model_version:
        problems.append("model version differs from the spec")
    if not report.ready:
        problems.append("service is not ready")
    expected = {secret.name for secret in spec.secrets if secret.required}
    missing = sorted(expected - set(report.secret_names))
    if missing:
        problems.append(f"required secrets not mounted: {', '.join(missing)}")
    extra = sorted(set(report.secret_names) - {s.name for s in spec.secrets})
    if extra:
        problems.append(f"undeclared secrets mounted: {', '.join(extra)}")
    if not report.read_only_root:
        problems.append("root filesystem is writable")
    if not report.run_as_non_root:
        problems.append("container runs as root")
    if (
        report.memory_limit_mb is not None
        and report.memory_limit_mb > spec.resources.memory_limit_mb
    ):
        problems.append("memory limit exceeds the spec")
    if (
        report.cpu_limit is not None
        and report.cpu_limit > spec.resources.cpu_limit
    ):
        problems.append("cpu limit exceeds the spec")
    return problems


class ProviderAdapter(abc.ABC):
    """What a platform adapter implements; nothing else is provider-specific."""

    provider: str = ""

    @abc.abstractmethod
    def supported_architectures(self) -> tuple[Architecture, ...]:
        """Architectures the platform can run."""

    @abc.abstractmethod
    def apply(self, spec: DeploymentSpec) -> RuntimeReport:
        """Makes the platform run `spec` and reports what runs."""

    @abc.abstractmethod
    def observe(self) -> RuntimeReport:
        """Reads the running state back from the platform."""

    def deploy(self, spec: DeploymentSpec) -> RuntimeReport:
        """Applies `spec` and refuses a platform that does not conform.

        Raises:
            ConfigError: If the platform cannot run an image architecture.
            ArtifactError: If the reported state differs from the spec.
        """
        missing = set(spec.image.architectures) - set(
            self.supported_architectures()
        )
        if missing:
            raise errors.ConfigError(
                f"{self.provider or 'adapter'} cannot run "
                + ", ".join(sorted(item.value for item in missing))
            )
        report = self.apply(spec)
        problems = conformance_problems(spec, report)
        if problems:
            raise errors.ArtifactError(
                "deployment does not match its spec: " + "; ".join(problems),
                {"problems": problems},
            )
        return report
