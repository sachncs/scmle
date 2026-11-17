"""What a platform reports as running, its conformance and the adapter."""

import abc
import dataclasses

from scmle.contracts import errors
from scmle.deploy.contract import deployment
from scmle.deploy.contract import references


@dataclasses.dataclass(frozen=True, slots=True)
class Runtime:
    """What a platform says is running, in the spec's own terms.

    Attributes:
        spec_digest: Digest of the spec the runtime was started from.
        image_digest: Digest of the running image.
        bundle_digest: Digest of the running model bundle.
        model_version: Version of the running model.
        ready: Whether the service reports itself ready.
        secret_names: Names of the secrets mounted.
        read_only_root: Whether the root filesystem is read-only.
        run_as_non_root: Whether the container runs as a non-root user.
        memory_limit_mb: Memory limit actually applied; None if unknown.
        cpu_limit: CPU limit actually applied; None if unknown.
    """

    __module__ = "scmle.deploy.contract"  # the path pinned signatures name

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
        references.require_digest(self.spec_digest, "report.spec_digest")
        references.require_digest(self.image_digest, "report.image_digest")
        references.require_digest(self.bundle_digest, "report.bundle_digest")


def divergences(spec: deployment.Deployment, report: Runtime) -> list[str]:
    """Lists the differences in what runs: spec, image, bundle and readiness.

    Args:
        spec: What was meant to run.
        report: What the platform says is running.

    Returns:
        One message per difference; empty if the identity matches.
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
    return problems


def mounts(spec: deployment.Deployment, report: Runtime) -> list[str]:
    """Lists the differences in which secrets are mounted.

    Args:
        spec: What was meant to run.
        report: What the platform says is running.

    Returns:
        One message per difference; empty if the mounted secrets match.
    """
    problems = []
    expected = {secret.name for secret in spec.secrets if secret.required}
    missing = sorted(expected - set(report.secret_names))
    if missing:
        problems.append(f"required secrets not mounted: {', '.join(missing)}")
    extra = sorted(set(report.secret_names) - {s.name for s in spec.secrets})
    if extra:
        problems.append(f"undeclared secrets mounted: {', '.join(extra)}")
    return problems


def diff(spec: deployment.Deployment, report: Runtime) -> list[str]:
    """Lists the differences in filesystem, user and resource limits.

    Args:
        spec: What was meant to run.
        report: What the platform says is running.

    Returns:
        One message per difference; empty if isolation and limits conform.
    """
    problems = []
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


def conformance_problems(
    spec: deployment.Deployment, report: Runtime
) -> list[str]:
    """Lists every way the running service differs from the spec.

    An empty list means the platform runs exactly what the spec says. The
    comparison is on digests and limits, never on tags or names.

    Args:
        spec: What was meant to run.
        report: What the platform says is running.

    Returns:
        One message per difference; empty if the runtime conforms.
    """
    return [
        *divergences(spec, report),
        *mounts(spec, report),
        *diff(spec, report),
    ]


class Provider(abc.ABC):
    """What a platform adapter implements; nothing else is provider-specific.

    Attributes:
        provider: Name of the platform, used in messages.
    """

    provider: str = ""

    @abc.abstractmethod
    def supported_architectures(self) -> tuple[references.Architecture, ...]:
        """Lists the architectures the platform can run.

        Returns:
            The supported architectures.
        """

    @abc.abstractmethod
    def apply(self, spec: deployment.Deployment) -> Runtime:
        """Makes the platform run `spec` and reports what runs.

        Args:
            spec: The deployment to run.

        Returns:
            The state the platform reports after applying the spec.
        """

    @abc.abstractmethod
    def observe(self) -> Runtime:
        """Reads the running state back from the platform.

        Returns:
            The state the platform currently reports.
        """

    def deploy(self, spec: deployment.Deployment) -> Runtime:
        """Applies `spec` and refuses a platform that does not conform.

        Args:
            spec: The deployment to run.

        Returns:
            The runtime report after the spec was applied and found to conform.

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
