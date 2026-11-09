"""Bounded host observations used by the pre-training capability gate."""

import datetime
import importlib.metadata
import os
import pathlib
import re
import shutil
import sys
from typing import Any
from typing import TYPE_CHECKING

from scmle.contracts import capabilities

if TYPE_CHECKING:
    from collections.abc import Iterable

OBSERVATION_SCHEMA_VERSION = 1
BYTES_PER_GB = 1024**3
WINDOWS = sys.platform == "win32"
PROJECT_DISTRIBUTION = "scmle"
CORE_EXTRA = "core"
EXTRA_MARKER = re.compile(r";\s*extra\s*==\s*['\"]([^'\"]+)['\"]")
REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


class Observation:
    """A bounded, serializable snapshot of facts used at a capability gate."""

    __slots__ = (
        "captured_at",
        "disk_path",
        "environment",
        "extra_observations",
        "sources",
    )

    def __init__(
        self,
        environment: capabilities.Environment,
        *,
        captured_at: str,
        disk_path: str,
        extra_observations: dict[str, bool | None],
        sources: dict[str, str],
    ) -> None:
        """Initializes an observation holding only plain values.

        Args:
            environment: The observed capability environment.
            captured_at: ISO 8601 timestamp of the observation.
            disk_path: Path whose filesystem free space was measured.
            extra_observations: For each project extra, whether it was found
                installed (None if unknown).
            sources: For each observed fact, how it was obtained.

        Raises:
            TypeError: If `environment` is not a `Environment`.
        """
        if not isinstance(environment, capabilities.Environment):
            raise TypeError("environment must be Environment")
        self.environment = environment
        self.captured_at = captured_at
        self.disk_path = disk_path
        self.extra_observations = dict(sorted(extra_observations.items()))
        self.sources = dict(sorted(sources.items()))

    def to_dict(self) -> dict[str, Any]:
        """Returns reviewable evidence, including unknown values as ``null``."""
        environment = self.environment
        values = {
            "cpu_cores": environment.cpu_cores,
            "memory_gb": environment.memory_gb,
            "disk_gb": environment.disk_gb,
            "gpu_available": environment.gpu_available,
            "network_available": environment.network_available,
        }
        return {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "captured_at": self.captured_at,
            "disk_path": self.disk_path,
            "environment": {
                **values,
                "available_extras": sorted(environment.available_extras),
            },
            "observations": {
                name: {
                    "value": value,
                    "source": self.sources.get(name, "unknown"),
                }
                for name, value in values.items()
            },
            "optional_extras": {
                name: {
                    "available": available,
                    "source": self.sources.get(f"extra:{name}", "unknown"),
                }
                for name, available in self.extra_observations.items()
            },
        }


def observe_runtime_resources(
    disk_path: str | pathlib.Path | None = None,
) -> Observation:
    """Captures only the bounded host facts that capability declarations need.

    A failed probe yields ``None``. The capability registry then rejects a
    declaration that requires that fact instead of treating the probe failure
    as proof that the resource exists.

    Args:
        disk_path: Path whose filesystem free space is measured; defaults to
            the current working directory.

    Returns:
        The observation: CPU cores, memory, free disk space and installed
        extras, with the source of each. GPU and network availability are
        never probed and are recorded as unknown.
    """
    cpu_cores, cpu_source = observe_cpu_cores()
    memory_gb, memory_source = observe_memory_gb()
    selected_path = pathlib.Path(disk_path or pathlib.Path.cwd())
    disk_gb, disk_source = observe_disk_gb(selected_path)
    extras, extra_observations, extra_sources = observe_extras()
    environment = capabilities.Environment(
        available_extras=extras,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        disk_gb=disk_gb,
        # These are intentionally not guessed here. A capability
        # that declares either requirement therefore fails closed.
        gpu_available=None,
        network_available=None,
    )
    return Observation(
        environment,
        captured_at=datetime.datetime.now(datetime.UTC).isoformat(),
        disk_path=str(selected_path),
        extra_observations=extra_observations,
        sources={
            "cpu_cores": cpu_source,
            "memory_gb": memory_source,
            "disk_gb": disk_source,
            **extra_sources,
        },
    )


def observe_cpu_cores() -> tuple[int | None, str]:
    """Returns CPUs available to this process, respecting affinity when able."""
    affinity_probe = getattr(os, "sched_getaffinity", None)
    try:
        affinity = affinity_probe(0) if affinity_probe is not None else None
    except OSError:
        affinity = None
    if affinity:
        return len(affinity), "os.sched_getaffinity"
    try:
        count = os.cpu_count()
    except NotImplementedError:
        count = None
    return (count, "os.cpu_count") if count else (None, "unknown")


def observe_memory_gb() -> tuple[float | None, str]:
    """Returns physical memory capacity from portable probes.

    POSIX hosts report it through ``sysconf``. Windows has no ``sysconf``, so
    it is read from ``psutil`` (a Windows-only core dependency).
    """
    try:
        pages: object = os.sysconf("SC_PHYS_PAGES")
        page_size: object = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return observe_windows_memory_gb() if WINDOWS else (None, "unknown")
    if not isinstance(pages, int) or not isinstance(page_size, int):
        return None, "unknown"
    if pages <= 0 or page_size <= 0:
        return None, "unknown"
    return pages * page_size / BYTES_PER_GB, "os.sysconf(SC_PHYS_PAGES)"


def observe_windows_memory_gb() -> tuple[float | None, str]:
    """Returns total physical memory on Windows from ``psutil``."""
    try:
        import psutil  # noqa: PLC0415 -- Windows-only dependency

        total = psutil.virtual_memory().total
    except (ImportError, OSError, AttributeError):
        return None, "unknown"
    if not isinstance(total, int) or total <= 0:
        return None, "unknown"
    return total / BYTES_PER_GB, "psutil.virtual_memory()"


def observe_disk_gb(path: pathlib.Path) -> tuple[float | None, str]:
    """Returns the free space of the filesystem containing ``path``.

    Args:
        path: Any path on the filesystem to measure.

    Returns:
        The free space in gigabytes and a description of the source, or
        None and "unknown" if the filesystem cannot be queried.
    """
    try:
        free = shutil.disk_usage(path).free
    except OSError:
        return None, "unknown"
    return free / BYTES_PER_GB, f"shutil.disk_usage({path})"


def observe_extras() -> tuple[
    frozenset[str], dict[str, bool | None], dict[str, str]
]:
    """Observes project extras by checking their declared distributions.

    An extra counts as installed only when every distribution it requires has
    readable metadata.

    Returns:
        A triple: the extras observed as installed, each extra's observation
        (True or False; the core extra is always True), and the source of
        each observation.
    """
    core_available = True
    observations: dict[str, bool | None] = {CORE_EXTRA: core_available}
    sources = {"extra:core": "running scmle package"}
    requirements = project_extra_requirements()
    for extra, packages in requirements.items():
        observations[extra] = all(is_installed(package) for package in packages)
        sources[f"extra:{extra}"] = "importlib.metadata"
    available = frozenset(
        extra for extra, observed in observations.items() if observed is True
    )
    return available, observations, sources


def project_extra_requirements() -> dict[str, tuple[str, ...]]:
    """Reads the installed distribution metadata without importing extras.

    Returns:
        A mapping from extra name to the sorted, de-duplicated names of the
        distributions it requires. Empty if the project metadata cannot be
        read.
    """
    try:
        requirements: Iterable[str] = (
            importlib.metadata.requires(PROJECT_DISTRIBUTION) or ()
        )
    except importlib.metadata.PackageNotFoundError:
        return {}
    except (ValueError, OSError):
        return {}
    grouped: dict[str, list[str]] = {}
    for requirement in requirements:
        marker = EXTRA_MARKER.search(requirement)
        name = REQUIREMENT_NAME.match(requirement)
        if marker is None or name is None:
            continue
        grouped.setdefault(marker.group(1), []).append(name.group(1))
    return {
        extra: tuple(sorted(set(packages)))
        for extra, packages in grouped.items()
    }


def is_installed(package: str) -> bool:
    """Returns whether distribution metadata for ``package`` is available.

    Args:
        package: Distribution name to look up.

    Returns:
        True if the installed version of the distribution can be read.
    """
    try:
        importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return False
    except (ValueError, OSError):
        return False
    return True
