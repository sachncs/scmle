"""Bounded host observations used by the pre-training capability gate."""

from collections.abc import Iterable
import datetime
import importlib.metadata
import importlib.util
import os
import pathlib
import re
import shutil
from typing import Any

from scml.contracts import capabilities

OBSERVATION_SCHEMA_VERSION = 1
BYTES_PER_GB = 1024**3
PROJECT_DISTRIBUTION = "super-charge-ml"
CORE_EXTRA = "core"
EXTRA_MARKER = re.compile(r";\s*extra\s*==\s*['\"]([^'\"]+)['\"]")
REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


class RuntimeResourceObservation:
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
        environment: capabilities.CapabilityEnvironment,
        *,
        captured_at: str,
        disk_path: str,
        extra_observations: dict[str, bool | None],
        sources: dict[str, str],
    ) -> None:
        """Initialize an observation without retaining host-specific objects."""
        if not isinstance(environment, capabilities.CapabilityEnvironment):
            raise TypeError("environment must be CapabilityEnvironment")
        self.environment = environment
        self.captured_at = captured_at
        self.disk_path = disk_path
        self.extra_observations = dict(sorted(extra_observations.items()))
        self.sources = dict(sorted(sources.items()))

    def to_dict(self) -> dict[str, Any]:
        """Return reviewable evidence, including unknown values as ``null``."""
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
) -> RuntimeResourceObservation:
    """Capture only bounded host facts needed by capability declarations.

    A failed probe yields ``None``. The capability registry then rejects a
    declaration that requires that fact instead of treating the probe failure
    as proof that the resource exists.
    """
    cpu_cores, cpu_source = observe_cpu_cores()
    memory_gb, memory_source = observe_memory_gb()
    selected_path = pathlib.Path(disk_path or pathlib.Path.cwd())
    disk_gb, disk_source = observe_disk_gb(selected_path)
    extras, extra_observations, extra_sources = observe_extras()
    environment = capabilities.CapabilityEnvironment(
        available_extras=extras,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        disk_gb=disk_gb,
        # These are intentionally not guessed by this slice. A capability
        # that declares either requirement therefore fails closed.
        gpu_available=None,
        network_available=None,
    )
    return RuntimeResourceObservation(
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
    """Return CPUs available to this process, respecting affinity when able."""
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
    """Return physical memory capacity from portable ``sysconf`` probes."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None, "unknown"
    if not isinstance(pages, int) or not isinstance(page_size, int):
        return None, "unknown"
    if pages <= 0 or page_size <= 0:
        return None, "unknown"
    return pages * page_size / BYTES_PER_GB, "os.sysconf(SC_PHYS_PAGES)"


def observe_disk_gb(path: pathlib.Path) -> tuple[float | None, str]:
    """Return free bytes on the filesystem containing ``path``."""
    try:
        free = shutil.disk_usage(path).free
    except OSError:
        return None, "unknown"
    return free / BYTES_PER_GB, f"shutil.disk_usage({path})"


def observe_extras() -> tuple[
    frozenset[str], dict[str, bool | None], dict[str, str]
]:
    """Observe project extras by checking their declared distributions."""
    core_available = True
    observations: dict[str, bool | None] = {CORE_EXTRA: core_available}
    sources = {"extra:core": "running scml package"}
    requirements = project_extra_requirements()
    for extra, packages in requirements.items():
        observations[extra] = all(
            is_installed(package) for package in packages
        )
        sources[f"extra:{extra}"] = "importlib.metadata"
    available = frozenset(
        extra for extra, observed in observations.items() if observed is True
    )
    return available, observations, sources


def project_extra_requirements() -> dict[str, tuple[str, ...]]:
    """Read the installed distribution metadata without importing extras."""
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
    """Return whether distribution metadata for ``package`` is available."""
    try:
        importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return False
    except (ValueError, OSError):
        return False
    return True
