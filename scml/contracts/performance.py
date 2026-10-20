"""Measured performance evidence and explicit release budgets."""

from collections.abc import Mapping
import dataclasses
import datetime
import enum
import math
from typing import Any
from typing import Self


class PerformanceOperation(enum.StrEnum):
    """Lifecycle operation whose resource cost can be released."""

    INGEST = "ingest"
    FIT = "fit"
    PACKAGE = "package"
    LOAD = "load"
    INFERENCE = "inference"


def parse_timestamp(value: str, field_name: str) -> datetime.datetime:
    """Parse an offset-aware ISO-8601 timestamp."""
    timestamp = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information")
    return timestamp


@dataclasses.dataclass(frozen=True, slots=True)
class PerformanceObservation:
    """One measured operation with the machine context for its result."""

    operation: PerformanceOperation
    elapsed_seconds: float
    peak_memory_gb: float | None
    sample_count: int
    machine: str
    observed_at: str

    def __post_init__(self) -> None:
        """Reject measurements that cannot support a budget decision."""
        if not isinstance(self.operation, PerformanceOperation):
            object.__setattr__(
                self, "operation", PerformanceOperation(self.operation)
            )
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        if self.peak_memory_gb is not None and (
            not math.isfinite(self.peak_memory_gb) or self.peak_memory_gb < 0
        ):
            raise ValueError(
                "peak_memory_gb must be finite and non-negative when present"
            )
        if isinstance(self.sample_count, bool) or self.sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        if not isinstance(self.machine, str) or not self.machine.strip():
            raise ValueError("machine must not be empty")
        parse_timestamp(self.observed_at, "observed_at")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-native performance evidence."""
        return {
            "operation": self.operation.value,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_memory_gb": self.peak_memory_gb,
            "sample_count": self.sample_count,
            "machine": self.machine,
            "observed_at": self.observed_at,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild an observation from JSON-native evidence."""
        return cls(
            operation=PerformanceOperation(record["operation"]),
            elapsed_seconds=float(record["elapsed_seconds"]),
            peak_memory_gb=(
                float(record["peak_memory_gb"])
                if record.get("peak_memory_gb") is not None
                else None
            ),
            sample_count=int(record["sample_count"]),
            machine=str(record["machine"]),
            observed_at=str(record["observed_at"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class PerformanceBudget:
    """Explicit time and optional memory ceilings for lifecycle operations."""

    max_seconds: Mapping[PerformanceOperation, float]
    max_peak_memory_gb: float | None = None

    def __post_init__(self) -> None:
        """Validate positive ceilings and normalize operation names."""
        normalized: dict[PerformanceOperation, float] = {}
        for operation, limit in self.max_seconds.items():
            resolved = (
                operation
                if isinstance(operation, PerformanceOperation)
                else PerformanceOperation(operation)
            )
            if not math.isfinite(limit) or limit <= 0:
                raise ValueError(
                    f"performance limit must be positive: {resolved.value}"
                )
            normalized[resolved] = float(limit)
        if not normalized:
            raise ValueError("max_seconds must not be empty")
        if self.max_peak_memory_gb is not None and (
            not math.isfinite(self.max_peak_memory_gb)
            or self.max_peak_memory_gb <= 0
        ):
            raise ValueError("max_peak_memory_gb must be positive when present")
        object.__setattr__(self, "max_seconds", normalized)

    def evaluate(
        self, observations: tuple[PerformanceObservation, ...]
    ) -> "PerformanceResult":
        """Evaluate observations without treating missing data as pass."""
        by_operation: dict[PerformanceOperation, PerformanceObservation] = {}
        violations: list[str] = []
        for observation in observations:
            if observation.operation in by_operation:
                raise ValueError(
                    "performance observations must be unique per operation"
                )
            by_operation[observation.operation] = observation
        for operation, limit in sorted(
            self.max_seconds.items(), key=lambda item: item[0].value
        ):
            measurement = by_operation.get(operation)
            if measurement is None:
                violations.append(
                    f"missing performance observation: {operation.value}"
                )
                continue
            if measurement.elapsed_seconds > limit:
                violations.append(
                    f"{operation.value} took "
                    f"{measurement.elapsed_seconds:.6g}s "
                    f"> {limit:.6g}s budget"
                )
            memory = measurement.peak_memory_gb
            if self.max_peak_memory_gb is not None and memory is None:
                violations.append(
                    f"missing peak memory observation: {operation.value}"
                )
            elif (
                self.max_peak_memory_gb is not None
                and memory is not None
                and memory > self.max_peak_memory_gb
            ):
                violations.append(
                    f"{operation.value} used "
                    f"{memory:.6g} GB "
                    f"> {self.max_peak_memory_gb:.6g} GB budget"
                )
        return PerformanceResult(
            passed=not violations,
            observations=observations,
            violations=tuple(violations),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-native budget policy."""
        return {
            "max_seconds": {
                operation.value: limit
                for operation, limit in sorted(
                    self.max_seconds.items(), key=lambda item: item[0].value
                )
            },
            "max_peak_memory_gb": self.max_peak_memory_gb,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class PerformanceResult:
    """Budget decision with the observations that justify it."""

    passed: bool
    observations: tuple[PerformanceObservation, ...]
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-native performance decision evidence."""
        return {
            "passed": self.passed,
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
            "violations": list(self.violations),
        }


__all__ = [
    "PerformanceBudget",
    "PerformanceObservation",
    "PerformanceOperation",
    "PerformanceResult",
    "parse_timestamp",
]
