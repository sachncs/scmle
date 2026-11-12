"""Measured performance evidence and explicit release budgets."""

from collections.abc import Mapping
import dataclasses
import datetime
import enum
import math
import statistics
from typing import Any
from typing import Self

MIN_REFERENCE_RUNS = 2


class Operation(enum.StrEnum):
    """Lifecycle operation whose resource cost can be released."""

    INGEST = "ingest"
    FIT = "fit"
    PACKAGE = "package"
    LOAD = "load"
    INFERENCE = "inference"


def parse_timestamp(value: str, field_name: str) -> datetime.datetime:
    """Parses an offset-aware ISO-8601 timestamp.

    Args:
        value: The timestamp; a trailing ``Z`` means UTC.
        field_name: The field name, for the error message.

    Returns:
        The timestamp as a timezone-aware datetime.

    Raises:
        ValueError: If the timestamp is malformed or has no timezone.
    """
    timestamp = datetime.datetime.fromisoformat(value)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information")
    return timestamp


@dataclasses.dataclass(frozen=True, slots=True)
class Observation:
    """One measured operation with the machine context for its result."""

    operation: Operation
    elapsed_seconds: float
    peak_memory_gb: float | None
    sample_count: int
    machine: str
    observed_at: str
    software: str

    def __post_init__(self) -> None:
        """Rejects measurements that cannot support a budget decision.

        Raises:
            ValueError: If the operation is not a known operation, the elapsed
                time or peak memory is not finite and non-negative, the sample
                count is negative (or a bool), the machine or software is
                blank, or the timestamp is malformed or not offset-aware.
        """
        operation: Any = self.operation  # may be a raw value at runtime
        if not isinstance(operation, Operation):
            object.__setattr__(self, "operation", Operation(operation))
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
        machine: object = self.machine  # may be a non-string at runtime
        if not isinstance(machine, str) or not machine.strip():
            raise ValueError("machine must not be empty")
        software: object = self.software  # may be a non-string at runtime
        if not isinstance(software, str) or not software.strip():
            raise ValueError("software must not be empty")
        parse_timestamp(self.observed_at, "observed_at")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native performance evidence."""
        return {
            "operation": self.operation.value,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_memory_gb": self.peak_memory_gb,
            "sample_count": self.sample_count,
            "machine": self.machine,
            "observed_at": self.observed_at,
            "software": self.software,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds an observation from JSON-native evidence.

        Args:
            record: A mapping as produced by `to_dict`.

        Returns:
            The observation.
        """
        return cls(
            operation=Operation(record["operation"]),
            elapsed_seconds=float(record["elapsed_seconds"]),
            peak_memory_gb=(
                float(record["peak_memory_gb"])
                if record.get("peak_memory_gb") is not None
                else None
            ),
            sample_count=int(record["sample_count"]),
            machine=str(record["machine"]),
            observed_at=str(record["observed_at"]),
            software=str(record["software"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Run:
    """One complete performance measurement run."""

    run_id: str
    observations: tuple[Observation, ...]

    def __post_init__(self) -> None:
        """Rejects empty or ambiguous run evidence.

        Raises:
            ValueError: If the run id is blank, there are no observations, an
                operation repeats, or the observations come from more than one
                environment.
        """
        run_id: object = self.run_id  # may be a non-string at runtime
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must not be empty")
        normalized = tuple(self.observations)
        operations = [observation.operation for observation in normalized]
        if not normalized:
            raise ValueError("performance run must contain observations")
        if len(set(operations)) != len(operations):
            raise ValueError(
                "performance run observations must be unique per operation"
            )
        environments = {
            (observation.machine, observation.software)
            for observation in normalized
        }
        if len(environments) != 1:
            raise ValueError(
                "performance run observations must share one environment"
            )
        object.__setattr__(self, "observations", normalized)

    def observation(self, operation: Operation) -> Observation | None:
        """Returns the observation for an operation, when present.

        Args:
            operation: The operation to look up.

        Returns:
            The observation, or None if the run has none for that operation.
        """
        return next(
            (
                observation
                for observation in self.observations
                if observation.operation is operation
            ),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native run evidence."""
        return {
            "run_id": self.run_id,
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a measured run from JSON-native evidence.

        Args:
            record: A mapping as produced by `to_dict`.

        Returns:
            The run.
        """
        return cls(
            run_id=str(record["run_id"]),
            observations=tuple(
                Observation.from_dict(observation)
                for observation in record["observations"]
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Comparison:
    """Decision and measurements from a baseline comparison."""

    passed: bool
    baseline_run_count: int
    current_run_id: str
    relative_regressions: dict[str, dict[str, float | None]]
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native comparison evidence."""
        return {
            "passed": self.passed,
            "baseline_run_count": self.baseline_run_count,
            "current_run_id": self.current_run_id,
            "relative_regressions": self.relative_regressions,
            "violations": list(self.violations),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Baseline:
    """Repeated-run performance baseline with a relative regression gate."""

    reference_runs: tuple[Run, ...]
    max_relative_regression: float = 0.10

    def __post_init__(self) -> None:
        """Validates repeated, schema-consistent reference measurements.

        Raises:
            ValueError: If there are fewer than two reference runs, the allowed
                regression is not finite and non-negative, the runs have no
                operations, differ in operations or sample counts, or come from
                more than one environment.
        """
        runs = tuple(self.reference_runs)
        if len(runs) < MIN_REFERENCE_RUNS:
            raise ValueError(
                "performance baseline requires at least two reference runs"
            )
        if (
            not math.isfinite(self.max_relative_regression)
            or self.max_relative_regression < 0
        ):
            raise ValueError(
                "max_relative_regression must be finite and non-negative"
            )
        expected = {
            observation.operation for observation in runs[0].observations
        }
        expected_sample_counts = {
            observation.operation: observation.sample_count
            for observation in runs[0].observations
        }
        if not expected:
            raise ValueError(
                "performance baseline operations must not be empty"
            )
        if any(
            {observation.operation for observation in run.observations}
            != expected
            for run in runs[1:]
        ):
            raise ValueError(
                "performance baseline runs must share the same operations"
            )
        for run in runs[1:]:
            for operation in expected:
                observation = run.observation(operation)
                if observation is None:
                    raise ValueError(
                        "performance baseline runs must share the same "
                        "operations"
                    )
                if (
                    observation.sample_count
                    != expected_sample_counts[operation]
                ):
                    raise ValueError(
                        "performance baseline runs must share sample counts"
                    )
        environments = {
            (observation.machine, observation.software)
            for run in runs
            for observation in run.observations
        }
        if len(environments) != 1:
            raise ValueError(
                "performance baseline runs must share one environment"
            )
        object.__setattr__(self, "reference_runs", runs)

    def evaluate(self, current: Run) -> Comparison:
        """Compares one run with the median of the repeated baseline.

        A run from a different environment, or with different sample counts,
        does not pass and is reported without regressions.

        Args:
            current: The run to judge.

        Returns:
            The comparison: whether every operation stayed within the allowed
            relative regression of the baseline median in time and, where
            every reference run recorded it, peak memory (a run lacking peak
            memory then is itself a violation); the regressions by operation
            value, each with ``elapsed_seconds`` and ``peak_memory_gb``
            entries; and the violations.

        Raises:
            RuntimeError: If an observation that construction guarantees is
                missing (defensive; not reachable for validated runs).
            ValueError: If the run covers different operations from the
                baseline.
        """
        operations = {
            observation.operation for observation in current.observations
        }
        expected = {
            observation.operation
            for observation in self.reference_runs[0].observations
        }
        if operations != expected:
            missing = sorted(
                operation.value for operation in expected - operations
            )
            extra = sorted(
                operation.value for operation in operations - expected
            )
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            raise ValueError(
                "performance comparison operations differ: "
                + ", ".join(details)
            )

        reference_environment = (
            self.reference_runs[0].observations[0].machine,
            self.reference_runs[0].observations[0].software,
        )
        current_environment = (
            current.observations[0].machine,
            current.observations[0].software,
        )
        if current_environment != reference_environment:
            return Comparison(
                passed=False,
                baseline_run_count=len(self.reference_runs),
                current_run_id=current.run_id,
                relative_regressions={},
                violations=(
                    "performance comparison environment differs from baseline",
                ),
            )

        sample_count_violations = self.sample_count_violations(
            current, expected
        )
        if sample_count_violations:
            return Comparison(
                passed=False,
                baseline_run_count=len(self.reference_runs),
                current_run_id=current.run_id,
                relative_regressions={},
                violations=sample_count_violations,
            )

        regressions: dict[str, dict[str, float | None]] = {}
        violations: list[str] = []
        for operation in sorted(expected, key=lambda item: item.value):
            regressions[operation.value], found = self.operation_regression(
                operation, current
            )
            violations.extend(found)
        return Comparison(
            passed=not violations,
            baseline_run_count=len(self.reference_runs),
            current_run_id=current.run_id,
            relative_regressions=regressions,
            violations=tuple(violations),
        )

    def sample_count_violations(
        self, current: Run, expected: set[Operation]
    ) -> tuple[str, ...]:
        """Lists the operations whose sample count differs from the baseline.

        Args:
            current: The run to judge.
            expected: The operations of the baseline, which `current` shares.

        Returns:
            One message per operation whose sample count differs from that of
            the first reference run; empty if none does.

        Raises:
            RuntimeError: If an observation that construction guarantees is
                missing (defensive; not reachable for validated runs).
        """
        violations: list[str] = []
        for operation in sorted(expected, key=lambda item: item.value):
            baseline_observation = self.reference_runs[0].observation(operation)
            current_observation = current.observation(operation)
            if baseline_observation is None or current_observation is None:
                raise RuntimeError(
                    "validated sample-count observation disappeared: "
                    f"{operation.value}"
                )
            if (
                current_observation.sample_count
                != baseline_observation.sample_count
            ):
                violations.append(
                    f"{operation.value} sample count differs from baseline: "
                    f"expected {baseline_observation.sample_count}, "
                    f"observed {current_observation.sample_count}"
                )
        return tuple(violations)

    def operation_regression(
        self, operation: Operation, current: Run
    ) -> tuple[dict[str, float | None], list[str]]:
        """Compares one operation of a run with the baseline median.

        Args:
            operation: The operation to compare.
            current: The run to judge.

        Returns:
            The relative regressions, with ``elapsed_seconds`` and
            ``peak_memory_gb`` entries, and the violations of the operation.

        Raises:
            RuntimeError: If an observation that construction guarantees is
                missing (defensive; not reachable for validated runs).
        """
        violations: list[str] = []
        current_observation = current.observation(operation)
        if current_observation is None:
            raise RuntimeError(
                "validated performance observation disappeared: "
                f"{operation.value}"
            )
        reference_observations = [
            run.observation(operation) for run in self.reference_runs
        ]
        if any(observation is None for observation in reference_observations):
            raise RuntimeError(
                f"validated baseline observation disappeared: {operation.value}"
            )
        reference_times = [
            observation.elapsed_seconds
            for observation in reference_observations
            if observation is not None
        ]
        time_regression = relative_change(
            statistics.median(reference_times),
            current_observation.elapsed_seconds,
        )
        memory_regression: float | None = None
        reference_memory = [
            observation.peak_memory_gb
            for observation in reference_observations
            if observation is not None
        ]
        if all(value is not None for value in reference_memory):
            if current_observation.peak_memory_gb is None:
                violations.append(
                    f"missing peak memory observation: {operation.value}"
                )
            else:
                memory_regression = relative_change(
                    statistics.median(
                        [
                            value
                            for value in reference_memory
                            if value is not None
                        ]
                    ),
                    current_observation.peak_memory_gb,
                )
        regression = {
            "elapsed_seconds": time_regression,
            "peak_memory_gb": memory_regression,
        }
        if time_regression > self.max_relative_regression:
            violations.append(
                f"{operation.value} time regression "
                f"{time_regression:.2%} exceeds "
                f"{self.max_relative_regression:.2%}"
            )
        if (
            memory_regression is not None
            and memory_regression > self.max_relative_regression
        ):
            violations.append(
                f"{operation.value} memory regression "
                f"{memory_regression:.2%} exceeds "
                f"{self.max_relative_regression:.2%}"
            )
        return regression, violations

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native baseline policy and reference evidence."""
        return {
            "max_relative_regression": self.max_relative_regression,
            "reference_runs": [run.to_dict() for run in self.reference_runs],
        }


def relative_change(reference: float, current: float) -> float:
    """Returns the non-negative relative increase from reference to current.

    Args:
        reference: The reference value.
        current: The value compared with it.

    Returns:
        The increase as a share of the reference, or 0 if the value fell. A
        zero reference gives 0 unless `current` is positive, which gives
        infinity.
    """
    if reference == 0:
        return 0.0 if current <= 0 else math.inf
    return max((current - reference) / reference, 0.0)


@dataclasses.dataclass(frozen=True, slots=True)
class Budget:
    """Explicit time and optional memory ceilings for lifecycle operations."""

    max_seconds: Mapping[Operation, float]
    max_peak_memory_gb: float | None = None

    def __post_init__(self) -> None:
        """Validates positive ceilings and normalizes operation names.

        Raises:
            ValueError: If an operation name is unknown, a time limit or the
                memory limit is not positive and finite, or there are no time
                limits.
        """
        normalized: dict[Operation, float] = {}
        for operation, limit in self.max_seconds.items():
            named: Any = operation  # keys may be raw values at runtime
            resolved = (
                named if isinstance(named, Operation) else Operation(named)
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

    def evaluate(self, observations: tuple[Observation, ...]) -> "Verdict":
        """Evaluates observations without treating missing data as pass.

        Args:
            observations: The observations, at most one for each operation.

        Returns:
            The result: whether no violation was found, the observations as
            given, and the violations. Only operations with a time limit are
            checked (extra observations are ignored). Each is checked for a
            missing observation, an exceeded time limit and, when
            `max_peak_memory_gb` is set, an exceeded or missing peak memory.

        Raises:
            ValueError: If an operation is observed more than once.
        """
        by_operation: dict[Operation, Observation] = {}
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
        return Verdict(
            passed=not violations,
            observations=observations,
            violations=tuple(violations),
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native budget policy."""
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
class Verdict:
    """Budget decision with the observations that justify it."""

    passed: bool
    observations: tuple[Observation, ...]
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native performance decision evidence."""
        return {
            "passed": self.passed,
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
            "violations": list(self.violations),
        }


__all__ = [
    "Baseline",
    "Budget",
    "Comparison",
    "Observation",
    "Operation",
    "Run",
    "Verdict",
    "parse_timestamp",
    "relative_change",
]
