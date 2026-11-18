"""Measured performance evidence and explicit release budgets."""

from collections.abc import Mapping
import dataclasses
import datetime
import enum
import math
import statistics
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.contracts import errors
from scmle.contracts import records

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
        raise errors.InputError(
            f"{field_name} must include timezone information"
        )
    return timestamp


@dataclasses.dataclass(frozen=True, slots=True)
class Observation:
    """One measured operation with the machine context for its result.

    Attributes:
        operation: The lifecycle operation measured.
        elapsed: Wall-clock time of the operation, in seconds.
        memory: Peak resident memory during the operation, in gigabytes, or
            None when it was not measured.
        samples: Number of samples the operation processed.
        machine: Machine the measurement ran on.
        observed: Time of the measurement, as an offset-aware ISO-8601
            timestamp.
        software: Software environment of the measurement.
        migrations: Version steps of the stored observation, which is
            versioned by the run that holds it.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "elapsed_seconds": "elapsed",
                "peak_memory_gb": "memory",
                "sample_count": "samples",
                "observed_at": "observed",
            }
        ),
    )

    operation: Operation
    elapsed: float
    memory: float | None
    samples: int
    machine: str
    observed: str
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
        if not math.isfinite(self.elapsed) or self.elapsed < 0:
            raise errors.InputError("elapsed must be finite and non-negative")
        if self.memory is not None and (
            not math.isfinite(self.memory) or self.memory < 0
        ):
            raise errors.InputError(
                "memory must be finite and non-negative when present"
            )
        if isinstance(self.samples, bool) or self.samples < 0:
            raise errors.InputError("samples must be non-negative")
        machine: object = self.machine  # may be a non-string at runtime
        if not isinstance(machine, str) or not machine.strip():
            raise errors.InputError("machine must not be empty")
        software: object = self.software  # may be a non-string at runtime
        if not isinstance(software, str) or not software.strip():
            raise errors.InputError("software must not be empty")
        parse_timestamp(self.observed, "observed")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native performance evidence."""
        return {
            "operation": self.operation.value,
            "elapsed": self.elapsed,
            "memory": self.memory,
            "samples": self.samples,
            "machine": self.machine,
            "observed": self.observed,
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
            elapsed=float(record["elapsed"]),
            memory=(
                float(record["memory"])
                if record.get("memory") is not None
                else None
            ),
            samples=int(record["samples"]),
            machine=str(record["machine"]),
            observed=str(record["observed"]),
            software=str(record["software"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Run:
    """One complete performance measurement run.

    Attributes:
        identifier: The run's own id.
        observations: One observation per measured operation.
        migrations: Version steps of the stored run and its observations.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            renames={"run_id": "identifier"},
            parts={"observations": Observation.migrations[0]},
        ),
    )

    identifier: str
    observations: tuple[Observation, ...]

    def __post_init__(self) -> None:
        """Rejects empty or ambiguous run evidence.

        Raises:
            ValueError: If the run id is blank, there are no observations, an
                operation repeats, or the observations come from more than one
                environment.
        """
        identifier: object = self.identifier  # may be a non-string at runtime
        if not isinstance(identifier, str) or not identifier.strip():
            raise errors.InputError("identifier must not be empty")
        normalized = tuple(self.observations)
        operations = [observation.operation for observation in normalized]
        if not normalized:
            raise errors.InputError("performance run must contain observations")
        if len(set(operations)) != len(operations):
            raise errors.InputError(
                "performance run observations must be unique per operation"
            )
        environments = {
            (observation.machine, observation.software)
            for observation in normalized
        }
        if len(environments) != 1:
            raise errors.InputError(
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
            "schema_version": records.ORIGIN + len(self.migrations),
            "identifier": self.identifier,
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a measured run from JSON-native evidence.

        A run stored at an older schema version is migrated first.

        Args:
            record: A mapping as produced by `to_dict`, at any supported
                schema version.

        Returns:
            The run.
        """
        record = records.load(cls.__name__, cls.migrations, record)
        return cls(
            identifier=str(record["identifier"]),
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
            raise errors.InputError(
                "performance baseline requires at least two reference runs"
            )
        if (
            not math.isfinite(self.max_relative_regression)
            or self.max_relative_regression < 0
        ):
            raise errors.InputError(
                "max_relative_regression must be finite and non-negative"
            )
        expected = {
            observation.operation for observation in runs[0].observations
        }
        expected_sample_counts = {
            observation.operation: observation.samples
            for observation in runs[0].observations
        }
        if not expected:
            raise errors.InputError(
                "performance baseline operations must not be empty"
            )
        if any(
            {observation.operation for observation in run.observations}
            != expected
            for run in runs[1:]
        ):
            raise errors.InputError(
                "performance baseline runs must share the same operations"
            )
        for run in runs[1:]:
            for operation in expected:
                observation = run.observation(operation)
                if observation is None:
                    raise errors.InputError(
                        "performance baseline runs must share the same "
                        "operations"
                    )
                if observation.samples != expected_sample_counts[operation]:
                    raise errors.InputError(
                        "performance baseline runs must share sample counts"
                    )
        environments = {
            (observation.machine, observation.software)
            for run in runs
            for observation in run.observations
        }
        if len(environments) != 1:
            raise errors.InputError(
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
        expected = self.match(current)
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
                current_run_id=current.identifier,
                relative_regressions={},
                violations=(
                    "performance comparison environment differs from baseline",
                ),
            )

        tally = self.tally(current, expected)
        if tally:
            return Comparison(
                passed=False,
                baseline_run_count=len(self.reference_runs),
                current_run_id=current.identifier,
                relative_regressions={},
                violations=tally,
            )

        regressions: dict[str, dict[str, float | None]] = {}
        violations: list[str] = []
        for operation in sorted(expected, key=lambda item: item.value):
            regressions[operation.value], found = self.regress(
                operation, current
            )
            violations.extend(found)
        return Comparison(
            passed=not violations,
            baseline_run_count=len(self.reference_runs),
            current_run_id=current.identifier,
            relative_regressions=regressions,
            violations=tuple(violations),
        )

    def match(self, current: Run) -> set[Operation]:
        """Returns the baseline operations, refusing a run that differs.

        Args:
            current: The run to judge.

        Returns:
            The operations of the first reference run.

        Raises:
            ValueError: If the run covers different operations from the
                baseline; the message names the missing and extra ones.
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
            raise errors.InputError(
                "performance comparison operations differ: "
                + ", ".join(details)
            )
        return expected

    def tally(self, current: Run, expected: set[Operation]) -> tuple[str, ...]:
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
            if current_observation.samples != baseline_observation.samples:
                violations.append(
                    f"{operation.value} sample count differs from baseline: "
                    f"expected {baseline_observation.samples}, "
                    f"observed {current_observation.samples}"
                )
        return tuple(violations)

    def regress(
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
            observation.elapsed
            for observation in reference_observations
            if observation is not None
        ]
        time_regression = rise(
            statistics.median(reference_times),
            current_observation.elapsed,
        )
        reference_memory = [
            observation.memory
            for observation in reference_observations
            if observation is not None
        ]
        memory_regression, violations = weigh(
            operation, reference_memory, current_observation.memory
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


def rise(reference: float, current: float) -> float:
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


def weigh(
    operation: Operation,
    reference: list[float | None],
    current: float | None,
) -> tuple[float | None, list[str]]:
    """Returns the peak-memory regression of one operation.

    Memory is compared only when every reference run recorded it.

    Args:
        operation: The operation compared.
        reference: Peak memory of each reference run, None where unrecorded.
        current: Peak memory of the judged run, or None.

    Returns:
        The relative regression from the reference median, or None when it
        is not compared, and the violations: a missing current observation
        where every reference run recorded one.
    """
    if not all(value is not None for value in reference):
        return None, []
    if current is None:
        return None, [f"missing peak memory observation: {operation.value}"]
    recorded = [value for value in reference if value is not None]
    return rise(statistics.median(recorded), current), []


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
                raise errors.InputError(
                    f"performance limit must be positive: {resolved.value}"
                )
            normalized[resolved] = float(limit)
        if not normalized:
            raise errors.InputError("max_seconds must not be empty")
        if self.max_peak_memory_gb is not None and (
            not math.isfinite(self.max_peak_memory_gb)
            or self.max_peak_memory_gb <= 0
        ):
            raise errors.InputError(
                "max_peak_memory_gb must be positive when present"
            )
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
                raise errors.InputError(
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
            if measurement.elapsed > limit:
                violations.append(
                    f"{operation.value} took "
                    f"{measurement.elapsed:.6g}s "
                    f"> {limit:.6g}s budget"
                )
            memory = measurement.memory
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
    "rise",
]
