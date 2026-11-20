"""The job spec, the states of an attempt and a run, and the cost meter."""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
import datetime
import enum
import hashlib
import json
import time
from typing import Any
from typing import ClassVar

from scmle.contracts import errors
from scmle.contracts import records
from scmle.runtime import retry

MAX_RUN_ID_LENGTH = 64
RUN_ID_PATTERN = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
)
SECRET_WORDS = ("SECRET", "PASSWORD", "TOKEN", "CREDENTIAL", "PRIVATE", "KEY")


class State(enum.StrEnum):
    """Where one attempt is.

    Attributes:
        PENDING: Submitted, not yet running.
        RUNNING: Executing.
        SUCCEEDED: Ended successfully.
        FAILED: Ended with an error.
        INTERRUPTED: Taken away by the platform (for example preemption).
        CANCELLED: Stopped on request.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


TERMINAL = frozenset(
    {
        State.SUCCEEDED,
        State.FAILED,
        State.INTERRUPTED,
        State.CANCELLED,
    }
)


class Progress(enum.StrEnum):
    """How a whole run ended (or that it has not).

    Attributes:
        STARTED: Not finished.
        SUCCEEDED: An attempt succeeded and left a result.
        FAILED: An attempt failed without a retry, or ended without a result.
        BUDGET_STOPPED: Cancelled because the estimated cost passed the budget.
        TIMED_OUT: Cancelled because an attempt passed its time limit.
        RETRIES_EXHAUSTED: Every allowed attempt was interrupted or failed
            retryably.
    """

    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BUDGET_STOPPED = "budget_stopped"
    TIMED_OUT = "timed_out"
    RETRIES_EXHAUSTED = "retries_exhausted"


FINAL = frozenset(set(Progress) - {Progress.STARTED})


class ConflictError(errors.ScmleError):
    """A run id is being reused for a different spec."""

    category = "run_conflict"


@dataclasses.dataclass(frozen=True, slots=True)
class Resources:
    """What a job asks for.

    Attributes:
        cpu: vCPUs.
        memory_mb: Memory in MiB.
        gpu: GPUs; more than zero needs a stated reason.
        gpu_reason: Why a GPU is needed (declared need).
    """

    # The public path: pinned public signatures name this class, and the
    # API snapshot spells it with its module.
    __module__ = "scmle.runners.base"

    cpu: float = 1.0
    memory_mb: int = 2048
    gpu: int = 0
    gpu_reason: str = ""

    def __post_init__(self) -> None:
        """Validates the request.

        Raises:
            ConfigError: If `cpu` or `memory_mb` is not positive, `gpu` is
                negative, or a GPU is requested with a blank `gpu_reason`.
        """
        if self.cpu <= 0 or self.memory_mb <= 0 or self.gpu < 0:
            raise errors.ConfigError("resources must be positive")
        if self.gpu > 0 and not self.gpu_reason.strip():
            raise errors.ConfigError(
                "specialised compute needs a declared reason (gpu_reason)"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class CostRate:
    """Declared prices, per hour, for the cost estimate.

    Attributes:
        per_cpu_hour: Price of one vCPU for an hour.
        per_gb_hour: Price of one GiB of memory for an hour.
        per_gpu_hour: Price of one GPU for an hour.
    """

    # The public path: pinned public signatures name this class, and the
    # API snapshot spells it with its module.
    __module__ = "scmle.runners.base"

    per_cpu_hour: float = 0.0
    per_gb_hour: float = 0.0
    per_gpu_hour: float = 0.0

    def cost(self, resources: Resources, seconds: float) -> float:
        """Estimates the cost of running resources for some time.

        Args:
            resources: The resources used.
            seconds: Elapsed seconds; a negative value counts as zero.

        Returns:
            The hourly price of the resources times the elapsed hours, in the
            currency of the rate.
        """
        hourly = (
            resources.cpu * self.per_cpu_hour
            + resources.memory_mb / 1024 * self.per_gb_hour
            + resources.gpu * self.per_gpu_hour
        )
        return hourly * max(0.0, seconds) / 3600.0


@dataclasses.dataclass(frozen=True, slots=True)
class CostBudget:
    """A ceiling on the estimated cost of a whole run.

    Attributes:
        max_cost: The ceiling (same currency as the rate).
        rate: The declared prices.
    """

    # The public path: pinned public signatures name this class, and the
    # API snapshot spells it with its module.
    __module__ = "scmle.runners.base"

    max_cost: float
    rate: CostRate

    def __post_init__(self) -> None:
        """Validates the ceiling.

        Raises:
            ConfigError: If `max_cost` is not positive.
        """
        if self.max_cost <= 0:
            raise errors.ConfigError("max_cost must be positive")


@dataclasses.dataclass(frozen=True, slots=True)
class Job:
    """One training run, independent of where it runs.

    Attributes:
        run_id: Durable id (letters, digits, `-`, `_`); the idempotency key.
        name: Human-readable job name.
        project: The project specification (JSON-ready), opaque to runners.
        identity: The training identity (role ARN or service account).
        resources: What the job asks for.
        timeout_seconds: Wall-clock limit of one attempt.
        retries: Attempts and backoff.
        budget: Cost ceiling for the whole run, if any.
        environment: Non-secret environment for the job.
        KEYS: Version-1 keys to the written ones (`identifier`, `timeout`
            in seconds, `digests.project`); `downgrade` restores version 1.
    """

    KEYS: ClassVar[records.Migration] = records.nest(
        {"run_id": "identifier", "timeout_seconds": "timeout"},
        groups={"digests": {"project_digest": "project"}},
    )

    run_id: str
    name: str
    project: Mapping[str, Any]
    identity: str
    resources: Resources = dataclasses.field(default_factory=Resources)
    timeout_seconds: float = 3600.0
    retries: retry.Retries = dataclasses.field(default_factory=retry.Retries)
    budget: CostBudget | None = None
    environment: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validates the spec.

        Raises:
            ConfigError: If the id is unsafe, the timeout is not positive, the
                identity is empty or the environment holds something
                secret-looking.
        """
        if (
            not self.run_id
            or len(self.run_id) > MAX_RUN_ID_LENGTH
            or set(self.run_id) - set(RUN_ID_PATTERN)
        ):
            raise errors.ConfigError(
                "run_id must be 1-64 characters of letters, digits, - and _"
            )
        if self.timeout_seconds <= 0 or not self.identity:
            raise errors.ConfigError("a timeout and an identity are required")
        for name, value in self.environment.items():
            if any(word in name.upper() for word in SECRET_WORDS):
                raise errors.ConfigError(
                    f"environment {name} looks like a secret; use a secret "
                    "reference of the platform instead"
                )
            if "-----BEGIN" in value:
                raise errors.ConfigError(f"environment {name} holds a key")

    @property
    def project_digest(self) -> str:
        """The SHA-256 of the canonical project specification."""
        text = json.dumps(self.project, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready form that is stored with the run.

        Returns:
            A dict of all fields, plus the project digest under
            `digests.project`, with nested objects as dicts.
        """
        return self.KEYS.upgrade(self.layout(self.retries.to_dict()))

    def anchor(self) -> records.Layout:
        """Builds the spec in its version-1 layout, the one `digest` names.

        Returns:
            The spec as version 1 wrote it (`run_id`, `project_digest`,
            `timeout_seconds`, the retry policy's field names).
        """
        return self.layout(self.retries.anchor())

    def layout(self, retries: Mapping[str, Any]) -> records.Layout:
        """Builds the version-1 layout around a written retry policy.

        Args:
            retries: The retry policy as it is to be written.

        Returns:
            The spec's fields with their version-1 keys.
        """
        return {
            "run_id": self.run_id,
            "name": self.name,
            "project": dict(self.project),
            "project_digest": self.project_digest,
            "identity": self.identity,
            "resources": dataclasses.asdict(self.resources),
            "timeout_seconds": self.timeout_seconds,
            "retries": dict(retries),
            "budget": None
            if self.budget is None
            else {
                "max_cost": self.budget.max_cost,
                "rate": dataclasses.asdict(self.budget.rate),
            },
            "environment": dict(self.environment),
        }

    @classmethod
    def read(cls, record: Mapping[str, Any]) -> "Job":
        """Rebuilds a spec from its stored form.

        Args:
            record: The spec as `to_dict` wrote it, in the current layout
                (`identifier`, `timeout`, `digests.project`) or in version 1
                (`run_id`, `timeout_seconds`, `project_digest`).

        Returns:
            The spec.

        Raises:
            ConfigError: If a field is missing or has the wrong shape.
        """
        try:
            fields = cls.KEYS.downgrade(dict(record))
            budget = fields["budget"]
            return cls(
                run_id=fields["run_id"],
                name=fields["name"],
                project=fields["project"],
                identity=fields["identity"],
                resources=Resources(**fields["resources"]),
                timeout_seconds=fields["timeout_seconds"],
                retries=retry.Retries.read(fields["retries"]),
                budget=None
                if budget is None
                else CostBudget(budget["max_cost"], CostRate(**budget["rate"])),
                environment=fields["environment"],
            )
        except (KeyError, TypeError, AttributeError) as error:
            raise errors.ConfigError(f"malformed job spec: {error}") from error

    def digest(self) -> str:
        """Digests the whole spec, anchored at version 1.

        Renaming a written key leaves the digest of a started run unchanged.

        Returns:
            The hex SHA-256 of the key-sorted JSON of `anchor()`.
        """
        text = json.dumps(self.anchor(), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class Ticket:
    """A started attempt.

    Attributes:
        job_id: The runner's identifier of the attempt; the local runner
            uses `local-<run_id>-<attempt>`.
        attempt: Attempt number, from 1.
    """

    job_id: str
    attempt: int


@dataclasses.dataclass(frozen=True, slots=True)
class Status:
    """What a platform reports about an attempt.

    Attributes:
        state: Current state.
        reason: Platform reason or exit description.
        retryable: For `FAILED`: whether another attempt may succeed.
    """

    state: State
    reason: str = ""
    retryable: bool = False

    def outcome(self) -> Progress | None:
        """Says how this terminal status ends the run.

        Returns:
            `SUCCEEDED` for a success; None for an interruption or a
            retryable failure, after which another attempt may follow;
            otherwise `FAILED`.
        """
        if self.state is State.SUCCEEDED:
            return Progress.SUCCEEDED
        if self.state is State.INTERRUPTED or (
            self.state is State.FAILED and self.retryable
        ):
            return None
        return Progress.FAILED


class Exit(enum.IntEnum):
    """How a job process ends: the exit status of the entrypoint.

    The numbers follow conventions a container platform already reports:
    `TRANSIENT` is `EX_TEMPFAIL` from sysexits.h, `INTERRUPTED` is 128 plus
    SIGTERM, and `KILLED` is 128 plus SIGKILL (a stop that outlived its
    grace period, or a reclaimed host).

    Attributes:
        SUCCEEDED: The workload returned.
        FAILED: The workload raised; another attempt would fail the same way.
        TRANSIENT: The workload raised a transient error; a retry may pass.
        INTERRUPTED: The platform asked the job to stop (SIGTERM).
        KILLED: The platform killed the job (SIGKILL).
    """

    __module__ = "scmle.runners.base"

    SUCCEEDED = 0
    FAILED = 1
    TRANSIENT = 75
    INTERRUPTED = 143
    KILLED = 137

    @classmethod
    def status(cls, code: int, reason: str = "") -> Status:
        """Reads the exit status of a job process.

        A negative code is how `subprocess` reports a child ended by a
        signal; any signal counts as an interruption by the platform.

        Args:
            code: The process exit status.
            reason: What the process said last (its stderr tail).

        Returns:
            `SUCCEEDED` for 0, an `INTERRUPTED` status for a signal, a
            retryable `FAILED` for `TRANSIENT`, otherwise a final `FAILED`
            whose reason names the code.
        """
        if code == cls.SUCCEEDED:
            return Status(State.SUCCEEDED)
        if code < 0 or code in {cls.INTERRUPTED, cls.KILLED}:
            return Status(State.INTERRUPTED, f"signal (exit {code})")
        detail = f"exit {code}" + (f": {reason}" if reason else "")
        return Status(State.FAILED, detail, retryable=code == cls.TRANSIENT)


class Clock:
    """Time source for supervision; tests replace both functions.

    Attributes:
        now: Returns monotonic seconds.
        sleep: Waits for the given seconds.
    """

    # The public path: pinned public signatures name this class, and the
    # API snapshot spells it with its module.
    __module__ = "scmle.runners.base"

    def __init__(
        self,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initializes the clock.

        Args:
            now: Monotonic seconds.
            sleep: Waits.
        """
        self.now = now
        self.sleep = sleep


def stamp() -> str:
    """Formats the current time.

    Returns:
        The current UTC time as an ISO 8601 string with second resolution.
    """
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


class Meter:
    """Accumulates the estimated cost of a run across attempts.

    Attributes:
        spec: The job.
        spent: Cost accumulated by finished attempts.
    """

    def __init__(self, spec: Job, spent: float = 0.0) -> None:
        """Initializes the meter.

        Args:
            spec: The job (its resources and budget).
            spent: Cost already incurred by earlier attempts.
        """
        self.spec = spec
        self.spent = spent

    def estimate(self, seconds: float) -> float:
        """Estimates the run's cost including a running attempt.

        Args:
            seconds: Elapsed seconds of the current attempt.

        Returns:
            The cost already spent plus the cost of this attempt; just the cost
            already spent if the job has no budget.
        """
        if self.spec.budget is None:
            return self.spent
        rate = self.spec.budget.rate
        return self.spent + rate.cost(self.spec.resources, seconds)

    def exceeded(self, seconds: float) -> bool:
        """Reports whether the run's cost passed its ceiling.

        Args:
            seconds: Elapsed seconds of the current attempt.

        Returns:
            True if the job has a budget and `estimate(seconds)` is above its
            ceiling.
        """
        budget = self.spec.budget
        return budget is not None and self.estimate(seconds) > budget.max_cost

    def close(self, seconds: float) -> float:
        """Adds a finished attempt's cost.

        Args:
            seconds: Elapsed seconds of the finished attempt.

        Returns:
            The new total cost.
        """
        self.spent = self.estimate(seconds)
        return self.spent

    def verdict(self, seconds: float) -> Progress | None:
        """Says whether the running attempt must stop.

        Args:
            seconds: Elapsed seconds of the current attempt.

        Returns:
            `BUDGET_STOPPED` if the run's cost passed its ceiling, else
            `TIMED_OUT` if the attempt passed the job's time limit, else None.
        """
        if self.exceeded(seconds):
            return Progress.BUDGET_STOPPED
        if seconds > self.spec.timeout_seconds:
            return Progress.TIMED_OUT
        return None
