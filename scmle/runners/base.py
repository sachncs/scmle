"""The job spec, the run manifest and the `Runner` template.

One `JobSpec` describes a training run independent of where it runs. A
`Runner` adapter knows how to start, poll and cancel a job in one place
(this process, a container job, a managed service). The shared `Runner.run`
does everything that must not differ between places:

* **Durable run id.** The spec is stored at ``runs/<id>/spec.json``; running
  an id again with a different spec is refused, and running a finished id
  returns its recorded outcome without starting anything.
* **Manifest.** ``runs/<id>/manifest.json`` is rewritten atomically at every
  transition: attempts, states, costs, reasons, and at the end the digests of
  the checkpoints and the result.
* **Controlled retries.** An attempt that ends interrupted (preemption,
  spot reclaim) or failed with a retryable cause is retried under the spec's
  `RetryPolicy`, up to its attempt limit; the retry resumes from the run's
  checkpoints. Other failures, timeouts and budget stops are final.
* **Cost budget.** Cost is the declared rate times resources times elapsed
  time, accumulated across attempts. When it passes the budget the job is
  cancelled and the run ends `budget_stopped`. It is an estimate from the
  declared rates, not billing data.
* **Separate roles.** A job may not run as the serving identity.
"""

import abc
from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
import datetime
import enum
import hashlib
import json
import time
from typing import Any

from scmle.contracts import errors
from scmle.core import retry
from scmle.data.store import objectstore
from scmle.runners import checkpoint as checkpoint_module
from scmle.runners import roles

MANIFEST_FORMAT = 1
RUN_ID_PATTERN = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
)
SECRET_WORDS = ("SECRET", "PASSWORD", "TOKEN", "CREDENTIAL", "PRIVATE", "KEY")


class JobState(enum.StrEnum):
    """Where one attempt is."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


TERMINAL = frozenset(
    {
        JobState.SUCCEEDED,
        JobState.FAILED,
        JobState.INTERRUPTED,
        JobState.CANCELLED,
    }
)


class RunState(enum.StrEnum):
    """How a whole run ended (or that it has not)."""

    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BUDGET_STOPPED = "budget_stopped"
    TIMED_OUT = "timed_out"
    RETRIES_EXHAUSTED = "retries_exhausted"


FINAL = frozenset(set(RunState) - {RunState.STARTED})


class RunConflictError(errors.ScmleError):
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

    cpu: float = 1.0
    memory_mb: int = 2048
    gpu: int = 0
    gpu_reason: str = ""

    def __post_init__(self) -> None:
        """Validates the request.

        Raises:
            ConfigError: If a size is not positive or a GPU has no reason.
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

    per_cpu_hour: float = 0.0
    per_gb_hour: float = 0.0
    per_gpu_hour: float = 0.0

    def cost(self, resources: Resources, seconds: float) -> float:
        """The estimated cost of `seconds` of these resources."""
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

    max_cost: float
    rate: CostRate

    def __post_init__(self) -> None:
        """Validates the ceiling."""
        if self.max_cost <= 0:
            raise errors.ConfigError("max_cost must be positive")


@dataclasses.dataclass(frozen=True, slots=True)
class JobSpec:
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
    """

    run_id: str
    name: str
    project: Mapping[str, Any]
    identity: str
    resources: Resources = dataclasses.field(default_factory=Resources)
    timeout_seconds: float = 3600.0
    retries: retry.RetryPolicy = dataclasses.field(
        default_factory=retry.RetryPolicy
    )
    budget: CostBudget | None = None
    environment: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validates the spec.

        Raises:
            ConfigError: If the id is unsafe, the timeout not positive or the
                environment holds something secret-looking.
        """
        if (
            not self.run_id
            or len(self.run_id) > 64
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
        """SHA-256 of the canonical project specification."""
        text = json.dumps(self.project, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form (stored with the run)."""
        return {
            "run_id": self.run_id,
            "name": self.name,
            "project": dict(self.project),
            "project_digest": self.project_digest,
            "identity": self.identity,
            "resources": dataclasses.asdict(self.resources),
            "timeout_seconds": self.timeout_seconds,
            "retries": self.retries.to_dict(),
            "budget": None
            if self.budget is None
            else {
                "max_cost": self.budget.max_cost,
                "rate": dataclasses.asdict(self.budget.rate),
            },
            "environment": dict(self.environment),
        }

    def digest(self) -> str:
        """SHA-256 of the whole spec."""
        text = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class JobHandle:
    """A started attempt.

    Attributes:
        job_id: The platform's id (task ARN, execution name, thread id).
        attempt: Attempt number, from 1.
    """

    job_id: str
    attempt: int


@dataclasses.dataclass(frozen=True, slots=True)
class JobStatus:
    """What a platform reports about an attempt.

    Attributes:
        state: Current state.
        reason: Platform reason or exit description.
        retryable: For `FAILED`: whether another attempt may succeed.
    """

    state: JobState
    reason: str = ""
    retryable: bool = False


class Clock:
    """Time source for supervision; tests replace both functions."""

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
    """The current UTC time, ISO 8601."""
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


@dataclasses.dataclass(slots=True)
class Attempt:
    """The record of one attempt in the manifest."""

    attempt: int
    job_id: str
    state: str
    started_at: str
    ended_at: str = ""
    seconds: float = 0.0
    cost: float = 0.0
    reason: str = ""


@dataclasses.dataclass(slots=True)
class RunManifest:
    """The durable record of a run."""

    run_id: str
    runner: str
    spec_digest: str
    project_digest: str
    identity: str
    state: str = RunState.STARTED.value
    attempts: list[Attempt] = dataclasses.field(default_factory=list)
    total_cost: float = 0.0
    budget: float | None = None
    checkpoints: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    result_sha256: str = ""
    evidence: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    started_at: str = ""
    ended_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        record = dataclasses.asdict(self)
        record["format"] = MANIFEST_FORMAT
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "RunManifest":
        """Rebuilds a manifest from its stored form."""
        fields = {k: v for k, v in record.items() if k != "format"}
        fields["attempts"] = [Attempt(**a) for a in fields["attempts"]]
        return cls(**fields)


class BudgetMeter:
    """Accumulates the estimated cost of a run across attempts."""

    def __init__(self, spec: JobSpec, spent: float = 0.0) -> None:
        """Initializes the meter.

        Args:
            spec: The job (its resources and budget).
            spent: Cost already incurred by earlier attempts.
        """
        self.spec = spec
        self.spent = spent

    def cost_of(self, seconds: float) -> float:
        """The cost of `seconds` of this attempt on top of `spent`."""
        if self.spec.budget is None:
            return self.spent
        rate = self.spec.budget.rate
        return self.spent + rate.cost(self.spec.resources, seconds)

    def exceeded(self, seconds: float) -> bool:
        """Whether the run's cost passed its ceiling."""
        budget = self.spec.budget
        return budget is not None and self.cost_of(seconds) > budget.max_cost

    def close(self, seconds: float) -> float:
        """Adds a finished attempt's cost and returns the new total."""
        self.spent = self.cost_of(seconds)
        return self.spent


class Runner(abc.ABC):
    """Runs a `JobSpec` in one place, with the shared supervision."""

    kind = "runner"

    def __init__(
        self,
        store: objectstore.ObjectStore,
        *,
        serving_identity: str,
        clock: Clock | None = None,
        poll_seconds: float = 5.0,
    ) -> None:
        """Initializes the runner.

        Args:
            store: Durable storage for the spec, manifest, checkpoints and
                result. It must outlive the job.
            serving_identity: The identity the serving side uses; a job may
                not run as it.
            clock: Time source (tests inject one).
            poll_seconds: Seconds between status polls.
        """
        self.store = store
        self.serving_identity = serving_identity
        self.clock = clock or Clock()
        self.poll_seconds = poll_seconds
        self.stop_reason: RunState | None = None
        self.watcher: Callable[[], RunState | None] = lambda: None

    @abc.abstractmethod
    def submit(self, spec: JobSpec, attempt: int) -> JobHandle:
        """Starts one attempt."""

    @abc.abstractmethod
    def poll(self, handle: JobHandle) -> JobStatus:
        """Reports the attempt's state."""

    @abc.abstractmethod
    def cancel(self, handle: JobHandle, reason: str) -> None:
        """Asks the platform to stop the attempt."""

    def manifest_key(self, run_id: str) -> str:
        """Where a run's manifest lives."""
        return f"runs/{run_id}/manifest.json"

    def spec_key(self, run_id: str) -> str:
        """Where a run's spec lives."""
        return f"runs/{run_id}/spec.json"

    def result_key(self, run_id: str) -> str:
        """Where a workload leaves its result."""
        return f"runs/{run_id}/result.json"

    def save(self, manifest: RunManifest) -> None:
        """Writes the manifest atomically."""
        body = json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
        self.store.put(self.manifest_key(manifest.run_id), body.encode())

    def load(self, run_id: str) -> RunManifest | None:
        """The stored manifest, if the run exists."""
        try:
            data, ignored = self.store.get(self.manifest_key(run_id))
        except objectstore.NotFoundError:
            return None
        return RunManifest.from_dict(json.loads(data))

    def register(self, spec: JobSpec) -> RunManifest:
        """Stores the spec, or recognises the same run, or refuses a clash.

        Raises:
            RunConflictError: If the id exists with a different spec.
            ConfigError: If the job would run as the serving identity.
        """
        roles.check_distinct(spec.identity, self.serving_identity)
        body = json.dumps(spec.to_dict(), indent=2, sort_keys=True).encode()
        try:
            self.store.put(self.spec_key(spec.run_id), body, if_absent=True)
        except objectstore.AlreadyExistsError:
            stored, ignored = self.store.get(self.spec_key(spec.run_id))
            if stored != body:
                raise RunConflictError(
                    f"run {spec.run_id} exists with a different spec"
                ) from None
        existing = self.load(spec.run_id)
        if existing is not None:
            return existing
        manifest = RunManifest(
            run_id=spec.run_id,
            runner=self.kind,
            spec_digest=spec.digest(),
            project_digest=spec.project_digest,
            identity=spec.identity,
            budget=None if spec.budget is None else spec.budget.max_cost,
            started_at=stamp(),
        )
        self.save(manifest)
        return manifest

    def run(self, spec: JobSpec) -> RunManifest:
        """Runs the spec to a final state, resuming and retrying as declared.

        Calling it again with the same spec after the run finished returns
        the recorded manifest and starts nothing; after an interruption of
        the caller itself it continues the numbering and resumes from the
        run's checkpoints.

        Args:
            spec: The job.

        Returns:
            The final manifest.

        Raises:
            RunConflictError: If the id belongs to a different spec.
            ConfigError: If the job would run as the serving identity.
        """
        manifest = self.register(spec)
        if manifest.state in {s.value for s in FINAL}:
            return manifest
        meter = BudgetMeter(spec, manifest.total_cost)
        policy = spec.retries
        while len(manifest.attempts) < policy.max_attempts:
            number = len(manifest.attempts) + 1
            ended = self.attempt(spec, manifest, meter, number)
            if ended is not None:
                return self.finish(spec, manifest, ended)
            if len(manifest.attempts) < policy.max_attempts:
                self.clock.sleep(policy.delay_after(number))
        return self.finish(spec, manifest, RunState.RETRIES_EXHAUSTED)

    def attempt(
        self,
        spec: JobSpec,
        manifest: RunManifest,
        meter: BudgetMeter,
        number: int,
    ) -> RunState | None:
        """Runs one attempt; returns a final state, or None to retry."""
        started = self.clock.now()
        self.stop_reason = None

        def watch() -> RunState | None:
            elapsed = self.clock.now() - started
            if meter.exceeded(elapsed):
                self.stop_reason = RunState.BUDGET_STOPPED
            elif elapsed > spec.timeout_seconds:
                self.stop_reason = RunState.TIMED_OUT
            return self.stop_reason

        self.watcher = watch
        handle = self.submit(spec, number)
        record = Attempt(number, handle.job_id, JobState.RUNNING.value, stamp())
        manifest.attempts.append(record)
        self.save(manifest)
        status = JobStatus(JobState.PENDING)
        stopped: RunState | None = None
        while True:
            status = self.poll(handle)
            if status.state in TERMINAL:
                stopped = self.stop_reason
                break
            stopped = watch()
            if stopped is not None:
                self.cancel(handle, stopped.value)
                status = self.confirm_stopped(handle)
                break
            self.clock.sleep(self.poll_seconds)
        elapsed = self.clock.now() - started
        record.seconds = round(elapsed, 3)
        record.ended_at = stamp()
        record.state = status.state.value
        record.reason = stopped.value if stopped else status.reason
        before = meter.spent
        manifest.total_cost = round(meter.close(elapsed), 6)
        record.cost = round(manifest.total_cost - before, 6)
        self.save(manifest)
        if stopped is not None:
            return stopped
        if status.state is JobState.SUCCEEDED:
            return RunState.SUCCEEDED
        if status.state is JobState.INTERRUPTED or (
            status.state is JobState.FAILED and status.retryable
        ):
            return None
        return RunState.FAILED

    def confirm_stopped(self, handle: JobHandle) -> JobStatus:
        """Polls a cancelled attempt a few times for its terminal state."""
        status = self.poll(handle)
        for ignored in range(5):
            if status.state in TERMINAL:
                return status
            self.clock.sleep(self.poll_seconds)
            status = self.poll(handle)
        return JobStatus(JobState.CANCELLED, "cancel requested")

    def finish(
        self, spec: JobSpec, manifest: RunManifest, state: RunState
    ) -> RunManifest:
        """Seals the manifest with checkpoints, the result and the state."""
        checkpoints = checkpoint_module.CheckpointStore(self.store, spec.run_id)
        manifest.checkpoints = [
            {"step": step, "sha256": found.sha256}
            for step in checkpoints.steps()
            if (found := checkpoints.load(step)) is not None
        ]
        if state is RunState.SUCCEEDED:
            try:
                ignored, meta = self.store.get(self.result_key(spec.run_id))
            except objectstore.NotFoundError:
                state = RunState.FAILED
                manifest.attempts[-1].reason = "job ended without a result"
            else:
                manifest.result_sha256 = meta.sha256
        manifest.evidence = [
            {"key": meta.key, "sha256": meta.sha256, "size": meta.size}
            for meta in self.store.list_objects(f"runs/{spec.run_id}/")
            if meta.key != self.manifest_key(spec.run_id)
        ]
        manifest.state = state.value
        manifest.ended_at = stamp()
        self.save(manifest)
        return manifest
