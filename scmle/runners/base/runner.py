"""The `Runner` template: one place to run a job, with shared supervision."""

import abc
import json
from typing import TYPE_CHECKING

from scmle.runners import roles
from scmle.runners.base import job
from scmle.runners.base import ledger
from scmle.storage import objectstore

if TYPE_CHECKING:
    from collections.abc import Callable


class Runner(abc.ABC):
    """Runs a `Job` in one place, with the shared supervision.

    Attributes:
        kind: Name of the kind of runner, recorded in the manifest.
        store: Durable storage.
        serving_identity: The identity a job may not run as.
        clock: Time source.
        poll_seconds: Seconds between status polls.
        stop_reason: Why the supervisor stopped the current attempt, if it did.
        watcher: Returns the stop reason of the current attempt, or None.
    """

    kind = "runner"

    def __init__(
        self,
        store: objectstore.Depot,
        *,
        serving_identity: str,
        clock: job.Clock | None = None,
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
        self.clock = clock or job.Clock()
        self.poll_seconds = poll_seconds
        self.stop_reason: job.Progress | None = None
        self.watcher: Callable[[], job.Progress | None] = lambda: None

    @abc.abstractmethod
    def submit(self, spec: job.Job, attempt: int) -> job.Ticket:
        """Starts one attempt.

        Args:
            spec: The job.
            attempt: Attempt number, from 1.

        Returns:
            The handle of the started attempt.
        """

    @abc.abstractmethod
    def poll(self, handle: job.Ticket) -> job.Status:
        """Reports the attempt's state.

        Args:
            handle: The attempt.

        Returns:
            The state the platform reports.
        """

    @abc.abstractmethod
    def cancel(self, handle: job.Ticket, reason: str) -> None:
        """Asks the platform to stop the attempt.

        Args:
            handle: The attempt.
            reason: Why it is stopped.
        """

    def locate(self, run_id: str) -> str:
        """Builds the key of a run's manifest.

        Args:
            run_id: The run.

        Returns:
            The object key.
        """
        return f"runs/{run_id}/manifest.json"

    def spec_key(self, run_id: str) -> str:
        """Builds the key of a run's spec.

        Args:
            run_id: The run.

        Returns:
            The object key.
        """
        return f"runs/{run_id}/spec.json"

    def result_key(self, run_id: str) -> str:
        """Builds the key where a workload leaves its result.

        Args:
            run_id: The run.

        Returns:
            The object key.
        """
        return f"runs/{run_id}/result.json"

    def save(self, manifest: ledger.RunManifest) -> None:
        """Writes the manifest atomically.

        Args:
            manifest: The manifest to store.
        """
        body = json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
        self.store.put(self.locate(manifest.identifier), body.encode())

    def load(self, run_id: str) -> ledger.RunManifest | None:
        """Reads a run's manifest.

        Args:
            run_id: The run.

        Returns:
            The stored manifest, or None if the run has none.
        """
        try:
            data = self.store.get(self.locate(run_id))[0]
        except objectstore.AbsentError:
            # handled: documented None when run has no manifest
            return None
        return ledger.RunManifest.from_dict(json.loads(data))

    def register(self, spec: job.Job) -> ledger.RunManifest:
        """Stores the spec, or recognises the same run, or refuses a clash.

        The job's identity is first checked against the serving identity, which
        refuses a job that would run as the serving side with a `ConfigError`.

        Args:
            spec: The job.

        Returns:
            The run's stored manifest, or a new one if the run is new.

        Raises:
            ConflictError: If the id exists with a different spec.
        """
        roles.check_distinct(spec.identity, self.serving_identity)
        body = json.dumps(spec.to_dict(), indent=2, sort_keys=True).encode()
        try:
            self.store.put(self.spec_key(spec.run_id), body, if_absent=True)
        except objectstore.ExistsError:
            stored = self.store.get(self.spec_key(spec.run_id))[0]
            if stored != body:
                raise job.ConflictError(
                    f"run {spec.run_id} exists with a different spec"
                ) from None
        existing = self.load(spec.run_id)
        if existing is not None:
            return existing
        manifest = ledger.RunManifest.begin(spec, self.kind)
        self.save(manifest)
        return manifest

    def run(self, spec: job.Job) -> ledger.RunManifest:
        """Runs the spec to a final state, resuming and retrying as declared.

        Calling it again with the same spec after the run finished returns the
        recorded manifest and starts nothing; after an interruption of the
        caller itself it continues the numbering and resumes from the run's
        checkpoints. A spec that clashes with the stored one is refused with
        `ConflictError`, and a job that would run as the serving identity
        with `ConfigError`.

        Args:
            spec: The job.

        Returns:
            The final manifest.
        """
        manifest = self.register(spec)
        if manifest.state in {s.value for s in job.FINAL}:
            return manifest
        meter = job.Meter(spec, manifest.cost)
        policy = spec.retries
        while len(manifest.attempts) < policy.attempts:
            number = len(manifest.attempts) + 1
            ended = self.attempt(spec, manifest, meter, number)
            if ended is not None:
                return self.finish(spec, manifest, ended)
            if len(manifest.attempts) < policy.attempts:
                self.clock.sleep(policy.delay(number))
        return self.finish(spec, manifest, job.Progress.RETRIES_EXHAUSTED)

    def attempt(
        self,
        spec: job.Job,
        manifest: ledger.RunManifest,
        meter: job.Meter,
        number: int,
    ) -> job.Progress | None:
        """Runs one attempt.

        Args:
            spec: The job.
            manifest: The run's manifest, updated in place.
            meter: The run's cost meter, updated in place.
            number: Attempt number, from 1.

        Returns:
            The final state of the run if the attempt ended it (success,
            failure, timeout or budget stop), or None if the attempt was
            interrupted or failed retryably and another attempt may follow.
        """
        started = self.clock.now()
        self.stop_reason = None

        def watch() -> job.Progress | None:
            """Checks the budget and the timeout of the running attempt.

            Returns:
                The reason the attempt must stop (`BUDGET_STOPPED` or
                `TIMED_OUT`), or None if it may continue; the reason is also
                kept as `stop_reason`.
            """
            elapsed = self.clock.now() - started
            self.stop_reason = meter.verdict(elapsed) or self.stop_reason
            return self.stop_reason

        self.watcher = watch
        handle = self.submit(spec, number)
        record = ledger.Attempt(
            number, handle.job_id, job.State.RUNNING.value, job.stamp()
        )
        manifest.attempts.append(record)
        self.save(manifest)
        stopped: job.Progress | None = None
        while True:
            status = self.poll(handle)
            if status.state in job.TERMINAL:
                stopped = self.stop_reason
                break
            stopped = watch()
            if stopped is not None:
                self.cancel(handle, stopped.value)
                status = self.confirm(handle)
                break
            self.clock.sleep(self.poll_seconds)
        elapsed = self.clock.now() - started
        record.seconds = round(elapsed, 3)
        record.ended_at = job.stamp()
        record.state = status.state.value
        record.reason = stopped.value if stopped else status.reason
        before = meter.spent
        manifest.cost = round(meter.close(elapsed), 6)
        record.cost = round(manifest.cost - before, 6)
        self.save(manifest)
        return stopped if stopped is not None else status.outcome()

    def confirm(self, handle: job.Ticket) -> job.Status:
        """Polls a cancelled attempt a few times for its terminal state.

        Args:
            handle: The cancelled attempt.

        Returns:
            The terminal status if it appears within about six polls; otherwise
            a `CANCELLED` status with the reason `cancel requested`.
        """
        status = self.poll(handle)
        remaining = 5
        while remaining > 0:
            remaining -= 1
            if status.state in job.TERMINAL:
                return status
            self.clock.sleep(self.poll_seconds)
            status = self.poll(handle)
        return job.Status(job.State.CANCELLED, "cancel requested")

    def finish(
        self, spec: job.Job, manifest: ledger.RunManifest, state: job.Progress
    ) -> ledger.RunManifest:
        """Seals the manifest with checkpoints, the result and the state.

        A run that succeeded but left no result is recorded as failed.

        Args:
            spec: The job.
            manifest: The run's manifest, updated in place.
            state: The final state of the run.

        Returns:
            The saved manifest.
        """
        manifest.seal(
            self.store,
            state,
            result=self.result_key(spec.run_id),
            own=self.locate(spec.run_id),
        )
        self.save(manifest)
        return manifest
