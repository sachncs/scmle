r"""Runners on this machine: inline, as a child process, or in a container.

`Runner` runs the workload, a callable that receives a `Context`, inline in
this process. The workload saves progress with `context.step(n, state)`, and
on a restart asks `context.resume()` for the last good checkpoint. `step` is
also where this runner checks the cost budget and the timeout, and where an
injected interruption (a stand-in for a preemption) strikes, right after the
checkpoint is saved.

`Process` starts each attempt as ``python -m scmle.runners.entrypoint`` in a
child process, and `Container` starts the same entrypoint in a container
image through the Docker CLI (a local daemon, or a remote one named by
``DOCKER_HOST``). Both hand the job the store by URI, read the attempt's end
from its exit status (`base.Exit`), stop it with SIGTERM and then SIGKILL
after a grace period, and keep its output at ``runs/<id>/logs/<n>.txt``.

    python -m scmle.runners.local --store store --spec job.json \
        --serving svc --runner container --image scmle:1.0 \
        --put data/train.csv=train.csv
"""

import argparse
from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
import json
import logging
import os
import pathlib
import subprocess  # nosec B404 -- runs the job's own entrypoint and the Docker CLI, never a shell
import sys
import tempfile
from typing import Any
from typing import IO

from scmle.contracts import errors
from scmle.runners import base
from scmle.runners import checkpoint
from scmle.runners import roles
from scmle.runtime import retry
from scmle.storage import objectstore
from scmle.storage import tenancy

LOGGER = logging.getLogger(__name__)

# Seconds between SIGTERM and SIGKILL when a job is stopped: the default of
# `docker stop`, long enough for a workload to finish writing the checkpoint
# it is on (a checkpoint is one atomic object write).
GRACE = 10.0
# Characters of a job's last output line kept as the attempt's reason, the
# same cut the inline runner applies to an exception's text.
TAIL = 300
ENTRYPOINT = ("-m", "scmle.runners.entrypoint")
DEFAULT_WORKLOAD = "scmle.runners.project:run_project"


class PreemptedError(errors.ScmleError):
    """The platform took the job away (preemption); retry from a checkpoint."""


class StoppedError(errors.ScmleError):
    """The supervisor stopped the job (budget or timeout)."""


@dataclasses.dataclass(slots=True)
class Context:
    """What a workload sees.

    Attributes:
        store: Durable storage, limited to the training role.
        run_id: The run.
        attempt: Attempt number, from 1.
        spec: The job spec.
        checkpoints: This run's checkpoints.
        hook: Called by `step` after saving (interruption, budget).
    """

    store: roles.RoleStore
    run_id: str
    attempt: int
    spec: base.Job
    checkpoints: checkpoint.CheckpointStore
    hook: Callable[[int], None] = lambda step: None  # noqa: ARG005 -- hook signature

    def resume(self) -> checkpoint.Checkpoint | None:
        """Finds where to resume.

        Corrupt checkpoints passed over on the way are recorded in durable
        storage (`CheckpointStore.record`), so a restart from an older step
        or from zero is on file (ERR-10).

        Returns:
            The newest checkpoint that verifies, or None at a fresh start.
        """
        found = self.checkpoints.latest()
        self.checkpoints.record(self.attempt, found)
        return found

    def step(self, number: int, state: Any) -> None:
        """Saves a checkpoint, then lets the supervisor act.

        The hook may then raise `PreemptedError` (the platform interrupts
        after this step) or `StoppedError` (the budget or timeout stops the
        job).

        Args:
            number: Step number.
            state: JSON-serializable state to save.
        """
        self.checkpoints.save(number, state)
        self.hook(number)

    def finish(self, result: Mapping[str, Any]) -> None:
        """Records the job's result (metrics, trial results, evidence).

        Args:
            result: JSON-serializable mapping, stored as the run's result.
        """
        body = json.dumps(dict(result), indent=2, sort_keys=True).encode()
        self.store.put(f"runs/{self.run_id}/result.json", body)


Workload = Callable[[Context], None]
Interruption = Callable[[int, int], bool]


@dataclasses.dataclass(slots=True)
class Outcome:
    """A finished local attempt.

    Attributes:
        status: How the attempt ended.
    """

    status: base.Status


class Runner(base.Runner):
    """Runs the workload inline, as the job.

    Attributes:
        workload: The training function.
        interruptions: Decides which steps are interrupted, or None.
        jobs: Outcome of each submitted attempt, by job id.
    """

    kind = "local"

    def __init__(
        self,
        store: Any,
        workload: Workload,
        *,
        serving_identity: str,
        interruptions: Interruption | None = None,
        **options: Any,
    ) -> None:
        """Initializes the runner.

        Args:
            store: Durable storage (any `Depot`).
            workload: The training function.
            serving_identity: The serving identity (see `Runner`).
            interruptions: `(attempt, step) -> bool`; true makes the job
                interrupted right after that step's checkpoint.
            **options: `clock` and `poll_seconds` (see `Runner`).
        """
        super().__init__(store, serving_identity=serving_identity, **options)
        self.workload = workload
        self.interruptions = interruptions
        self.jobs: dict[str, Outcome] = {}

    def submit(self, spec: base.Job, attempt: int) -> base.Ticket:
        """Runs the workload to completion and remembers how it ended.

        An interruption becomes `INTERRUPTED`, a supervisor stop `CANCELLED`, a
        transient error a retryable `FAILED`, and any other exception a
        non-retryable `FAILED` carrying its type and message (cut to 300
        characters).

        Args:
            spec: The job.
            attempt: Attempt number, from 1.

        Returns:
            The handle of the finished attempt.
        """
        handle = base.Ticket(f"local-{spec.run_id}-{attempt}", attempt)

        def hook(step: int) -> None:
            """Lets the supervisor act after a checkpoint.

            Args:
                step: The step just saved.

            Raises:
                StoppedError: If the supervisor wants the job stopped.
                PreemptedError: If the injected interruption strikes at
                    this attempt and step.
            """
            if self.watcher() is not None:
                raise StoppedError
            if self.interruptions and self.interruptions(attempt, step):
                raise PreemptedError

        context = Context(
            roles.RoleStore(self.store, roles.TRAINING_ROLE),
            spec.run_id,
            attempt,
            spec,
            checkpoint.CheckpointStore(self.store, spec.run_id),
            hook,
        )
        try:
            self.workload(context)
            status = base.Status(base.State.SUCCEEDED)
        except PreemptedError:
            # handled: interruption recorded as INTERRUPTED job status
            status = base.Status(base.State.INTERRUPTED, "interrupted")
        except StoppedError:
            # handled: supervisor stop recorded as CANCELLED job status
            status = base.Status(base.State.CANCELLED, "stopped")
        except retry.TransientError as error:
            # handled: transient error recorded as retryable FAILED status
            status = base.Status(
                base.State.FAILED, f"transient: {error}", retryable=True
            )
        except Exception as error:
            LOGGER.exception("job %s failed", handle.job_id)
            reason = errors.Failure.of(error).text[:300]
            status = base.Status(base.State.FAILED, reason)
        self.jobs[handle.job_id] = Outcome(status)
        return handle

    def poll(self, handle: base.Ticket) -> base.Status:
        """Reports the remembered outcome.

        Args:
            handle: The attempt.

        Returns:
            The status the attempt ended with.
        """
        return self.jobs[handle.job_id].status

    def cancel(self, handle: base.Ticket, reason: str) -> None:
        """Does nothing: the attempt already ended.

        Args:
            handle: The attempt. Unused.
            reason: Why it is stopped. Unused.
        """


@dataclasses.dataclass(slots=True)
class Child:
    """A started job process.

    Attributes:
        process: The child process (the entrypoint, or the Docker client).
        log: Temporary file that receives its stdout and stderr.
        run: The run the attempt belongs to.
        cancelled: Whether the supervisor stopped it.
        status: Its terminal status, once read.
    """

    process: "subprocess.Popen[bytes]"
    log: IO[bytes]
    run: str
    cancelled: bool = False
    status: base.Status | None = None


class Process(base.Runner):
    """Runs each attempt as a child process of the entrypoint.

    Attributes:
        root: The shared store's directory on this machine (a tenant's store
            lies under it; the job confines itself by `SCMLE_TENANT`).
        workload: The workload the entrypoint runs, as `module:function`.
        grace: Seconds between SIGTERM and SIGKILL on a stop.
        directory: Working directory of the job process; None keeps ours.
        jobs: The started attempts, by job id.
    """

    kind = "process"

    def __init__(
        self,
        store: objectstore.Depot,
        *,
        serving_identity: str,
        workload: str = DEFAULT_WORKLOAD,
        grace: float = GRACE,
        directory: str | pathlib.Path | None = None,
        **options: Any,
    ) -> None:
        """Initializes the runner.

        Args:
            store: Durable storage; a `LocalObjectStore`, or a tenant's
                `Tenancy` over one, so the job can open it by its `file://`
                URI and the tenant (`environment`).
            serving_identity: The serving identity (see `base.Runner`).
            workload: The workload as `module:function`, importable in the
                job's environment.
            grace: Seconds between SIGTERM and SIGKILL on a stop.
            directory: Working directory of the job process.
            **options: `clock` and `poll_seconds` (see `base.Runner`).

        Raises:
            ConfigError: If the store is not a local object store, or a
                tenancy over one, the workload is not `module:function` or the
                grace is negative.
        """
        inner = store.inner if isinstance(store, tenancy.Tenancy) else store
        if not isinstance(inner, objectstore.LocalObjectStore):
            raise errors.ConfigError(
                f"the {self.kind} runner needs a LocalObjectStore the job "
                f"can open by URI, not {type(inner).__name__}"
            )
        if ":" not in workload or grace < 0:
            raise errors.ConfigError(
                "workload must be module:function and grace not negative"
            )
        super().__init__(store, serving_identity=serving_identity, **options)
        self.root = inner.root
        self.workload = workload
        self.grace = grace
        self.directory = directory
        self.jobs: dict[str, Child] = {}

    def uri(self) -> str:
        """Names the store as the job opens it.

        Returns:
            The `file://` URI of the store.
        """
        return self.root.as_uri()

    def environment(self, spec: base.Job, attempt: int) -> dict[str, str]:
        """Builds the variables the entrypoint reads, after the spec's own.

        Args:
            spec: The job.
            attempt: Attempt number, from 1.

        Returns:
            The job's environment. `SCMLE_TENANT` names the runner's tenant,
            or is empty when its store is not confined, whatever the spec
            sets, so a job opens the namespace its runner was given.
        """
        tenant = (
            self.store.tenant if isinstance(self.store, tenancy.Tenancy) else ""
        )
        return {
            **spec.environment,
            tenancy.VARIABLE: tenant,
            "SCMLE_RUN_ID": spec.run_id,
            "SCMLE_ATTEMPT": str(attempt),
            "SCMLE_STORE_URI": self.uri(),
            "SCMLE_SPEC_DIGEST": spec.digest(),
            "SCMLE_WORKLOAD": self.workload,
        }

    def command(self, spec: base.Job, attempt: int) -> list[str]:  # noqa: ARG002 -- the container runner uses them
        """Builds the command that runs one attempt.

        Args:
            spec: The job.
            attempt: Attempt number, from 1.

        Returns:
            The argument vector.
        """
        return [sys.executable, *ENTRYPOINT]

    def label(self, spec: base.Job, attempt: int) -> str:
        """Names an attempt (its job id).

        Args:
            spec: The job.
            attempt: Attempt number, from 1.

        Returns:
            `scmle-<kind>-<run_id>-<attempt>`, also the container name.
        """
        return f"scmle-{self.kind}-{spec.run_id}-{attempt}"

    def prepare(self, name: str) -> None:
        """Clears what an earlier supervisor left under `name` (nothing here).

        Args:
            name: The attempt's job id.
        """

    def submit(self, spec: base.Job, attempt: int) -> base.Ticket:
        """Starts the attempt's process.

        Args:
            spec: The job.
            attempt: Attempt number, from 1.

        Returns:
            The handle of the started attempt.

        Raises:
            ConfigError: If the command cannot be started (for example, the
                Docker CLI is not installed).
        """
        name = self.label(spec, attempt)
        self.prepare(name)
        command = self.command(spec, attempt)
        log = tempfile.TemporaryFile()  # noqa: SIM115 -- closed when the attempt is settled
        try:
            process = subprocess.Popen(  # noqa: S603 -- a fixed argument vector, no shell  # nosec B603
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, **self.environment(spec, attempt)},
                cwd=self.directory,
            )
        except OSError as error:
            log.close()
            raise errors.ConfigError(
                f"cannot start {command[0]}: {error}"
            ) from error
        self.jobs[name] = Child(process, log, spec.run_id)
        return base.Ticket(name, attempt)

    def poll(self, handle: base.Ticket) -> base.Status:
        """Reports the attempt's state from its process.

        Args:
            handle: The attempt.

        Returns:
            `RUNNING` while the process runs, then its terminal status.
        """
        child = self.jobs[handle.job_id]
        if child.status is None:
            code = child.process.poll()
            if code is None:
                return base.Status(base.State.RUNNING)
            child.status = self.settle(handle, child, code)
        return child.status

    def settle(
        self, handle: base.Ticket, child: Child, code: int
    ) -> base.Status:
        """Reads how an ended attempt went and keeps its output.

        Args:
            handle: The attempt.
            child: Its process.
            code: Its exit status.

        Returns:
            `CANCELLED` if the supervisor stopped it, otherwise what the exit
            status means (`base.Exit.status`).
        """
        child.log.seek(0)
        output = child.log.read()
        child.log.close()
        key = f"runs/{child.run}/logs/{handle.attempt}.txt"
        self.store.put(key, output)
        if child.cancelled:
            return base.Status(base.State.CANCELLED, "stopped")
        lines = output.decode("utf-8", "replace").strip().splitlines()
        return base.Exit.status(code, lines[-1][-TAIL:] if lines else "")

    def cancel(self, handle: base.Ticket, reason: str) -> None:
        """Stops the attempt: SIGTERM, then SIGKILL after the grace period.

        Args:
            handle: The attempt.
            reason: Why it is stopped.
        """
        child = self.jobs[handle.job_id]
        if child.process.poll() is not None:
            return
        LOGGER.info("stopping %s: %s", handle.job_id, reason)
        child.cancelled = True
        self.stop(handle.job_id, child)

    def stop(self, name: str, child: Child) -> None:  # noqa: ARG002 -- the container runner uses the name
        """Ends a running attempt.

        Args:
            name: The attempt's job id.
            child: Its process.
        """
        child.process.terminate()
        try:
            child.process.wait(self.grace)
        except subprocess.TimeoutExpired:
            # handled: past the grace period the process is killed
            child.process.kill()
            child.process.wait()


class Container(Process):
    """Runs each attempt in a container image, through the Docker CLI.

    The store's directory is bind-mounted at `MOUNT`; the root file system is
    read-only, with a `/tmp` in memory, every capability dropped and the
    job's resources applied as limits. The container keeps its name until its
    end is read, so an out-of-memory kill is told apart from a preemption,
    and is then removed.

    Attributes:
        image: The image reference, ideally pinned by digest.
        client: The Docker CLI.
        user: `uid:gid` the job runs as.
    """

    kind = "container"
    MOUNT = "/store"

    def __init__(
        self,
        store: objectstore.Depot,
        *,
        serving_identity: str,
        image: str,
        client: str = "docker",
        user: str | None = None,
        **options: Any,
    ) -> None:
        """Initializes the runner.

        Args:
            store: Durable storage, a `LocalObjectStore` on a path the Docker
                daemon can mount.
            serving_identity: The serving identity (see `base.Runner`).
            image: The image that holds scmle.
            client: The Docker CLI.
            user: `uid:gid` for the job; defaults to ours, so the files it
                writes are ours, or to the image's 10001 when we are root.
            **options: `workload`, `grace`, `clock` and `poll_seconds` (see
                `Process`).

        Raises:
            ConfigError: If `image` is empty.
        """
        if not image.strip():
            raise errors.ConfigError("the container runner needs an image")
        super().__init__(store, serving_identity=serving_identity, **options)
        self.image = image
        self.client = client
        owner = getattr(os, "getuid", lambda: 0)()
        group = getattr(os, "getgid", lambda: 0)()
        self.user = user or (f"{owner}:{group}" if owner else "10001:10001")

    def uri(self) -> str:
        """Names the store as the job opens it, inside the container.

        Returns:
            The `file://` URI of the mount.
        """
        return f"file://{self.MOUNT}"

    def command(self, spec: base.Job, attempt: int) -> list[str]:
        """Builds the `docker run` command of one attempt.

        Values travel in the client's environment (`--env NAME`), not on the
        command line; the job's home is the in-memory `/tmp`.

        Args:
            spec: The job.
            attempt: Attempt number, from 1.

        Returns:
            The argument vector.
        """
        resources = spec.resources
        names = sorted(self.environment(spec, attempt))
        gpus = ["--gpus", str(resources.gpu)] if resources.gpu else []
        return [
            self.client,
            "run",
            "--name",
            self.label(spec, attempt),
            "--init",
            "--read-only",
            "--tmpfs",
            "/tmp",  # noqa: S108 -- the container's in-memory /tmp  # nosec B108
            "--workdir",
            "/tmp",  # noqa: S108 -- the container's in-memory /tmp  # nosec B108
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            self.user,
            "--cpus",
            f"{resources.cpu:g}",
            "--memory",
            f"{resources.memory_mb}m",
            *gpus,
            "--mount",
            f"type=bind,source={self.root},target={self.MOUNT}",
            "--env",
            "HOME=/tmp",
            *[part for name in names for part in ("--env", name)],
            self.image,
            "python",
            *ENTRYPOINT,
        ]

    def call(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        """Runs a Docker CLI command and returns its result.

        Args:
            *arguments: The command after `docker`.

        Returns:
            The finished command, its output captured as text.

        Raises:
            ConfigError: If the Docker CLI cannot be started.
        """
        try:
            return subprocess.run(  # noqa: S603 -- a fixed argument vector, no shell  # nosec B603
                [self.client, *arguments],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as error:
            raise errors.ConfigError(
                f"cannot start {self.client}: {error}"
            ) from error

    def prepare(self, name: str) -> None:
        """Removes a container an earlier supervisor left under `name`.

        Args:
            name: The attempt's job id.
        """
        self.call("rm", "--force", name)

    def settle(
        self, handle: base.Ticket, child: Child, code: int
    ) -> base.Status:
        """Reads the end, tells an out-of-memory kill apart, then removes it.

        Args:
            handle: The attempt.
            child: Its Docker client.
            code: The container's exit status.

        Returns:
            A final `FAILED` for an out-of-memory kill; otherwise the status
            `Process.settle` reads.
        """
        found = self.call(
            "inspect", "--format", "{{.State.OOMKilled}}", handle.job_id
        )
        self.call("rm", "--force", handle.job_id)
        status = super().settle(handle, child, code)
        if found.stdout.strip() == "true" and not child.cancelled:
            return base.Status(
                base.State.FAILED, f"out of memory (exit {code})"
            )
        return status

    def stop(self, name: str, child: Child) -> None:
        """Stops the container: `docker stop` with the grace period.

        Args:
            name: The attempt's job id (the container's name).
            child: Its Docker client.
        """
        self.call("stop", "--time", f"{int(self.grace)}", name)
        super().stop(name, child)


RUNNERS: dict[str, type[Process]] = {"process": Process, "container": Container}


def main(arguments: list[str] | None = None) -> int:
    """Runs a job spec on this machine (the command line).

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 if the run succeeded, otherwise 1 (the final state or the error goes
        to stderr).
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])  # pyright: ignore[reportOptionalMemberAccess]  # the module docstring is always present
    parser.add_argument("--store", required=True, help="store directory")
    parser.add_argument("--spec", required=True, help="job spec JSON file")
    parser.add_argument("--serving", required=True, help="serving identity")
    parser.add_argument("--runner", choices=sorted(RUNNERS), default="process")
    parser.add_argument("--image", default="", help="for --runner container")
    parser.add_argument("--workload", default=DEFAULT_WORKLOAD)
    parser.add_argument("--poll", type=float, default=1.0, help="seconds")
    parser.add_argument(
        "--put", action="append", default=[], help="KEY=PATH, stored first"
    )
    options = parser.parse_args(arguments)
    try:
        store = tenancy.confine(
            objectstore.LocalObjectStore(options.store), os.environ
        )
        for item in options.put:
            key, separator, path = item.partition("=")
            if not separator:
                raise errors.ConfigError(f"--put needs KEY=PATH, got {item!r}")
            store.put(key, pathlib.Path(path).read_bytes())
        record = json.loads(pathlib.Path(options.spec).read_text("utf-8"))
        extra = (
            {"image": options.image} if options.runner == "container" else {}
        )
        runner = RUNNERS[options.runner](
            store,
            serving_identity=options.serving,
            workload=options.workload,
            poll_seconds=options.poll,
            **extra,
        )
        manifest = runner.run(base.Job.read(record))
    except (errors.ScmleError, OSError, json.JSONDecodeError) as error:
        # handled: a refused or unreadable job is reported, exit status 1
        print(f"error: {error}", file=sys.stderr)  # noqa: T201 -- command-line output
        return 1
    print(f"{manifest.identifier}: {manifest.state}")  # noqa: T201 -- command-line output
    return 0 if manifest.state == base.Progress.SUCCEEDED.value else 1


if __name__ == "__main__":
    sys.exit(main())
