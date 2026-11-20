"""The durable record of a run and of each of its attempts."""

from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import ClassVar

from scmle.contracts import records
from scmle.runners import checkpoint
from scmle.runners.base import job
from scmle.storage import objectstore

MANIFEST_FORMAT = 2  # 1: multi-word keys; 2: the vocabulary of phase 5


@dataclasses.dataclass(slots=True)
class Attempt:
    """The record of one attempt in the manifest.

    Attributes:
        attempt: Attempt number, from 1.
        job_id: The platform's id of the attempt.
        state: Final (or current) `State` value.
        started_at: UTC start time.
        ended_at: UTC end time; empty while running.
        seconds: Elapsed seconds.
        cost: Estimated cost of this attempt.
        reason: Why the attempt ended as it did.
    """

    # The public path: pinned public signatures name this class, and the
    # API snapshot spells it with its module.
    __module__ = "scmle.runners.base"

    attempt: int
    job_id: str
    state: str
    started_at: str
    ended_at: str = ""
    seconds: float = 0.0
    cost: float = 0.0
    reason: str = ""


@dataclasses.dataclass(slots=True)
class Manifest:
    """The durable record of a run.

    Attributes:
        identifier: The run.
        runner: Kind of the runner that executed it.
        digests: `spec` (the job spec), `project` (the project specification,
            the anchored identity of `Job.project`) and `result` (the result
            object; empty if none).
        identity: The training identity.
        state: `Progress` value.
        attempts: One record per attempt.
        cost: Estimated cost across attempts.
        budget: The cost ceiling, or None if there was none.
        checkpoints: Step and SHA-256 of each verified checkpoint at the end;
            a corrupt one has `sha256` None and its `damage` reason.
        evidence: Key, SHA-256 and size of every other object stored under the
            run at the end.
        started: UTC time the run was registered.
        ended: UTC time the run ended; empty while running.
        migrations: The manifest's format steps, oldest first; the format
            is stored under `format` (its attempts keep their keys).
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            {
                "run_id": "identifier",
                "total_cost": "cost",
                "started_at": "started",
                "ended_at": "ended",
            },
            groups={
                "digests": {
                    "spec_digest": "spec",
                    "project_digest": "project",
                    "result_sha256": "result",
                }
            },
        ),
    )

    identifier: str
    runner: str
    digests: dict[str, str]
    identity: str
    state: str = job.Progress.STARTED.value
    attempts: list[Attempt] = dataclasses.field(default_factory=list)
    cost: float = 0.0
    budget: float | None = None
    checkpoints: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    evidence: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    started: str = ""
    ended: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready form.

        Returns:
            A dict of all fields plus the manifest format number.
        """
        record = dataclasses.asdict(self)
        record["format"] = MANIFEST_FORMAT
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Manifest":
        """Rebuilds a manifest from its stored form.

        Args:
            record: A dict produced by `to_dict`; format 1 is migrated.

        Returns:
            The manifest.

        Raises:
            InputError: If the format is not a positive integer or is newer
                than this build.
        """
        fields = records.load("run manifest", cls.migrations, record, "format")
        del fields[records.KEY]
        fields["attempts"] = [Attempt(**a) for a in fields["attempts"]]
        return cls(**fields)

    @classmethod
    def begin(cls, spec: job.Job, runner: str) -> "Manifest":
        """Starts the manifest of a new run.

        Args:
            spec: The job.
            runner: Kind of the runner that executes it.

        Returns:
            A manifest with no attempts, started now.
        """
        return cls(
            identifier=spec.identifier,
            runner=runner,
            digests={
                "spec": spec.digest(),
                "project": spec.fingerprint,
                "result": "",
            },
            identity=spec.identity,
            budget=None if spec.budget is None else spec.budget.max_cost,
            started=job.stamp(),
        )

    def seal(
        self,
        store: objectstore.Depot,
        state: job.Progress,
        *,
        result: str,
        own: str,
    ) -> None:
        """Records the checkpoints, the result, the evidence and the state.

        A run that succeeded but left no result is recorded as failed.

        Args:
            store: The run's durable storage.
            state: The final state of the run.
            result: The key where the workload leaves its result.
            own: The key of this manifest, left out of the evidence.
        """
        checkpoints = checkpoint.Store(store, self.identifier)
        self.checkpoints = []
        for step in checkpoints.steps():
            reading = checkpoints.load(step)
            if reading.checkpoint is not None:
                self.checkpoints.append(
                    {"step": step, "sha256": reading.checkpoint.sha256}
                )
            elif reading.condition is checkpoint.Condition.CORRUPT:
                self.checkpoints.append(
                    {"step": step, "sha256": None, "damage": reading.reason}
                )
        if state is job.Progress.SUCCEEDED:
            try:
                meta = store.get(result)[1]
            except objectstore.AbsentError:
                # handled: run marked FAILED with reason
                state = job.Progress.FAILED
                self.attempts[-1].reason = "job ended without a result"
            else:
                self.digests["result"] = meta.sha256
        self.evidence = [
            {"key": meta.key, "sha256": meta.sha256, "size": meta.size}
            for meta in store.catalog(f"runs/{self.identifier}/")
            if meta.key != own
        ]
        self.state = state.value
        self.ended = job.stamp()
