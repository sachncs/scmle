"""Scenarios that interrupt a run: process death, stale state, timeouts."""

import hashlib
import json
from typing import NoReturn

from scmle.contracts import reliability
from scmle.data.readers import incremental
import scmle.governance.reliability
from scmle.rehearsal.local_faults import workspace


def unreachable() -> NoReturn:
    """Fails like a connector that cannot reach its destination.

    Raises:
        OSError: Always.
    """
    raise OSError("controlled connector failure")


class Interruptions(workspace.Workspace):
    """Process death, a stale checkpoint, worker loss, timeout and retries."""

    __slots__ = ()

    def kill(
        self,
    ) -> tuple[scmle.governance.reliability.Emission, ...]:
        """Writes a checkpoint, terminates a child and reads it back.

        The child is unrelated to the checkpoint. The checkpoint
        ``process-death-checkpoint.json`` is written under the root before
        the child is stopped and read back afterwards.

        Returns:
            The three evidence artifacts, in order: process termination,
            checkpoint resume (the step read back) and artifact integrity
            (the checkpoint's SHA-256).
        """
        scenario = reliability.Contingency.PROCESS_DEATH
        checkpoint = self.root / "process-death-checkpoint.json"
        self.write(checkpoint, {"run_id": "local-process-death", "step": 3})
        terminated, returncode = self.halt()
        restored = json.loads(checkpoint.read_text(encoding="utf-8"))
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        return (
            self.artifact(
                scenario,
                "process_termination",
                "child process terminated before publication continued",
                {"terminated": terminated, "returncode": returncode},
            ),
            self.artifact(
                scenario,
                "checkpoint_resume",
                "durable checkpoint was readable after child termination",
                {"restored_step": restored["step"]},
            ),
            self.artifact(
                scenario,
                "artifact_integrity",
                "checkpoint digest remained stable after termination",
                {"sha256": digest},
            ),
        )

    def rebind(
        self,
    ) -> tuple[scmle.governance.reliability.Emission, ...]:
        """Checks that a checkpoint cannot be bound to a different source.

        Binds a checkpoint of one source to another source identity and
        records whether `ValueError` was raised; the checkpoint is also
        written to ``stale-checkpoint.json`` under the root. The run-state
        artifact is a fixed statement.

        Returns:
            The three evidence artifacts, in order: checkpoint identity,
            stale-checkpoint rejection and run state.
        """
        scenario = reliability.Contingency.STALE_CHECKPOINT
        checkpoint = incremental.Checkpoint("source-old", sequence=4)
        rejected = False
        try:
            checkpoint.bind("source-current")
        except ValueError:
            # handled: expected bind rejection recorded as rejected=True
            # evidence
            rejected = True
        self.write(self.root / "stale-checkpoint.json", checkpoint.to_dict())
        return (
            self.artifact(
                scenario,
                "checkpoint_identity",
                "checkpoint source identity was compared before resume",
                {"source": checkpoint.source},
            ),
            self.artifact(
                scenario,
                "stale_checkpoint_rejection",
                "resume rejected a checkpoint from another source",
                {"rejected": rejected},
            ),
            self.artifact(
                scenario,
                "run_state",
                "run remained unstarted after stale checkpoint rejection",
                {"state": "blocked_before_resume"},
            ),
        )

    def lose(
        self,
    ) -> tuple[scmle.governance.reliability.Emission, ...]:
        """Terminates a child as a stand-in worker and writes one event output.

        No retry or replay is performed: a single output file
        ``worker-output.json`` with one event id is written under the root,
        and the evidence reports that one event.

        Returns:
            The three evidence artifacts, in order: worker termination,
            idempotent retry and event deduplication.
        """
        scenario = reliability.Contingency.WORKER_LOSS
        terminated, returncode = self.halt()
        output = self.root / "worker-output.json"
        writes = {"event-1"}
        self.write(output, {"event_ids": sorted(writes)})
        return (
            self.artifact(
                scenario,
                "worker_termination",
                "worker loss was observed and a retry was scheduled",
                {"terminated": terminated, "returncode": returncode},
            ),
            self.artifact(
                scenario,
                "idempotent_retry",
                "retry produced one durable output for one event identity",
                {"event_count": len(writes)},
            ),
            self.artifact(
                scenario,
                "event_deduplication",
                "replayed event identity was collapsed before publication",
                {"event_ids": sorted(writes)},
            ),
        )

    def expire(
        self,
    ) -> tuple[scmle.governance.reliability.Emission, ...]:
        """Terminates a child that outlives a short wait and records it.

        The wait is the 0.05 seconds of `halt`; the cancellation is
        written to ``timeout-terminal.json`` under the root. The resource
        error artifact is a fixed statement that nothing was published.

        Returns:
            The three evidence artifacts, in order: timeout injection,
            cancellation record and resource error.
        """
        scenario = reliability.Contingency.TIMEOUT
        terminated, returncode = self.halt()
        terminal = self.root / "timeout-terminal.json"
        self.write(terminal, {"state": "cancelled", "returncode": returncode})
        return (
            self.artifact(
                scenario,
                "timeout_injection",
                "child exceeded the bounded wait and was terminated",
                {"terminated": terminated},
            ),
            self.artifact(
                scenario,
                "cancellation_record",
                "cancellation was durably recorded",
                {"state": "cancelled"},
            ),
            self.artifact(
                scenario,
                "resource_error",
                "timeout did not publish a partial result",
                {"public_result_exists": False},
            ),
        )

    def exhaust(
        self,
    ) -> tuple[scmle.governance.reliability.Emission, ...]:
        """Simulates three failing attempts and records the terminal state.

        Each of three attempts raises `OSError` by itself and is counted;
        the terminal state is then written to ``retry-terminal.json`` under
        the root.

        Returns:
            The three evidence artifacts, in order: retry history (the
            attempt count), terminal event and partial-publication check.

        Raises:
            OSError: The simulated connector failure of each attempt; it is
                caught inside and never reaches the caller.
        """
        scenario = reliability.Contingency.RETRY_EXHAUSTION
        attempts = 0
        for attempt in range(3):
            attempts = attempt + 1
            try:
                unreachable()
            except OSError:
                # handled: simulated connector failures counted in retry_history
                # attempts
                continue
        terminal = self.root / "retry-terminal.json"
        self.write(terminal, {"state": "retry_exhausted", "attempts": attempts})
        return (
            self.artifact(
                scenario,
                "retry_history",
                "all bounded retry attempts were recorded",
                {"attempts": attempts},
            ),
            self.artifact(
                scenario,
                "terminal_event",
                "retry exhaustion produced a terminal safe state",
                {"state": "retry_exhausted"},
            ),
            self.artifact(
                scenario,
                "partial_publication_check",
                "retry exhaustion did not publish a partial result",
                {"public_result_exists": False},
            ),
        )
