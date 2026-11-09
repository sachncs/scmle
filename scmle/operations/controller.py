"""The unattended controller: one idempotent tick of the closed loop.

    controller = Controller(deploy_dir, retrain=train_and_stage)
    result = controller.tick(inbox)          # run from cron or a workflow

A tick applies the new files in an inbox to `scmle.governance.loop.Loop`
(``inbox/batches/*.csv`` are served cases, ``inbox/labels/*.csv`` are arrived
losses, with optional paired losses of challengers), steps the loop, and
journals what it did in the deployment ledger. It decides nothing itself:
alarms, challengers, proposals and escalations are the loop's, and promotion,
rollback and the other never-list actions still wait for a person. What the
controller adds is what a person did by typing commands:

* **Exactly once.** Each file is identified by the SHA-256 of its bytes. An
  ``intent`` row (with the loop's batch counter) is chained before the loop is
  called and an ``applied`` row after, so a tick that is killed and run again
  applies the file once: it reads the counter to see whether the loop got
  there, and halts and escalates if the counter fits neither case. A file that
  cannot be read is ``rejected`` once and escalated, never retried.
* **Authority.** The loop calls its retrain hook only when its policy says an
  alarm may gate retraining (`Preregistration.alarm_gates_retrain`, off until
  the alarm design is certified). Even then the hook first spends one use of an
  envelope (`scmle.governance.envelope`); without one it escalates and trains
  nothing. A training that fails is escalated and the loop goes on.
* **Dead-man rules.** Ticks with no new batch, or with matured batches that
  never get enough labels, open one escalation each; they do not stop the loop.
* **A halted loop is left alone.** While the ledger is halted, or its state
  and ledger disagree, a tick changes nothing and reports ``halted``. Only a
  person's recalibration releases a halt.

Limits. Between the ``applied`` row and the loop's own state file there is a
window in which a crash leaves the state behind the ledger; `Loop` then starts
halted instead of guessing, as it does for its own rows. The simulation in
``tests/simulation`` kills the controller between calls, not inside one.
"""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
import datetime
import hashlib
import pathlib
from typing import Any

import pandas

from scmle.governance import deployment_ledger
from scmle.governance import envelope
import scmle.governance.loop

BATCHES = "batches"
LABELS = "labels"
HALTED = "halted"
KIND = {BATCHES: "batch", LABELS: "labels"}
CHALLENGER_PREFIX = "challenger:"
Checkpoint = Callable[[str], None]


@dataclasses.dataclass(frozen=True)
class Limits:
    """When the controller asks a person because nothing is happening.

    Attributes:
        max_idle_ticks: Consecutive ticks without a new batch before the
            escalation ``controller_no_new_data``.
        max_unlabelled_ticks: Consecutive ticks whose loop status was
            ``insufficient_labels`` before ``controller_labels_not_arriving``.
    """

    max_idle_ticks: int = 3
    max_unlabelled_ticks: int = 5

    def __post_init__(self) -> None:
        """Both counts are at least one.

        Raises:
            ValueError: If a tick limit is below one.
        """
        if self.max_idle_ticks < 1 or self.max_unlabelled_ticks < 1:
            raise ValueError("tick limits must be at least 1")


@dataclasses.dataclass(frozen=True)
class Tick:
    """What one tick did.

    Attributes:
        status: ``ok``, ``insufficient_labels`` or ``halted``.
        observed: Batch files applied in this tick.
        matured: Cases whose label matched a pending batch in this tick.
        rejected: Names of the files rejected in this tick (unreadable or
            malformed), sorted.
        events: Ledger events the loop wrote while stepping.
        escalations: Ids of escalations open after the tick.
    """

    status: str
    observed: int
    matured: int
    rejected: tuple[str, ...]
    events: tuple[tuple[str, str], ...]
    escalations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form.

        Returns:
            The status, counts, rejected file names, events and open
            escalations as a dict.
        """
        return {
            "status": self.status,
            "observed": self.observed,
            "matured": self.matured,
            "rejected": list(self.rejected),
            "events": [list(event) for event in self.events],
            "escalations": list(self.escalations),
        }


def file_digest(path: pathlib.Path) -> str:
    """SHA-256 of a file's bytes.

    Args:
        path: The file.

    Returns:
        The hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Controller:
    """Runs ticks of one deployment's loop (see the module docstring)."""

    def __init__(
        self,
        deploy_dir: str | pathlib.Path,
        *,
        retrain: Callable[[], str | None] | None = None,
        verification_keys: Mapping[str, str | bytes] | None = None,
        system_keys: tuple[str | bytes, ...] = (),
        today: Callable[[], datetime.date] = datetime.date.today,
        policy: Limits | None = None,
        checkpoint: Checkpoint | None = None,
    ) -> None:
        """Opens an existing deployment (`scmle loop init` creates one).

        Args:
            deploy_dir: Directory holding the deployment ledger.
            retrain: Trains and stages a challenger and returns its name. It
                runs only with a policy that lets an alarm gate retraining and
                an envelope that allows it.
            verification_keys: Map from key id to the key that verifies a
                person's attestations and envelopes.
            system_keys: Keys the system holds; signatures under them are
                refused.
            today: Callable that returns today's date; replaceable in tests.
            policy: Dead-man limits.
            checkpoint: Called with a label at fixed points while a file is
                applied (after the intent, after the loop took it, after the
                result) and after the loop has stepped; a seam for crash
                tests, absent in production.
        """
        self.retrain = retrain
        self.policy = policy or Limits()
        self.checkpoint = checkpoint
        self.loop = scmle.governance.loop.Loop.resume(
            deploy_dir,
            hooks=scmle.governance.loop.Hooks(retrain=self.authorized_retrain),
            verification_keys=verification_keys,
            system_keys=system_keys,
            today=today,
        )
        self.envelopes = envelope.EnvelopeGuard(
            self.loop.ledger, today, verification_keys
        )
        self.envelopes.on_write = self.loop.save_state

    # ------------------------------------------------------------ authority

    def authorized_retrain(self) -> str | None:
        """The loop's retrain hook: spend an envelope, then train.

        Returns:
            The name returned by the retrain callable. None if no envelope
            authorized the retrain, no callable is configured or the training
            raised (each of these opens an escalation), or if the callable
            itself returned None.
        """
        loop = self.loop
        try:
            used = self.envelopes.authorize(
                envelope.EnvelopeAction.TRAIN_CHALLENGER,
                loop.policy.scope,
                {"batch": loop.batch - 1},
            )
        except envelope.EnvelopeRefused as refused:
            loop.escalate("retrain_not_authorized", {"reason": refused.reason})
            return None
        if self.retrain is None:
            loop.escalate("retrain_not_configured", {"envelope": used})
            return None
        try:
            return self.retrain()
        except Exception as error:  # noqa: BLE001 -- a failed training must not stop the loop
            loop.escalate(
                "retrain_failed",
                {"envelope": used, "error": type(error).__name__},
            )
            return None

    # -------------------------------------------------------------- journal

    def journal(self, body: Mapping[str, Any]) -> None:
        """Chains a controller row and moves the loop's state with it.

        Args:
            body: The row content.
        """
        self.loop.ledger.commit(deployment_ledger.CONTROLLER, dict(body))
        self.loop.save_state()

    def mark(self, label: str) -> None:
        """Reports progress to the crash-test seam.

        Args:
            label: Names the point reached; passed to the checkpoint callback
                if one is set.
        """
        if self.checkpoint is not None:
            self.checkpoint(label)

    def trailing(self, test: Callable[[Mapping[str, Any]], bool]) -> int:
        """Consecutive newest tick rows for which `test(body)` holds.

        Args:
            test: Called with a tick row's body.

        Returns:
            The number of consecutive newest tick rows that satisfy the test;
            other rows are skipped.
        """
        count = 0
        for row in reversed(self.loop.ledger.rows):
            body = row["body"]
            if (
                row["kind"] != deployment_ledger.CONTROLLER
                or body["phase"] != deployment_ledger.TICK
            ):
                continue
            if not test(body):
                break
            count += 1
        return count

    # --------------------------------------------------------------- inputs

    def reject(
        self, what: str, path: pathlib.Path, digest: str, why: str
    ) -> None:
        """Records a file that cannot be applied, once, and asks a person.

        Args:
            what: The kind of input, batches or labels.
            path: The rejected file.
            digest: The file's digest.
            why: Why it cannot be applied.
        """
        self.journal(
            {
                "phase": deployment_ledger.REJECTED,
                "what": KIND[what],
                "digest": digest,
                "file": path.name,
                "reason": why,
            }
        )
        self.loop.escalate(
            "controller_input_rejected", {"file": path.name, "reason": why}
        )

    @staticmethod
    def read(path: pathlib.Path, required: tuple[str, ...]) -> pandas.DataFrame:
        """A CSV with the required columns and at least one row.

        Args:
            path: The CSV file.
            required: Column names that must be present.

        Returns:
            The file's rows.

        Raises:
            ValueError: If a required column is missing or the file has no rows.
                An unreadable or unparseable file raises the error of the CSV
                reader or of the file system instead.
        """
        frame = pandas.read_csv(path)
        for name in required:
            if name not in frame.columns:
                raise ValueError(f"missing column {name!r}")
        if frame.empty:
            raise ValueError("the file has no rows")
        return frame

    def run_batch(self, path: pathlib.Path) -> None:
        """Gives the loop one batch file.

        Args:
            path: A CSV with ``prediction`` and ``decision`` columns.
        """
        frame = self.read(path, ("prediction", "decision"))
        self.loop.observe(
            frame, frame["prediction"].tolist(), frame["decision"].tolist()
        )

    def run_labels(self, path: pathlib.Path) -> int:
        """Gives the loop one labels file; returns the cases it matched.

        Columns: ``case_id`` and ``loss``; optionally ``covered`` and one
        ``challenger:<name>`` column per challenger with its loss on the same
        case (empty where it was not scored).

        Args:
            path: The labels CSV.

        Returns:
            The number of cases the loop matched to earlier batches.
        """
        frame = self.read(path, ("case_id", "loss"))
        identities = frame["case_id"].astype(str)
        covered = (
            dict(zip(identities, frame["covered"], strict=True))
            if "covered" in frame.columns
            else None
        )
        challengers = {}
        for column in frame.columns:
            if column.startswith(CHALLENGER_PREFIX):
                scored = frame[column].notna()
                challengers[column[len(CHALLENGER_PREFIX) :]] = dict(
                    zip(
                        identities[scored],
                        frame.loc[scored, column],
                        strict=True,
                    )
                )
        return self.loop.mature(
            dict(zip(identities, frame["loss"], strict=True)),
            covered,
            challengers or None,
        )

    def run(self, what: str, path: pathlib.Path) -> int:
        """Gives the loop one file of either kind.

        Args:
            what: The kind of input, batches or labels.
            path: The file.

        Returns:
            The number of cases matched for a labels file; 0 for a batch file.
        """
        if what == BATCHES:
            self.run_batch(path)
            return 0
        return self.run_labels(path)

    def apply(self, what: str, path: pathlib.Path, digest: str) -> int:
        """Applies one file once: intent row, loop call, applied row.

        A file that cannot be read, lacks a required column or has no rows is
        instead journalled as rejected and escalated, before any intent row.

        Args:
            what: The kind of input, batches or labels.
            path: The file.
            digest: The file's digest.

        Returns:
            The number of cases matched for a labels file; 0 for a batch file
            or a file that was rejected.
        """
        try:
            if what == BATCHES:
                self.read(path, ("prediction", "decision"))
            else:
                self.read(path, ("case_id", "loss"))
        except (ValueError, OSError, pandas.errors.ParserError) as error:
            self.reject(what, path, digest, str(error))
            return 0
        self.journal(
            {
                "phase": deployment_ledger.INTENT,
                "what": KIND[what],
                "digest": digest,
                "file": path.name,
                "batch_before": self.loop.batch,
            }
        )
        self.mark(f"intent:{path.name}")
        matched = self.run(what, path)
        self.mark(f"applied-by-loop:{path.name}")
        self.journal(
            {
                "phase": deployment_ledger.APPLIED,
                "what": KIND[what],
                "digest": digest,
            }
        )
        self.mark(f"applied:{path.name}")
        return matched

    def recover(self, inbox: pathlib.Path) -> bool:
        """Finishes inputs whose intent has no result; False if it cannot.

        A batch whose counter already moved by one is only marked applied; a
        batch whose counter fits neither case, and a file that is missing or
        changed since its intent, halt the loop and open an escalation. A
        labels file is applied again.

        Args:
            inbox: The inbox directory holding the batch and label files.

        Returns:
            True if every unfinished input was completed; False if the loop
            state is unclear or a file is missing or changed, in which case
            the loop has been halted.
        """
        ledger = self.loop.ledger
        for digest, intent in list(ledger.intents.items()):
            if digest in ledger.consumed:
                continue
            batch = intent["what"] == KIND[BATCHES]
            what = BATCHES if batch else LABELS
            done = {
                "phase": deployment_ledger.APPLIED,
                "what": intent["what"],
                "digest": digest,
            }
            if batch:
                before = int(intent["batch_before"])
                if self.loop.batch == before + 1:
                    self.journal(done)
                    continue
                if self.loop.batch != before:
                    self.stop("controller_state_unclear", intent["file"])
                    return False
            path = inbox / what / intent["file"]
            if not path.is_file() or file_digest(path) != digest:
                self.stop("controller_input_changed", intent["file"])
                return False
            self.run(what, path)
            self.journal(done)
        return True

    def stop(self, reason: str, name: str) -> None:
        """Halts the loop and asks a person; used when it cannot go on.

        Args:
            reason: Why the controller stops.
            name: The file involved.
        """
        self.loop.enter_halt(reason, {"file": name})
        self.loop.escalate(reason, {"file": name})
        self.loop.save_state()

    def fresh(
        self, inbox: pathlib.Path, folder: str
    ) -> list[tuple[pathlib.Path, str]]:
        """CSV files of a folder not yet applied or rejected, by name.

        Args:
            inbox: The inbox directory.
            folder: The sub-folder to scan.

        Returns:
            Pairs of path and digest for the CSV files whose digest is in
            neither the applied nor the rejected set, in name order.
        """
        ledger = self.loop.ledger
        found = []
        for path in sorted((inbox / folder).glob("*.csv")):
            digest = file_digest(path)
            if digest not in ledger.consumed and digest not in ledger.rejected:
                found.append((path, digest))
        return found

    # ----------------------------------------------------------------- tick

    def tick(self, inbox: str | pathlib.Path) -> Tick:
        """Applies the new inbox files, steps the loop and journals the tick.

        Args:
            inbox: The inbox directory with its batch and label folders.

        Returns:
            The report of the tick. If the loop was already halted or recovery
            fails, nothing is applied or journalled and the status is
            ``halted``; otherwise the status is the loop's status after
            stepping.
        """
        folder = pathlib.Path(inbox)
        loop = self.loop
        if loop.halted or not self.recover(folder):
            return self.result(HALTED, 0, 0, (), ())
        rejected_before = set(loop.ledger.rejected)
        observed = 0
        matured = 0
        for path, digest in self.fresh(folder, BATCHES):
            self.apply(BATCHES, path, digest)
            observed += digest in loop.ledger.consumed
        for path, digest in self.fresh(folder, LABELS):
            matured += self.apply(LABELS, path, digest)
        step = loop.step()
        self.mark("stepped")
        rejected = tuple(
            sorted(
                str(loop.ledger.rejected[digest]["file"])
                for digest in set(loop.ledger.rejected) - rejected_before
            )
        )
        self.journal(
            {
                "phase": deployment_ledger.TICK,
                "observed": observed,
                "matured": matured,
                "rejected": len(rejected),
                "status": step.status,
            }
        )
        self.dead_man()
        return self.result(
            step.status, observed, matured, rejected, step.events
        )

    def dead_man(self) -> None:
        """Opens an escalation when data or labels have stopped arriving."""
        idle = self.trailing(lambda body: body["observed"] == 0)
        if idle >= self.policy.max_idle_ticks:
            self.loop.escalate(
                "controller_no_new_data", {"ticks": self.policy.max_idle_ticks}
            )
        starved = self.trailing(
            lambda body: (
                body["status"] == scmle.governance.loop.INSUFFICIENT_LABELS
            )
        )
        if starved >= self.policy.max_unlabelled_ticks:
            self.loop.escalate(
                "controller_labels_not_arriving",
                {"ticks": self.policy.max_unlabelled_ticks},
            )
        self.loop.save_state()

    def result(
        self,
        status: str,
        observed: int,
        matured: int,
        rejected: tuple[str, ...],
        events: tuple[tuple[str, str], ...],
    ) -> Tick:
        """Builds the report, with the escalations now open.

        Args:
            status: The loop's status.
            observed: Number of batches applied.
            matured: Number of cases matched.
            rejected: Names of the rejected files.
            events: The loop's events.

        Returns:
            The tick report.
        """
        return Tick(
            status,
            observed,
            matured,
            rejected,
            events,
            tuple(e.id for e in self.loop.pending_escalations()),
        )

    def status(self) -> dict[str, Any]:
        """The controller's state as JSON-ready data.

        Returns:
            Whether the loop is halted, the batch counter, the numbers of
            applied and rejected inputs, the open escalations, the uses left
            per envelope and whether an alarm gates retraining.
        """
        ledger = self.loop.ledger
        return {
            "halted": self.loop.halted,
            "batch": self.loop.batch,
            "applied": len(ledger.consumed),
            "rejected": len(ledger.rejected),
            "pending_escalations": [
                e.id for e in self.loop.pending_escalations()
            ],
            "envelopes": {
                identity: self.envelopes.remaining(identity)
                for identity in ledger.envelopes
            },
            "alarm_gates_retrain": self.loop.policy.alarm_gates_retrain,
        }
