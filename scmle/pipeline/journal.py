"""The records a run keeps: files, events, trials and decisions.

`Journal` is the part of the run context that writes: run files, events,
work skipped for the budget, trial results and evidence-based decisions.
Every write is skipped when the run has no store (the plan preview).
"""

import logging
from typing import Any

import scmle.artifacts.store
from scmle.contracts import errors
from scmle.contracts import events
from scmle.contracts import experiments
from scmle.contracts import records
from scmle.contracts import tasks
from scmle.storage import stores

LOGGER = logging.getLogger(__name__)


class Journal:
    """Writes the records of a run; the run context is one.

    Attributes:
        spec: Project specification.
        store: Run directory, or None for previews.
        log: Event log, or None for previews.
        tracker: Experiment tracking backend.
        trials: Trial results.
        outcomes: Evidence-based decisions made while running.
        trial_counter: Number of trial identifiers handed out so far.
        skipped: Work reduced or skipped to keep the time budget.
    """

    spec: tasks.Config
    store: scmle.artifacts.store.Workspace | None
    log: events.Log | None
    tracker: stores.Store
    trials: list[experiments.Trial]
    outcomes: list[experiments.Decision]
    trial_counter: int
    skipped: list[dict[str, str]]

    def write_json(self, relative: str, payload: Any) -> None:
        """Writes a JSON artifact if the run is persisted.

        Args:
            relative: Artifact path relative to the run directory.
            payload: JSON-serializable content.
        """
        if self.store is not None:
            self.store.write_json(relative, payload)

    def write_text(self, relative: str, text: str) -> None:
        """Writes a text artifact if the run is persisted.

        Args:
            relative: Artifact path relative to the run directory.
            text: The content to write.
        """
        if self.store is not None:
            self.store.write_text(relative, text)

    def emit(
        self,
        kind: str,
        reason: str = "",
        artifacts: tuple[str, ...] = (),
        data: records.Layout | None = None,
    ) -> None:
        """Records an event if the run is persisted.

        Args:
            kind: Dotted event type.
            reason: Why it happened.
            artifacts: Run-relative artifact paths it refers to.
            data: Structured payload (for example hashes and counts).
        """
        if self.log is not None and self.store is not None:
            self.log.emit(
                events.Event(
                    self.store.run_id,
                    kind,
                    reason,
                    self.spec.config_hash(),
                    artifacts,
                    data or {},
                )
            )

    def forgo(self, work: str, reason: str) -> None:
        """Records work reduced or skipped to keep the time budget.

        Args:
            work: What was reduced or skipped.
            reason: Why, with the estimate and the time left.
        """
        self.skipped.append({"work": work, "reason": reason})
        LOGGER.warning("%s skipped or reduced: %s", work, reason)
        self.emit(
            "budget.skipped",
            f"{work} skipped or reduced: {reason}",
            data={"work": work, "reason": reason},
        )

    def allocate(self, count: int) -> list[str]:
        """Hands out the next `count` sequential trial identifiers.

        Args:
            count: How many identifiers to reserve.

        Returns:
            Identifiers of the form `trial_001`, continuing from the trials
            already allocated.
        """
        start = self.trial_counter
        self.trial_counter += count
        return [f"trial_{start + offset + 1:03d}" for offset in range(count)]

    def enter(
        self,
        technique: str,
        action: experiments.Edict,
        reason: str,
    ) -> None:
        """Records an evidence-based decision for the decision log.

        Args:
            technique: The technique the decision concerns.
            action: What was decided (the decision action).
            reason: The evidence behind the decision.
        """
        self.outcomes.append(experiments.Decision(technique, action, reason))

    def transcribe(self, outcome: experiments.Trial) -> None:
        """Appends a trial and persists, announces and tracks it.

        Args:
            outcome: The finished trial to record.

        Raises:
            ScmleError: If a recorded trial already has the same identifier,
                which would overwrite its checkpoint.
        """
        if any(t.trial_id == outcome.trial_id for t in self.trials):
            raise errors.ScmleError(
                f"duplicate trial id {outcome.trial_id}",
                {"trial_id": outcome.trial_id},
            )
        self.trials.append(outcome)
        path = f"trials/{outcome.trial_id}.json"
        self.write_json(path, outcome.to_dict())
        self.emit(f"trial.{outcome.status}", outcome.model_name, (path,))
        if self.store is not None:
            self.tracker.transcribe(self.store.run_id, outcome)

    def track(self, **fields: Any) -> None:
        """Updates the tracked run row if the run is persisted.

        Args:
            **fields: Run-row fields to update.
        """
        if self.store is not None:
            self.tracker.revise(self.store.run_id, **fields)
