"""Audit of a finished (or interrupted) run against governance invariants.

The audit reads only the evidence a run left behind: its event log, decision
log and referenced artifacts. It never re-executes anything.
"""

import dataclasses
import json
import pathlib
from typing import Any

from scmle.core import state

EVENTS_FILE = "events.jsonl"
DECISION_LOG = "diagnostics/decision_log.json"
RESUME_REASON = "run resumed"
TEST_OPENED = "test.opened"
SELECTION_FROZEN = "selection.frozen"
TRIAL_PREFIX = "trial."


@dataclasses.dataclass(frozen=True)
class AuditFinding:
    """One violated invariant.

    Attributes:
        code: Stable identifier of the invariant.
        message: What is wrong, with the evidence.
    """

    code: str
    message: str


@dataclasses.dataclass(frozen=True)
class Audit:
    """The result of auditing one run.

    Attributes:
        run_id: Name of the audited run directory.
        events: Number of events read.
        decisions: Automated decisions in order: planner decisions first,
            then the selection, test-opening, strategy, ensemble, plan and
            approval events with their reasons.
        findings: Violated invariants; empty when the run is clean.
    """

    run_id: str
    events: int
    decisions: tuple[dict[str, Any], ...]
    findings: tuple[AuditFinding, ...]

    @property
    def ok(self) -> bool:
        """Whether every invariant held."""
        return not self.findings


class RunAuditor:
    """Checks a run directory against the governance invariants.

    Invariants: the log is ordered and consistent; state transitions follow
    the lifecycle; the locked test set is opened at most once, and only after
    the selection was frozen and every trial event was logged; referenced
    artifacts exist; the run ends in a terminal state.
    """

    def audit(self, run_dir: str | pathlib.Path) -> Audit:
        """Audits one run directory.

        A malformed log (invalid JSON, or a record without the expected keys)
        is not guarded against and raises.

        Args:
            run_dir: Directory of a run.

        Returns:
            The decisions made and any violated invariants; for a directory
            with no event log, a report with no events and a `log.missing`
            finding.
        """
        root = pathlib.Path(run_dir)
        path = root / EVENTS_FILE
        if not path.exists():
            finding = AuditFinding("log.missing", f"{EVENTS_FILE} is absent")
            return Audit(root.name, 0, (), (finding,))
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        findings = [
            *self.check_log(records),
            *self.check_states(records),
            *self.check_test_access(records),
            *self.check_artifacts(root, records),
        ]
        return Audit(
            root.name,
            len(records),
            self.decisions(root, records),
            tuple(findings),
        )

    @staticmethod
    def check_log(records: list[dict[str, Any]]) -> list[AuditFinding]:
        """Checks ordering and consistency of the log itself.

        Args:
            records: The event records, in log order.

        Returns:
            A finding if timestamps go backwards and one if the records
            belong to more than one run; empty if neither occurs.
        """
        found = []
        stamps = [r["timestamp"] for r in records]
        if stamps != sorted(stamps):
            found.append(AuditFinding("log.order", "timestamps go backwards"))
        runs = {r["run_id"] for r in records}
        if len(runs) > 1:
            found.append(
                AuditFinding("log.runs", f"mixed run ids: {sorted(runs)}")
            )
        return found

    @staticmethod
    def check_states(records: list[dict[str, Any]]) -> list[AuditFinding]:
        """Checks that state events follow the lifecycle and end terminal.

        A first `CREATED` event, or one whose reason starts with "run resumed",
        starts the sequence again without a finding.

        Args:
            records: The event records, in log order.

        Returns:
            A finding for each state entered out of order or after a terminal
            state, and one if the last state is not terminal (which includes a
            log with no state events).
        """
        found = []
        current: state.RunState | None = None
        for record in records:
            if not record["kind"].startswith("state."):
                continue
            entered = state.RunState(record["kind"].split(".", 1)[1])
            restarted = entered is state.RunState.CREATED
            if restarted and (
                current is None or record["reason"].startswith(RESUME_REASON)
            ):
                current = entered
                continue
            if current is None or current in state.TERMINAL_STATES:
                found.append(
                    AuditFinding(
                        "state.order", f"{entered} entered after {current}"
                    )
                )
            elif entered not in state.ABORT_STATES:
                expected = state.FORWARD_ORDER.index(current) + 1
                if state.FORWARD_ORDER.index(entered) != expected:
                    found.append(
                        AuditFinding(
                            "state.order",
                            f"{current} -> {entered} skips a step",
                        )
                    )
            current = entered
        if current not in state.TERMINAL_STATES:
            found.append(
                AuditFinding("state.unfinished", f"last state is {current}")
            )
        return found

    @staticmethod
    def check_test_access(records: list[dict[str, Any]]) -> list[AuditFinding]:
        """Checks that the locked test set was opened once, and late enough.

        Args:
            records: The event records, in log order.

        Returns:
            Findings for a test set opened more than once, opened before the
            selection was frozen (or with no freeze logged), or followed by
            trial events; empty if it was never opened or none applies.
        """
        kinds = [r["kind"] for r in records]
        opened = [i for i, kind in enumerate(kinds) if kind == TEST_OPENED]
        if not opened:
            return []
        found = []
        if len(opened) > 1:
            found.append(
                AuditFinding(
                    "test.reopened", f"test set opened {len(opened)} times"
                )
            )
        frozen = [i for i, kind in enumerate(kinds) if kind == SELECTION_FROZEN]
        if not frozen or frozen[0] > opened[0]:
            found.append(
                AuditFinding(
                    "test.before_selection",
                    "test set opened before the selection was frozen",
                )
            )
        late = [
            i
            for i, kind in enumerate(kinds)
            if kind.startswith(TRIAL_PREFIX) and i > opened[0]
        ]
        if late:
            found.append(
                AuditFinding(
                    "test.before_trials",
                    f"{len(late)} trial event(s) after the test set opened",
                )
            )
        return found

    @staticmethod
    def check_artifacts(
        root: pathlib.Path, records: list[dict[str, Any]]
    ) -> list[AuditFinding]:
        """Checks that every artifact an event refers to exists.

        Args:
            root: The run directory.
            records: The event records, each listing its artifacts.

        Returns:
            One finding per referenced artifact that is absent, sorted by path.
        """
        missing = sorted(
            {
                artifact
                for record in records
                for artifact in record["artifacts"]
                if not (root / artifact).exists()
            }
        )
        return [
            AuditFinding(
                "artifact.missing", f"{artifact} is referenced but absent"
            )
            for artifact in missing
        ]

    @staticmethod
    def decisions(
        root: pathlib.Path, records: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], ...]:
        """Collects the automated decisions in the order they were taken.

        Args:
            root: The run directory, whose decision log holds the planner's
                decisions.
            records: The event records, from which the selection, test opening
                and strategy, ensemble, plan and approval events are taken.

        Returns:
            The planner's decisions first, each with the source "planner", then
            the matching events in log order, each as a mapping of the event
            kind as `source`, its `reason` and its `timestamp`.
        """
        found: list[dict[str, Any]] = []
        log = root / DECISION_LOG
        if log.exists():
            for item in json.loads(log.read_text(encoding="utf-8")).get(
                "plan", []
            ):
                found.append({"source": "planner", **item})
        for record in records:
            kind = record["kind"]
            if kind in {SELECTION_FROZEN, TEST_OPENED} or kind.startswith(
                ("strategy.", "ensemble.", "plan.", "approval.")
            ):
                found.append(
                    {
                        "source": kind,
                        "reason": record["reason"],
                        "timestamp": record["timestamp"],
                    }
                )
        return tuple(found)
