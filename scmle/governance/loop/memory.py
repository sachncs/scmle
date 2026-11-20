"""The loop's running state: monitors, the state file and the ledger check."""

from collections.abc import Callable
from collections.abc import Mapping
import datetime
import json
import math
from pathlib import Path
from typing import Any

from scmle.governance import deployment_ledger
from scmle.governance import never_list
from scmle.governance.loop import terms
from scmle.monitoring import labels
from scmle.monitoring import sequential_tests


class Memory:
    """The running state of a loop and its persistence beside the ledger."""

    def __init__(
        self,
        directory: Path,
        ledger: deployment_ledger.Ledger,
        policy: terms.Preregistration,
        guard: never_list.Guard,
        hooks: terms.Hooks,
        keys: Mapping[str, str | bytes],
        today: Callable[[], datetime.date],
    ) -> None:
        """Builds a loop from parts that `Loop.open` assembles.

        Hooks the guard's ``on_write`` callback to `save` and creates
        the monitors. It does not load the saved state; `Loop.open` does.

        Args:
            directory: Directory of the ledger and loop state files.
            ledger: The deployment ledger.
            policy: The pre-registered policy.
            guard: The never-list guard.
            hooks: Optional callbacks.
            keys: Map from key id to the key that verifies a
                person's attestation.
            today: Clock, replaceable in tests.
        """
        self.directory = directory
        self.ledger = ledger
        self.policy = policy
        self.guard = guard
        self.hooks = hooks
        self.keys = dict(keys)
        self.today = today
        self.batch = 0
        self.tau = math.inf if policy.tau is None else policy.tau
        self.batches: list[dict[str, Any]] = []
        self.escalation: dict[str, int] = {}
        self.rollbacks = 0
        self.stopped: dict[str, float] = {}
        self.queue: list[str] = []
        self.active: dict[str, dict[str, Any]] = {}
        self.watch: dict[str, Any] | None = None
        self.mechanism = labels.Mechanism.COMPLETE
        self.reason: str | None = None
        self.frozen = False
        self.naive_flag = False
        self.calibrated = False
        self.feedback: dict[str, Any] | None = None
        self.guard.observer = self.save
        self.arm()

    def arm(self) -> None:
        """Creates fresh monitors from the policy."""
        policy = self.policy
        self.alarm = sequential_tests.Tripwire(
            policy.reference,
            policy.tolerance,
            policy.alarm,
            policy.scale,
        )
        self.naive_alarm = sequential_tests.Tripwire(
            policy.reference,
            policy.tolerance,
            policy.alarm,
            policy.loss,
        )
        self.selective = sequential_tests.Tripwire(
            policy.target, 0.0, policy.delta, policy.scale
        )
        self.test: sequential_tests.Mixture | None = (
            None
            if policy.coverage is None
            else sequential_tests.Mixture(
                policy.coverage,
                sequential_tests.LESS,
                policy.alarm,
            )
        )
        self.novelty = sequential_tests.Mixture(
            min(max(policy.novelty, 1e-6), 1 - 1e-6),
            sequential_tests.GREATER,
            policy.alarm,
        )

    @property
    def statefile(self) -> Path:
        """The state file."""
        return self.directory / terms.STATE_FILE

    def save(self) -> None:
        """Writes the running state with the ledger head it belongs to."""
        challengers = {
            name: {
                "process": item["process"].to_dict(),
                "batches": item["batches"],
            }
            for name, item in self.active.items()
        }
        state = {
            "version": terms.STATE_VERSION,
            "ledger_head": self.ledger.head,
            "batch": self.batch,
            "tau": None if math.isinf(self.tau) else self.tau,
            "mechanism": self.mechanism.value,
            "halt_reason": self.reason,
            "naive_flag": self.naive_flag,
            "calibrated": self.calibrated,
            "feedback": self.feedback,
            "consecutive_rollbacks": self.rollbacks,
            "escalation_batch": self.escalation,
            "queue": self.queue,
            "stopped": self.stopped,
            "alarm": self.alarm.to_dict(),
            "naive_alarm": self.naive_alarm.to_dict(),
            "selective": self.selective.to_dict(),
            "coverage_test": (
                None if self.test is None else self.test.to_dict()
            ),
            "novelty_test": self.novelty.to_dict(),
            "active": challengers,
            "rollback_watch": (
                None
                if self.watch is None
                else {
                    "previous": self.watch["previous"],
                    "process": self.watch["process"].to_dict(),
                }
            ),
            "pending_batches": self.batches,
        }
        temporary = self.statefile.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(terms.jsonify(state), sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(self.statefile)

    def restore(self) -> None:
        """Reads the state file; halts if it does not match the ledger."""
        if not self.statefile.exists():
            self.frozen = self.ledger.halted
            return
        state = json.loads(self.statefile.read_text(encoding="utf-8"))
        if state["ledger_head"] != self.ledger.head:
            self.frozen = True
            self.reason = terms.HALT_STATE
            return
        self.batch = state["batch"]
        self.tau = math.inf if state["tau"] is None else state["tau"]
        self.mechanism = labels.Mechanism(state["mechanism"])
        self.reason = state["halt_reason"]
        self.naive_flag = state["naive_flag"]
        self.calibrated = state["calibrated"]
        self.feedback = state.get("feedback")
        self.rollbacks = state["consecutive_rollbacks"]
        self.escalation = dict(state["escalation_batch"])
        self.queue = list(state["queue"])
        self.stopped = dict(state["stopped"])
        self.alarm = sequential_tests.Tripwire.from_dict(state["alarm"])
        self.naive_alarm = sequential_tests.Tripwire.from_dict(
            state["naive_alarm"]
        )
        self.selective = sequential_tests.Tripwire.from_dict(state["selective"])
        if state["coverage_test"] is not None:
            self.test = sequential_tests.Mixture.from_dict(
                state["coverage_test"]
            )
        self.novelty = sequential_tests.Mixture.from_dict(state["novelty_test"])
        self.active = {
            name: {
                "process": sequential_tests.Comparison.from_dict(
                    item["process"]
                ),
                "batches": item["batches"],
            }
            for name, item in state["active"].items()
        }
        watch = state["rollback_watch"]
        if watch is not None:
            self.watch = {
                "previous": watch["previous"],
                "process": sequential_tests.Comparison.from_dict(
                    watch["process"]
                ),
            }
        self.batches = state["pending_batches"]
        self.frozen = self.ledger.halted

    @property
    def halted(self) -> bool:
        """Whether every case is escalated."""
        return self.ledger.halted or self.frozen

    def recheck(self) -> bool:
        """Re-checks the ledger file; enters a halt (in memory) if it fails.

        Returns:
            True if the ledger verifies; False if it fails, in which case the
            loop is halted in memory.
        """
        try:
            self.ledger.verify()
        except deployment_ledger.MisuseError:
            # handled: loop halted with reason; returns False
            self.frozen = True
            self.reason = terms.HALT_LEDGER
            return False
        return True
