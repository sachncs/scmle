"""The continuous evaluation and the replacement it guards."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import pathlib
import time
from typing import Any

from scmle.deploy import control
from scmle.deploy import gate
from scmle.deploy import release_state
from scmle.governance import deployment_ledger
import scmle.governance.loop
import scmle.monitoring.champion
from scmle.operations.continuous import policies
from scmle.operations.continuous import promotion
from scmle.operations.continuous import records
from scmle.operations.continuous import retraining


@dataclasses.dataclass(frozen=True, slots=True)
class Replacement:
    """What an attempt to replace the production model did.

    Attributes:
        done: Whether the runtime confirmed the new model.
        reasons: Why not, when it was not done.
        outcome: The release registry's outcome, when promotion was tried.
    """

    __module__ = "scmle.operations.continuous"

    done: bool
    reasons: tuple[str, ...]
    outcome: release_state.Outcome | None = None

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready record.

        Returns:
            Whether it was done, the reasons and the release outcome (or None)
            as a dict.
        """
        return {
            "done": self.done,
            "reasons": list(self.reasons),
            "outcome": None if self.outcome is None else self.outcome.to_dict(),
        }


class Evaluation(retraining.Retraining):
    """Feeds the loop, applies the retraining policy and guards replacement."""

    __module__ = "scmle.operations.continuous"

    def __init__(
        self,
        loop: scmle.governance.loop.Loop,
        directory: str | pathlib.Path,
        policy: policies.Policy = policies.Policy(),  # noqa: B008
        retrain: Callable[[], str | None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Creates the evaluation around an opened loop.

        Args:
            loop: The closed loop of the deployment.
            directory: Where history, retained outcomes and datasets live.
            policy: The retraining policy.
            retrain: Hook that trains and stages a challenger and returns its
                name, or None if it started none. Without it a trigger or an
                approved retrain starts no challenger.
            clock: Callable that returns the current time in epoch seconds.
        """
        self.loop = loop
        self.directory = pathlib.Path(directory)
        self.policy = policy
        self.retrain = retrain
        self.clock = clock
        self.history = records.History(self.directory / records.HISTORY_FILE)
        self.alarms = sum(
            1 for r in loop.ledger.rows if r["kind"] == deployment_ledger.ALARM
        )
        proposals = self.history.select("retrain_decision")
        self.proposed: int | None = (
            proposals[-1]["facts"].get("batch") if proposals else None
        )

    def replace(
        self,
        challenger: str,
        release: release_state.Registry,
        plane: control.Platform,
        candidate: gate.Candidate,
        key: str | bytes,
        *,
        health: release_state.Health | None = None,
        steps: tuple[int, ...] = (100,),
        comparison: scmle.monitoring.champion.Comparison | None = None,
    ) -> Replacement:
        """Puts a challenger into an environment, or says why not.

        Refuses unless (1) the loop recorded a promotion proposal for the
        challenger, (2) the promotion gate admits the candidate and (3) the
        release registry has a valid approval for it. If an offline
        `comparison` is given its decision must be ``promote``. The registry
        changes only after the runtime confirms. A refusal or an unconfirmed
        rollout does not raise; it returns ``done=False``. Invalid `steps` or
        a candidate container digest that is not a digest raise the registry's
        ``ConfigError``. Every outcome is written to the history.

        Args:
            challenger: Name of the challenger.
            release: The release registry.
            plane: The control plane of the platform.
            candidate: The candidate the gate judges.
            key: Key that verifies the approval and the gate evidence.
            health: Called after each traffic step; reasons stop the rollout.
            steps: Traffic shares in percent to pass through, increasing and
                ending in 100.
            comparison: An offline comparison; if given, its decision must be
                ``promote``.

        Returns:
            Whether the replacement was done and, if not, the reasons.
        """
        reasons = []
        if challenger not in self.nominations():
            reasons.append(
                f"the loop has no promotion proposal for {challenger!r}"
            )
        if comparison is not None and comparison.decision != "promote":
            reasons.append("the offline comparison does not favour it")
        decision = gate.admit(candidate, key)
        reasons.extend(decision.reasons)
        if reasons:
            self.history.record(
                "replacement_refused",
                self.clock(),
                challenger=challenger,
                reasons=reasons,
            )
            return Replacement(done=False, reasons=tuple(reasons))
        outcome = release.promote(
            candidate.model.model,
            candidate.model.version,
            plane,
            candidate.environment,
            candidate.container,
            steps=steps,
            health=health,
            signing_key=key,
        )
        self.history.record(
            "replacement" if outcome.done else "replacement_refused",
            self.clock(),
            challenger=challenger,
            environment=candidate.environment,
            outcome=outcome.to_dict(),
        )
        return Replacement(outcome.done, outcome.reasons, outcome)

    def followup(
        self,
        specs: Sequence[promotion.Indicator],
        before: Mapping[str, Sequence[float]],
        after: Mapping[str, Sequence[float]],
        **options: Any,
    ) -> dict[str, Any]:
        """Runs `reassess` and records the result.

        Args:
            specs: The metrics to judge.
            before: Observations of each metric in the earlier period.
            after: Observations of each metric in the later period.
            **options: Further options of `reassess`.

        Returns:
            The result of `reassess`.
        """
        result = promotion.reassess(specs, before, after, **options)
        self.history.record("post_promotion_check", self.clock(), **result)
        return result
