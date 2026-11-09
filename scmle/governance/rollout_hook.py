"""The decision layer's rollout, run through the lifecycle manager.

`scmle.decision.rollout` decides from an e-process; it knows nothing of
registries. This hook is the one place that joins the two:

* an experiment is started with the level the `DecisionBudget` assigns, spent
  whether or not the experiment promotes (the budget after the assignment is
  written to the deployment ledger);
* units are assigned by the salted hash of the rollout plan and every batch is
  written to the deployment ledger with the arm and the probability that arm
  had (the propensity), and to the hash-chained decision log;
* a `promote` decision is a proposal row; the alias moves only through
  `LifecycleManager.promote_challenger`, which still needs the registry's
  independent signed approval (and the attestation of a guarded registry);
* binding the hook makes the manager's bootstrap rules descriptive
  (`LifecycleManager.defer_to_rollout`): `review` and `check_rollback` no
  longer decide promotion or rollback, and the hook's `review` and
  `check_rollback` write their results to the decision log as
  `bootstrap_diagnostic` rows with `descriptive: true` and `decides: false`;
* a promoted champion gets its rollback monitor at the rollout plan's
  rollback level (once per champion); a rollback decision is a proposal
  row, and the alias moves back only through
  `LifecycleManager.restore_previous`.

The statistical guarantees are those of `scmle.decision.rollout`: a false
promotion has probability at most the experiment's level, and the levels sum
to at most `alpha_promote` by the union bound over experiments.
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
import pandas as pd

from scmle.contracts import errors
from scmle.data.charter import decision as charter_decision
from scmle.data.ledger import budget as budget_module
from scmle.decision import design
from scmle.decision import rollout
from scmle.governance import deployment_ledger
from scmle.governance import lifecycle
from scmle.governance import registry as registry_module


class RolloutHook:
    """One model's rollouts under a plan, a budget and a deployment ledger."""

    def __init__(
        self,
        manager: lifecycle.LifecycleManager,
        plan: charter_decision.RolloutPlan,
        ledger: deployment_ledger.DeploymentLedger,
        *,
        budget: budget_module.DecisionBudget | None = None,
        log: rollout.DecisionLog | None = None,
    ) -> None:
        """Binds the hook.

        Args:
            manager: The model's lifecycle manager.
            plan: The pre-registered rollout plan.
            ledger: The deployment ledger the rows are chained in.
            budget: The decision budget so far (the rollout plan's fresh one
                when omitted).
            log: The decision log to continue (a new one when omitted).
        """
        self.manager = manager
        manager.defer_to_rollout()
        self.plan = plan
        self.ledger = ledger
        self.budget = budget if budget is not None else plan.budget()
        self.log = log if log is not None else rollout.DecisionLog()
        self.experiment: rollout.Rollout | None = None
        self.experiment_id: str | None = None
        self.decision: rollout.Decision | None = None
        self.challenger: registry_module.VersionRecord | None = None
        self.monitor: rollout.RollbackMonitor | None = None
        self.champion_id: str | None = None
        self.units = 0

    def diagnose(self, stage: str, comparison: Any) -> None:
        """Logs a bootstrap result as a descriptive diagnostic.

        Args:
            stage: Which review the result belongs to (for example `review` or
                `rollback`).
            comparison: The bootstrap comparison, or None; its decision,
                advantage and interval are recorded when present. It decides
                nothing.
        """
        body: dict[str, Any] = {
            "stage": stage,
            "descriptive": True,
            "decides": False,
        }
        if comparison is not None:
            body.update(
                decision=comparison.decision,
                advantage=comparison.advantage,
                interval=list(comparison.interval),
            )
        self.log.append("bootstrap_diagnostic", body)

    def review(
        self,
        fresh: pd.DataFrame,
        target: str,
        retrain: bool = True,
        project_options: dict[str, Any] | None = None,
    ) -> lifecycle.LifecycleOutcome:
        """Runs `LifecycleManager.review` and logs its comparison.

        The challenger is staged for the rollout; the bootstrap comparison
        decides nothing and goes to the decision log as a diagnostic.

        Args:
            fresh: Fresh labelled rows.
            target: Name of the target column.
            retrain: Whether the manager may retrain a challenger.
            project_options: Options for the retraining project.

        Returns:
            The lifecycle manager's outcome.
        """
        outcome = self.manager.review(fresh, target, retrain, project_options)
        if outcome.comparison is not None:
            self.diagnose("review", outcome.comparison)
        return outcome

    def check_rollback(
        self,
        fresh: pd.DataFrame,
        target: str,
        policy: lifecycle.RollbackPolicy | None = None,
        approval: registry_module.PromotionApproval | None = None,
        signing_key: str | bytes | None = None,
    ) -> lifecycle.RollbackOutcome:
        """Runs `LifecycleManager.check_rollback` as a logged diagnostic.

        Nothing moves, whatever the approval: only `watch` and `roll_back`
        take the previous champion back.

        Args:
            fresh: Fresh labelled rows.
            target: Name of the target column.
            policy: The rollback policy, or None for the default.
            approval: A promotion approval, accepted but not acted on here.
            signing_key: Key that verifies the approval.

        Returns:
            The lifecycle manager's rollback outcome.
        """
        outcome = self.manager.check_rollback(
            fresh, target, policy, approval, signing_key
        )
        if outcome.comparison is not None:
            self.diagnose("rollback", outcome.comparison)
        return outcome

    def started(self) -> rollout.Rollout:
        """The running experiment.

        Returns:
            The experiment started by `start`.

        Raises:
            ScmleError: If none was started.
        """
        if self.experiment is None:
            raise errors.ScmleError("no experiment was started")
        return self.experiment

    def start(self, experiment_id: str) -> float:
        """Starts an experiment of the staged challenger against the champion.

        The level is assigned before any traffic and is spent now. A reused
        id is refused (`ValueError`), an exhausted budget with
        `BudgetExhaustedError`, and a missing staged challenger with
        `ArtifactError`.

        Args:
            experiment_id: A new identifier.

        Returns:
            The level assigned from the decision budget.
        """
        challenger = self.manager.registry.resolve(
            self.manager.model, registry_module.CHALLENGER
        )
        budget, level = budget_module.assign_experiment(
            self.budget, experiment_id
        )
        self.budget = budget
        self.challenger = challenger
        self.experiment_id = experiment_id
        self.experiment = rollout.Rollout(
            self.plan, experiment_id, float(level)
        )
        self.decision = None
        self.units = 0
        self.ledger.commit(
            deployment_ledger.ROLLOUT_EXPERIMENT,
            {
                "experiment_id": experiment_id,
                "model": self.manager.model,
                "challenger_version": challenger.version,
                "level": str(level),
                "budget": budget.to_dict(),
            },
        )
        self.log.append(
            "experiment_started",
            {
                "experiment_id": experiment_id,
                "level": str(level),
                "challenger_version": challenger.version,
            },
        )
        return float(level)

    def assign(self, unit_ids: ArrayLike) -> NDArray[np.float64]:
        """Assigns units and writes the propensity-logged batch.

        Args:
            unit_ids: Identifiers of the randomisation unit.

        Returns:
            1 for the challenger and 0 for the champion, per unit.
        """
        self.started()
        names = [str(u) for u in np.asarray(unit_ids).ravel()]
        arms = design.assign(names, self.plan.salt, self.plan.assignment_prob)
        q = self.plan.assignment_prob
        odds = [q if a else 1.0 - q for a in arms]
        self.ledger.commit(
            deployment_ledger.ROLLOUT_ASSIGNMENT,
            {
                "experiment_id": self.experiment_id,
                "first_unit": self.units,
                "units": names,
                "arm": [int(a) for a in arms],
                "probability": odds,
            },
        )
        self.units += len(names)
        return arms

    def observe(
        self,
        arm: ArrayLike,
        primary: ArrayLike,
        guardrails: Any = None,
    ) -> rollout.Decision:
        """Adds a batch of outcomes and records the decision so far.

        A promotion also writes a proposal row; nothing moves.

        Args:
            arm: The arm of each unit (0 or 1).
            primary: The primary outcome of each unit.
            guardrails: Guardrail outcomes, if any.

        Returns:
            The experiment's decision so far.
        """
        experiment = self.started()
        decision = experiment.observe(arm, primary, guardrails)
        self.decision = decision
        body = {
            "experiment_id": self.experiment_id,
            "stage": "promotion",
            **decision.to_dict(),
        }
        self.ledger.commit(deployment_ledger.ROLLOUT_DECISION, body)
        self.log.append("decision", decision.to_dict())
        if decision.status == rollout.PROMOTE:
            self.ledger.commit(
                deployment_ledger.PROMOTION_PROPOSAL,
                {
                    "experiment_id": self.experiment_id,
                    "model": self.manager.model,
                    "version": self.challenger.version
                    if self.challenger
                    else None,
                    "source": "rollout e-process",
                    "level": decision.level,
                },
            )
        return decision

    def promote(
        self,
        approval: registry_module.PromotionApproval,
        signing_key: str | bytes | None = None,
    ) -> registry_module.VersionRecord:
        """Moves the champion alias after a `promote` decision, with approval.

        Without a valid independent approval the registry refuses with an
        `ArtifactError`.

        Args:
            approval: The independent approval of the challenger bundle.
            signing_key: Key that verifies the approval.

        Returns:
            The record of the version now champion.

        Raises:
            ScmleError: If the experiment did not end in `promote`.
        """
        if self.decision is None or self.decision.status != rollout.PROMOTE:
            raise errors.ScmleError(
                "the experiment has no promote decision; nothing to promote"
            )
        record = self.manager.promote_challenger(approval, signing_key)
        self.champion_id = f"{self.manager.model}:v{record.version}"
        budget, level = budget_module.start_rollback_monitor(
            self.budget, self.champion_id
        )
        self.budget = budget
        self.monitor = rollout.RollbackMonitor(
            self.plan.tolerance,
            float(level),
            self.plan.assignment_prob,
            self.plan.low,
            self.plan.high,
        )
        self.log.append(
            "promoted",
            {"champion": self.champion_id, "rollback_level": str(level)},
        )
        return record

    def watch(self, arm: ArrayLike, outcome: ArrayLike) -> str:
        """Feeds the promoted champion's rollback monitor on fresh traffic.

        Args:
            arm: The arm of each unit (0 or 1).
            outcome: The outcome of each unit.

        Returns:
            The monitor's status; a rollback decision is also recorded in the
            ledger and the decision log.

        Raises:
            ScmleError: If nothing was promoted through this hook.
        """
        if self.monitor is None:
            raise errors.ScmleError("no promoted champion is monitored")
        status = self.monitor.observe(arm, outcome)
        if status == rollout.ROLLBACK:
            self.ledger.commit(
                deployment_ledger.ROLLBACK_PROPOSAL,
                {
                    "experiment_id": self.experiment_id,
                    "model": self.manager.model,
                    "champion": self.champion_id,
                    "source": "rollback e-process",
                    "units": self.monitor.done_at,
                },
            )
            self.ledger.commit(
                deployment_ledger.ROLLOUT_DECISION,
                {
                    "experiment_id": self.experiment_id,
                    "stage": "rollback",
                    "status": status,
                    "units": self.monitor.done_at,
                    "champion": self.champion_id,
                },
            )
            self.log.append(
                "rollback_decision",
                {"champion": self.champion_id, "units": self.monitor.done_at},
            )
        return status

    def roll_back(
        self,
        approval: registry_module.PromotionApproval,
        signing_key: str | bytes | None = None,
    ) -> int:
        """Restores the previous champion after a rollback decision.

        Without a valid independent approval the registry refuses with an
        `ArtifactError`.

        Args:
            approval: The independent approval of the restored bundle.
            signing_key: Key that verifies the approval.

        Returns:
            The version number restored as champion.

        Raises:
            ScmleError: If the monitor has not decided a rollback.
        """
        if self.monitor is None or self.monitor.status != rollout.ROLLBACK:
            raise errors.ScmleError(
                "the monitor has no rollback decision; nothing to restore"
            )
        version = self.manager.restore_previous(approval, signing_key)
        self.log.append("rolled_back", {"restored_version": version})
        return version
