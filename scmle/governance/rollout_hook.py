"""The decision layer's rollout, run through the lifecycle manager (F11).

`scmle.decision.rollout` decides from an e-process; it knows nothing of
registries. This hook is the one place that joins the two:

* an experiment is started with the level the `DecisionBudget` assigns, spent
  whether or not the experiment promotes (the budget after the assignment is
  written to the deployment ledger);
* a `promote` decision is a proposal row; the alias moves only through
  `LifecycleManager.promote_challenger`, which still needs the registry's
  independent signed approval (and the attestation of a guarded registry);
* a promoted champion gets its rollback monitor at the plan's rollback level
  (once per champion); a rollback decision is a proposal row, and the alias
  moves back only through `LifecycleManager.restore_previous`.

The statistical guarantees are those of `scmle.decision.rollout`: a false
promotion has probability at most the experiment's level, and the levels sum
to at most `alpha_promote` by the union bound over experiments.
"""

from typing import Any

from numpy.typing import ArrayLike

from scmle.contracts import errors
from scmle.data.charter import decision as charter_decision
from scmle.data.ledger import budget as budget_module
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
            budget: The decision budget so far (the plan's fresh one when
                omitted).
            log: The decision log to continue (a new one when omitted).
        """
        self.manager = manager
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

    def started(self) -> rollout.Rollout:
        """The running experiment.

        Raises:
            ScmleError: If none was started.
        """
        if self.experiment is None:
            raise errors.ScmleError("no experiment was started")
        return self.experiment

    def start(self, experiment_id: str) -> float:
        """Starts an experiment of the staged challenger against the champion.

        The level is assigned before any traffic and is spent now.

        Args:
            experiment_id: A new identifier.

        Returns:
            The level assigned from the decision budget.

        Raises:
            ValueError: If the id was used.
            BudgetExhaustedError: If the budget cannot cover another level.
            ArtifactError: If no challenger is staged.
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

    def observe(
        self,
        arm: ArrayLike,
        primary: ArrayLike,
        guardrails: Any = None,
    ) -> rollout.Decision:
        """Adds a batch of outcomes and records the decision so far.

        A promotion also writes a proposal row; nothing moves.
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

        Raises:
            ScmleError: If the experiment did not end in `promote`.
            ArtifactError: Without a valid independent approval.
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

        Raises:
            ScmleError: If the monitor has not decided a rollback.
            ArtifactError: Without a valid independent approval.
        """
        if self.monitor is None or self.monitor.status != rollout.ROLLBACK:
            raise errors.ScmleError(
                "the monitor has no rollback decision; nothing to restore"
            )
        version = self.manager.restore_previous(approval, signing_key)
        self.log.append("rolled_back", {"restored_version": version})
        return version
