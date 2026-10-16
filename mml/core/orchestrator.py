"""Orchestrator: runs the stage sequence under the state machine.

Stages (state machine): VALIDATING -> PROFILING -> PLANNING -> TRAINING ->
SELECTING -> TESTING -> PACKAGING -> COMPLETED, with FAILED / CANCELLED /
BLOCKED as exits. Every transition emits an event, and finished trials are
checkpointed to disk as they complete.
"""

import functools
from importlib import metadata
import json
import pathlib
import threading
from typing import Any
from typing import Self

import pandas as pd

from mml.api import results
from mml.artifacts import store as store_module
from mml.contracts import errors
from mml.contracts import events
from mml.contracts import experiments
from mml.contracts import tasks
from mml.core import compiler
from mml.core import context as context_module
from mml.core import limits
from mml.core import policies as checkpoints_module
from mml.core import scheduler
from mml.core import stages as stages_module
from mml.core import state
from mml.evaluation import metrics
from mml.evaluation import selection
from mml.experimentation import stores as tracking_stores
from mml.models import registry as registry_module


def environment_lock() -> str:
    """Lists installed distributions as sorted `name==version` lines."""
    pins = {
        f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
    }
    return "\n".join(sorted(pins))


class Orchestrator:
    """Compiles a project specification into a run and executes it."""

    def __init__(
        self,
        spec: tasks.ProjectSpec,
        frame: pd.DataFrame | None = None,
        registry: registry_module.ModelRegistry | None = None,
        policy: selection.SelectionPolicy | None = None,
        stages: tuple[stages_module.Stage, ...] | None = None,
        tracker: tracking_stores.ExperimentStore | None = None,
        checkpoints: tuple[checkpoints_module.PolicyCheckpoint, ...] = (),
    ) -> None:
        """Initializes the orchestrator.

        Args:
            spec: Project specification.
            frame: In-memory dataset, if not read from `spec.data.uri`.
            registry: Model registry; defaults to the built-in adapters plus
                every installed plugin that passes the conformance suite.
            policy: Selection policy; defaults to best mean CV score.
            stages: Stage sequence; defaults to the standard workflow.
            checkpoints: Approval gates consulted after the planning and
                selection stages; any rejection blocks the run.
            tracker: Experiment store; defaults to a SQLite database next to
                the run directories.
        """
        self.spec = spec
        self.frame = frame
        if registry is None:
            registry = registry_module.ModelRegistry.with_builtins()
            registry.load_plugins()
        self.registry = registry
        self.policy = policy or selection.BestMeanPolicy(spec.selection)
        self.stages = compiler.compile_workflow(
            stages or stages_module.default_stages()
        )
        self.tracker = tracker
        self.checkpoints = checkpoints
        self.tracking_path: pathlib.Path | None = None
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        """Requests cooperative cancellation of a running workflow."""
        self.cancel_event.set()

    def new_context(
        self,
        store: store_module.RunStore | None = None,
        log: events.EventLog | None = None,
        approved_plan: experiments.ExperimentPlan | None = None,
    ) -> context_module.RunContext:
        """Creates a run context wired with this orchestrator's services.

        Args:
            store: Run directory, or None for a side-effect-free preview.
            log: Event log, or None for a preview.
            approved_plan: Plan restricting which models may be considered.

        Returns:
            A fresh context.
        """
        return context_module.RunContext(
            spec=self.spec,
            registry=self.registry,
            catalog=metrics.CATALOG,
            policy=self.policy,
            backend=scheduler.create_backend(
                limits.ResourceGovernor(self.spec.resources).workers,
                self.spec.resources.executor,
            ),
            cancel_event=self.cancel_event,
            input_frame=self.frame,
            store=store,
            log=log,
            approved_plan=approved_plan,
        )

    def load_history(
        self, tracker: tracking_stores.ExperimentStore | None
    ) -> tuple[experiments.HistoricalRun, ...]:
        """Reads past runs for meta-learning without creating a database.

        Args:
            tracker: The store to read, or None to look for an existing
                database at `tracking_path`.

        Returns:
            Past runs, empty when no tracking database exists yet.
        """
        if tracker is None:
            path = self.tracking_path
            if path is None or not path.exists():
                return ()
            tracker = tracking_stores.SqliteExperimentStore(path)
        return tuple(tracker.history())

    def apply_plan(self, approved: experiments.ExperimentPlan | None) -> None:
        """Makes an approved or modified plan binding on the specification.

        The plan's validation strategy, time budget and metric replace the
        configured ones (and are validated with the rest of the
        specification); its candidate models restrict the search, and its
        task must match what the data resolves to.

        Args:
            approved: The plan, or None to keep the specification as is.

        Raises:
            ValidationError: If the plan contradicts the specification, for
                example group validation without a group column.
        """
        if approved is None:
            return
        record = self.spec.model_dump()
        record["validation"]["strategy"] = approved.validation_strategy
        record["resources"]["time_budget_seconds"] = (
            approved.optimization_budget_seconds
        )
        record["metric"] = approved.metric
        self.spec = tasks.ProjectSpec.model_validate(record)

    def prepare(
        self, approved: experiments.ExperimentPlan | None = None
    ) -> context_module.RunContext:
        """Ingests, profiles and plans without training or writing files.

        Args:
            approved: Plan restricting which models may be considered.

        Returns:
            A context whose `profile` and `plan` are populated.
        """
        self.apply_plan(approved)
        context = self.new_context(approved_plan=approved)
        context.history = self.load_history(self.tracker)
        stages_module.IngestStage().execute(context)
        stages_module.ProfileStage().execute(context)
        context.plan = stages_module.build_plan(context)
        stages_module.advise(context)
        return context

    def run(
        self,
        output_dir: str | pathlib.Path = "runs",
        approved: experiments.ExperimentPlan | None = None,
    ) -> results.RunResult:
        """Executes the full workflow in a new run directory.

        Args:
            output_dir: Parent directory for run directories and the
                tracking database.
            approved: Approved or modified plan restricting candidates.

        Returns:
            The completed run.

        Raises:
            BlockedError: The framework could not safely continue.
            CancelledError: The run was cancelled.
            MagicMLError: Any other framework failure; unexpected errors are
                wrapped. The run directory is in `details["run_dir"]`.
        """
        self.apply_plan(approved)
        store = store_module.RunStore.create(output_dir)
        store.write_json("config.json", json.loads(self.spec.canonical_json()))
        store.write_text("environment.lock", environment_lock())
        return self.execute(store, approved, {}, "run created")

    def resume(self, run_directory: str | pathlib.Path) -> results.RunResult:
        """Continues an interrupted run, reusing its completed trials.

        The configuration must be identical (same hash). Stages are
        re-executed deterministically; trials whose checkpoints are valid
        are reused instead of recomputed.

        Args:
            run_directory: Directory of a cancelled, failed or crashed run.

        Returns:
            The completed run.

        Raises:
            ConfigError: If the run already completed or the saved
                configuration differs from this orchestrator's.
        """
        store = store_module.RunStore.open(run_directory)
        saved = tasks.ProjectSpec.model_validate_json(
            store.path("config.json").read_text()
        )
        if saved.config_hash() != self.spec.config_hash():
            raise errors.ConfigError(
                "cannot resume: configuration differs from the saved run",
                {
                    "saved": saved.config_hash(),
                    "current": self.spec.config_hash(),
                },
            )
        if store.path("evaluation.json").exists():
            raise errors.ConfigError("run already completed; nothing to resume")
        plan_path = store.path("execution_plan.json")
        approved = (
            experiments.ExperimentPlan.from_dict(
                json.loads(plan_path.read_text())
            )
            if plan_path.exists()
            else None
        )
        checkpoints = self.load_checkpoints(store)
        return self.execute(store, approved, checkpoints, "run resumed")

    @classmethod
    def for_run_directory(
        cls, run_directory: str | pathlib.Path, **kwargs: Any
    ) -> Self:
        """Builds an orchestrator from a saved run's configuration.

        Args:
            run_directory: An existing run directory.
            **kwargs: Passed to the constructor (registry, policy, ...).

        Returns:
            An orchestrator whose spec is the run's saved configuration.
        """
        store = store_module.RunStore.open(run_directory)
        spec = tasks.ProjectSpec.model_validate_json(
            store.path("config.json").read_text()
        )
        return cls(spec, **kwargs)

    def load_checkpoints(
        self, store: store_module.RunStore
    ) -> dict[str, experiments.TrialResult]:
        """Reads completed trial checkpoints from a run directory."""
        checkpoints: dict[str, experiments.TrialResult] = {}
        for path in sorted(store.path("trials").glob("trial_*.json")):
            record = json.loads(path.read_text())
            record["status"] = experiments.TrialStatus(record["status"])
            checkpoints[record["trial_id"]] = experiments.TrialResult(**record)
        return checkpoints

    def recoverable(self, store: store_module.RunStore) -> list[str]:
        """Lists the trial checkpoints a failed or interrupted run can resume.

        Args:
            store: The run directory.
        """
        trials_dir = store.path("trials")
        if not trials_dir.exists():
            return []
        return sorted(path.stem for path in trials_dir.glob("trial_*.json"))

    def consult_checkpoints(
        self, gate: state.RunState, context: context_module.RunContext
    ) -> None:
        """Asks every checkpoint registered for `gate` and audits the verdicts.

        Raises:
            BlockedError: If any checkpoint rejects.
        """
        for checkpoint in self.checkpoints:
            if gate not in checkpoint.gates:
                continue
            verdict = checkpoint.review(
                checkpoints_module.describe(gate, context)
            )
            context.emit(
                f"policy.{'approved' if verdict.approved else 'rejected'}",
                verdict.reason,
                data={"gate": gate.value, "reviewer": verdict.reviewer},
            )
            if not verdict.approved:
                raise errors.BlockedError(
                    f"{gate.value} not approved by {verdict.reviewer}: "
                    f"{verdict.reason or 'no reason given'}"
                )

    def execute(
        self,
        store: store_module.RunStore,
        approved: experiments.ExperimentPlan | None,
        checkpoints: dict[str, experiments.TrialResult],
        reason: str,
    ) -> results.RunResult:
        """Runs the stage sequence under the state machine.

        Args:
            store: Run directory.
            approved: Plan restricting candidates.
            checkpoints: Reusable trial results (empty for a fresh run).
            reason: Reason attached to the first state event.

        Returns:
            The completed run.
        """
        log = events.EventLog(store.path("events.jsonl"))
        tracker = self.tracker or tracking_stores.SqliteExperimentStore(
            store.directory.parent / "experiments.sqlite"
        )
        tracker.start_run(
            store.run_id,
            self.spec.config_hash(),
            str(store.directory),
            events.utc_timestamp(),
        )
        tracker.update_run(store.run_id, config_json=self.spec.canonical_json())
        log.subscribe(functools.partial(tracker.listen, store.run_id))
        machine = state.StateMachine(
            store.run_id, log, self.spec.config_hash(), reason
        )
        context = self.new_context(store=store, log=log, approved_plan=approved)
        context.tracker = tracker
        context.history = self.load_history(tracker)
        context.checkpoints = checkpoints
        for name, reason in self.registry.plugin_failures.items():
            context.emit("plugin.rejected", f"{name}: {reason}")
        try:
            for stage in self.stages:
                context.check_cancelled()
                machine.transition(stage.state, stage.describe(context))
                stage.execute(context)
                self.consult_checkpoints(stage.state, context)
            machine.transition(
                state.RunState.COMPLETED,
                "run complete",
                ("model/", "evaluation.json"),
            )
            return context_module.require(context.result, "result")
        except errors.CancelledError as error:
            error.details.setdefault("run_dir", str(store.directory))
            error.details["recoverable_trials"] = self.recoverable(store)
            machine.transition(state.RunState.CANCELLED, str(error))
            raise
        except KeyboardInterrupt as error:
            machine.transition(state.RunState.CANCELLED, "interrupted")
            raise errors.CancelledError(
                "interrupted", {"run_dir": str(store.directory)}
            ) from error
        except errors.BlockedError as error:
            error.details["run_dir"] = str(store.directory)
            error.details["recoverable_trials"] = self.recoverable(store)
            machine.transition(state.RunState.BLOCKED, str(error))
            raise
        except errors.MagicMLError as error:
            error.details["run_dir"] = str(store.directory)
            error.details["recoverable_trials"] = self.recoverable(store)
            machine.transition(
                state.RunState.FAILED, f"{error.category}: {error}"
            )
            raise
        except Exception as error:
            machine.transition(state.RunState.FAILED, f"internal: {error!r}")
            raise errors.MagicMLError(
                str(error),
                {
                    "run_dir": str(store.directory),
                    "recoverable_trials": self.recoverable(store),
                },
            ) from error
