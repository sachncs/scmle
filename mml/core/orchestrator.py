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

import pandas as pd

from mml.api import results
from mml.artifacts import store as store_module
from mml.contracts import errors
from mml.contracts import events
from mml.contracts import experiments
from mml.contracts import tasks
from mml.core import context as context_module
from mml.core import execution
from mml.core import stages as stages_module
from mml.core import state
from mml.evaluation import metrics
from mml.evaluation import selection
from mml.models import registry as registry_module
from mml.tracking import stores as tracking_stores


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
    ) -> None:
        """Initializes the orchestrator.

        Args:
            spec: Project specification.
            frame: In-memory dataset, if not read from `spec.data.uri`.
            registry: Model registry; defaults to the built-in adapters.
            policy: Selection policy; defaults to best mean CV score.
            stages: Stage sequence; defaults to the standard workflow.
            tracker: Experiment store; defaults to a SQLite database next to
                the run directories.
        """
        self.spec = spec
        self.frame = frame
        self.registry = (
            registry or registry_module.ModelRegistry.with_builtins()
        )
        self.policy = policy or selection.BestMeanPolicy()
        self.stages = stages or stages_module.default_stages()
        self.tracker = tracker
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
            backend=execution.create_backend(self.spec.resources.workers),
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

    def prepare(
        self, approved: experiments.ExperimentPlan | None = None
    ) -> context_module.RunContext:
        """Ingests, profiles and plans without training or writing files.

        Args:
            approved: Plan restricting which models may be considered.

        Returns:
            A context whose `profile` and `plan` are populated.
        """
        context = self.new_context(approved_plan=approved)
        context.history = self.load_history(self.tracker)
        stages_module.IngestStage().execute(context)
        stages_module.ProfileStage().execute(context)
        context.plan = stages_module.build_plan(context)
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
    ) -> Orchestrator:
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
        log.subscribe(functools.partial(tracker.listen, store.run_id))
        machine = state.StateMachine(
            store.run_id, log, self.spec.config_hash(), reason
        )
        context = self.new_context(store=store, log=log, approved_plan=approved)
        context.tracker = tracker
        context.history = self.load_history(tracker)
        context.checkpoints = checkpoints
        try:
            for stage in self.stages:
                context.check_cancelled()
                machine.transition(stage.state, stage.describe(context))
                stage.execute(context)
            machine.transition(
                state.RunState.COMPLETED,
                "run complete",
                ("model/", "evaluation.json"),
            )
            return context_module.require(context.result, "result")
        except errors.CancelledError as error:
            error.details.setdefault("run_dir", str(store.directory))
            machine.transition(state.RunState.CANCELLED, str(error))
            raise
        except KeyboardInterrupt as error:
            machine.transition(state.RunState.CANCELLED, "interrupted")
            raise errors.CancelledError(
                "interrupted", {"run_dir": str(store.directory)}
            ) from error
        except errors.BlockedError as error:
            error.details["run_dir"] = str(store.directory)
            machine.transition(state.RunState.BLOCKED, str(error))
            raise
        except errors.MagicMLError as error:
            error.details["run_dir"] = str(store.directory)
            machine.transition(
                state.RunState.FAILED, f"{error.category}: {error}"
            )
            raise
        except Exception as error:
            machine.transition(state.RunState.FAILED, f"internal: {error!r}")
            raise errors.MagicMLError(
                str(error), {"run_dir": str(store.directory)}
            ) from error
