"""Orchestrator: runs the stage sequence under the state machine.

Stages (state machine): VALIDATING -> PROFILING -> PLANNING -> TRAINING ->
SELECTING -> TESTING -> PACKAGING -> COMPLETED, with FAILED / CANCELLED /
BLOCKED as exits. Every transition emits an event, and finished trials are
checkpointed to disk as they complete.
"""

from importlib import metadata
import json
import pathlib
import threading

import pandas as pd

from mml.api import results
from mml.artifacts import store as store_module
from mml.contracts import errors
from mml.contracts import events
from mml.contracts import experiments
from mml.contracts import tasks
from mml.core import context as context_module
from mml.core import stages as stages_module
from mml.core import state
from mml.evaluation import metrics
from mml.evaluation import selection
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
    ) -> None:
        """Initializes the orchestrator.

        Args:
            spec: Project specification.
            frame: In-memory dataset, if not read from `spec.data.uri`.
            registry: Model registry; defaults to the built-in adapters.
            policy: Selection policy; defaults to best mean CV score.
            stages: Stage sequence; defaults to the standard workflow.
        """
        self.spec = spec
        self.frame = frame
        self.registry = (
            registry or registry_module.ModelRegistry.with_builtins()
        )
        self.policy = policy or selection.BestMeanPolicy()
        self.stages = stages or stages_module.default_stages()
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
            cancel_event=self.cancel_event,
            input_frame=self.frame,
            store=store,
            log=log,
            approved_plan=approved_plan,
        )

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
        stages_module.IngestStage().execute(context)
        stages_module.ProfileStage().execute(context)
        context.plan = stages_module.build_plan(context)
        return context

    def run(
        self,
        output_dir: str | pathlib.Path = "runs",
        approved: experiments.ExperimentPlan | None = None,
    ) -> results.RunResult:
        """Executes the full workflow.

        Args:
            output_dir: Parent directory for run directories.
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
        log = events.EventLog(store.path("events.jsonl"))
        machine = state.StateMachine(store.run_id, log, self.spec.config_hash())
        store.write_json("config.json", json.loads(self.spec.canonical_json()))
        store.write_text("environment.lock", environment_lock())
        context = self.new_context(store=store, log=log, approved_plan=approved)
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
            return context.result
        except errors.CancelledError as error:
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
