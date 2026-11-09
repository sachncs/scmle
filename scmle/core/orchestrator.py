"""Orchestrator: runs the stage sequence under the state machine.

Stages (state machine): VALIDATING -> PROFILING -> PLANNING -> TRAINING ->
SELECTING -> TESTING -> PACKAGING -> COMPLETED, with FAILED / CANCELLED /
BLOCKED as exits. Every transition emits an event, and finished trials are
checkpointed to disk as they complete.
"""

import functools
import hashlib
from importlib import metadata
import json
import pathlib
import threading
from typing import Any
from typing import Self

import pandas

from scmle.api import results
import scmle.artifacts.store
from scmle.contracts import capabilities
from scmle.contracts import errors
from scmle.contracts import events
from scmle.contracts import experiments
from scmle.contracts import tasks
from scmle.core import compiler
from scmle.core import limits
from scmle.core import policies
from scmle.core import resource_observation
from scmle.core import scheduler
from scmle.core import state
import scmle.core.context
import scmle.core.stages
from scmle.evaluation import metrics
from scmle.evaluation import selection
from scmle.experimentation import stores
import scmle.models.registry


def environment_lock() -> str:
    """Lists installed distributions as sorted `name==version` lines.

    Returns:
        One `name==version` line per installed distribution, sorted and
        joined with newlines.
    """
    pins = {
        f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
    }
    return "\n".join(sorted(pins))


class Orchestrator:
    """Compiles a project specification into a run and executes it."""

    def __init__(
        self,
        spec: tasks.Config,
        frame: pandas.DataFrame | None = None,
        registry: scmle.models.registry.Catalog | None = None,
        policy: selection.Selector | None = None,
        stages: tuple[scmle.core.stages.Stage, ...] | None = None,
        tracker: stores.ExperimentStore | None = None,
        checkpoints: tuple[policies.Reviewer, ...] = (),
        capability_registry: capabilities.Registry | None = None,
    ) -> None:
        """Initializes the orchestrator.

        Args:
            spec: Project specification.
            frame: In-memory dataset, if not read from `spec.data.uri`.
            registry: Model registry; defaults to the built-in adapters plus
                every installed plugin that passes the conformance suite.
            policy: Selection policy; defaults to `MeanSelector` (best mean
                cross-validation score with the configured tie rule).
            stages: Stage sequence; defaults to the standard workflow.
            checkpoints: Approval gates, each consulted after the stages in
                its own `gates` (planning and selection by default); any
                rejection blocks the run.
            tracker: Experiment store; defaults to a SQLite database next to
                the run directories.
            capability_registry: Explicit capability declarations; defaults
                to the built-in tabular beta registry.
        """
        self.spec = spec
        self.frame = frame
        if registry is None:
            registry = scmle.models.registry.Catalog.with_builtins()
            registry.load_plugins()
        self.registry = registry
        self.policy = policy or selection.MeanSelector(spec.selection)
        self.stages = compiler.compile_workflow(
            stages or scmle.core.stages.default_stages()
        )
        self.tracker = tracker
        self.checkpoints = checkpoints
        self.capability_registry = capability_registry
        self.tracking_path: pathlib.Path | None = None
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        """Requests cooperative cancellation of a running workflow."""
        self.cancel_event.set()

    def new_context(
        self,
        store: scmle.artifacts.store.RunStore | None = None,
        log: events.Log | None = None,
        approved_plan: experiments.Plan | None = None,
    ) -> scmle.core.context.Context:
        """Creates a run context wired with this orchestrator's services.

        Args:
            store: Run directory, or None for a side-effect-free preview.
            log: Event log, or None for a preview.
            approved_plan: Plan restricting which models may be considered.

        Returns:
            A fresh context.
        """
        return scmle.core.context.Context(
            spec=self.spec,
            registry=self.registry,
            catalog=metrics.CATALOG,
            policy=self.policy,
            capability_registry=(
                self.capability_registry
                if self.capability_registry is not None
                else scmle.core.context.tabular_beta_capability_registry()
            ),
            capability_observation=resource_observation.observe_runtime_resources(
                store.directory if store is not None else None
            ),
            backend=scheduler.create_backend(
                limits.Governor(self.spec.resources).workers,
                self.spec.resources.executor,
                self.spec.resources.cluster_address,
            ),
            cancel_event=self.cancel_event,
            input_frame=self.frame,
            store=store,
            log=log,
            approved_plan=approved_plan,
        )

    def load_history(
        self, tracker: stores.ExperimentStore | None
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
            tracker = stores.SqliteExperimentStore(path)
        return tuple(tracker.history())

    def apply_plan(self, approved: experiments.Plan | None) -> None:
        """Makes an approved or modified plan binding on the specification.

        Replaces `self.spec` with a copy whose validation strategy, time
        budget and metric are those of `approved`, validated with the rest
        of the specification. A plan that contradicts the specification,
        for example group validation without a group column, fails that
        validation with a `ValidationError`. The candidate models and task of
        the plan are not applied here; the stages read them from the
        context's approved plan.

        Args:
            approved: The plan, or None to keep the specification as is.
        """
        if approved is None:
            return
        record = self.spec.model_dump()
        record["validation"]["strategy"] = approved.validation_strategy
        record["resources"]["time_budget_seconds"] = (
            approved.optimization_budget_seconds
        )
        record["metric"] = approved.metric
        self.spec = tasks.Config.model_validate(record)

    def prepare(
        self, approved: experiments.Plan | None = None
    ) -> scmle.core.context.Context:
        """Ingests, profiles and plans without training or writing files.

        Args:
            approved: Plan restricting which models may be considered.

        Applies `approved` to the specification first (see `apply_plan`).

        Returns:
            A context, with no run store or log, whose `profile` and `plan`
            are populated.
        """
        self.apply_plan(approved)
        context = self.new_context(approved_plan=approved)
        context.history = self.load_history(self.tracker)
        scmle.core.stages.IngestStage().execute(context)
        scmle.core.stages.ProfileStage().execute(context)
        context.plan = scmle.core.stages.build_plan(context)
        scmle.core.stages.advise(context)
        return context

    def run(
        self,
        output_dir: str | pathlib.Path = "runs",
        approved: experiments.Plan | None = None,
        *,
        parameter_estimation_declaration: dict[str, Any] | None = None,
    ) -> results.Run:
        """Executes the full workflow in a new run directory.

        Failures surface from `execute`: a run that cannot safely continue
        raises `BlockedError`, a cancelled run raises `CancelledError`, and
        any other failure is raised as `ScmleError` (unexpected errors are
        wrapped). The run directory is in the error's `details["run_dir"]`.

        Args:
            output_dir: Parent directory for run directories and, unless a
                tracker was given, the tracking database.
            approved: Approved or modified plan restricting candidates.
            parameter_estimation_declaration: Declaration for the optional
                parameter-estimation evidence track, written (with the
                project config hash and its digest) to the run directory;
                a disabled declaration is written when None.

        Returns:
            The completed run.
        """
        self.apply_plan(approved)
        store = scmle.artifacts.store.RunStore.create(output_dir)
        declaration = dict(
            parameter_estimation_declaration
            or {"schema_version": 1, "enabled": False}
        )
        declaration.pop("digest", None)
        declaration["project_config_hash"] = self.spec.config_hash()
        digest = hashlib.sha256(
            json.dumps(
                declaration, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        store.write_json(
            "parameter_estimation_declaration.json",
            {**declaration, "digest": digest},
        )
        store.write_text("parameter_estimation_declaration.lock", f"{digest}\n")
        store.write_json("config.json", json.loads(self.spec.canonical_json()))
        store.write_text("environment.lock", environment_lock())
        return self.execute(store, approved, {}, "run created")

    def resume(self, run_directory: str | pathlib.Path) -> results.Run:
        """Continues an interrupted run, reusing its completed trials.

        The configuration must be identical (same hash). Stages are
        re-executed; trials whose checkpoints were saved in the run directory
        are reused instead of recomputed, and a saved `execution_plan.json`
        is applied as the approved plan.

        Args:
            run_directory: Directory of a cancelled, failed or crashed run.

        Returns:
            The completed run.

        Raises:
            ConfigError: If the run already completed or the saved
                configuration differs from this orchestrator's.
        """
        store = scmle.artifacts.store.RunStore.open(run_directory)
        saved = tasks.Config.model_validate_json(
            store.path("config.json").read_text(encoding="utf-8")
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
            experiments.Plan.from_dict(
                json.loads(plan_path.read_text(encoding="utf-8"))
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
        store = scmle.artifacts.store.RunStore.open(run_directory)
        spec = tasks.Config.model_validate_json(
            store.path("config.json").read_text(encoding="utf-8")
        )
        return cls(spec, **kwargs)

    def load_checkpoints(
        self, store: scmle.artifacts.store.RunStore
    ) -> dict[str, experiments.Trial]:
        """Reads completed trial checkpoints from a run directory.

        Args:
            store: The run directory.

        Returns:
            The stored trial results keyed by trial id, read from the run's
            `trials/trial_*.json` files in file-name order.
        """
        checkpoints: dict[str, experiments.Trial] = {}
        for path in sorted(store.path("trials").glob("trial_*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            record["status"] = experiments.TrialStatus(record["status"])
            checkpoints[record["trial_id"]] = experiments.Trial(**record)
        return checkpoints

    def recoverable(self, store: scmle.artifacts.store.RunStore) -> list[str]:
        """Lists the trial checkpoints a failed or interrupted run can resume.

        Args:
            store: The run directory.

        Returns:
            The sorted names (without extension) of the trial checkpoint
            files in the run, empty if there is no trials directory.
        """
        trials_dir = store.path("trials")
        if not trials_dir.exists():
            return []
        return sorted(path.stem for path in trials_dir.glob("trial_*.json"))

    def consult_checkpoints(
        self, gate: state.RunState, context: scmle.core.context.Context
    ) -> None:
        """Asks every checkpoint registered for `gate` and audits the verdicts.

        Args:
            gate: The state of the stage that just completed.
            context: The run context, used to build each review and to record
                the verdict.

        Raises:
            BlockedError: If any checkpoint rejects.
        """
        for checkpoint in self.checkpoints:
            if gate not in checkpoint.gates:
                continue
            verdict = checkpoint.review(policies.describe(gate, context))
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
        store: scmle.artifacts.store.RunStore,
        approved: experiments.Plan | None,
        checkpoints: dict[str, experiments.Trial],
        reason: str,
    ) -> results.Run:
        """Runs the stage sequence under the state machine.

        The run ends in the completed, cancelled, blocked or failed state. On a
        cancellation or failure the error's `details` gain the run directory and
        the recoverable trial checkpoints.

        Args:
            store: Run directory.
            approved: Plan restricting candidates.
            checkpoints: Reusable trial results (empty for a fresh run).
            reason: Reason attached to the first state event.

        Returns:
            The completed run.

        Raises:
            BlockedError: If a stage or an approval gate blocks the run.
            CancelledError: If the run is cancelled, or interrupted from the
                keyboard.
            ScmleError: If a stage fails; any other exception is wrapped in it.
        """
        log = events.Log(store.path("events.jsonl"))
        tracker = self.tracker or stores.SqliteExperimentStore(
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
        machine = state.Tracker(
            store.run_id, log, self.spec.config_hash(), reason
        )
        context = self.new_context(store=store, log=log, approved_plan=approved)
        context.tracker = tracker
        context.history = self.load_history(tracker)
        context.checkpoints = checkpoints
        for name, failure in self.registry.plugin_failures.items():
            context.emit("plugin.rejected", f"{name}: {failure}")
        try:
            for stage in self.stages:
                context.check_cancelled()
                machine.transition(stage.state, stage.describe(context))
                if stage.state is state.RunState.PLANNING:
                    context.persist_capability_observation()
                stage.execute(context)
                self.consult_checkpoints(stage.state, context)
            machine.transition(
                state.RunState.COMPLETED,
                "run complete",
                ("model/", "evaluation.json"),
            )
            return scmle.core.context.require(context.result, "result")
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
        except errors.ScmleError as error:
            error.details["run_dir"] = str(store.directory)
            error.details["recoverable_trials"] = self.recoverable(store)
            machine.transition(
                state.RunState.FAILED, f"{error.category}: {error}"
            )
            raise
        except Exception as error:
            machine.transition(state.RunState.FAILED, f"internal: {error!r}")
            raise errors.ScmleError(
                str(error),
                {
                    "run_dir": str(store.directory),
                    "recoverable_trials": self.recoverable(store),
                },
            ) from error
        finally:
            context.close()
