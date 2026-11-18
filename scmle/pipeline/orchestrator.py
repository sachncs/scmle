"""Orchestrator: runs the stage sequence under the state machine.

Stages (state machine): VALIDATING -> PROFILING -> PLANNING -> TRAINING ->
SELECTING -> TESTING -> PACKAGING -> COMPLETED, with FAILED / CANCELLED /
BLOCKED as exits. Every transition emits an event, finished trials are
checkpointed to disk as they complete, each finished stage leaves a record
and a completed run its marker (`archive`).
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

from scmle.artifacts import results
import scmle.artifacts.store
from scmle.assurance.custody import journal
from scmle.contracts import capabilities
from scmle.contracts import errors
from scmle.contracts import events
from scmle.contracts import experiments
from scmle.contracts import records
from scmle.contracts import tasks
from scmle.evaluation import metrics
from scmle.evaluation import selection
import scmle.models.registry
from scmle.pipeline import archive
from scmle.pipeline import engine
from scmle.pipeline import policies
from scmle.pipeline import roster
import scmle.pipeline.context
import scmle.pipeline.stages
import scmle.runtime.engine
from scmle.runtime import limits
from scmle.runtime import logs
from scmle.runtime import resource_observation
from scmle.runtime import scheduler
from scmle.runtime import state
from scmle.storage import stores


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
        stages: tuple[scmle.pipeline.stages.Stage, ...] | None = None,
        tracker: stores.Store | None = None,
        checkpoints: tuple[policies.Reviewer, ...] = (),
        capability_registry: capabilities.Registry | None = None,
    ) -> None:
        """Initializes the orchestrator.

        Args:
            spec: Project specification.
            frame: In-memory dataset, if not read from `spec.data.uri`.
            registry: Model registry; defaults to the built-in adapters plus
                every installed plugin that passes the conformance suite.
            policy: Selection policy; defaults to `Averager` (best mean
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
            registry = scmle.models.registry.Catalog.stock()
            registry.load_plugins()
        self.registry = registry
        self.policy = policy or selection.Averager(spec.selection)
        self.stages = scmle.runtime.engine.order(
            stages or scmle.pipeline.stages.default_stages()
        )
        self.pipeline = engine.Pipeline(self.stages)
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
        store: scmle.artifacts.store.Workspace | None = None,
        log: events.Log | None = None,
        approved_plan: experiments.Plan | None = None,
    ) -> scmle.pipeline.context.Context:
        """Creates a run context wired with this orchestrator's services.

        Args:
            store: Run directory, or None for a side-effect-free preview.
            log: Event log, or None for a preview.
            approved_plan: Plan restricting which models may be considered.

        Returns:
            A fresh context.
        """
        return scmle.pipeline.context.Context(
            spec=self.spec,
            registry=self.registry,
            catalog=metrics.CATALOG,
            policy=self.policy,
            capability_registry=self.capability_registry or roster.enrol(),
            capability_observation=resource_observation.observe_runtime_resources(
                store.directory if store is not None else None
            ),
            backend=scheduler.select(
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

    def prepare(
        self, approved: experiments.Plan | None = None
    ) -> scmle.pipeline.context.Context:
        """Ingests, profiles and plans without training or writing files.

        Runs the planning stage and the stages it needs, with no state
        machine, reviewers, run store or log.

        Args:
            approved: Plan restricting which models may be considered.

        Applies `approved` to the specification first (see `bind`).

        Returns:
            A context, with no run store or log, whose `profile` and `plan`
            are populated.
        """
        self.spec = bind(self.spec, approved)
        context = self.new_context(approved_plan=approved)
        context.history = recall(self.tracker, self.tracking_path)
        self.pipeline = engine.Pipeline(self.stages)
        self.pipeline.advance(context, until=scmle.pipeline.stages.Planning)
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
        self.spec = bind(self.spec, approved)
        store = scmle.artifacts.store.Workspace.create(output_dir)
        declare(store, parameter_estimation_declaration, self.spec)
        store.write_json("config.json", json.loads(self.spec.canonize()))
        store.write_text("environment.lock", environment_lock())
        journal.Journal.locate(store.directory).start(self.spec.config_hash())
        docket = archive.Directory(store.directory, self.spec.config_hash())
        with logs.bind(run=store.run_id):
            return self.execute(store, approved, docket, "run created")

    def resume(self, run_directory: str | pathlib.Path) -> results.Run:
        """Continues an interrupted run, keeping the time it already spent.

        The configuration must be identical (same hash). The stage records
        are verified and the elapsed seconds they carry are kept, so the
        time budget is not restarted. A finished stage whose product is
        kept, authenticates with the signing key and was made from the same
        inputs is not run again: its product is loaded (`products`). The
        other stages before the locked test set are derived again from the
        same data and seeds; trials whose checkpoints were saved in the run
        directory are reused instead of recomputed, and a saved
        `execution_plan.json` is applied as the approved plan. The test set
        is never opened twice: the custody journal refuses a run whose test
        set was opened.

        Args:
            run_directory: Directory of a cancelled, failed or crashed run.

        Returns:
            The completed run.

        Raises:
            ConfigError: If the run already completed (it has
                `complete.json`) or the saved configuration differs from this
                orchestrator's.
            BlockedError: If a stage record was edited or has another
                format, or the run's custody journal (kept outside the run
                directory) is missing or edited, or records that its test set
                was opened (`Journal.admit`).
        """
        store = scmle.artifacts.store.Workspace.open(run_directory)
        docket = archive.Directory.open(store.directory, self.spec)
        with logs.bind(run=store.run_id):
            return self.execute(store, docket.plan(), docket, "run resumed")

    @classmethod
    def reload(cls, run_directory: str | pathlib.Path, **kwargs: Any) -> Self:
        """Builds an orchestrator from a saved run's configuration.

        Args:
            run_directory: An existing run directory.
            **kwargs: Passed to the constructor (registry, policy, ...).

        Returns:
            An orchestrator whose spec is the run's saved configuration.
        """
        store = scmle.artifacts.store.Workspace.open(run_directory)
        return cls(archive.saved(store.directory), **kwargs)

    def consult_checkpoints(
        self, gate: state.Progress, context: scmle.pipeline.context.Context
    ) -> None:
        """Asks every checkpoint registered for `gate` and audits the verdicts.

        Args:
            gate: The state of the stage that just completed.
            context: The run context, used to build each review and to record
                the verdict.

        A gate the run's custody journal records as rejected stays rejected
        until a checkpoint of this orchestrator approves it, so resuming
        without the checkpoint, or after deleting `events.jsonl`, does not
        lift a rejection.

        Raises:
            BlockedError: If any checkpoint rejects, or the gate was rejected
                before and no checkpoint here reviews it.
        """
        record = journal.Journal.locate(context.workspace.directory)
        reviewers = [c for c in self.checkpoints if gate in c.gates]
        earlier = record.rejected().get(gate.value)
        if earlier is not None and not reviewers:
            raise errors.BlockedError(
                f"{gate.value} was rejected by {earlier}; resume with a "
                "checkpoint that reviews it"
            )
        for checkpoint in reviewers:
            verdict = checkpoint.review(policies.describe(gate, context))
            context.emit(
                f"policy.{'approved' if verdict.approved else 'rejected'}",
                verdict.reason,
                data={"gate": gate.value, "reviewer": verdict.reviewer},
            )
            record.decide(
                gate.value, verdict.approved, verdict.reviewer, verdict.reason
            )
            if not verdict.approved:
                raise errors.BlockedError(
                    f"{gate.value} not approved by {verdict.reviewer}: "
                    f"{verdict.reason or 'no reason given'}"
                )

    def execute(
        self,
        store: scmle.artifacts.store.Workspace,
        approved: experiments.Plan | None,
        docket: archive.Directory,
        reason: str,
    ) -> results.Run:
        """Runs the stage sequence under the state machine.

        The run ends in the completed, cancelled, blocked or failed state. On a
        cancellation or failure the error's `details` gain the run directory and
        the recoverable trial checkpoints.

        Args:
            store: Run directory.
            approved: Plan restricting candidates.
            docket: The run's stage records; their clock, which counts the
                seconds earlier attempts spent, starts the time budget, and
                their trial checkpoints are reused.
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
        tracker = enlist(self.tracker, store, self.spec, log)
        machine = state.Tracker(
            store.run_id, log, self.spec.config_hash(), reason
        )
        context = self.new_context(store=store, log=log, approved_plan=approved)
        context.started_at = docket.started
        context.tracker = tracker
        context.history = recall(tracker, self.tracking_path)
        context.checkpoints = docket.trials()
        for name, failure in self.registry.plugin_failures.items():
            context.emit("plugin.rejected", f"{name}: {failure}")
        self.pipeline = engine.Pipeline(self.stages, docket)
        try:
            self.pipeline.run(
                context,
                machine,
                self.consult_checkpoints,
                lambda: {
                    "run_dir": str(store.directory),
                    "recoverable_trials": docket.recoverable(),
                },
                ("model/", "evaluation.json"),
            )
            return scmle.pipeline.context.require(context.result, "result")
        finally:
            context.close()


def enlist(
    tracker: stores.Store | None,
    store: scmle.artifacts.store.Workspace,
    spec: tasks.Config,
    log: events.Log,
) -> stores.Store:
    """Opens the run's row in the experiment store and has it follow the log.

    Args:
        tracker: The experiment store, or None for a SQLite database next to
            the run directories.
        store: The run directory.
        spec: The project specification.
        log: The run's event log.

    Returns:
        The experiment store.
    """
    tracker = tracker or stores.SqliteExperimentStore(
        store.directory.parent / "experiments.sqlite"
    )
    tracker.commence(
        store.run_id, spec.config_hash(), str(store.directory), events.stamp()
    )
    tracker.revise(store.run_id, config_json=spec.canonize())
    log.listeners.append(functools.partial(tracker.listen, store.run_id))
    return tracker


def recall(
    tracker: stores.Store | None, path: pathlib.Path | None
) -> tuple[experiments.Precedent, ...]:
    """Reads past runs for meta-learning without creating a database.

    Args:
        tracker: The store to read, or None to look for an existing database
            at `path`.
        path: Where the tracking database is, or None.

    Returns:
        Past runs, empty when no tracking database exists yet.
    """
    if tracker is None:
        if path is None or not path.exists():
            return ()
        tracker = stores.SqliteExperimentStore(path)
    return tuple(tracker.history())


def bind(spec: tasks.Config, approved: experiments.Plan | None) -> tasks.Config:
    """Makes an approved or modified plan binding on a specification.

    The copy's validation strategy, time budget and metric are those of
    `approved`, validated with the rest of the specification. A plan that
    contradicts the specification, for example group validation without a
    group column, fails that validation with a `ValidationError`. The
    candidate models and task of the plan are not applied here; the stages
    read them from the context's approved plan.

    Args:
        spec: The specification.
        approved: The plan, or None to keep the specification as is.

    Returns:
        The specification the run uses.
    """
    if approved is None:
        return spec
    record = spec.model_dump()
    record["validation"]["strategy"] = approved.validation
    record["resources"]["budget"] = approved.budget
    record["metric"] = approved.metric
    return tasks.Config.model_validate(record)


def declare(
    store: scmle.artifacts.store.Workspace,
    declaration: records.Layout | None,
    spec: tasks.Config,
) -> None:
    """Writes the parameter-estimation declaration and its lock.

    Args:
        store: The run directory.
        declaration: The declaration, or None for a disabled one.
        spec: The project specification, whose hash the declaration names.
    """
    body: records.Layout = dict(
        declaration or {"schema_version": 1, "enabled": False}
    )
    body.pop("digest", None)
    body["project_config_hash"] = spec.config_hash()
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    store.write_json(
        "parameter_estimation_declaration.json", {**body, "digest": digest}
    )
    store.write_text("parameter_estimation_declaration.lock", f"{digest}\n")
