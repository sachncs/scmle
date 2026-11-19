"""What a run is given: its services, fixed when the run starts.

`Services` holds the inputs and the collaborators every stage reads: the
specification, the registries, the store and log, the tracker, the trial
checkpoints of an interrupted run, the execution backend, the governor of
the resource limits, the cancellation event, and the clock and the random
streams. It is frozen: a stage reads it and never assigns to it. What a run
records (files, events, trials, decisions) goes through `journal.Journal`.

The clock and the random streams are objects so a test can replace them
(`Clock`, `Random`); the run uses the system clock and the keyed streams of
`runtime.randomness`.
"""

import dataclasses
import datetime
import threading
import time
from typing import Any

import numpy
import pandas

import scmle.artifacts.store
from scmle.contracts import capabilities
from scmle.contracts import errors
from scmle.contracts import events
from scmle.contracts import experiments
from scmle.contracts import tasks
from scmle.evaluation import metrics
from scmle.evaluation import selection
import scmle.models.registry
from scmle.pipeline import roster
from scmle.runtime import limits
from scmle.runtime import randomness
from scmle.runtime import resource_observation
from scmle.runtime import scheduler
from scmle.storage import stores


class Clock:
    """The run's source of time; a test replaces it with a fixed one."""

    def monotonic(self) -> float:
        """Returns seconds on the monotonic clock, for durations."""
        return time.monotonic()

    def now(self) -> datetime.datetime:
        """Returns the current UTC time, for timestamps."""
        return datetime.datetime.now(datetime.UTC)


class Random:
    """The run's source of randomness; a test replaces it.

    Every stream is keyed by the run seed, a component name and an index
    (`runtime.randomness`), so a stream does not depend on the order in which
    the stages draw.
    """

    def generator(
        self, seed: int, component: str, index: int = 0
    ) -> numpy.random.Generator:
        """Returns the random generator of one component of a run.

        Args:
            seed: The run's seed.
            component: Stable name of the component drawing.
            index: Repeat counter for a component that draws several streams.

        Returns:
            A generator that depends only on the three arguments.
        """
        return randomness.component_rng(seed, component, index)


@dataclasses.dataclass(frozen=True)
class Services:
    """The fixed inputs and collaborators of one run.

    When `store` is None (the plan preview) the journal skips every write
    and event.

    Attributes:
        spec: Project specification.
        registry: Model registry.
        catalog: Metric catalog.
        policy: Model selection policy.
        governor: Applies and reports the resource limits; its start time
            starts the time budget.
        capability_registry: Declared capabilities that planning checks the
            run against (the tabular beta registry by default).
        capability_observation: Runtime resource facts captured before the
            capability gate.
        store: Run directory, or None for previews.
        log: Event log, or None for previews.
        tracker: Experiment tracking backend (no-op by default).
        checkpoints: Completed trials from an interrupted run, by trial id,
            reused instead of recomputed when resuming.
        backend: Where independent trials run (sequential by default).
        cancel_event: Set to request cancellation.
        input_frame: In-memory data, if the project was built from one.
        approved_plan: Human-approved plan restricting candidates.
        history: Past runs available for meta-learning.
        clock: Where the stages read the time.
        random: Where the stages draw random streams.
    """

    spec: tasks.Config
    registry: scmle.models.registry.Catalog
    catalog: metrics.Repertoire
    policy: selection.Selector
    governor: limits.Governor
    capability_registry: capabilities.Registry = dataclasses.field(
        default_factory=roster.enrol
    )
    capability_observation: resource_observation.Observation | None = None
    store: scmle.artifacts.store.Workspace | None = None
    log: events.Log | None = None
    tracker: stores.Store = dataclasses.field(default_factory=stores.Null)
    checkpoints: dict[str, experiments.Trial] = dataclasses.field(
        default_factory=dict
    )
    backend: scheduler.Executor = dataclasses.field(
        default_factory=scheduler.Serial
    )
    cancel_event: threading.Event = dataclasses.field(
        default_factory=threading.Event
    )
    input_frame: pandas.DataFrame | None = None
    approved_plan: experiments.Plan | None = None
    history: tuple[experiments.Precedent, ...] = ()
    clock: Clock = dataclasses.field(default_factory=Clock)
    random: Random = dataclasses.field(default_factory=Random)

    @property
    def deadline(self) -> float:
        """Monotonic time at which the budget is exhausted."""
        return (
            self.governor.started_at + self.spec.resources.time_budget_seconds
        )

    @property
    def workspace(self) -> scmle.artifacts.store.Workspace:
        """The run store.

        Raises:
            ScmleError: For a preview, which has none.
        """
        if self.store is None:
            raise errors.ScmleError(
                "'store' is unavailable: its stage has not run yet"
            )
        return self.store

    @property
    def directory(self) -> str:
        """Runs directory as a string, empty for previews."""
        return str(self.store.directory) if self.store else ""

    def elapsed(self) -> float:
        """Returns the seconds the run has spent of its time budget."""
        return self.clock.monotonic() - self.governor.started_at

    def check(self) -> None:
        """Raises if cancellation was requested.

        Raises:
            CancelledError: Carrying the run directory in its details.
        """
        if self.cancel_event.is_set():
            raise errors.CancelledError(
                "run cancelled", {"run_dir": self.directory}
            )


def start(
    spec: tasks.Config,
    registry: scmle.models.registry.Catalog,
    policy: selection.Selector,
    *,
    store: scmle.artifacts.store.Workspace | None = None,
    capability_registry: capabilities.Registry | None = None,
    started: float | None = None,
    **fields: Any,
) -> Services:
    """Builds the services of a run, observing its runtime resources.

    The resource facts are observed and bound to the capability registry,
    the execution backend is chosen from the resource specification and the
    governor's clock starts the time budget.

    Args:
        spec: Project specification.
        registry: Model registry.
        policy: Model selection policy.
        store: Run directory, or None for a preview.
        capability_registry: Declared capabilities, or None for the
            tabular beta registry.
        started: Monotonic time the time budget counts from (kept across
            a resume), or None for now.
        **fields: The other `Services` fields.

    Returns:
        The services, with the metric catalog.
    """
    observation = resource_observation.observe_runtime_resources(
        store.directory if store is not None else None
    )
    declared = capability_registry or roster.enrol()
    declared.bind(observation.environment)
    governor = (
        limits.Governor(spec.resources)
        if started is None
        else limits.Governor(spec.resources, started)
    )
    return Services(
        spec,
        registry,
        metrics.CATALOG,
        policy,
        governor,
        capability_registry=declared,
        capability_observation=observation,
        store=store,
        backend=scheduler.select(
            limits.Governor(spec.resources).workers,
            spec.resources.executor,
            spec.resources.cluster_address,
        ),
        **fields,
    )
