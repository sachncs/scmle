"""Fault catalogue, declared objectives and the report of a recovery run.

Each `Fault` names what is broken, which stateful components it touches and
what a run on a managed platform would need. `Runner` executes the
scenarios it is given, records for each component the measured recovery
time (RTO) and the acknowledged data lost (RPO) with the method used,
compares them with the objectives declared *before* the run, and lists
every fault it did not run with the reason. A fault with no scenario is
``not_run``, never ``passed``.

Objectives are declared in `OBJECTIVES` and are not tuned to a result. A
local pass says the local service, emulators and files behaved within them
under the stated method. It says nothing about a cloud, a machine crash or
power loss, a network partition or the availability of anything.

A slow shared CI runner can start a Python service several times slower than
a workstation. ``SCMLE_RECOVERY_TIME_SCALE`` (default 1.0) multiplies the
waits of the scenarios and the RTO each objective allows; the declared
objectives are not edited, and the scale used is written into every
measurement and into the report so a scaled run is visibly not a full-speed
claim.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import math
import os
import time
import traceback
from typing import Any

PASSED = "passed"
FAILED = "failed"
NOT_RUN = "not_run"
TIME_SCALE_ENV = "SCMLE_RECOVERY_TIME_SCALE"


def time_scale() -> float:
    """The factor applied to scenario waits and allowed RTO (default 1.0).

    Returns:
        The positive factor read from the environment, or 1.0 if the
        variable is unset or empty.

    Raises:
        ValueError: If the environment variable is not a positive number.
    """
    text = os.environ.get(TIME_SCALE_ENV, "").strip()
    if not text:
        return 1.0
    try:
        scale = float(text)
    except ValueError:
        raise ValueError(
            f"{TIME_SCALE_ENV} is not a number: {text!r}"
        ) from None
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"{TIME_SCALE_ENV} must be positive, not {text!r}")
    return scale


def scaled(seconds: float) -> float:
    """A wait in seconds multiplied by the run's time scale.

    Args:
        seconds: The unscaled wait.

    Returns:
        `seconds` times the time scale.
    """
    return seconds * time_scale()


@dataclasses.dataclass(frozen=True, slots=True)
class Objective:
    """What recovery of one stateful component must achieve.

    Attributes:
        component: Component name.
        rto_seconds: Longest tolerated time from the fault to service.
        rpo_lost: Most acknowledged units that may be lost (0 = none).
        rpo_unit: What a unit is.
        scope: What the objective covers and what it does not.
    """

    component: str
    rto_seconds: float
    rpo_lost: int
    rpo_unit: str
    scope: str


OBJECTIVES = (
    Objective(
        "service",
        30.0,
        0,
        "requests answered differently after restart",
        "the serving process; stateless, so recovery is a restart to ready",
    ),
    Objective(
        "ledger",
        10.0,
        0,
        "acknowledged ledger rows",
        "reload and chain verification after the writer is killed; a "
        "machine crash or power loss is not covered (no fsync is claimed)",
    ),
    Objective(
        "registry_state",
        10.0,
        0,
        "acknowledged state writes",
        "atomic replacement of a state file with its writer killed",
    ),
    Objective(
        "object_store",
        30.0,
        0,
        "acknowledged objects",
        "the local object-store emulator, from access restored to a "
        "verified bundle serving; not a managed object store",
    ),
    Objective(
        "secrets",
        60.0,
        0,
        "none: secrets are held outside this system",
        "from detection of a credential mismatch to ready under the "
        "corrected or rotated key; a local environment variable, not a "
        "secret manager",
    ),
    Objective(
        "telemetry_spool",
        30.0,
        0,
        "spooled samples",
        "from the destination returning to the spool drained; delivery is "
        "at least once, so duplicates are possible",
    ),
    Objective(
        "job_state",
        60.0,
        0,
        "completed steps",
        "a workflow run killed mid-step and continued from its persisted "
        "state, including the wait for the lease to expire",
    ),
)
OBJECTIVE_OF = {item.component: item for item in OBJECTIVES}


@dataclasses.dataclass(frozen=True, slots=True)
class Fault:
    """One entry of the catalogue.

    Attributes:
        id: Stable identifier.
        title: Short name.
        summary: What is broken and what recovery means.
        components: Stateful components it touches.
        local: Whether a local scenario can run it.
        local_limits: What the local scenario is not.
        cloud_needs: What a run on a managed platform needs.
    """

    id: str
    title: str
    summary: str
    components: tuple[str, ...]
    local: bool
    local_limits: str
    cloud_needs: str


CATALOGUE = (
    Fault(
        "process_kill",
        "Process kill",
        "SIGKILL the serving process; a supervisor starts a new one. Also "
        "kills a state writer at random points.",
        ("service", "registry_state"),
        local=True,
        local_limits=(
            "The runner is the supervisor; no orchestrator restarts anything."
        ),
        cloud_needs=(
            "Kill a task or revision instance and read the platform's own "
            "restart and health record."
        ),
    ),
    Fault(
        "unhealthy_container_rolling_restart",
        "Unhealthy instance and rolling restart",
        "A replacement that never becomes ready is not admitted; a ready "
        "replacement takes over without dropped requests.",
        ("service",),
        local=True,
        local_limits=(
            "Processes, not containers; readiness-gated switching is done by "
            "the runner, not by a managed platform."
        ),
        cloud_needs=(
            "A failing health check and a rolling deployment on a managed "
            "platform with the load balancer's request log."
        ),
    ),
    Fault(
        "invalid_credentials",
        "Invalid credentials",
        "The service starts with a key that does not verify its bundle; it "
        "stays alive and unready until the key is corrected.",
        ("secrets",),
        local=True,
        local_limits=(
            "The credential is the bundle signing key in an environment "
            "variable; no IAM, token issuer or secret manager is involved."
        ),
        cloud_needs=(
            "Revoked or wrong workload identity and secret versions on a "
            "managed platform."
        ),
    ),
    Fault(
        "lost_object_store_access",
        "Lost object-store access",
        "The store stops answering; the running service keeps serving; when "
        "access returns the bundle is fetched again by digest.",
        ("object_store",),
        local=True,
        local_limits=(
            "A local emulator (scmle.deploy.mocks); no IAM, bucket policy, "
            "versioning or replication."
        ),
        cloud_needs=(
            "A denied bucket policy or key on a managed object store and a "
            "restore from versioned objects."
        ),
    ),
    Fault(
        "dependency_outage",
        "Dependency outage",
        "The metrics destination is down; telemetry is spooled and drained "
        "when it returns; serving is unaffected.",
        ("telemetry_spool",),
        local=True,
        local_limits="A local metrics sink; not a managed metrics backend.",
        cloud_needs="An outage or throttling of a managed metrics backend.",
    ),
    Fault(
        "corrupted_bundle",
        "Corrupted bundle",
        "A modified model file never becomes ready; restoring the bundle "
        "from a digest-checked copy recovers.",
        ("service", "object_store"),
        local=True,
        local_limits="The copy is in a local folder and an emulator.",
        cloud_needs=(
            "A corrupted object in a managed object store and an image whose "
            "bundle digest differs from the release record."
        ),
    ),
    Fault(
        "incompatible_schema",
        "Incompatible schema",
        "Requests with a schema the service does not serve are refused with "
        "a precise status; the service keeps serving; the caller recovers by "
        "reading the schema.",
        ("service",),
        local=True,
        local_limits="One local service and one client.",
        cloud_needs=(
            "A client and a model version released on a platform that differ "
            "in schema."
        ),
    ),
    Fault(
        "bad_release",
        "Bad release",
        "A release that serves wrong answers is detected from delayed "
        "labels and rolled back to the previous revision; a release that "
        "never becomes ready is not promoted and the previous revision is "
        "restored.",
        ("service",),
        local=True,
        local_limits=(
            "Revisions are local processes behind a local control plane that "
            "implements the same interface as a managed platform's control "
            "plane."
        ),
        cloud_needs=(
            "The same on a managed platform with its real revisions and "
            "traffic shares."
        ),
    ),
    Fault(
        "job_interruption",
        "Job interruption with persistent state",
        "A workflow run is killed mid-step; after the lease expires it "
        "continues; finished steps are not repeated.",
        ("job_state",),
        local=True,
        local_limits="The local executor and a local state directory.",
        cloud_needs=(
            "Managed workflow executions with their own retry and state."
        ),
    ),
    Fault(
        "audit_log_continuity",
        "Audit-log continuity",
        "The ledger writer is killed; the chain verifies after restart and "
        "continues without a gap.",
        ("ledger",),
        local=True,
        local_limits=(
            "Process kill only; no machine crash, disk full or torn page."
        ),
        cloud_needs=(
            "The ledger on the platform's durable storage across a task "
            "replacement."
        ),
    ),
    Fault(
        "secret_rotation_recovery",
        "Secret rotation",
        "The signing key is rotated; a service with the old key cannot "
        "verify the new bundle and recovers once given the new key.",
        ("secrets",),
        local=True,
        local_limits=(
            "A rotated environment variable; no secret manager versions."
        ),
        cloud_needs=(
            "A rotation in a managed secret store with the workload picking up "
            "the new version."
        ),
    ),
    Fault(
        "rollback_to_managed_revision",
        "Rollback to a managed revision",
        "Traffic returns to the previous revision of the managed service "
        "and the registry follows only after the platform confirms.",
        ("service",),
        local=False,
        local_limits=(
            "The registry logic is exercised in bad_release against local "
            "processes; no managed revision exists locally."
        ),
        cloud_needs=(
            "A previous revision of a managed service with the traffic split "
            "read back from the platform."
        ),
    ),
    Fault(
        "zone_or_region_loss",
        "Zone or region loss",
        "A zone or region of the platform is unavailable.",
        ("service", "object_store", "ledger"),
        local=False,
        local_limits="Nothing local resembles it.",
        cloud_needs="Two zones or regions and a deliberate failover.",
    ),
)
FAULT_IDS = tuple(item.id for item in CATALOGUE)


@dataclasses.dataclass(frozen=True, slots=True)
class Measurement:
    """What was measured for one component in one fault.

    Attributes:
        component: Component name.
        rto_seconds: Seconds from the fault (or its detection, per
            ``method``) to the component serving again; None if it did
            not.
        rpo_lost: Acknowledged units found missing after recovery; None if
            not measured.
        method: How both were measured.
    """

    component: str
    rto_seconds: float | None
    rpo_lost: int | None
    method: str

    def problems(self) -> list[str]:
        """Ways the measurement misses the declared objective.

        Compares the measurement with the objective declared for its
        component; the allowed RTO is multiplied by the time scale.

        Returns:
            One message for each way the measurement misses the declared
            objective: no recovery, a recovery time over the (scaled) objective,
            an unmeasured recovery point or too much lost; empty if it meets it.

        Raises:
            KeyError: If the component has no declared objective.
            ValueError: If the time-scale variable is set to something that
                is not a positive number.
        """
        goal = OBJECTIVE_OF[self.component]
        found = []
        if self.rto_seconds is None:
            found.append(f"{self.component}: did not recover")
        elif self.rto_seconds > scaled(goal.rto_seconds):
            note = (
                ""
                if time_scale() == 1.0
                else f" at time scale {time_scale():g}"
            )
            found.append(
                f"{self.component}: RTO {self.rto_seconds:.2f}s exceeds "
                f"{scaled(goal.rto_seconds):.0f}s{note}"
            )
        if self.rpo_lost is None:
            found.append(f"{self.component}: RPO not measured")
        elif self.rpo_lost > goal.rpo_lost:
            found.append(
                f"{self.component}: {self.rpo_lost} {goal.rpo_unit} lost"
            )
        return found

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready record.

        Returns:
            The component, the measured RTO (rounded to milliseconds) and
            RPO, the objective RTO, RPO and RPO unit, the time scale and the
            method as a dict.

        Raises:
            KeyError: If the component has no declared objective.
            ValueError: If the time-scale variable is set to something that
                is not a positive number.
        """
        return {
            "component": self.component,
            "rto_seconds": None
            if self.rto_seconds is None
            else round(self.rto_seconds, 3),
            "rto_objective_seconds": OBJECTIVE_OF[self.component].rto_seconds,
            "time_scale": time_scale(),
            "rpo_lost": self.rpo_lost,
            "rpo_objective": OBJECTIVE_OF[self.component].rpo_lost,
            "rpo_unit": OBJECTIVE_OF[self.component].rpo_unit,
            "method": self.method,
        }


@dataclasses.dataclass(slots=True)
class Outcome:
    """The outcome of one fault.

    Attributes:
        fault_id: The catalogue id.
        status: passed, failed or not_run.
        environment: Where it ran (or why it did not).
        checks: Named boolean observations, all of which must hold.
        measurements: RTO and RPO per component.
        detail: Facts worth keeping.
        problems: Why it failed.
        seconds: Wall time of the scenario.
    """

    fault_id: str
    status: str = PASSED
    environment: str = ""
    checks: dict[str, bool] = dataclasses.field(default_factory=dict)
    measurements: list[Measurement] = dataclasses.field(default_factory=list)
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)
    problems: list[str] = dataclasses.field(default_factory=list)
    seconds: float = 0.0

    def judge(self) -> None:
        """Sets the problems and the status from the checks and measurements.

        The result fails if any check is false or any measurement has
        problems, and passes otherwise.
        """
        found = [f"check failed: {k}" for k, v in self.checks.items() if not v]
        for item in self.measurements:
            found.extend(item.problems())
        self.problems = found
        self.status = FAILED if found else PASSED

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready record.

        Returns:
            The fault id, status, environment, seconds (rounded to two
            places), checks, measurements as dicts, detail and problems as a
            dict.
        """
        return {
            "fault_id": self.fault_id,
            "status": self.status,
            "environment": self.environment,
            "seconds": round(self.seconds, 2),
            "checks": dict(self.checks),
            "measurements": [m.to_dict() for m in self.measurements],
            "detail": self.detail,
            "problems": list(self.problems),
        }


Scenario = Callable[[], Outcome]


class Runner:
    """Runs scenarios and builds the report."""

    def __init__(
        self,
        scenarios: Mapping[str, Scenario],
        catalogue: Sequence[Fault] = CATALOGUE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Binds scenarios to the catalogue.

        Args:
            scenarios: Scenario functions by fault id.
            catalogue: The faults that exist.
            clock: Source of the monotonic time, in seconds.

        Raises:
            ValueError: If a scenario names a fault that is not in the
                catalogue or that the catalogue says cannot run locally.
        """
        known = {fault.id: fault for fault in catalogue}
        for name in scenarios:
            if name not in known:
                raise ValueError(f"{name} is not in the catalogue")
            if not known[name].local:
                raise ValueError(f"{name} cannot run locally")
        self.scenarios = dict(scenarios)
        self.catalogue = tuple(catalogue)
        self.clock = clock

    def run(self, only: Sequence[str] | None = None) -> dict[str, Any]:
        """Runs every scenario (or ``only``) and reports all faults.

        A scenario that raises an ``Exception`` (or whose result cannot be
        judged) is recorded as failed with the error and the end of its
        traceback as problems, not propagated. Each result carries the
        seconds the scenario took, by the runner's clock.

        Args:
            only: Fault ids to run; all scenarios if None. Ids that are not
                in the catalogue are ignored.

        Returns:
            The report built by `report`.
        """
        results = []
        for fault in self.catalogue:
            scenario = self.scenarios.get(fault.id)
            if scenario is None or (only is not None and fault.id not in only):
                results.append(self.not_run(fault, only is not None))
                continue
            started = self.clock()
            try:
                result = scenario()
                result.judge()
            except Exception as error:  # noqa: BLE001 -- a crash is a failed fault
                result = Outcome(fault.id, FAILED, "crashed")
                result.problems = [
                    f"{type(error).__name__}: {error}",
                    traceback.format_exc()[-600:],
                ]
            result.seconds = self.clock() - started
            results.append(result)
        return self.report(results)

    def not_run(self, fault: Fault, selected: bool) -> Outcome:
        """Builds the not-run result of a fault.

        Args:
            fault: The fault.
            selected: Whether a subset of faults was selected for this run.

        Returns:
            A not-run result whose problem states why: not selected (a local
            fault when a subset was selected), no scenario provided (a local
            fault otherwise) or that it needs a real cloud environment (a
            fault that cannot run locally).
        """
        if fault.local and selected:
            reason = "not selected in this run"
        elif fault.local:
            reason = "no scenario was provided"
        else:
            reason = "needs a real cloud environment: " + fault.cloud_needs
        return Outcome(
            fault.id,
            NOT_RUN,
            "not run",
            problems=[reason],
        )

    def report(self, results: Sequence[Outcome]) -> dict[str, Any]:
        """Builds the report of a run from its results.

        Args:
            results: The results, normally one for each fault in the
                catalogue.

        Returns:
            A dict with the declared ``objectives``, the ``time_scale``, the
            catalogue as ``faults``, the ``results`` as dicts, a ``summary``
            of counts (faults, ran_locally, passed, failed, not_run) and
            ``all_local_passed``, which is true only if at least one fault
            ran and every fault that ran passed.
        """
        ran = [r for r in results if r.status != NOT_RUN]
        return {
            "objectives": [dataclasses.asdict(o) for o in OBJECTIVES],
            "time_scale": time_scale(),
            "faults": [
                {
                    "id": f.id,
                    "title": f.title,
                    "summary": f.summary,
                    "components": list(f.components),
                    "local_limits": f.local_limits,
                    "cloud_needs": f.cloud_needs,
                }
                for f in self.catalogue
            ],
            "results": [r.to_dict() for r in results],
            "summary": {
                "faults": len(results),
                "ran_locally": len(ran),
                "passed": sum(r.status == PASSED for r in ran),
                "failed": sum(r.status == FAILED for r in ran),
                "not_run": sum(r.status == NOT_RUN for r in results),
            },
            "all_local_passed": bool(ran)
            and all(r.status == PASSED for r in ran),
        }
