"""The scenario table the recovery runner dispatches by fault id."""

from collections.abc import Callable
from typing import Any

from scmle.operations import recovery
from scmle.rehearsal.faults import crashes
from scmle.rehearsal.faults import outages
from scmle.rehearsal.faults import release
from scmle.rehearsal.faults import service

SCENARIOS: dict[str, Callable[[], recovery.Outcome]] = {
    "process_kill": crashes.process,
    "unhealthy_container_rolling_restart": service.restart,
    "invalid_credentials": service.invalid_credentials,
    "lost_object_store_access": outages.disconnect,
    "dependency_outage": outages.blackout,
    "corrupted_bundle": service.bundle,
    "incompatible_schema": service.incompatible_schema,
    "bad_release": release.release,
    "job_interruption": crashes.resume,
    "audit_log_continuity": crashes.resilience,
    "secret_rotation_recovery": service.rekey,
}


def rehearse(only: list[str] | None = None) -> dict[str, Any]:
    """Runs the local scenarios of the fault catalogue and builds the report.

    Args:
        only: Fault ids to run; all if None. Faults that are not selected or
            have no local scenario are reported as not run, and ids outside
            the catalogue are ignored.

    Returns:
        The report built by `Runner.run`. A scenario that raises is recorded
        in it as a failed fault rather than propagated.
    """
    return recovery.Runner(SCENARIOS).run(only)
