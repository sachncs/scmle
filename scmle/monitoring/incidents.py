"""Durable, auditable incident state transitions for ML operations."""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
import enum
import functools
import hashlib
import json
import os
import pathlib
from typing import Any
from typing import TypeVar

from scmle import durability
from scmle.monitoring import slo

Method = TypeVar("Method", bound=Callable[..., Any])


def serialized(method: Method) -> Method:
    """Runs a workflow method while holding the journal's lock.

    Every transition reads the journal, decides, then appends; the lock makes
    that one step, so concurrent threads see one hash chain.

    Args:
        method: A method of `IncidentWorkflow`.

    Returns:
        The method wrapped so it holds the workflow's lock while it runs.
    """

    @functools.wraps(method)
    def wrapper(self: "IncidentWorkflow", *args: Any, **kwargs: Any) -> Any:
        """Call the method under the workflow's lock.

        Args:
            self: The workflow.
            *args: Positional arguments of the method.
            **kwargs: Keyword arguments of the method.

        Returns:
            Whatever the method returns.
        """
        with self.lock:
            return method(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


class IncidentStatus(enum.StrEnum):
    """States an operator can record for an SLO incident.

    Attributes:
        OPEN: Opened, nobody has taken it yet.
        ACKNOWLEDGED: An operator took it.
        RESOLVED: Resolved with evidence.
    """

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


def require_text(value: str, field_name: str) -> str:
    """Returns non-empty incident metadata.

    Args:
        value: The field value.
        field_name: Name used in the error message.

    Returns:
        The value with surrounding whitespace removed.

    Raises:
        ValueError: If the value is not a string or is blank.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value.strip()


def incident_id(target: slo.SLOTarget, observed_at: str) -> str:
    """Returns an idempotent identity for one target evaluation window.

    Args:
        target: The objective.
        observed_at: Evaluation time as given to the SLO monitor.

    Returns:
        The hex SHA-256 of the target name, evaluation time and alert route.
    """
    payload = json.dumps(
        {
            "target": target.name,
            "observed_at": observed_at,
            "route": target.alert_route,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class Incident:
    """Current operator-facing state of one SLO incident.

    Attributes:
        incident_id: Identity of the incident.
        target_name: Name of the SLO target.
        status: Current state.
        opened_at: Time it was opened.
        observed_at: Time of the SLO evaluation that caused it.
        owner: Who answers for the target.
        runbook: Where the response procedure is described.
        alert_route: Where alerts go.
        reason: Why the SLO needs an incident.
        run_id: Serving run the incident concerns.
        rollback_target: Where to roll back to.
        acknowledged_at: Time it was acknowledged, if it was.
        acknowledged_by: Who acknowledged it.
        resolved_at: Time it was resolved, if it was.
        resolved_by: Who resolved it.
        resolution: How it was resolved.
        evidence_digest: Digest of the resolution evidence.
    """

    incident_id: str
    target_name: str
    status: IncidentStatus
    opened_at: str
    observed_at: str
    owner: str
    runbook: str
    alert_route: str
    reason: str
    run_id: str
    rollback_target: str
    acknowledged_at: str | None = None
    acknowledged_by: str | None = None
    resolved_at: str | None = None
    resolved_by: str | None = None
    resolution: str | None = None
    evidence_digest: str | None = None

    def __post_init__(self) -> None:
        """Validates the state record before it enters the journal.

        Raises:
            ValueError: If a required text field is blank, a timestamp is not
                ISO-8601 with timezone, or the acknowledgement and resolution
                fields do not fit the status.
        """
        for field_name in (
            "incident_id",
            "target_name",
            "opened_at",
            "observed_at",
            "owner",
            "runbook",
            "alert_route",
            "reason",
            "run_id",
            "rollback_target",
        ):
            require_text(getattr(self, field_name), field_name)
        if not isinstance(self.status, IncidentStatus):
            object.__setattr__(self, "status", IncidentStatus(self.status))
        slo.parse_timestamp(self.opened_at, "opened_at")
        slo.parse_timestamp(self.observed_at, "observed_at")
        if self.acknowledged_at is not None:
            slo.parse_timestamp(self.acknowledged_at, "acknowledged_at")
            require_text(self.acknowledged_by or "", "acknowledged_by")
        if self.resolved_at is not None:
            slo.parse_timestamp(self.resolved_at, "resolved_at")
            require_text(self.resolved_by or "", "resolved_by")
            require_text(self.resolution or "", "resolution")
            require_text(self.evidence_digest or "", "evidence_digest")
        if self.status is IncidentStatus.OPEN and self.acknowledged_at:
            raise ValueError("open incidents must not have acknowledgement")
        if (
            self.status is IncidentStatus.ACKNOWLEDGED
            and not self.acknowledged_at
        ):
            raise ValueError("acknowledged incidents require acknowledgement")
        if self.status is IncidentStatus.RESOLVED and not self.resolved_at:
            raise ValueError("resolved incidents require resolution")

    def to_dict(self) -> dict[str, Any]:
        """Returns deterministic incident evidence.

        Returns:
            A dict of all fields, with the status as its value.
        """
        return {
            "incident_id": self.incident_id,
            "target_name": self.target_name,
            "status": self.status.value,
            "opened_at": self.opened_at,
            "observed_at": self.observed_at,
            "owner": self.owner,
            "runbook": self.runbook,
            "alert_route": self.alert_route,
            "reason": self.reason,
            "run_id": self.run_id,
            "rollback_target": self.rollback_target,
            "acknowledged_at": self.acknowledged_at,
            "acknowledged_by": self.acknowledged_by,
            "resolved_at": self.resolved_at,
            "resolved_by": self.resolved_by,
            "resolution": self.resolution,
            "evidence_digest": self.evidence_digest,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Incident":
        """Rebuilds an incident from a journal snapshot.

        Args:
            record: A dict produced by `to_dict`.

        Returns:
            The incident.
        """
        return cls(
            incident_id=str(record["incident_id"]),
            target_name=str(record["target_name"]),
            status=IncidentStatus(record["status"]),
            opened_at=str(record["opened_at"]),
            observed_at=str(record["observed_at"]),
            owner=str(record["owner"]),
            runbook=str(record["runbook"]),
            alert_route=str(record["alert_route"]),
            reason=str(record["reason"]),
            run_id=str(record["run_id"]),
            rollback_target=str(record["rollback_target"]),
            acknowledged_at=record.get("acknowledged_at"),
            acknowledged_by=record.get("acknowledged_by"),
            resolved_at=record.get("resolved_at"),
            resolved_by=record.get("resolved_by"),
            resolution=record.get("resolution"),
            evidence_digest=record.get("evidence_digest"),
        )


def event_digest(event: Mapping[str, Any]) -> str:
    """Hashes a journal event without trusting its stored digest.

    Args:
        event: A journal event.

    Returns:
        The hex SHA-256 of the event's other fields.
    """
    payload = {
        key: value for key, value in event.items() if key != "event_digest"
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def check_transition(incident: Incident, previous: Incident) -> None:
    """Checks that a journal event continues the previous state of an incident.

    Args:
        incident: The incident as recorded by the event.
        previous: The incident as recorded by the previous event.

    Raises:
        ValueError: If the opening time changed, a timestamp needed for the
            comparison is missing, or the transition is dated before the
            previous state.
    """
    if incident.opened_at != previous.opened_at:
        raise ValueError("incident journal identity changed")
    transition_at = (
        incident.acknowledged_at
        if incident.status is IncidentStatus.ACKNOWLEDGED
        else incident.resolved_at
    )
    previous_at = (
        previous.opened_at
        if previous.status is IncidentStatus.OPEN
        else previous.acknowledged_at
    )
    if transition_at is None or previous_at is None:
        raise ValueError("incident journal timestamps are invalid")
    if slo.parse_timestamp(
        transition_at, "incident.transition_at"
    ) < slo.parse_timestamp(previous_at, "incident.previous_at"):
        raise ValueError("incident journal timestamps are invalid")


class IncidentWorkflow:
    """Append-only incident workflow with idempotent transitions.

    Attributes:
        path: The JSONL event journal.
        lock: Serializes transitions per journal.
    """

    def __init__(self, path: str | pathlib.Path) -> None:
        """Initializes a workflow backed by a JSONL event journal.

        Transitions are serialized per journal file within this process.

        Args:
            path: The journal file.
        """
        self.path = pathlib.Path(path)
        self.lock = durability.path_lock(self.path)

    @serialized
    def read_events(self) -> tuple[dict[str, Any], ...]:
        """Reads and verifies the complete journal, failing closed on tamper.

        Returns:
            The events in journal order; empty if there is no journal.

        Raises:
            ValueError: If an event is not an object, the hash chain or an event
                digest is invalid, an event type does not match its status, a
                transition is out of order, an incident's identity changes, or
                timestamps go backwards.
        """
        if not self.path.exists():
            return ()
        events: list[dict[str, Any]] = []
        previous_digest = ""
        statuses: dict[str, IncidentStatus] = {}
        previous_records: dict[str, Incident] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("incident journal events must be objects")
            if record.get("previous_digest") != previous_digest:
                raise ValueError("incident journal chain is invalid")
            if record.get("event_digest") != event_digest(record):
                raise ValueError("incident journal event digest is invalid")
            incident = Incident.from_dict(record["incident"])
            event_type = str(record.get("event_type", ""))
            expected_status = {
                "opened": IncidentStatus.OPEN,
                "acknowledged": IncidentStatus.ACKNOWLEDGED,
                "resolved": IncidentStatus.RESOLVED,
            }.get(event_type)
            if (
                expected_status is None
                or incident.status is not expected_status
            ):
                raise ValueError("incident journal event type is invalid")
            previous_status = statuses.get(incident.incident_id)
            allowed_previous = {
                IncidentStatus.OPEN: (None,),
                IncidentStatus.ACKNOWLEDGED: (IncidentStatus.OPEN,),
                IncidentStatus.RESOLVED: (IncidentStatus.ACKNOWLEDGED,),
            }[incident.status]
            if previous_status not in allowed_previous:
                raise ValueError("incident journal transition is invalid")
            if previous_status is not None:
                check_transition(
                    incident, previous_records[incident.incident_id]
                )
            statuses[incident.incident_id] = incident.status
            previous_records[incident.incident_id] = incident
            events.append(record)
            previous_digest = str(record["event_digest"])
        return tuple(events)

    @serialized
    def records(self) -> tuple[Incident, ...]:
        """Returns the latest state for every incident in journal order.

        Returns:
            One incident per identity, in the order first opened.
        """
        latest: dict[str, Incident] = {}
        for record in self.read_events():
            incident = Incident.from_dict(record["incident"])
            latest[incident.incident_id] = incident
        return tuple(latest.values())

    @serialized
    def open(
        self,
        evaluation: slo.SLOEvaluation,
        *,
        opened_at: str,
        run_id: str,
        rollback_target: str,
    ) -> Incident:
        """Open or return the existing incident for an SLO evaluation.

        Args:
            evaluation: The SLO evaluation that needs an incident.
            opened_at: Time of opening, an ISO-8601 timestamp with timezone.
            run_id: Serving run the incident concerns.
            rollback_target: Where to roll back to.

        Returns:
            The incident, which is the existing one if this evaluation window
            already opened one.

        Raises:
            TypeError: If `evaluation` is not an `SLOEvaluation`.
            ValueError: If the SLO passed, so no incident is required.
        """
        if not isinstance(evaluation, slo.SLOEvaluation):
            raise TypeError("evaluation must be SLOEvaluation")
        if not evaluation.incident_required:
            raise ValueError("a passing SLO cannot open an incident")
        slo.parse_timestamp(opened_at, "opened_at")
        require_text(run_id, "run_id")
        require_text(rollback_target, "rollback_target")
        identity = incident_id(evaluation.target, evaluation.observed_at)
        current = {record.incident_id: record for record in self.records()}
        if identity in current:
            return current[identity]
        incident = Incident(
            incident_id=identity,
            target_name=evaluation.target.name,
            status=IncidentStatus.OPEN,
            opened_at=opened_at,
            observed_at=evaluation.observed_at,
            owner=evaluation.target.owner,
            runbook=evaluation.target.runbook,
            alert_route=evaluation.target.alert_route,
            reason=evaluation.reason,
            run_id=run_id,
            rollback_target=rollback_target,
        )
        self.append_event("opened", incident)
        return incident

    @serialized
    def acknowledge(
        self,
        identity: str,
        *,
        actor: str,
        acknowledged_at: str,
    ) -> Incident:
        """Acknowledge an open incident exactly once.

        Args:
            identity: Identity of the incident.
            actor: Who acknowledges.
            acknowledged_at: Time, an ISO-8601 timestamp with timezone.

        Returns:
            The acknowledged incident; the existing one if it already was.

        Raises:
            ValueError: If the incident is resolved, or the acknowledgement
                precedes the opening.
        """
        incident = self.require_record(identity)
        if incident.status is IncidentStatus.ACKNOWLEDGED:
            return incident
        if incident.status is not IncidentStatus.OPEN:
            raise ValueError("only open incidents can be acknowledged")
        require_text(actor, "actor")
        timestamp = slo.parse_timestamp(acknowledged_at, "acknowledged_at")
        if timestamp < slo.parse_timestamp(incident.opened_at, "opened_at"):
            raise ValueError("acknowledgement cannot precede opening")
        updated = dataclasses.replace(
            incident,
            status=IncidentStatus.ACKNOWLEDGED,
            acknowledged_at=acknowledged_at,
            acknowledged_by=actor,
        )
        self.append_event("acknowledged", updated)
        return updated

    @serialized
    def resolve(
        self,
        identity: str,
        *,
        actor: str,
        resolved_at: str,
        resolution: str,
        evidence_digest: str,
    ) -> Incident:
        """Resolves an acknowledged incident with evidence.

        Args:
            identity: Identity of the incident.
            actor: Who resolves.
            resolved_at: Time, an ISO-8601 timestamp with timezone.
            resolution: How it was resolved.
            evidence_digest: Digest of the evidence.

        Returns:
            The resolved incident; the existing one if it already was.

        Raises:
            ValueError: If the incident is not acknowledged, or the resolution
                precedes the acknowledgement.
        """
        incident = self.require_record(identity)
        if incident.status is IncidentStatus.RESOLVED:
            return incident
        if incident.status is not IncidentStatus.ACKNOWLEDGED:
            raise ValueError("only acknowledged incidents can be resolved")
        require_text(actor, "actor")
        require_text(resolution, "resolution")
        require_text(evidence_digest, "evidence_digest")
        timestamp = slo.parse_timestamp(resolved_at, "resolved_at")
        if timestamp < slo.parse_timestamp(
            incident.acknowledged_at or "", "acknowledged_at"
        ):
            raise ValueError("resolution cannot precede acknowledgement")
        updated = dataclasses.replace(
            incident,
            status=IncidentStatus.RESOLVED,
            resolved_at=resolved_at,
            resolved_by=actor,
            resolution=resolution,
            evidence_digest=evidence_digest,
        )
        self.append_event("resolved", updated)
        return updated

    def require_record(self, identity: str) -> Incident:
        """Returns the incident with the given identity.

        A blank or non-string identity raises ValueError from `require_text`.

        Args:
            identity: Identity of the incident.

        Returns:
            The incident.

        Raises:
            KeyError: If no incident has that identity.
        """
        require_text(identity, "incident_id")
        for record in self.records():
            if record.incident_id == identity:
                return record
        raise KeyError(f"unknown incident '{identity}'")

    @serialized
    def append_event(self, event_type: str, incident: Incident) -> None:
        """Appends one chained state snapshot durably.

        Args:
            event_type: `opened`, `acknowledged` or `resolved`.
            incident: The incident state to record.
        """
        require_text(event_type, "event_type")
        existing = self.read_events()
        previous_digest = str(existing[-1]["event_digest"]) if existing else ""
        event: dict[str, Any] = {
            "event_type": event_type,
            "incident": incident.to_dict(),
            "previous_digest": previous_digest,
        }
        event["event_digest"] = event_digest(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        durability.sync_directory(self.path.parent)


__all__ = [
    "Incident",
    "IncidentStatus",
    "IncidentWorkflow",
    "event_digest",
    "incident_id",
    "require_text",
]
