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
from typing import ClassVar
from typing import TypeVar

from scmle import durability
from scmle.contracts import errors
from scmle.contracts import records
from scmle.monitoring import slo

Method = TypeVar("Method", bound=Callable[..., Any])


def serialized(method: Method) -> Method:
    """Runs a workflow method while holding the journal's lock.

    Every transition reads the journal, decides, then appends; the lock makes
    that one step, so concurrent threads see one hash chain.

    Args:
        method: A method of `Incidents`.

    Returns:
        The method wrapped so it holds the workflow's lock while it runs.
    """

    @functools.wraps(method)
    def wrapper(self: "Incidents", *args: Any, **kwargs: Any) -> Any:
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


class Phase(enum.StrEnum):
    """States an operator can record for an SLO incident.

    Attributes:
        OPEN: Opened, nobody has taken it yet.
        ACKNOWLEDGED: An operator took it.
        RESOLVED: Resolved with evidence.
    """

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


def require(value: object, field_name: str) -> str:
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
        raise errors.InputError(f"{field_name} must not be empty")
    return value.strip()


def identify(target: slo.Slo, observed_at: str) -> str:
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
            "route": target.route,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class Incident:
    """Current operator-facing state of one SLO incident.

    Attributes:
        identifier: Identity of the incident.
        target: Name of the SLO target.
        status: Current state.
        opened: Time it was opened.
        observed: Time of the SLO evaluation that caused it.
        owner: Who answers for the target.
        runbook: Where the response procedure is described.
        route: Where alerts go.
        reason: Why the SLO needs an incident.
        run: Serving run the incident concerns.
        rollback: Where to roll back to.
        acknowledged: Time it was acknowledged, if it was.
        acknowledger: Who acknowledged it.
        resolved: Time it was resolved, if it was.
        resolver: Who resolved it.
        resolution: How it was resolved.
        digests: `evidence`, the digest of the resolution evidence, or None.
        migrations: The snapshot's version steps, oldest first.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            {
                "incident_id": "identifier",
                "target_name": "target",
                "opened_at": "opened",
                "observed_at": "observed",
                "alert_route": "route",
                "run_id": "run",
                "rollback_target": "rollback",
                "acknowledged_at": "acknowledged",
                "acknowledged_by": "acknowledger",
                "resolved_at": "resolved",
                "resolved_by": "resolver",
            },
            groups={"digests": {"evidence_digest": "evidence"}},
        ),
    )

    identifier: str
    target: str
    status: Phase
    opened: str
    observed: str
    owner: str
    runbook: str
    route: str
    reason: str
    run: str
    rollback: str
    acknowledged: str | None = None
    acknowledger: str | None = None
    resolved: str | None = None
    resolver: str | None = None
    resolution: str | None = None
    digests: Mapping[str, str | None] = dataclasses.field(
        default_factory=lambda: {"evidence": None}
    )

    def __post_init__(self) -> None:
        """Validates the state record before it enters the journal.

        Raises:
            ValueError: If a required text field is blank, a timestamp is not
                ISO-8601 with timezone, or the acknowledgement and resolution
                fields do not fit the status.
        """
        for field_name in (
            "identifier",
            "target",
            "opened",
            "observed",
            "owner",
            "runbook",
            "route",
            "reason",
            "run",
            "rollback",
        ):
            require(getattr(self, field_name), field_name)
        status: Any = self.status  # may be text from a persisted record
        if not isinstance(status, Phase):
            object.__setattr__(self, "status", Phase(status))
        slo.parse(self.opened, "opened")
        slo.parse(self.observed, "observed")
        if self.acknowledged is not None:
            slo.parse(self.acknowledged, "acknowledged")
            require(self.acknowledger or "", "acknowledger")
        if self.resolved is not None:
            slo.parse(self.resolved, "resolved")
            require(self.resolver or "", "resolver")
            require(self.resolution or "", "resolution")
            require(self.digests.get("evidence") or "", "digests.evidence")
        if self.status is Phase.OPEN and self.acknowledged:
            raise errors.InputError(
                "open incidents must not have acknowledgement"
            )
        if self.status is Phase.ACKNOWLEDGED and not self.acknowledged:
            raise errors.InputError(
                "acknowledged incidents require acknowledgement"
            )
        if self.status is Phase.RESOLVED and not self.resolved:
            raise errors.InputError("resolved incidents require resolution")

    def to_dict(self) -> dict[str, Any]:
        """Returns deterministic incident evidence at the current version.

        Returns:
            A dict of all fields, with the status as its value, and the
            `schema_version`.
        """
        record = dataclasses.asdict(self)
        record["status"] = self.status.value
        record["digests"] = dict(self.digests)
        return {records.KEY: records.ORIGIN + len(self.migrations), **record}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Incident":
        """Rebuilds an incident from a journal snapshot.

        Journal rows are verified as stored (`Incidents.read`) before
        they are rebuilt; a snapshot without `schema_version` is version 1
        and is migrated.

        Args:
            record: A dict produced by `to_dict`.

        Returns:
            The incident.

        Raises:
            ValueError: If the version is not a positive integer or is newer
                than this build.
        """
        values = records.load("incident", cls.migrations, record)
        digests = values.get("digests")
        return cls(
            identifier=str(values["identifier"]),
            target=str(values["target"]),
            status=Phase(values["status"]),
            opened=str(values["opened"]),
            observed=str(values["observed"]),
            owner=str(values["owner"]),
            runbook=str(values["runbook"]),
            route=str(values["route"]),
            reason=str(values["reason"]),
            run=str(values["run"]),
            rollback=str(values["rollback"]),
            acknowledged=values.get("acknowledged"),
            acknowledger=values.get("acknowledger"),
            resolved=values.get("resolved"),
            resolver=values.get("resolver"),
            resolution=values.get("resolution"),
            digests={
                "evidence": (
                    digests.get("evidence")
                    if isinstance(digests, Mapping)
                    else None
                )
            },
        )


def digest(event: Mapping[str, Any]) -> str:
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


def proceed(incident: Incident, previous: Incident) -> None:
    """Checks that a journal event continues the previous state of an incident.

    Args:
        incident: The incident as recorded by the event.
        previous: The incident as recorded by the previous event.

    Raises:
        ValueError: If the opening time changed, a timestamp needed for the
            comparison is missing, or the transition is dated before the
            previous state.
    """
    if incident.opened != previous.opened:
        raise errors.InputError("incident journal identity changed")
    transition_at = (
        incident.acknowledged
        if incident.status is Phase.ACKNOWLEDGED
        else incident.resolved
    )
    previous_at = (
        previous.opened
        if previous.status is Phase.OPEN
        else previous.acknowledged
    )
    if transition_at is None or previous_at is None:
        raise errors.InputError("incident journal timestamps are invalid")
    if slo.parse(transition_at, "incident.transition_at") < slo.parse(
        previous_at, "incident.previous_at"
    ):
        raise errors.InputError("incident journal timestamps are invalid")


class Incidents:
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
        self.lock = durability.latch(self.path)

    @serialized
    def read(self) -> tuple[dict[str, Any], ...]:
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
        statuses: dict[str, Phase] = {}
        previous_records: dict[str, Incident] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if not isinstance(record, dict):
                raise errors.InputError(
                    "incident journal events must be objects"
                )
            if record.get("previous_digest") != previous_digest:
                raise errors.InputError("incident journal chain is invalid")
            if record.get("event_digest") != digest(record):
                raise errors.InputError(
                    "incident journal event digest is invalid"
                )
            incident = Incident.from_dict(record["incident"])
            event_type = str(record.get("event_type", ""))
            expected_status = {
                "opened": Phase.OPEN,
                "acknowledged": Phase.ACKNOWLEDGED,
                "resolved": Phase.RESOLVED,
            }.get(event_type)
            if (
                expected_status is None
                or incident.status is not expected_status
            ):
                raise errors.InputError(
                    "incident journal event type is invalid"
                )
            previous_status = statuses.get(incident.identifier)
            allowed_previous = {
                Phase.OPEN: (None,),
                Phase.ACKNOWLEDGED: (Phase.OPEN,),
                Phase.RESOLVED: (Phase.ACKNOWLEDGED,),
            }[incident.status]
            if previous_status not in allowed_previous:
                raise errors.InputError(
                    "incident journal transition is invalid"
                )
            if previous_status is not None:
                proceed(incident, previous_records[incident.identifier])
            statuses[incident.identifier] = incident.status
            previous_records[incident.identifier] = incident
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
        for record in self.read():
            incident = Incident.from_dict(record["incident"])
            latest[incident.identifier] = incident
        return tuple(latest.values())

    @serialized
    def open(
        self,
        evaluation: slo.Assessment,
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
            TypeError: If `evaluation` is not an `Assessment`.
            ValueError: If the SLO passed, so no incident is required.
        """
        if not isinstance(evaluation, slo.Assessment):
            raise errors.KindError("evaluation must be Assessment")
        if not evaluation.required:
            raise errors.InputError("a passing SLO cannot open an incident")
        slo.parse(opened_at, "opened_at")
        require(run_id, "run_id")
        require(rollback_target, "rollback_target")
        identity = identify(evaluation.target, evaluation.observed)
        current = {record.identifier: record for record in self.records()}
        if identity in current:
            return current[identity]
        incident = Incident(
            identifier=identity,
            target=evaluation.target.name,
            status=Phase.OPEN,
            opened=opened_at,
            observed=evaluation.observed,
            owner=evaluation.target.owner,
            runbook=evaluation.target.runbook,
            route=evaluation.target.route,
            reason=evaluation.reason,
            run=run_id,
            rollback=rollback_target,
        )
        self.log("opened", incident)
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
        incident = self.require(identity)
        if incident.status is Phase.ACKNOWLEDGED:
            return incident
        if incident.status is not Phase.OPEN:
            raise errors.InputError("only open incidents can be acknowledged")
        require(actor, "actor")
        timestamp = slo.parse(acknowledged_at, "acknowledged_at")
        if timestamp < slo.parse(incident.opened, "opened"):
            raise errors.InputError("acknowledgement cannot precede opening")
        updated = dataclasses.replace(
            incident,
            status=Phase.ACKNOWLEDGED,
            acknowledged=acknowledged_at,
            acknowledger=actor,
        )
        self.log("acknowledged", updated)
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
        incident = self.require(identity)
        if incident.status is Phase.RESOLVED:
            return incident
        if incident.status is not Phase.ACKNOWLEDGED:
            raise errors.InputError(
                "only acknowledged incidents can be resolved"
            )
        require(actor, "actor")
        require(resolution, "resolution")
        require(evidence_digest, "evidence_digest")
        timestamp = slo.parse(resolved_at, "resolved_at")
        if timestamp < slo.parse(incident.acknowledged or "", "acknowledged"):
            raise errors.InputError("resolution cannot precede acknowledgement")
        updated = dataclasses.replace(
            incident,
            status=Phase.RESOLVED,
            resolved=resolved_at,
            resolver=actor,
            resolution=resolution,
            digests={"evidence": evidence_digest},
        )
        self.log("resolved", updated)
        return updated

    def require(self, identity: str) -> Incident:
        """Returns the incident with the given identity.

        A blank or non-string identity raises ValueError from `require`.

        Args:
            identity: Identity of the incident.

        Returns:
            The incident.

        Raises:
            KeyError: If no incident has that identity.
        """
        require(identity, "identifier")
        for record in self.records():
            if record.identifier == identity:
                return record
        raise KeyError(f"unknown incident '{identity}'")

    @serialized
    def log(self, event_type: str, incident: Incident) -> None:
        """Appends one chained state snapshot durably.

        Args:
            event_type: `opened`, `acknowledged` or `resolved`.
            incident: The incident state to record.
        """
        require(event_type, "event_type")
        existing = self.read()
        previous_digest = str(existing[-1]["event_digest"]) if existing else ""
        event: dict[str, Any] = {
            "event_type": event_type,
            "incident": incident.to_dict(),
            "previous_digest": previous_digest,
        }
        event["event_digest"] = digest(event)
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
    "Incidents",
    "Phase",
    "digest",
    "identify",
    "require",
]
