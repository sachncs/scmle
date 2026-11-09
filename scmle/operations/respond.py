"""Detect, classify, notify, retain and record what the operator did.

`ResponseDesk` ties the pieces of an incident together:

1. the signals of a window are classified (`scmle.operations.classify`);
2. every window's signals are appended to a telemetry file, so the evidence
   for an alert outlives the process that raised it;
3. a classified incident is sent to the pager and ticket senders with a
   trace id and the model version;
4. each step, and each later operator action (acknowledge, resolve), is an
   ``admin`` row in the hash-chained audit ledger
   (`scmle.security.audit.AuditLog`), naming the actor and the incident.

The ledger shows that rows were not edited or reordered; it does not prove who
the actor was, which is the identity system's job.
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import hashlib
import json
import pathlib
import time
from typing import Any

from scmle import durability
from scmle.operations import classify
from scmle.operations import logs
from scmle.operations import notify
from scmle.operations import signals
from scmle.security import audit as audit_module

DESK = "operations-desk"
TELEMETRY_FILE = "telemetry.jsonl"


@dataclasses.dataclass(frozen=True, slots=True)
class Response:
    """What the desk did for one window.

    Attributes:
        incident_id: Id of the incident, or None when the verdict is none
            (no signal breached, or no signals at all).
        classification: The verdict and its evidence.
        deliveries: Pager and ticket outcomes.
        telemetry_digest: SHA-256 of the line retained for the window.
        audit_head: Hash of the newest audit row: the one that opened the
            incident, or the current ledger head if none was opened.
    """

    incident_id: str | None
    classification: classify.Classification
    deliveries: tuple[notify.Delivery, ...]
    telemetry_digest: str
    audit_head: str

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready record.

        Returns:
            The incident id, classification, deliveries, telemetry digest and
            audit head as a dict.
        """
        return {
            "incident_id": self.incident_id,
            "classification": self.classification.to_dict(),
            "deliveries": [d.to_dict() for d in self.deliveries],
            "telemetry_digest": self.telemetry_digest,
            "audit_head": self.audit_head,
        }


class ResponseDesk:
    """Classifies windows, notifies, retains telemetry and audits."""

    def __init__(
        self,
        directory: str | pathlib.Path,
        audit: audit_module.AuditLog,
        notifier: notify.Notifier,
        model_version: str,
        artifact_digest: str,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Creates a desk that keeps its telemetry in ``directory``.

        Args:
            directory: Directory for the retained telemetry; it is created
                when the first window is retained.
            audit: The audit log that records incidents and actions.
            notifier: Sends the notifications.
            model_version: Version of the model that is served.
            artifact_digest: Digest of the served artifact.
            clock: Source of the current time, in seconds.
        """
        self.directory = pathlib.Path(directory)
        self.audit = audit
        self.notifier = notifier
        self.model_version = model_version
        self.artifact_digest = artifact_digest
        self.clock = clock

    @property
    def telemetry_path(self) -> pathlib.Path:
        """The file of retained windows."""
        return self.directory / TELEMETRY_FILE

    def retain(self, at: float, signal_rows: Sequence[signals.Signal]) -> str:
        """Appends a window's signals to the telemetry file.

        The line is JSON with the time (rounded to milliseconds), model
        version, artifact digest and the signals as dicts.

        Args:
            at: Time of the window, in seconds.
            signal_rows: The window's signals.

        Returns:
            The SHA-256 of the line that was appended, without its newline.
        """
        line = json.dumps(
            {
                "at": round(at, 3),
                "model_version": self.model_version,
                "artifact_digest": self.artifact_digest,
                "signals": [s.to_dict() for s in signal_rows],
            },
            sort_keys=True,
        )
        durability.append_text(self.telemetry_path, line + "\n")
        return hashlib.sha256(line.encode()).hexdigest()

    def retained(self, since: float = 0.0) -> list[dict[str, Any]]:
        """Retained windows at or after ``since``.

        Args:
            since: Earliest time to include, in seconds.

        Returns:
            The retained windows as decoded, in the order they were
            appended; empty if there is no telemetry file.
        """
        if not self.telemetry_path.exists():
            return []
        rows = [
            json.loads(line)
            for line in self.telemetry_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        return [row for row in rows if row["at"] >= since]

    def expire(self, older_than: float) -> int:
        """Removes retained windows older than ``older_than``.

        Rewrites the telemetry file atomically with the windows kept (an
        empty file if none are kept; the file is created if absent).

        Args:
            older_than: Time before which windows are removed, in seconds.

        Returns:
            The number of windows removed.
        """
        rows = self.retained()
        keep = [row for row in rows if row["at"] >= older_than]
        text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in keep)
        durability.write_atomically(self.telemetry_path, text.encode())
        return len(rows) - len(keep)

    def handle(
        self,
        signal_rows: Sequence[signals.Signal],
        trace_id: str | None = None,
    ) -> Response:
        """Processes one window of signals.

        Retains the window, then classifies it. For any verdict other than
        none (an indeterminate verdict included) it sends the notifications
        and then records an ``incident_opened`` row in the audit log.

        Args:
            signal_rows: The window's signals.
            trace_id: Trace id for the notification; a new one if None or
                empty.

        Returns:
            The response. When the verdict is none it has no incident id and
            no deliveries and nothing is audited; otherwise it carries the
            incident id (derived from the verdict and the window's digest),
            the deliveries and the hash of the new audit row.
        """
        now = self.clock()
        trace = trace_id or logs.new_trace_id()
        digest = self.retain(now, signal_rows)
        found = classify.classify(signal_rows)
        if found.verdict is classify.Verdict.NONE:
            return Response(None, found, (), digest, self.audit.ledger.head)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(now))
        incident_id = (
            "inc-"
            + hashlib.sha256(
                f"{found.verdict.value}|{digest}".encode()
            ).hexdigest()[:12]
        )
        deliveries = self.notifier.notify(
            classify.notification(
                found,
                incident_id,
                self.model_version,
                self.artifact_digest,
                trace,
                stamp,
            )
        )
        row = self.audit.admin(
            DESK,
            "incident_opened",
            incident_id=incident_id,
            verdict=found.verdict.value,
            breached=list(found.breached),
            telemetry_digest=digest,
            trace_id=trace,
            model_version=self.model_version,
            artifact_digest=self.artifact_digest,
            deliveries=[
                {"kind": d.kind, "delivered": d.delivered, "reason": d.reason}
                for d in deliveries
            ],
        )
        return Response(
            incident_id, found, tuple(deliveries), digest, row["hash"]
        )

    def act(
        self, incident_id: str, operator: str, action: str, note: str
    ) -> dict[str, Any]:
        """Records an operator's action on an incident.

        Args:
            incident_id: The incident acted on.
            operator: Who acted.
            action: ``acknowledged``, ``mitigated`` or ``resolved``.
            note: What was done (no credentials).

        Returns:
            The audit row, written under the operator as actor with the
            action ``incident_<action>``, that records the action.

        Raises:
            ValueError: If the action is unknown, or the incident was never
                opened in this ledger.
        """
        if action not in {"acknowledged", "mitigated", "resolved"}:
            raise ValueError("unknown operator action")
        opened = [
            row
            for row in self.audit.rows(audit_module.ADMIN)
            if row["body"].get("incident_id") == incident_id
            and row["body"].get("action") == "incident_opened"
        ]
        if not opened:
            raise ValueError(f"{incident_id} was not opened here")
        return self.audit.admin(
            operator,
            f"incident_{action}",
            incident_id=incident_id,
            note=note,
        )
