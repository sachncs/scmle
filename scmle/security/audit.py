"""Security audit rows in the hash-chained deployment ledger.

Three kinds of event are recorded, each as an ``audit`` row: ``activation``
(a model became ready to serve), ``access`` (a request was allowed or
denied) and ``admin`` (a key rotation or another administrative action). A
row names who acted, what, the outcome and the request id; it never holds a
credential or a request body. The ledger chain shows that rows were not
edited, removed from the middle or reordered; it does not show that the
newest rows were not removed (compare `head` with a copy kept elsewhere).
"""

from collections.abc import Callable
import pathlib
import threading
import time
from typing import Any

from scmle.governance import deployment_ledger
from scmle.security import tokens

ACTIVATION = "activation"
ACCESS = "access"
ADMIN = "admin"
EVENTS = (ACTIVATION, ACCESS, ADMIN)
ALLOWED = "allowed"
DENIED = "denied"
SERVICE_CHARTER = "service-audit"


class AuditLog:
    """Appends audit rows to a deployment ledger, one at a time.

    Attributes:
        ledger: The ledger rows are committed to.
        clock: Source of epoch seconds.
        lock: Serializes commits.
    """

    def __init__(
        self,
        ledger: deployment_ledger.DeploymentLedger,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Initializes the log.

        Args:
            ledger: An opened ledger.
            clock: Source of epoch seconds.
        """
        self.ledger = ledger
        self.clock = clock
        self.lock = threading.Lock()

    @classmethod
    def open(
        cls,
        path: str | pathlib.Path | None,
        clock: Callable[[], float] = time.time,
    ) -> "AuditLog":
        """Loads the ledger at `path` (verifying its chain) or creates it.

        Args:
            path: Ledger file, or None for an in-memory ledger.
            clock: Source of epoch seconds.

        Returns:
            The log. An existing file that is broken raises
            `DeploymentLedgerError` from the ledger's loader.
        """
        if path is not None and pathlib.Path(path).exists():
            ledger = deployment_ledger.DeploymentLedger.load(path)
        else:
            ledger = deployment_ledger.DeploymentLedger.create(
                path,
                charter_hash=SERVICE_CHARTER,
                policy={"purpose": "security audit"},
                alpha_promo="1/20",
            )
        return cls(ledger, clock)

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        """Appends one audit row.

        Args:
            event: `activation`, `access` or `admin`.
            **fields: JSON-ready facts (never credentials).

        Returns:
            The chained row.

        Raises:
            DeploymentLedgerError: If `event` is unknown.
        """
        if event not in EVENTS:
            raise deployment_ledger.DeploymentLedgerError(
                f"audit event must be one of {EVENTS}"
            )
        body = {"event": event, "at": round(self.clock(), 3), **fields}
        with self.lock:
            return self.ledger.commit(deployment_ledger.AUDIT, body)

    def activation(
        self, model_version: str, artifact_digest: str, **facts: Any
    ) -> dict[str, Any]:
        """Records that a model became ready to serve.

        Args:
            model_version: Version of the model.
            artifact_digest: Digest of the model artifact.
            **facts: Further JSON-ready facts.

        Returns:
            The chained row.
        """
        return self.record(
            ACTIVATION,
            actor="service",
            model_version=model_version,
            artifact_digest=artifact_digest,
            outcome=ALLOWED,
            **facts,
        )

    def access(
        self,
        principal: tokens.Principal | None,
        route: str,
        outcome: str,
        status: int,
        reason: str,
        request_id: str,
    ) -> dict[str, Any]:
        """Records an access decision.

        Args:
            principal: The verified caller, or None if authentication failed.
            route: The request path.
            outcome: `allowed` or `denied`.
            status: HTTP status given.
            reason: The precise cause.
            request_id: Request id.

        Returns:
            The chained row.
        """
        return self.record(
            ACCESS,
            actor=None if principal is None else principal.subject,
            tenant=None if principal is None else principal.tenant,
            key_id=None if principal is None else principal.key_id,
            token_id=None if principal is None else principal.token_id,
            route=route,
            outcome=outcome,
            status=status,
            reason=reason,
            request_id=request_id,
        )

    def admin(self, actor: str, action: str, **detail: Any) -> dict[str, Any]:
        """Records an administrative action such as a key rotation.

        Args:
            actor: Who acted.
            action: What was done.
            **detail: Further JSON-ready facts.

        Returns:
            The chained row.
        """
        return self.record(
            ADMIN, actor=actor, action=action, outcome=ALLOWED, **detail
        )

    def rows(self, event: str | None = None) -> list[dict[str, Any]]:
        """Lists the audit rows so far, optionally of one event kind.

        Args:
            event: `activation`, `access` or `admin` to keep only that kind;
                None for all.

        Returns:
            The chained rows, in ledger order.
        """
        found = [
            row
            for row in self.ledger.rows
            if row["kind"] == deployment_ledger.AUDIT
        ]
        if event is None:
            return found
        return [row for row in found if row["body"]["event"] == event]
