"""Authority granted in advance for the actions an unattended loop may take.

The never list (`scmle.governance.never_list`) names eleven actions that need
a person's attestation for each exact request. Everything else the loop does
on its own (observe, mature, step, escalate, propose) needs no authority. One
step in between does: training a challenger, which spends compute and puts a
new candidate in front of the promotion process. An *envelope* is how a person
authorises that step ahead of time:

    guard = EnvelopeGuard(ledger, verification_keys={key_id: key})
    guard.grant(Envelope.create(key, envelope_id="e1", limits={...}, ...))
    guard.authorize(EnvelopeAction.TRAIN_CHALLENGER, scope, {"batch": 7})
    guard.revoke("e1", "model retired")

What an envelope can and cannot do:

* It can cover only the actions in `EnvelopeAction`. An action on the never
  list cannot be put in one; `Envelope` refuses it when it is built and the
  ledger refuses it when it is read. Promotion and rollback stay behind the
  registry's signed approval and the never-list guard, one person's approval
  for each exact model.
* It has a scope (model version hash and stream id), a first and last day, and
  a use limit per action. A use is chained in the ledger before the action
  runs, so a crash cannot make an action that is not on the record, and the
  ledger itself refuses a use that is unknown, revoked, out of scope, not
  covered or over its limit.
* It is signed with a key the system does not hold. When the guard is given
  the verification keys it checks the signature again at every use; without
  them it relies on the grant row, which was checked when it was written. The
  limit is the one stated for attestations: this is code around the entry
  points of this package, and a process with file access can still write the
  ledger.
* Revoking removes authority only, so anyone may revoke.
"""

from collections.abc import Callable
from collections.abc import Mapping
import datetime
import enum
from typing import Any

from scmle.contracts import errors
from scmle.governance import deployment_ledger as ledger_module
from scmle.governance import never_list


class EnvelopeAction(enum.StrEnum):
    """The actions an envelope can authorise."""

    TRAIN_CHALLENGER = "train_challenger"


class EnvelopeRefused(errors.ArtifactError):
    """No envelope allows the requested action."""

    def __init__(self, action: str, reason: str) -> None:
        """Records the action and why every envelope was refused."""
        super().__init__(f"no envelope allows {action}: {reason}")
        self.action = action
        self.reason = reason


class EnvelopeGuard:
    """Grants, revokes and spends envelopes in the deployment ledger."""

    def __init__(
        self,
        ledger: ledger_module.DeploymentLedger,
        today: Callable[[], datetime.date] = datetime.date.today,
        verification_keys: Mapping[str, str | bytes] | None = None,
    ) -> None:
        """Creates the guard.

        Args:
            ledger: The deployment ledger.
            today: Clock, replaceable in tests.
            verification_keys: Optional map from key id to the key that
                verifies envelopes signed under it. When given (even empty), a
                grant needs a verifying signature and every use checks it
                again; None relies on the grant row.
        """
        self.ledger = ledger
        self.today = today
        self.verification_keys = (
            None if verification_keys is None else dict(verification_keys)
        )
        self.on_write: Callable[[], None] | None = None

    def wrote(self) -> None:
        """Tells the owner of the running state that the ledger moved."""
        if self.on_write is not None:
            self.on_write()

    def signature_ok(self, grant: ledger_module.Envelope) -> bool:
        """Whether the signature verifies, when keys were supplied."""
        if self.verification_keys is None:
            return True
        key = self.verification_keys.get(grant.key_id)
        return key is not None and grant.verify(key)

    def grant(self, envelope: ledger_module.Envelope) -> dict[str, Any]:
        """Writes a person's signed envelope to the ledger.

        Raises:
            EnvelopeRefused: If the signature does not verify.
            DeploymentLedgerError: If the id is taken or the key is one the
                system holds.
        """
        if not self.signature_ok(envelope):
            raise EnvelopeRefused(
                "grant", "the signature does not verify under a known key"
            )
        row = self.ledger.commit(ledger_module.ENVELOPE, envelope.to_dict())
        self.wrote()
        return row

    def revoke(self, envelope_id: str, reason: str) -> dict[str, Any]:
        """Removes an envelope's authority."""
        row = self.ledger.commit(
            ledger_module.ENVELOPE_REVOKED,
            {"envelope_id": envelope_id, "reason": reason},
        )
        self.wrote()
        return row

    def remaining(self, envelope_id: str) -> dict[str, int]:
        """Uses left per action; empty for an unknown or revoked envelope."""
        grant = self.ledger.envelopes.get(envelope_id)
        if grant is None or envelope_id in self.ledger.envelope_revoked:
            return {}
        used = self.ledger.envelope_uses[envelope_id]
        return {
            action: max(limit - used.get(action, 0), 0)
            for action, limit in grant["limits"].items()
        }

    def refusal(
        self, identity: str, grant: Mapping[str, Any], action: str, scope: str
    ) -> str | None:
        """Why one envelope does not allow the request, or None if it does."""
        if identity in self.ledger.envelope_revoked:
            return "revoked"
        if grant["scope"] != scope:
            return "a different scope"
        if action not in grant["limits"]:
            return f"does not cover {action}"
        today = self.today()
        issued = ledger_module.parse_date(grant["date"], "date")
        expires = ledger_module.parse_date(grant["expires"], "expires")
        if not issued <= today <= expires:
            return "not valid today"
        if not self.signature_ok(ledger_module.Envelope.from_dict(grant)):
            return "its signature does not verify"
        if self.remaining(identity).get(action, 0) < 1:
            return "its uses are spent"
        return None

    def authorize(
        self,
        action: EnvelopeAction,
        scope: str,
        detail: Mapping[str, Any] | None = None,
    ) -> str:
        """Spends one use of the first envelope that allows the request.

        The use is chained before this returns, so the caller acts only after
        it is on the record.

        Args:
            action: What is about to be done.
            scope: Model version hash and stream id.
            detail: JSON-ready description of the request, kept in the row.

        Returns:
            The id of the envelope that was used.

        Raises:
            EnvelopeRefused: If no envelope allows it; nothing is written.
        """
        if action.value in {item.value for item in never_list.NeverAction}:
            raise EnvelopeRefused(action.value, "it is on the never list")
        reasons = []
        for identity, grant in self.ledger.envelopes.items():
            reason = self.refusal(identity, grant, action.value, scope)
            if reason is None:
                self.ledger.commit(
                    ledger_module.ENVELOPE_USED,
                    {
                        "envelope_id": identity,
                        "action": action.value,
                        "scope": scope,
                        "date": self.today().isoformat(),
                        "detail": dict(detail or {}),
                    },
                )
                self.wrote()
                return identity
            reasons.append(f"{identity}: {reason}")
        raise EnvelopeRefused(
            action.value, "; ".join(reasons) or "no envelope was granted"
        )
