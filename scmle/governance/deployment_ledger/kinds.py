"""Row kinds, phases and the checks every part of the ledger shares."""

import datetime
import enum
import hashlib
from typing import Any

from scmle.assurance.ledger import ledger
from scmle.contracts import errors

FILE_NAME = "deployment_ledger.jsonl"
LEDGER_VERSION = 1
OPEN = "open"
OBSERVE_SUMMARY = "observe_summary"
ALARM = "alarm"
RETRAIN = "retrain"
CHALLENGER_RESULT = "challenger_result"
PROMOTION_PROPOSAL = "promotion_proposal"
ROLLBACK_PROPOSAL = "rollback_proposal"
ESCALATION = "escalation"
ATTESTATION = "attestation"
ATTESTATION_USED = "attestation_used"
HALT = "halt"
INCIDENT = "incident"
AUDIT = "audit"
DEPLOYMENT = "deployment"
ROLLOUT_EXPERIMENT = "rollout_experiment"
ROLLOUT_ASSIGNMENT = "rollout_assignment"
ROLLOUT_DECISION = "rollout_decision"
ENVELOPE = "envelope"
ENVELOPE_USED = "envelope_used"
ENVELOPE_REVOKED = "envelope_revoked"
CONTROLLER = "controller"
ENVELOPE_ACTIONS = ("train_challenger",)
INTENT = "intent"
APPLIED = "applied"
REJECTED = "rejected"
TICK = "tick"
CONTROLLER_PHASES = (INTENT, APPLIED, REJECTED, TICK)
CONTROLLER_INPUTS = ("batch", "labels")
KINDS = (
    OPEN,
    OBSERVE_SUMMARY,
    ALARM,
    RETRAIN,
    CHALLENGER_RESULT,
    PROMOTION_PROPOSAL,
    ROLLBACK_PROPOSAL,
    ESCALATION,
    ATTESTATION,
    ATTESTATION_USED,
    HALT,
    INCIDENT,
    AUDIT,
    DEPLOYMENT,
    ROLLOUT_EXPERIMENT,
    ROLLOUT_ASSIGNMENT,
    ROLLOUT_DECISION,
    ENVELOPE,
    ENVELOPE_USED,
    ENVELOPE_REVOKED,
    CONTROLLER,
)
REGISTERED = "registered"
DECIDED = "decided"
PHASE_OPEN = "open"
PHASE_TIMEOUT = "timeout"
ENTER = "enter"
RELEASE = "release"


class MisuseError(ledger.LedgerError):
    """The deployment ledger was misused or its contents are inconsistent."""


class Reply(enum.StrEnum):
    """What a human reply decides."""

    APPROVE = "approve"
    REJECT = "reject"
    OVERRIDE = "override"
    CANNOT_DECIDE = "cannot_decide"


def mapped(body: Any) -> None:
    """Checks that a ledger row body is an object.

    Args:
        body: The body of a row.

    Raises:
        TypeError: If `body` is not a dict.
    """
    if not isinstance(body, dict):
        raise errors.KindError("body must be an object")


def thumbprint(key: str | bytes) -> str:
    """A short public fingerprint of a signing key (never the key).

    Args:
        key: The signing key.

    Returns:
        The first 16 hex characters of a domain-separated SHA-256 of the
        key; it cannot be used to recover the key.
    """
    raw = key.encode() if isinstance(key, str) else key
    return hashlib.sha256(b"scmle-key-fingerprint:" + raw).hexdigest()[:16]


def parse_date(value: str, name: str) -> datetime.date:
    """Parses an ISO 8601 calendar date, normally ``YYYY-MM-DD``.

    Args:
        value: The text to parse; whatever `datetime.date.fromisoformat`
            accepts, which is wider than ``YYYY-MM-DD``.
        name: The field name, for the error message.

    Returns:
        The date.

    Raises:
        MisuseError: If `value` is not a string that
            `datetime.date.fromisoformat` accepts.
    """
    try:
        return datetime.date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise MisuseError(f"{name} must be YYYY-MM-DD") from error
