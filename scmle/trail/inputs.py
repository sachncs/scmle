"""The statements people make about a model: KPIs, acceptance, thresholds.

``inputs.json`` has two signed sections. The stakeholder's holds the business
KPIs mapped to metrics, the acceptance criteria and the approved expectation
of the observed outcome. The reviewer's holds the domain quality thresholds
and the evidence of the review, which names the records read and their
digests. Each section is signed with that person's own HMAC key
(``scmle.contracts.signing``); the signature proves possession of the key,
not identity.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import hmac
from typing import Any

from scmle.contracts import signing
from scmle.trail import records

SECTIONS = ("stakeholder", "reviewer")
EXPECTATION_FIELDS = ("kpi", "metric", "low", "high", "level", "basis")


def section_payload(record: Mapping[str, Any], role: str) -> dict[str, Any]:
    """Builds the content that a section's signature covers.

    Args:
        record: The inputs.
        role: `stakeholder` or `reviewer`.

    Returns:
        A mapping with the role, model, version, charter hash and the section
        body without its signature.
    """
    body = {
        key: value
        for key, value in dict(record[role]).items()
        if key != "signature"
    }
    return {
        "role": role,
        "model": record["model"],
        "version": record["version"],
        "charter_hash": record["charter_hash"],
        "body": body,
    }


def sign_inputs(
    record: Mapping[str, Any], keys: Mapping[str, str | bytes]
) -> dict[str, Any]:
    """Returns a copy of ``record`` with both sections signed.

    Args:
        record: The inputs, without signatures.
        keys: Signing key by role (``stakeholder``, ``reviewer``).

    Returns:
        The record with a `signature` in each section.
    """
    signed = {
        key: dict(value) if isinstance(value, Mapping) else value
        for key, value in record.items()
    }
    for role in SECTIONS:
        signed[role]["signature"] = signing.sign_payload(
            section_payload(signed, role), keys[role]
        )
    return signed


def signature_problem(
    record: Mapping[str, Any], role: str, key: str | bytes | None
) -> str | None:
    """Explains why a section's signature is not valid.

    Args:
        record: The inputs.
        role: `stakeholder` or `reviewer`.
        key: The role's key, or None if none was supplied.

    Returns:
        The reason (section absent, unsigned, no key supplied, or the signature
        does not verify), or None when the signature is valid.
    """
    section = record.get(role)
    if not isinstance(section, Mapping):
        return f"the {role} section is absent"
    signature = section.get("signature")
    if not signature:
        return f"the {role} section is unsigned"
    if key is None:
        return f"no {role} key was supplied to check the signature"
    expected = signing.sign_payload(section_payload(record, role), key)
    if not hmac.compare_digest(str(signature), expected):
        return f"the {role} signature does not verify"
    return None


def missing_fields(
    section: Mapping[str, Any], names: Sequence[str], message: str
) -> list[str]:
    """Lists a message for every required field a section lacks.

    Args:
        section: The mapping to check.
        names: The required field names, in the order to report them.
        message: A template with a `{name}` placeholder for the field.

    Returns:
        One formatted message per missing field.
    """
    return [message.format(name=name) for name in names if name not in section]


def structure_problems(record: Mapping[str, Any]) -> list[str]:
    """Lists missing or malformed fields in the inputs.

    Args:
        record: The inputs.

    Returns:
        One message per missing top-level field (`model`, `version`,
        `charter_hash`, `builder`), section field, KPI field, criterion field
        (for both acceptance criteria and reviewer thresholds), criterion
        operator not in `records.OPERATORS`, or expectation field.
    """
    found = missing_fields(
        record,
        ("model", "version", "charter_hash", "builder"),
        "missing {name}",
    )
    stakeholder = record.get("stakeholder", {})
    reviewer = record.get("reviewer", {})
    found.extend(
        missing_fields(
            stakeholder,
            ("name", "kpis", "acceptance", "expectation"),
            "stakeholder section lacks {name}",
        )
    )
    found.extend(
        missing_fields(
            reviewer,
            ("name", "thresholds", "evidence"),
            "reviewer section lacks {name}",
        )
    )
    for kpi in stakeholder.get("kpis", []):
        found.extend(
            missing_fields(
                kpi,
                ("kpi", "owner", "metric", "direction"),
                "a KPI lacks {name}",
            )
        )
    for criterion in [
        *stakeholder.get("acceptance", []),
        *reviewer.get("thresholds", []),
    ]:
        found.extend(
            missing_fields(
                criterion,
                ("id", "path", "op", "value"),
                "a criterion lacks {name}",
            )
        )
        if criterion.get("op") not in records.OPERATORS:
            found.append(f"criterion {criterion.get('id')} has a bad operator")
    found.extend(
        missing_fields(
            stakeholder.get("expectation", {}),
            EXPECTATION_FIELDS,
            "the expectation lacks {name}",
        )
    )
    return found
