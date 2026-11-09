"""Storing, signing and verifying a charter.

A charter file wraps the charter with its SHA-256 hash and, when a key is
available, an HMAC-SHA256 signature over the canonical payload. The signature
proves possession of the key, not the signer's identity. An unsigned charter
is allowed: it is hash-only and reported as unsigned.
"""

import dataclasses
import hmac
import json
from pathlib import Path
from typing import Any

from scmle.contracts import errors
from scmle.contracts import signing
from scmle.data.charter import contracts

SIGNATURE_FIELD = "signature"
RULES_FORMAT = 3
CAUSAL_FORMAT = 4
DECISION_FORMAT = 5
UNIT_FORMAT = 6
INGEST_FORMAT = 7
INGEST_ENTITY_FORMAT = 8
SUBGROUP_FORMAT = 9
UNCERTAINTY_FORMAT = 10
CONFIRMATION_COVERAGE_FORMAT = 11
BLOCK_FORMATS = (
    ("decision", "a 'decision' block", DECISION_FORMAT),
    ("unit_of_prediction", "'unit_of_prediction'", UNIT_FORMAT),
    ("ingest", "'ingest'", INGEST_FORMAT),
    ("uncertainty", "'uncertainty'", UNCERTAINTY_FORMAT),
)


@dataclasses.dataclass(frozen=True, slots=True)
class VerifiedCharter:
    """A charter read back from disk after its checks.

    Attributes:
        charter: The charter.
        charter_hash: Its recomputed SHA-256.
        signed: Whether a signature was present and verified.
    """

    charter: contracts.AnalysisCharter
    charter_hash: str
    signed: bool


def envelope(
    charter: contracts.AnalysisCharter, signing_key: str | bytes | None
) -> dict[str, Any]:
    """Builds the JSON wrapper that is stored on disk.

    Args:
        charter: The validated charter.
        signing_key: Key to sign with; when None the key is taken from the
            environment. An empty or missing key gives an unsigned wrapper.

    Returns:
        A dict with the charter format, the charter record, its hash, the
        signature algorithm and the signature (None when no key is
        available).
    """
    key = signing.resolve_key(signing_key)
    return {
        "format": charter.format_version,
        "charter": charter.to_dict(),
        "charter_hash": charter.charter_hash,
        "algorithm": signing.SIGNATURE_ALGORITHM,
        SIGNATURE_FIELD: (
            None
            if key is None
            else signing.sign_payload(charter.to_dict(), key)
        ),
    }


def write_charter(
    path: str | Path,
    charter: contracts.AnalysisCharter,
    signing_key: str | bytes | None = None,
) -> Path:
    """Writes the charter with its hash and an optional signature.

    Args:
        path: Destination file.
        charter: The validated charter.
        signing_key: Key to sign with; defaults to `SCMLE_SIGNING_KEY`.

    Returns:
        The path written, as a `Path`. Missing parent directories are
        created and the file is written as UTF-8 JSON with sorted keys.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(envelope(charter, signing_key), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target


def check_format(path: str | Path, record: dict[str, Any]) -> None:
    """Refuses a wrapper from a newer format, or a field its format lacks.

    A wrapper without a `format` is read as format 1. Format 3 added `rules`,
    format 4 the causal block, format 5 the decision layer, format 6 the unit
    of prediction, format 7 the ingest block and format 8 its entity and
    reconcile sub-blocks, format 9 causal subgroups and attestors, format 10
    the uncertainty block and format 11 its coverage claim on the
    confirmation rows; an older wrapper that carries one of them was edited
    or mislabelled.

    Args:
        path: Charter file, used in the error messages.
        record: The decoded wrapper.

    Raises:
        ConfigError: If the declared format is not an integer or is newer
            than `contracts.FORMAT_VERSION`, or an older format carries a
            field that a later format introduced.
    """
    declared = record.get("format", contracts.OLDEST_READABLE_FORMAT)
    if not isinstance(declared, int) or declared > contracts.FORMAT_VERSION:
        raise errors.ConfigError(
            f"charter {path} has format {declared!r}; this version reads "
            f"formats up to {contracts.FORMAT_VERSION}"
        )
    if declared < RULES_FORMAT and "rules" in record.get("charter", {}):
        raise errors.ConfigError(
            f"charter {path} declares format {declared} but carries 'rules', "
            f"which need format {RULES_FORMAT}"
        )
    estimand = record.get("charter", {}).get("estimand", {})
    if declared < CAUSAL_FORMAT and "causal" in estimand:
        raise errors.ConfigError(
            f"charter {path} declares format {declared} but carries a "
            f"'causal' block, which needs format {CAUSAL_FORMAT}"
        )
    for name, described, introduced in BLOCK_FORMATS:
        if declared < introduced and name in record.get("charter", {}):
            raise errors.ConfigError(
                f"charter {path} declares format {declared} but carries "
                f"{described}, which needs format {introduced}"
            )
    if declared < CONFIRMATION_COVERAGE_FORMAT and (
        record.get("charter", {}).get("uncertainty") or {}
    ).get("coverage_on_confirmation"):
        raise errors.ConfigError(
            f"charter {path} declares format {declared} but its uncertainty "
            "block carries 'coverage_on_confirmation', which needs format "
            f"{CONFIRMATION_COVERAGE_FORMAT}"
        )
    block = record.get("charter", {}).get("ingest") or {}
    causal_block = estimand.get("causal") or {}
    later = [
        name
        for name in ("subgroups", "attestors", "attestations")
        if name in causal_block
    ]
    if declared < SUBGROUP_FORMAT and later:
        raise errors.ConfigError(
            f"charter {path} declares format {declared} but its causal block "
            f"carries {later}, which need format {SUBGROUP_FORMAT}"
        )
    for name in ("entity", "reconcile"):
        if declared < INGEST_ENTITY_FORMAT and name in block:
            raise errors.ConfigError(
                f"charter {path} declares format {declared} but its ingest "
                f"block carries {name!r}, which needs format "
                f"{INGEST_ENTITY_FORMAT}"
            )


def read_verified(
    path: str | Path,
    signing_key: str | bytes | None = None,
    require_signature: bool = False,
) -> VerifiedCharter:
    """Reads a charter and verifies its hash and signature.

    Args:
        path: Charter file, wrapped or bare.
        signing_key: Key to verify with; defaults to `SCMLE_SIGNING_KEY`.
        require_signature: Fail unless a verified signature is present.

    A file that cannot be read, is not a JSON object, is not a valid
    charter, or fails `check_format` raises `ConfigError` (from
    `contracts.load_json`, `contracts.charter_from_record` and
    `check_format`). A bare charter has no wrapper and so no hash or
    signature to check.

    Returns:
        The charter, its recomputed hash, and whether a signature was
        present, a key was available and the signature verified. A signed
        file read without a key is returned with `signed` false unless
        `require_signature` is set.

    Raises:
        ArtifactError: If the algorithm is unknown, the stored hash differs
            from the recomputed one, the signature does not verify, or
            `require_signature` is set and the charter is bare, unsigned, or
            signed with no key available.
    """
    record = contracts.load_json(path)
    if "charter" not in record:
        charter = contracts.charter_from_record(record)
        if require_signature:
            raise errors.ArtifactError(f"charter {path} is unsigned")
        return VerifiedCharter(charter, charter.charter_hash, signed=False)
    if record.get("algorithm") != signing.SIGNATURE_ALGORITHM:
        raise errors.ArtifactError(f"charter {path}: unknown algorithm")
    check_format(path, record)
    charter = contracts.charter_from_record(record["charter"])
    if record.get("charter_hash") != charter.charter_hash:
        raise errors.ArtifactError(
            f"charter hash mismatch in {path}: the file was edited"
        )
    signature = record.get(SIGNATURE_FIELD)
    key = signing.resolve_key(signing_key)
    if signature is None:
        if require_signature:
            raise errors.ArtifactError(f"charter {path} is unsigned")
        return VerifiedCharter(charter, charter.charter_hash, signed=False)
    if key is None:
        if require_signature:
            raise errors.ArtifactError(
                f"charter {path} is signed but no signing key is configured"
            )
        return VerifiedCharter(charter, charter.charter_hash, signed=False)
    expected = signing.sign_payload(charter.to_dict(), key)
    if not hmac.compare_digest(str(signature), expected):
        raise errors.ArtifactError(
            f"charter signature does not verify for {path}"
        )
    return VerifiedCharter(charter, charter.charter_hash, signed=True)


def tighter(old: Any, new: Any) -> bool:
    """Compares two limits where a smaller value is tighter.

    Args:
        old: The current limit.
        new: The proposed limit.

    Returns:
        True when `new <= old`.
    """
    return bool(new <= old)


def check_amendment(
    old: contracts.AnalysisCharter, new: contracts.AnalysisCharter
) -> None:
    """Allows an amendment only if it tightens the budget.

    Only a smaller or equal `alpha_confirm`, `k_max` or initial exploration
    wealth may change; every other field must be identical.

    Args:
        old: The charter as signed.
        new: The amended charter.

    Raises:
        ConfigError: If any of those three values is larger in `new`, or any
            other field differs.
    """
    before, after = old.to_dict(), new.to_dict()
    budget_old, budget_new = old.error_budget, new.error_budget
    ok = (
        tighter(
            budget_old.confirmation.alpha_confirm,
            budget_new.confirmation.alpha_confirm,
        )
        and tighter(old.k_max, new.k_max)
        and tighter(
            budget_old.exploration.initial, budget_new.exploration.initial
        )
    )
    for record in (before, after):
        record.pop("k_max")
        record["error_budget"] = {
            key: value
            for key, value in record["error_budget"].items()
            if key not in ("confirmation", "exploration")
        }
    if not ok or before != after:
        raise errors.ConfigError(
            "an amendment may only tighten the budget (smaller alpha_confirm,"
            " K_max or exploration wealth); everything else must be unchanged"
        )
