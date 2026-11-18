"""The files of a charter run and how they are read."""

import json
from pathlib import Path
from typing import Any

from scmle.assurance.ledger import confirm
import scmle.assurance.ledger.ledger
from scmle.contracts import errors
from scmle.contracts import records

CHARTER_FILE = "analysis_charter.json"
CUSTODY_FILE = "custody_log.jsonl"
CONTROL_FILE = "negative_control.json"
BASELINE_FILE = "baseline_report.json"
CAUSAL_FILE = "causal_claims.json"


def migrate(kind: Any, row: Any) -> records.Layout:
    """Moves a stored claim or check row to its kind's current layout.

    Only the keys move; the row is not validated, so a malformed row is
    reported by the check that reads it rather than here.

    Args:
        kind: `qualification.Claim` or `qualification.Check`.
        row: The stored row, at any supported schema version.

    Returns:
        The row at the current version.
    """
    return records.load(kind.__name__, kind.migrations, row)


def recall(run_dir: str | Path) -> dict[str, Any]:
    """Reads ``confirmed_claims.json`` of a run.

    Args:
        run_dir: The run directory.

    Returns:
        The parsed claims document at the current version (`alpha`,
        `digests.survivors`, `heads.ledger`, `second`), whichever version
        the run wrote.

    Raises:
        ArtifactError: If the file is missing or unreadable, is not a JSON
            object with a `claims` key, or has an unsupported version.
    """
    path = Path(run_dir) / confirm.CLAIMS_FILE
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise errors.ArtifactError(f"cannot read {path}: {error}") from error
    if not isinstance(document, dict) or "claims" not in document:
        raise errors.ArtifactError(f"{path} is not a claims document")
    try:
        return records.load(
            "Confirmation", confirm.Confirmation.migrations, document
        )
    except errors.InputError as error:
        raise errors.ArtifactError(f"{path}: {error}") from error


def rehydrate(run_dir: str | Path) -> scmle.assurance.ledger.ledger.Ledger:
    """Loads the run's ledger (verifying its chain and replaying it).

    Args:
        run_dir: The run directory.

    Returns:
        The loaded ledger.

    Raises:
        ArtifactError: If the ledger is missing or fails verification.
    """
    try:
        return scmle.assurance.ledger.ledger.Ledger.load(
            Path(run_dir) / scmle.assurance.ledger.ledger.FILE_NAME
        )
    except scmle.assurance.ledger.ledger.LedgerError as error:
        raise errors.ArtifactError(str(error)) from error


def read_optional(path: Path) -> dict[str, Any] | None:
    """Reads a JSON object file.

    Args:
        path: The file.

    Returns:
        The object, or None when the file is missing, unreadable or not a JSON
        object.
    """
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # handled: documented None for missing/unreadable file
        return None
    return loaded if isinstance(loaded, dict) else None
