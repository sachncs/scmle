"""`stability_report.json` and `shift_report.json` for a charter run.

Both documents carry a `schema_version` and are checked by `conform`
and `validate_shift`, which raise `ValueError` on a missing key or an unknown
status. Specification-curve data (one effect and interval per specification)
and the choice-to-result attribution are inside each finding.
"""

from collections.abc import Mapping
from typing import Any

from scmle.assurance.robustness import periods
from scmle.assurance.robustness import qualify
from scmle.assurance.robustness import shift
from scmle.assurance.robustness import stability
from scmle.contracts import errors

# Version 2: the nested records spell their keys in single words (the
# stability rule, the permutation curve, coverage cells, forward findings and
# drift rows; see each record's `to_dict`). Version 1 documents still pass
# `conform` and `validate_shift`, whose keys did not change.
SCHEMA_VERSION = 2
STABILITY_KEYS = (
    "schema_version",
    "status",
    "reason",
    "estimator",
    "rule",
    "grid",
    "certificate",
    "counts",
    "fragile",
    "findings",
    "confirmation_filter",
    "guarantee",
)
SHIFT_KEYS = (
    "schema_version",
    "status",
    "reason",
    "agreement",
    "c2st",
    "mmd",
    "features",
    "coverage",
    "periods",
    "forward",
    "certificate",
    "assumptions",
)
FINDING_KEYS = ("name", "status", "reason", "specs", "attribution")
STABILITY_STATUSES = ("evaluated", "not_evaluated")
FINDING_STATUSES = (
    stability.STABLE,
    stability.FRAGILE,
    stability.NOT_EVALUABLE,
)


def stability_report(
    found: qualify.Robustness, filter_note: str | None = None
) -> dict[str, Any]:
    """Builds the `stability_report.json` document.

    Args:
        found: The robustness results.
        filter_note: What the second stage did with the fragile findings, or
            None when no second stage ran (the findings were confirmed or
            left unconfirmed on their own merits).

    Returns:
        The JSON-ready document, with the keys of `STABILITY_KEYS`.
    """
    grid = found.grid
    counts = {
        status: sum(1 for f in found.findings if f.status == status)
        for status in FINDING_STATUSES
    }
    evaluated = bool(found.findings)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "evaluated" if evaluated else "not_evaluated",
        "reason": (
            "survivors re-run over the declared specification grid"
            if evaluated
            else found.stability_note
        ),
        "estimator": "Spearman rank correlation of one feature with the "
        "target on the exploration set X; bootstrap resamples whole groups "
        "or time blocks when the data have them",
        "rule": found.rule.to_dict(),
        "grid": {
            "declaration_sha256": None
            if grid is None
            else grid.declaration_sha256,
            "total": None if grid is None else grid.total,
            "run": None if grid is None else len(grid.specs),
            "subset": None if grid is None else grid.subset,
            "choices": [c.to_dict() for c in found.applicable],
            "not_run": list(found.notes),
            "inclusion_note": "first_half and second_half refer to the "
            "stored row order",
        },
        "certificate": {
            "gate": found.stability_gate.value,
            "note": found.store_note
            or found.rule_note
            or (
                "a current passing certificate lets a fragile verdict "
                "quarantine"
                if found.stability_gate.value == "gating"
                else "no current passing certificate: a fragile verdict is a "
                "warning"
            ),
        },
        "counts": counts,
        "fragile": list(found.fragile),
        "findings": [f.to_dict() for f in found.findings],
        "confirmation_filter": filter_note
        or "not applied: the charter does not ask for fragile survivors to "
        "be withdrawn, so fragile findings were still confirmed or left "
        "unconfirmed on their own merits",
        "guarantee": "Stability is relative to the declared specification "
        "set and is not a confirmation. Confirmation comes only from the "
        "single test on the confirmation set.",
    }


def shift_report(found: qualify.Robustness) -> dict[str, Any]:
    """Builds the `shift_report.json` document.

    Args:
        found: The robustness results.

    Returns:
        The JSON-ready document, with the keys of `SHIFT_KEYS`.
    """
    result = found.shift.to_dict()
    gates = {
        shift.C2ST_ID: found.c2st_gate.value,
        shift.MMD_ID: found.mmd_gate.value,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": result["status"],
        "reason": result["reason"],
        "agreement": result["agreement"],
        "c2st": result["c2st"],
        "mmd": result["mmd"],
        "features": result["features"],
        "coverage": [row.to_dict() for row in found.coverage],
        "periods": found.drift.to_dict(),
        "forward": found.forward.to_dict(),
        "certificate": {"gates": gates, "note": found.store_note},
        "assumptions": [
            "Selection into the deployment sample is assumed random.",
            (
                "Two-sample tests detect covariate shift, not label or concept "
                "shift."
            ),
            "A pass means no shift was detected, not that none exists.",
            (
                "Exact only under exchangeability of the sides; clustered or "
                "temporal data need block-matched samples."
            ),
        ],
    }


def require(document: dict[str, Any], keys: tuple[str, ...], name: str) -> None:
    """Checks that a document has the expected keys and schema version.

    Args:
        document: The document to check.
        keys: The keys it must contain.
        name: What the document is, for the error message.

    Raises:
        ValueError: When a key is missing or `schema_version` is not a
            supported one (1 up to `SCHEMA_VERSION`).
    """
    missing = [k for k in keys if k not in document]
    if missing:
        raise errors.InputError(f"{name} is missing {missing}")
    if document["schema_version"] not in range(1, SCHEMA_VERSION + 1):
        raise errors.InputError(f"{name} has an unsupported schema_version")


def conform(document: dict[str, Any]) -> None:
    """Checks the structure of a stability report.

    Args:
        document: The stability report.

    Raises:
        ValueError: On a missing key, an unknown status or a finding that
            lacks its curve data.
    """
    require(document, STABILITY_KEYS, "stability report")
    if document["status"] not in STABILITY_STATUSES:
        raise errors.InputError("stability report has an unknown status")
    for finding in document["findings"]:
        require(
            {"schema_version": SCHEMA_VERSION, **finding},
            ("schema_version", *FINDING_KEYS),
            "finding",
        )
        if finding["status"] not in FINDING_STATUSES:
            raise errors.InputError("a finding has an unknown status")


def validate_shift(document: dict[str, Any]) -> None:
    """Checks the structure of a shift report.

    Args:
        document: The shift report.

    Raises:
        ValueError: On a missing key, an unknown status, or a `pass` without
            a classifier test (a pass needs a reference).
    """
    require(document, SHIFT_KEYS, "shift report")
    status = document["status"]
    if status not in {s.value for s in shift.Status}:
        raise errors.InputError("shift report has an unknown status")
    if status == shift.Status.PASS.value and document["c2st"] is None:
        raise errors.InputError("a pass needs a deployment sample and a test")
    periodic(document["periods"])


def periodic(record: Mapping[str, Any]) -> None:
    """Checks the per-period drift record of a shift report.

    Reports written before per-period drift was implemented say `not_run`,
    which still passes.

    Args:
        record: The `periods` entry of the shift report.

    Raises:
        ValueError: On a missing `status` or `reason`, or an unknown status.
    """
    missing = [k for k in ("status", "reason") if k not in record]
    if missing:
        raise errors.InputError(f"shift report periods are missing {missing}")
    known = {s.value for s in shift.Status} | {periods.LEGACY}
    if record["status"] not in known:
        raise errors.InputError("shift report has an unknown per-period status")
