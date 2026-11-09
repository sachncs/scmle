"""The registered coverage claim, tested on the sealed rows `C`.

A charter whose `uncertainty` block asks for it (format 11,
`coverage_on_confirmation`) turns the coverage claim into a registered claim
of the ledger, next to the baseline and causal claims. It is about one served
interval or set, named before any row is read: the `residual` interval for a
regression bundle, the pooled prediction set for a classification bundle, at
`conformal.DEFAULT_COVERAGE`. The registration hashes that choice, the
tolerance, the test level and the hash of `coverage_claim.json` (the claim
fixed before the final test rows), so the claim is frozen before `C` opens.

`C` is opened once, by the ledger, after the model and its calibration are
frozen. The claim holds if coverage is at least `level - tolerance`: a
one-sided exact binomial test of "coverage is at most `level - tolerance`"
on the covered count. It is `confirmed` when that test rejects at `ALPHA`
(equivalently, the exact one-sided Clopper-Pearson lower limit exceeds
`level - tolerance`) and `unverified` otherwise, with the numbers reported
either way. Nothing is retuned after the rows are seen. The other methods
and levels of the bundle are reported on the same rows as description; they
carry no status.

`ALPHA` is fixed here and is not drawn from the charter's error budget: one
test of one claim is registered. The rows of `C` must be exchangeable with
the calibration rows for the guarantee to mean anything; the report says so
and the status makes no claim about that.
"""

from collections.abc import Mapping
from typing import Any

import numpy
import pandas
from scipy import stats

from scmle.contracts import tasks
from scmle.data.ledger import confirm
from scmle.data.ledger import ledger
from scmle.evaluation import conformal
from scmle.evaluation import coverage
from scmle.evaluation import coverage_run
from scmle.evaluation import exchangeability
from scmle.evaluation import task_gate

KIND = "coverage"
CLAIM_ID = "coverage_nominal"
DOCUMENT = "coverage_confirmation.json"
ALPHA = 0.05
CONFIRMED = "confirmed"
UNVERIFIED = "unverified"
NOTE = (
    "the claim is exact for exchangeable rows; the status does not test "
    "exchangeability between the calibration rows and the confirmation rows"
)


def primary_choice(task: str) -> dict[str, Any] | None:
    """The one served interval or set the claim is about, or None.

    Args:
        task: The task type value.

    Returns:
        The method (`residual` for interval tasks, `pooled` for set tasks),
        the level, the tolerance and the test level; None for any other
        task.
    """
    if task in coverage_run.INTERVAL_TASKS:
        method = "residual"
    elif task in coverage_run.SET_TASKS:
        method = "pooled"
    else:
        return None
    return {
        "method": method,
        "level": conformal.DEFAULT_COVERAGE,
        "tolerance": coverage.DEFAULT_TOLERANCE,
        "alpha": ALPHA,
    }


def registration_body(
    claim_sha256: str, choice: Mapping[str, Any]
) -> dict[str, Any]:
    """What is frozen before `C` opens.

    Args:
        claim_sha256: Hash of `coverage_claim.json`.
        choice: The primary choice from `primary_choice`.

    Returns:
        The claim hash, the primary choice and the decision rule in words.
    """
    return {
        "coverage_claim_sha256": claim_sha256,
        "primary": dict(choice),
        "rule": "one-sided exact binomial test that coverage is at least "
        "level - tolerance, at ALPHA, on the one opening of C",
    }


def registration(
    claim_sha256: str, choice: Mapping[str, Any]
) -> confirm.RegisteredClaim:
    """The ledger registration of the claim.

    Args:
        claim_sha256: Hash of `coverage_claim.json`.
        choice: The primary choice from `primary_choice`.

    Returns:
        The registered claim, whose statement names the method, level and
        floor `level - tolerance` and whose digest covers the frozen body.
    """
    floor = float(choice["level"]) - float(choice["tolerance"])
    return confirm.RegisteredClaim(
        CLAIM_ID,
        KIND,
        f"the {choice['method']} interval or set at level {choice['level']:g} "
        f"covers at least {floor:g} of the rows of C",
        ledger.digest(registration_body(claim_sha256, choice)),
    )


def floor_p_value(covered: int, rows: int, floor: float) -> float:
    """One-sided exact p-value of "coverage is at most `floor`".

    Small values say the coverage is above the floor.

    Args:
        covered: Number of rows inside their interval or set.
        rows: Number of rows scored.
        floor: The coverage level the null hypothesis assumes at most.

    Returns:
        The probability of at least `covered` hits in `rows` trials at rate
        `floor`; 1.0 when `rows` is zero.
    """
    if rows == 0:
        return 1.0
    return float(stats.binom.sf(covered - 1, rows, floor))


def decide(
    covered: int, rows: int, choice: Mapping[str, Any]
) -> dict[str, Any]:
    """The test and its status for a covered count.

    Args:
        covered: Number of rows inside their interval or set.
        rows: Number of rows scored.
        choice: The primary choice from `primary_choice`.

    Returns:
        The counts, observed coverage (None without rows), floor, p-value,
        one-sided lower limit, `ALPHA` and the status: `confirmed` when the
        p-value is at most `ALPHA`, otherwise `unverified`.
    """
    floor = float(choice["level"]) - float(choice["tolerance"])
    p_value = floor_p_value(covered, rows, floor)
    lower = (
        coverage.clopper_pearson(covered, rows, 1.0 - 2.0 * ALPHA)[0]
        if rows
        else 0.0
    )
    return {
        "rows": rows,
        "covered": covered,
        "coverage": covered / rows if rows else None,
        "floor": floor,
        "p_value": p_value,
        "one_sided_lower_limit": lower,
        "alpha": ALPHA,
        "status": CONFIRMED if p_value <= ALPHA else UNVERIFIED,
    }


def capped(task: str, status: str) -> str:
    """The status after the task type's gate (`task_gate.tier_ceiling`).

    A task type that has not passed its measured gate cannot carry a
    `confirmed` status; the cap never raises a status.

    Args:
        task: The task type value.
        status: The status from the coverage test.

    Returns:
        `status` unchanged unless it is `confirmed`; a `confirmed` status
        stays so if the task type passed its gate and becomes `unverified`
        otherwise.
    """
    if status != CONFIRMED:
        return status
    return task_gate.tier_ceiling(tasks.TaskType(task), status)


def inside_of(
    bundle: Any,
    features: pandas.DataFrame,
    target: Any,
    choice: Mapping[str, Any],
) -> numpy.ndarray:
    """Whether the primary interval or set holds, per released row.

    Args:
        bundle: The frozen bundle that serves intervals and sets.
        features: Features of the released rows.
        target: True values (regression) or labels (classification).
        choice: The primary choice from `primary_choice`.

    Returns:
        A boolean array: True where the value lies in the interval, or the
        label is in the pooled prediction set.
    """
    level = float(choice["level"])
    if choice["method"] == "residual":
        band = bundle.predict_interval(
            features, coverage=level, method="residual"
        )
        values = numpy.asarray(target, dtype=float)
        return numpy.asarray(
            (values >= band["lower"].to_numpy())
            & (values <= band["upper"].to_numpy())
        )
    sets = bundle.predict_set(features, level, conditional=None)
    return numpy.array(
        [
            label in found
            for label, found in zip(list(target), sets, strict=True)
        ]
    )


def evaluate_document(
    bundle: Any,
    features: pandas.DataFrame,
    target: Any,
    claim_sha256: str,
    choice: Mapping[str, Any],
    task: str = "regression",
) -> dict[str, Any]:
    """`coverage_confirmation.json` for the released rows of `C`.

    The task type's gate (`task_gate_status.json`) caps the status: a type
    that has not passed its gate is reported `unverified` whatever the test
    says, and the document records the gate and whether the cap applied.

    Args:
        bundle: The frozen bundle, calibration included.
        features: Features of the released rows of `C`.
        target: True values or labels of those rows.
        claim_sha256: Hash of `coverage_claim.json`.
        choice: The primary choice from `primary_choice`.
        task: The task type value.

    Returns:
        The document: the claim identity, the tested result with its
        status after the cap, the task gate, and a description of the other
        methods and levels and of exchangeability on the same rows.
    """
    inside = inside_of(bundle, features, target, choice)
    result = decide(int(inside.sum()), len(inside), choice)
    tested = result["status"]
    result["status"] = capped(task, tested)
    kind = (
        bundle.conformal.kind if bundle.conformal is not None else "regression"
    )
    levels = (0.5, 0.8, 0.9, 0.95)
    others = (
        coverage_run.regression_reports(bundle, features, target, levels)
        if kind == "regression"
        else coverage_run.classification_reports(
            bundle, features, target, levels
        )
    )
    return {
        "schema_version": 1,
        "claim_id": CLAIM_ID,
        "coverage_claim_sha256": claim_sha256,
        "primary": dict(choice),
        **result,
        "task_gate": {
            "task": task,
            "gate": task_gate.status(tasks.TaskType(task)),
            "capped": result["status"] != tested,
        },
        "rows_read": "once, by the ledger's opening of C",
        "note": NOTE,
        "described_on_the_same_rows": {
            "note": "description only; these carry no status",
            "methods": others,
            "exchangeability": exchangeability.describe(
                bundle, features, target
            ),
        },
    }


def not_evaluated(reason: str) -> dict[str, Any]:
    """The document of a claim that `C` could not test.

    Args:
        reason: Why the claim was not evaluated.

    Returns:
        A document with the not-evaluated status and the reason.
    """
    return {
        "schema_version": 1,
        "claim_id": CLAIM_ID,
        "status": confirm.NOT_EVALUATED,
        "reasons": [reason],
    }


def build_runner(
    bundle: Any,
    task: str,
    claim_sha256: str,
) -> confirm.ClaimRunner | None:
    """The runner of the registered claim, or None when there is none.

    There is none when the task type has no served interval or set, or the
    bundle holds no calibration.

    Args:
        bundle: The frozen bundle, calibration included.
        task: The task value.
        claim_sha256: Hash of `coverage_claim.json`.

    Returns:
        A runner holding the registration and the evaluation of `C`, or
        None.
    """
    choice = primary_choice(task)
    if choice is None or bundle.conformal is None:
        return None

    def evaluate(
        features: pandas.DataFrame, target: pandas.Series
    ) -> confirm.ClaimRun:
        document = evaluate_document(
            bundle, features, target, claim_sha256, choice, task
        )
        return confirm.ClaimRun(
            {CLAIM_ID: str(document["status"])}, {DOCUMENT: document}
        )

    return confirm.ClaimRunner((registration(claim_sha256, choice),), evaluate)
