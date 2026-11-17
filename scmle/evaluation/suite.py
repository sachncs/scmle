"""The post-selection diagnostic suite and its scheduling.

`run_suite` runs diagnostics against the run's finish deadline: it times
the model once (`gauge`) and shares the time left among the diagnostics by
their cost models (`Diagnostic.estimate`).
"""

import dataclasses
from typing import Any

from scmle.contracts import errors
from scmle.evaluation import bootstrap
from scmle.evaluation import diagnostics
from scmle.evaluation import fairness
from scmle.evaluation import robustness
from scmle.evaluation import shift
from scmle.evaluation import slices
from scmle.explainability import suite

# Rows predicted once to time the model (`pace`). Derivation: 1,000 rows
# took at most 10 ms on the runs described at `diagnostics.MARGIN`, and a
# smaller probe is dominated by the
# fixed call overhead, which only makes the estimate more cautious.
PROBE_ROWS = 1000


def lineup() -> tuple[diagnostics.Diagnostic, ...]:
    """Returns every diagnostic run after the final evaluation.

    Performance reports, subgroup scores, stress tests, distribution shift
    between development and test rows, and model explanations.

    Returns:
        The default diagnostics, then slice, fairness, robustness and
        adversarial-validation diagnostics, then the explanation
        diagnostics.
    """
    return (
        *panoply(),
        slices.Slices(),
        fairness.Fairness(),
        robustness.Robustness(),
        shift.Adversary(),
        *suite.panoply(),
    )


def panoply() -> tuple[diagnostics.Diagnostic, ...]:
    """Returns the standard diagnostics.

    Returns:
        One instance of each diagnostic, in the order they are run.
    """
    return (
        diagnostics.Contingency(),
        diagnostics.Calibration(),
        diagnostics.Reliability(),
        diagnostics.Sweep(),
        diagnostics.Residuals(),
        diagnostics.Errors(),
        bootstrap.Uncertainty(),
    )


def run_suite(
    inputs: diagnostics.DiagnosticInput,
    suite: tuple[diagnostics.Diagnostic, ...] | None = None,
    total_seconds: float | None = None,
    deadline: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Runs every applicable diagnostic; one failure never hides the others.

    The suite ends by `finish`, the earlier of `total_seconds` from now and
    `deadline`. The model is timed once on up to `PROBE_ROWS` rows (`pace`),
    then each diagnostic gets the larger of its smallest useful cost
    (`least`) and a share of the time left in proportion to its full cost
    (`estimate`), never past `finish`. A diagnostic whose smallest useful
    cost does not fit before `finish` is not started.

    Args:
        inputs: Fitted candidate and evaluation data.
        suite: Diagnostics to run; defaults to `panoply()`.
        total_seconds: Time allowance for the whole suite (None =
            unlimited).
        deadline: Monotonic time (on `diagnostics.clock`) by which the
            suite must end, the run's finish deadline (None = no deadline).

    Returns:
        Report per name for the diagnostics whose `applies_to` contains the
        task; a failure appears as `{"error": "<ErrorType>: <message>"}`
        and one that did not fit as `skipped` (`available` False, with the
        reason).
    """
    reports: dict[str, dict[str, Any]] = {}
    chosen = [
        d
        for d in (suite if suite is not None else panoply())
        if inputs.task in d.applies_to
    ]
    limits = [
        limit
        for limit in (
            None
            if total_seconds is None
            else diagnostics.clock() + total_seconds,
            deadline,
        )
        if limit is not None
    ]
    finish = min(limits) if limits else None
    if finish is not None and chosen and diagnostics.clock() < finish:
        inputs = dataclasses.replace(inputs, pace=gauge(inputs))
    for position, diagnostic in enumerate(chosen):
        limited = inputs
        if finish is not None:
            left = finish - diagnostics.clock()
            need = diagnostic.least(inputs)
            if need > left:
                reports[diagnostic.name] = diagnostics.skipped(
                    f"needs about {need:.2f} s, {max(left, 0.0):.2f} s left "
                    "before the finish deadline"
                )
                continue
            wanted = [d.estimate(inputs) for d in chosen[position:]]
            share = left * wanted[0] / max(sum(wanted), 1e-9)
            limited = dataclasses.replace(
                inputs,
                deadline=min(diagnostics.clock() + max(share, need), finish),
            )
        try:
            reports[diagnostic.name] = diagnostic.compute(limited)
        except Exception as error:  # noqa: BLE001 -- isolation: diagnostics are advisory
            # handled: recorded in the diagnostic report with its traceback
            failure = errors.Failure.of(error)
            reports[diagnostic.name] = {
                "error": failure.text,
                "failure": dict(failure.dump()),
            }
    return reports


def gauge(inputs: diagnostics.DiagnosticInput) -> float | None:
    """Times one prediction over the evaluation rows.

    Predicts the first `PROBE_ROWS` rows and scales linearly to all rows,
    so the fixed call overhead only makes the result more cautious.

    Args:
        inputs: Fitted candidate and evaluation data.

    Returns:
        The predicted seconds, or None when the model cannot predict (the
        diagnostics then report that failure themselves).
    """
    probe = inputs.features.iloc[:PROBE_ROWS]
    if probe.empty:
        return None
    began = diagnostics.clock()
    try:
        inputs.estimator.predict(probe)
    except Exception:  # noqa: BLE001 -- the diagnostics report the failure
        # handled: None means untimed; each diagnostic records its own error
        return None
    return (diagnostics.clock() - began) * len(inputs.features) / len(probe)
