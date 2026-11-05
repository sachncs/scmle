"""Tell an infrastructure fault from bad data from a worse model.

The three have different owners and different responses, so mixing them up
costs time: restarting a service does not fix a drifted feed, and retraining
does not fix a failing dependency. `classify` reads the breached `Signal`
rows and answers with one of

``none``
    Nothing breached.
``infrastructure``, ``data_quality``, ``model_performance``
    Breaches of exactly one kind.
``mixed``
    Breaches of more than one kind. The classifier does not rank them: a
    data fault can cause a performance drop and an outage can bias the labels
    that arrive, so which came first is a question for the responder, and
    guessing would send them to the wrong runbook.
``indeterminate``
    Not one signal had enough data to be judged.

Caveats are returned with the verdict where it rests on less than it seems
to: drift with no labels yet does not show harm, and a performance drop with
no data-quality signals judged does not exclude a data cause.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import enum
from typing import Any

from scmle.operations import signals as signals_module

Category = signals_module.Category


class Verdict(enum.StrEnum):
    """What the breached signals add up to."""

    NONE = "none"
    INFRASTRUCTURE = "infrastructure"
    DATA_QUALITY = "data_quality"
    MODEL_PERFORMANCE = "model_performance"
    MIXED = "mixed"
    INDETERMINATE = "indeterminate"


RESPONSE = {
    Verdict.INFRASTRUCTURE: (
        "check the runtime and its dependencies; restart or roll back the "
        "release"
    ),
    Verdict.DATA_QUALITY: (
        "inspect the upstream feed and the request schema; do not retrain "
        "on this data"
    ),
    Verdict.MODEL_PERFORMANCE: (
        "open a retraining proposal through the escalation path; a person "
        "decides"
    ),
    Verdict.MIXED: (
        "more than one kind of fault; start with infrastructure, then data, "
        "and keep the performance question open"
    ),
}


@dataclasses.dataclass(frozen=True, slots=True)
class Classification:
    """The verdict and the evidence behind it.

    Attributes:
        verdict: See `Verdict`.
        categories: Kinds with at least one breached signal.
        breached: Names of the breached signals.
        unobserved: Names of signals with too little data.
        caveats: What the verdict does not establish.
        severity: ``info``, ``warning`` or ``critical``.
        response: The suggested first step.
    """

    verdict: Verdict
    categories: tuple[Category, ...]
    breached: tuple[str, ...]
    unobserved: tuple[str, ...]
    caveats: tuple[str, ...]
    severity: str
    response: str

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready record."""
        return {
            "verdict": self.verdict.value,
            "categories": [c.value for c in self.categories],
            "breached": list(self.breached),
            "unobserved": list(self.unobserved),
            "caveats": list(self.caveats),
            "severity": self.severity,
            "response": self.response,
        }


def caveats_for(
    verdict: Verdict,
    breached: Mapping[str, signals_module.Signal],
    unobserved: Sequence[str],
) -> list[str]:
    """What a verdict leaves open."""
    found = []
    drift_only = set(breached) <= {"feature_psi_max", "unseen_category_rate"}
    if (
        verdict is Verdict.DATA_QUALITY
        and "windowed_loss" in unobserved
        and drift_only
    ):
        found.append(
            "drift is not harm: no outcomes have arrived to show an effect "
            "on the model"
        )
    if verdict is Verdict.MODEL_PERFORMANCE:
        data_unjudged = [
            n
            for n in unobserved
            if n
            in {
                "feature_psi_max",
                "feature_missing_increase",
                "unseen_category_rate",
            }
        ]
        if data_unjudged:
            found.append(
                "a data cause is not excluded: "
                + ", ".join(sorted(data_unjudged))
                + " could not be judged"
            )
    if verdict is Verdict.INFRASTRUCTURE and "windowed_loss" in unobserved:
        found.append("model quality was not observed during the fault")
    return found


def classify(signals: Sequence[signals_module.Signal]) -> Classification:
    """Classifies a set of signals (see the module docstring)."""
    breached = {s.name: s for s in signals if s.breached}
    unobserved = tuple(
        sorted(
            s.name
            for s in signals
            if s.status is signals_module.Status.INSUFFICIENT_DATA
        )
    )
    categories = tuple(
        sorted({s.category for s in breached.values()}, key=lambda c: c.value)
    )
    if signals and len(unobserved) == len(signals):
        verdict = Verdict.INDETERMINATE
    elif not categories:
        verdict = Verdict.NONE
    elif len(categories) > 1:
        verdict = Verdict.MIXED
    else:
        verdict = Verdict(categories[0].value)
    severity = "info"
    if verdict is not Verdict.NONE:
        severity = "warning"
    if "availability" in breached or (
        verdict is Verdict.MIXED and Category.INFRASTRUCTURE in categories
    ):
        severity = "critical"
    if verdict is Verdict.INDETERMINATE:
        severity = "warning"
    return Classification(
        verdict,
        categories,
        tuple(sorted(breached)),
        unobserved,
        tuple(caveats_for(verdict, breached, unobserved)),
        severity,
        RESPONSE.get(
            verdict,
            "collect more data"
            if verdict is Verdict.INDETERMINATE
            else "no action",
        ),
    )


def notification(
    classification: Classification,
    incident_id: str,
    model_version: str,
    artifact_digest: str,
    trace_id: str,
    sent_at: str,
) -> dict[str, Any]:
    """The document a pager or ticket sender posts for a classification."""
    return {
        "incident_id": incident_id,
        "severity": classification.severity,
        "verdict": classification.verdict.value,
        "summary": (
            f"{classification.verdict.value}: "
            + (", ".join(classification.breached) or "no breached signal")
        ),
        "breached": list(classification.breached),
        "caveats": list(classification.caveats),
        "response": classification.response,
        "model_version": model_version,
        "artifact_digest": artifact_digest,
        "trace_id": trace_id,
        "sent_at": sent_at,
    }
