"""Tells an infrastructure fault from bad data from a worse model.

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
    There are signals and every one of them lacked enough data to be
    judged. An empty list of signals gives ``none``.

Caveats are returned with the verdict where it rests on less than it seems
to: drift with no labels yet does not show harm, a performance drop while
some data-quality signals could not be judged does not exclude a data cause,
and an infrastructure fault with no loss observed says nothing about quality.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import enum
from typing import Any

import scmle.operations.signals


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
        categories: Kinds with at least one breached signal, ordered by
            value.
        breached: Names of the breached signals, sorted.
        unobserved: Names of signals with too little data, sorted.
        caveats: What the verdict does not establish.
        severity: ``info`` for no breach, ``warning`` for any other verdict
            (including indeterminate), ``critical`` when `availability` is
            breached or the verdict is mixed with an infrastructure breach.
        response: The suggested first step; ``no action`` for no breach and
            ``collect more data`` for an indeterminate verdict.
    """

    verdict: Verdict
    categories: tuple[scmle.operations.signals.Category, ...]
    breached: tuple[str, ...]
    unobserved: tuple[str, ...]
    caveats: tuple[str, ...]
    severity: str
    response: str

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-ready record.

        Returns:
            The verdict, categories, breached and unobserved signal names,
            caveats, severity and response, with enums as their values.
        """
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
    breached: Mapping[str, scmle.operations.signals.Signal],
    unobserved: Sequence[str],
) -> list[str]:
    """Lists what a verdict leaves open.

    Args:
        verdict: The verdict reached.
        breached: The breached signals, by name.
        unobserved: Names of the signals with insufficient data.

    Returns:
        Messages for what the verdict cannot exclude: drift without
        outcomes (a data-quality verdict whose breaches are only
        `feature_psi_max` or `unseen_category_rate` while `windowed_loss` is
        unobserved), a data cause not judged (a model-performance verdict
        with any data signal unobserved), or model quality not observed
        (an infrastructure verdict with `windowed_loss` unobserved); empty
        if none applies.
    """
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


def classify(
    signals: Sequence[scmle.operations.signals.Signal],
) -> Classification:
    """Classifies a set of signals (see the module docstring).

    Args:
        signals: The evaluated signals.

    Returns:
        The verdict with its categories, breached and unobserved signals,
        caveats, severity and recommended response. No signals at all gives
        the ``none`` verdict.
    """
    breached = {s.name: s for s in signals if s.breached}
    unobserved = tuple(
        sorted(
            s.name
            for s in signals
            if s.status is scmle.operations.signals.Status.INSUFFICIENT_DATA
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
        verdict is Verdict.MIXED
        and scmle.operations.signals.Category.INFRASTRUCTURE in categories
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
    """Builds the document a pager or ticket sender posts.

    Args:
        classification: The classification to announce.
        incident_id: Identifier of the incident.
        model_version: Version of the model that is served.
        artifact_digest: Digest of the served artifact.
        trace_id: Trace id of the evaluation.
        sent_at: Timestamp of the notification.

    Returns:
        The notification fields: severity, verdict, a one-line summary
        (`<verdict>: <breached names>`, or `no breached signal`), breached
        signals, caveats, response and the identifying fields
        (`incident_id`, `model_version`, `artifact_digest`, `trace_id`,
        `sent_at`).
    """
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
