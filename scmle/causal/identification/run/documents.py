"""The five documents of a causal run and the claim file that hashes them."""

from collections.abc import Mapping
import dataclasses
import hashlib
import json
from typing import Any

from scmle.assurance.charter import causal
from scmle.causal.identification import select
from scmle.causal.identification import tier

DOCUMENTS = (
    "identification_plan.json",
    "identification_report.json",
    "falsification_report.json",
    "sensitivity_report.json",
    "causal_claims.json",
)


LIMITS = (
    "confirmed means: under the attested assumptions the interval has its "
    "measured coverage and the battery would have flagged a violation of the "
    "certified size; it does not say the assumptions are true, and a "
    "confounder the negative controls do not share passes undetected"
)


@dataclasses.dataclass(frozen=True)
class Outcome:
    """The result of a causal run.

    Attributes:
        documents: File name to JSON-ready content (`DOCUMENTS`).
        decision: The tier decision.
        hashes: File name to SHA-256 of the canonical JSON.
    """

    documents: dict[str, dict[str, Any]]
    decision: tier.Assignment
    hashes: dict[str, str]


def digest(document: Mapping[str, Any]) -> str:
    """Hashes the canonical JSON of a document.

    Args:
        document: The JSON-ready document; values that are not JSON types are
            converted with `str`.

    Returns:
        The SHA-256 hex digest of the document's JSON with sorted keys and
        compact separators.
    """
    text = json.dumps(
        document, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def refuse(
    plan: select.Plan,
    spec: causal.Question,
    reason: str | None = None,
    dag_checks: Mapping[str, Any] | None = None,
) -> Outcome:
    """Builds the five documents of a claim that is not identified.

    Args:
        plan: The selector's plan.
        spec: The charter's causal block.
        reason: Why the claim is refused, or None.
        dag_checks: What was found about the declared DAG, or None.

    Returns:
        The documents, their hashes and a tier decision with no tier.
    """
    report = dict(select.refusal(plan, spec))
    if reason:
        report["reason"] = reason
    if dag_checks is not None:
        report["dag"] = dict(dag_checks)
    decision = tier.decide(
        tier.Inputs(
            identified=False,
            needed=(),
            attestors={},
            confirmatory=False,
            sound=False,
            gates=(),
            interval="",
            ran=False,
            passed=False,
            negative=False,
            gate="",
            replicated=None,
            holds=None,
            sensitivity=None,
            unit=False,
            instrument=False,
            certified=(),
        )
    )
    return assemble(plan, spec, report, {}, {}, decision, "", ())


def assemble(
    plan: select.Plan,
    spec: causal.Question,
    report: dict[str, Any],
    falsification: dict[str, Any],
    sens: dict[str, Any],
    decision: tier.Assignment,
    strategy: str,
    needed: tuple[str, ...],
    attestors: Mapping[str, str] | None = None,
    certified: tuple[str, ...] = (),
    subgroups: Mapping[str, Any] | None = None,
    attestation_report: Mapping[str, Any] | None = None,
) -> Outcome:
    """Builds the documents and the claim file that hashes the others.

    Args:
        plan: The selector's plan.
        spec: The charter's causal block.
        report: The identification report.
        falsification: The falsification report.
        sens: The sensitivity report.
        decision: The tier decision.
        strategy: Name of the chosen strategy, or an empty string (recorded
            as null in the claim file).
        needed: The assumptions the strategy relies on.
        attestors: Assumption to the named attestor.
        certified: Certificate gates used.
        subgroups: The labelled subgroup report, or None.
        attestation_report: What was decided about the signed attestations, or
            None.

    Returns:
        The documents (the claim file included), the tier decision and the hash
        of every document.
    """
    documents = {
        "identification_plan.json": {
            **plan.to_dict(),
            "causal_spec": spec.to_dict(),
        },
        "identification_report.json": report,
        "falsification_report.json": falsification,
        "sensitivity_report.json": sens,
    }
    hashes = {k: digest(v) for k, v in documents.items()}
    claims: dict[str, Any] = {
        "status": decision.status,
        "strategy": strategy or None,
        "estimand": spec.estimand,
        "tier": None if decision.tier is None else decision.tier.value,
        "conditions": decision.conditions,
        "reasons": list(decision.reasons),
        "action": decision.action,
        "assumptions": {a: (attestors or {}).get(a, "") for a in needed},
        "certified_gates_used": sorted(certified),
        "hashes": dict(hashes),
        "limits": LIMITS,
    }
    if subgroups is not None:
        claims["subgroups"] = {
            "rule": subgroups["rule"],
            "alpha": subgroups["alpha"],
            "family_size": subgroups["family_size"],
            "rejected": list(subgroups["rejected"]),
            "gate": subgroups["gate"]["status"],
        }
    if attestation_report is not None:
        claims["attestations"] = dict(attestation_report)
    documents["causal_claims.json"] = claims
    hashes["causal_claims.json"] = digest(claims)
    return Outcome(documents, decision, hashes)


def arbitrate(spec: causal.Question) -> dict[str, Any]:
    """States the primary hypothesis `theta = 0` in words.

    Args:
        spec: The charter's causal block.

    Returns:
        The hypothesis sentence and how the result is labelled.
    """
    return {
        "hypothesis": (
            f"{spec.estimand} of {spec.treatment} on {spec.outcome} = 0"
        ),
        "labelled": "causal, under the attested assumptions",
    }


def prerequisites(strategy: str, spec: causal.Question) -> tuple[str, ...]:
    """Lists the attested assumptions the chosen strategy relies on.

    Args:
        strategy: Registry name of the strategy.
        spec: The charter's causal block.

    Returns:
        The strategy's required assumptions, with the unconfoundedness
        requirement replaced by the charter's matching assumption words.
    """
    required = select.STRATEGIES[strategy].assumptions
    out: list[str] = []
    for word in required:
        if word == "unconfoundedness_given":
            out.extend(
                item
                for item in spec.assumptions
                if item.startswith(causal.UNCONFOUNDEDNESS_PREFIX)
            )
        else:
            out.append(word)
    return tuple(out)
