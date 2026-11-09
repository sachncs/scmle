"""The admissibility rule by estimand type.

Per field the decision is `admit`, `quarantine` (excluded, reversible, reason
logged) or `admit_flagged`.

* `predictive`: a field is admissible iff its value is known before the
  prediction moment, at least `declared_lag_seconds` before it, and it is
  available at serving. Mediators, colliders and post-treatment variables
  known by then are NOT excluded: causal roles do not govern a pure
  prediction task, so this rule never reads a graph.
* `causal`: the timing rule above, and additionally the field must be an
  admissible covariate for the declared DAG: not a descendant of the
  treatment, and adding it to the smallest valid back-door set keeps the set
  valid (a collider whose conditioning opens a path fails this). The graph is
  a hypothesis, so a causal admission is only as good as the graph.

A field whose timing or serving availability is unknown is `unverified`: in
strict mode it is quarantined, in permissive mode admitted with a flag. A
log that disagrees with the stored values quarantines the field in both
modes. A leakage suspicion alone quarantines a field only when the leakage
detector has a current certificate and the field has no log-derived as-of
evidence that it precedes the moment.
"""

from collections.abc import Mapping
import dataclasses
import datetime
import enum
from typing import Any
from typing import Self

from scmle.contracts import qualification
from scmle.data.admissibility import dag as dag_module
from scmle.data.admissibility import provenance
from scmle.data.detectors import certificates

FORMAT_VERSION = 1
KINDS = ("predictive", "causal")


class Mode(enum.StrEnum):
    """What to do with fields whose provenance is unverified."""

    STRICT = "strict"
    PERMISSIVE = "permissive"


class Admission(enum.StrEnum):
    """What happens to a field."""

    ADMIT = "admit"
    QUARANTINE = "quarantine"
    ADMIT_FLAGGED = "admit_flagged"


@dataclasses.dataclass(frozen=True, slots=True)
class FieldDecision:
    """The decision for one field and why.

    Attributes:
        name: Field name.
        admission: The decision.
        tier: Evidence tier of the field's as-of rule.
        reasons: Plain-language reasons, in the order they applied.
        suspicion_kept: A leakage suspicion remains on record.
    """

    name: str
    admission: Admission
    tier: qualification.Tier
    reasons: tuple[str, ...]
    suspicion_kept: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert the decision to a JSON-ready record.

        Returns:
            A dict of the fields; enums as strings, reasons as a list.
        """
        return {
            "name": self.name,
            "admission": self.admission.value,
            "tier": self.tier.value,
            "reasons": list(self.reasons),
            "suspicion_kept": self.suspicion_kept,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild a decision, rejecting unknown or missing fields.

        Args:
            record: A mapping in the form `to_dict` returns.

        Returns:
            The decision. Unexpected or missing fields and unknown enum
            values are rejected with a `ValueError`.
        """
        qualification.require_fields(cls, record)
        return cls(
            record["name"],
            Admission(record["admission"]),
            qualification.Tier(record["tier"]),
            tuple(record["reasons"]),
            record["suspicion_kept"],
        )


def unverified_outcome(mode: Mode) -> Admission:
    """Decide the admission of a field whose provenance cannot be verified.

    Args:
        mode: Strict or permissive.

    Returns:
        `quarantine` in strict mode, `admit_flagged` otherwise.
    """
    if mode is Mode.STRICT:
        return Admission.QUARANTINE
    return Admission.ADMIT_FLAGGED


def decide_timing(
    record: provenance.FieldProvenance,
    mode: Mode,
    declared_lag_seconds: float,
    today: datetime.date | None,
    suspect: bool,
    leakage_gate: certificates.GateStatus,
) -> FieldDecision:
    """Apply the timing rule to one field.

    Args:
        record: The field's provenance.
        mode: Strict or permissive.
        declared_lag_seconds: Delay the charter declares between the
            as-of time and the prediction moment; a value known exactly this
            long before the moment is admissible.
        today: Day on which attestations must be valid, or None to skip.
        suspect: Whether the leakage tests called the field a suspect.
        leakage_gate: Whether the leakage detector may gate.

    Returns:
        The decision, with the reasons that applied. A failed log check,
        unavailability at serving, or a lead below the declared lag
        quarantines the field. An unknown serving availability or an
        unverified tier gives the unverified outcome of `mode`, which is
        quarantine for a leakage suspect when the detector is gating. A
        suspect with a confirmed tier is admitted; with an attested tier it
        is admitted flagged.
    """
    tier = record.tier(today)
    reasons: list[str] = []
    check = record.log_check
    if check is not None and not check.passed:
        reasons.append(
            f"change log disagrees with {check.mismatched} of {check.rows} "
            "stored values"
        )
        return FieldDecision(
            record.name, Admission.QUARANTINE, tier, tuple(reasons), suspect
        )
    if record.available_at_serving is provenance.Availability.NO:
        reasons.append("not available at serving")
        return FieldDecision(
            record.name, Admission.QUARANTINE, tier, tuple(reasons), suspect
        )
    lead = record.as_of.lead_seconds
    if lead is not None and lead < declared_lag_seconds:
        reasons.append(
            f"known {lead:g} s before the moment, less than the declared "
            f"{declared_lag_seconds:g} s"
        )
        return FieldDecision(
            record.name, Admission.QUARANTINE, tier, tuple(reasons), suspect
        )
    unknown_serving = (
        record.available_at_serving is provenance.Availability.UNKNOWN
    )
    if unknown_serving:
        reasons.append("availability at serving is unknown")
    if tier is qualification.Tier.UNVERIFIED:
        reasons.append("as-of rule is unverified")
    if unknown_serving or tier is qualification.Tier.UNVERIFIED:
        admission = unverified_outcome(mode)
        if suspect:
            reasons.append("leakage suspicion with no as-of evidence")
            if leakage_gate is certificates.GateStatus.GATING:
                admission = Admission.QUARANTINE
        return FieldDecision(
            record.name, admission, tier, tuple(reasons), suspect
        )
    admission = Admission.ADMIT
    if suspect:
        if tier is qualification.Tier.CONFIRMED:
            reasons.append(
                "leakage suspicion kept: log-derived as-of precedes the moment"
            )
        else:
            admission = Admission.ADMIT_FLAGGED
            reasons.append("leakage suspicion; as-of is attested, not logged")
    else:
        reasons.append("known before the moment and available at serving")
    return FieldDecision(record.name, admission, tier, tuple(reasons), suspect)


def decide_predictive(
    record: provenance.FieldProvenance,
    mode: Mode = Mode.STRICT,
    declared_lag_seconds: float = 0.0,
    today: datetime.date | None = None,
    suspect: bool = False,
    leakage_gate: certificates.GateStatus = certificates.GateStatus.DESCRIPTIVE,
) -> FieldDecision:
    """Decide the admissibility of one field for a predictive estimand.

    Causal roles (mediator, collider, post-treatment) are not consulted.

    Args:
        record: The field's provenance.
        mode: Strict or permissive.
        declared_lag_seconds: Declared delay before the moment, >= 0.
        today: Day on which attestations must be valid, or None.
        suspect: Whether the leakage tests called the field a suspect.
        leakage_gate: Whether the leakage detector may gate.

    Returns:
        The decision.

    Raises:
        ValueError: If `declared_lag_seconds` is negative or not finite.
    """
    lag = qualification.finite(declared_lag_seconds, "declared_lag_seconds")
    if lag < 0.0:
        raise ValueError("declared_lag_seconds must be non-negative")
    return decide_timing(record, mode, lag, today, suspect, leakage_gate)


def decide_causal(
    record: provenance.FieldProvenance,
    graph: dag_module.Dag,
    treatment: str,
    outcome: str,
    mode: Mode = Mode.STRICT,
    declared_lag_seconds: float = 0.0,
    today: datetime.date | None = None,
    suspect: bool = False,
    leakage_gate: certificates.GateStatus = certificates.GateStatus.DESCRIPTIVE,
) -> FieldDecision:
    """Decide the admissibility of one field as a covariate.

    Applies the predictive timing rule first, then the graph rule.

    Args:
        record: The field's provenance.
        graph: The DAG hypothesis.
        treatment: Treatment node.
        outcome: Outcome node.
        mode: Strict or permissive.
        declared_lag_seconds: Declared delay before the moment, >= 0.
        today: Day on which attestations must be valid, or None.
        suspect: Whether the leakage tests called the field a suspect.
        leakage_gate: Whether the leakage detector may gate.

    Returns:
        The decision. The treatment and outcome themselves are not
        covariates and are decided on timing alone. A negative or
        non-finite lag is rejected with a `ValueError`, as is a
        treatment or outcome that is not a node of `graph` or a search for
        the adjustment set that `Dag.adjustment_set` refuses as too large,
        when the graph rule is reached.
    """
    timing = decide_predictive(
        record, mode, declared_lag_seconds, today, suspect, leakage_gate
    )
    if timing.admission is Admission.QUARANTINE:
        return timing
    name = record.name
    if name in (treatment, outcome):
        return dataclasses.replace(
            timing, reasons=(*timing.reasons, "treatment or outcome node")
        )
    if name not in graph.nodes:
        return dataclasses.replace(
            timing,
            admission=(
                Admission.QUARANTINE
                if mode is Mode.STRICT
                else Admission.ADMIT_FLAGGED
            ),
            reasons=(*timing.reasons, "field is not in the declared DAG"),
        )
    if name in graph.descendants(treatment):
        return dataclasses.replace(
            timing,
            admission=Admission.QUARANTINE,
            reasons=(*timing.reasons, "descendant of the treatment"),
        )
    base = graph.adjustment_set(treatment, outcome)
    if base is None:
        return dataclasses.replace(
            timing,
            admission=Admission.ADMIT_FLAGGED,
            reasons=(
                *timing.reasons,
                "no valid adjustment set from observed variables",
            ),
        )
    if graph.backdoor_valid(treatment, outcome, {*base, name}):
        return dataclasses.replace(
            timing,
            reasons=(*timing.reasons, "keeps the back-door set valid"),
        )
    return dataclasses.replace(
        timing,
        admission=Admission.QUARANTINE,
        reasons=(
            *timing.reasons,
            "adjusting for it opens a back-door path (collider)",
        ),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class AdmissibilityTable:
    """The table written as `admissibility_table.json`.

    Attributes:
        kind: Estimand kind, `predictive` or `causal`.
        mode: Strict or permissive.
        decisions: One per field, in the order of the provenance table.
        dag_hash: Hash of the graph used, or None.
        adjustment_set: The back-door set found, for a causal estimand.
    """

    kind: str
    mode: Mode
    decisions: tuple[FieldDecision, ...]
    dag_hash: str | None
    adjustment_set: tuple[str, ...] | None

    def __post_init__(self) -> None:
        """Validate the table.

        Raises:
            ValueError: If `kind` is not `predictive` or `causal`, or field
                names repeat.
        """
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        names = [d.name for d in self.decisions]
        if len(set(names)) != len(names):
            raise ValueError("field names must be unique")

    def names(self, admission: Admission) -> tuple[str, ...]:
        """List the fields that received an admission.

        Args:
            admission: The admission to select.

        Returns:
            Field names with that admission, in decision order.
        """
        return tuple(d.name for d in self.decisions if d.admission is admission)

    def to_dict(self) -> dict[str, Any]:
        """Convert the table to a JSON-ready document.

        Returns:
            A dict with the format version, kind, mode, DAG hash,
            adjustment set (as a list, or None) and the decisions.
        """
        return {
            "format_version": FORMAT_VERSION,
            "kind": self.kind,
            "mode": self.mode.value,
            "dag_hash": self.dag_hash,
            "adjustment_set": None
            if self.adjustment_set is None
            else list(self.adjustment_set),
            "decisions": [d.to_dict() for d in self.decisions],
        }


def build_table(
    table: provenance.ProvenanceTable,
    kind: str,
    mode: Mode = Mode.STRICT,
    declared_lag_seconds: float = 0.0,
    today: datetime.date | None = None,
    suspects: Mapping[str, bool] | None = None,
    leakage_gate: (
        certificates.GateStatus | Mapping[str, certificates.GateStatus]
    ) = certificates.GateStatus.DESCRIPTIVE,
    graph: dag_module.Dag | None = None,
    treatment: str | None = None,
    outcome: str | None = None,
) -> AdmissibilityTable:
    """Decide every field of a provenance table.

    Args:
        table: Provenance of every field.
        kind: `predictive` or `causal`.
        mode: Strict or permissive.
        declared_lag_seconds: Declared delay before the moment.
        today: Day on which attestations must be valid, or None.
        suspects: Leakage suspicion per field name; absent means no.
        leakage_gate: Whether the leakage detector may gate, for every field
            or, as a mapping, per field (absent means descriptive).
        graph: DAG hypothesis; required for `causal`, ignored for
            `predictive`.
        treatment: Treatment node; required for `causal`.
        outcome: Outcome node; required for `causal`.

    Returns:
        The table, with one decision per field in table order.

    Raises:
        ValueError: If `kind` is unknown, or a causal estimand lacks a
            graph, treatment or outcome.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    flags = suspects or {}

    def gate_of(name: str) -> certificates.GateStatus:
        if isinstance(leakage_gate, Mapping):
            return leakage_gate.get(name, certificates.GateStatus.DESCRIPTIVE)
        return leakage_gate

    if kind == "predictive":
        decisions = tuple(
            decide_predictive(
                f,
                mode,
                declared_lag_seconds,
                today,
                flags.get(f.name, False),
                gate_of(f.name),
            )
            for f in table.fields
        )
        return AdmissibilityTable(kind, mode, decisions, None, None)
    if graph is None or treatment is None or outcome is None:
        raise ValueError("a causal estimand needs a graph, treatment, outcome")
    decisions = tuple(
        decide_causal(
            f,
            graph,
            treatment,
            outcome,
            mode,
            declared_lag_seconds,
            today,
            flags.get(f.name, False),
            gate_of(f.name),
        )
        for f in table.fields
    )
    return AdmissibilityTable(
        kind,
        mode,
        decisions,
        graph.digest(),
        graph.adjustment_set(treatment, outcome),
    )
