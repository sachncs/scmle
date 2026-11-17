"""The loop's pre-registered policy, its records and shared helpers."""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
from fractions import Fraction
import math
from typing import Any

import numpy

from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.governance import deployment_ledger
from scmle.governance import escalation
from scmle.monitoring import labels

STATE_FILE = "loop_state.json"
RISK_COVERAGE_FILE = "risk_coverage.json"
LABEL_MECHANISM_FILE = "label_mechanism.json"
STATE_VERSION = 1
ESCALATING_ROLLBACKS = 2  # consecutive rollbacks that force an escalation
OK = "ok"
HALTED = "halted"
INSUFFICIENT_LABELS = "insufficient_labels"
HALT_LEDGER = "ledger_verification_failed"
HALT_STATE = "state_does_not_match_ledger"
HALT_SELECTIVE = "selective_risk_above_target"
HALT_UNVERIFIED = "alarm_on_unverified_label_mechanism"
LOOSER = "looser"
TIGHTER = "tighter"
NUMERIC_ORIENTATION = {
    # field: +1 if a larger value is stricter, -1 if a smaller one is.
    "r_star": -1,
    "delta": -1,
    "c_max": -1,
    "alpha_alarm": -1,
    "alpha_rb": -1,
    "tolerance": -1,
    "delta_rb": -1,
    "delta_promo": 1,
    "coverage_target": 1,
}


class RefusalError(errors.ArtifactError):
    """A reply was not recorded, and why."""


@dataclasses.dataclass(frozen=True)
class Preregistration:
    """Pre-registered inputs of the loop (changed only by `Loop.amend_policy`).

    Risks, tolerances and margins are in loss units, ``alpha_*`` and ``delta``
    are probabilities, and times are counted in batches.

    Attributes:
        reference_risk: Reference risk `r_ref` of the bundle.
        tolerance: `Delta`, the rise over `r_ref` that is worth an alarm.
        alpha_alarm: Level of the L1 alarm.
        delta_promo: Margin the challenger must beat the champion by.
        alpha_promo: FDR level of e-LOND over challengers, as ``"p/q"``.
        alpha_rb: Level of the rollback process.
        delta_rb: Margin by which the new champion must be worse.
        r_star: Target selective risk.
        delta: Confidence ``1 - delta`` of the threshold.
        c_max: Largest escalated share of a batch.
        alarm_gates_retrain: Whether an L1 alarm may trigger a retrain. Set
            it True only if the loop certificate for the detection delay
            (a rise of `2 * tolerance` found within the charter's
            `D_max` with probability at least 0.9) passed for these
            settings; otherwise an alarm records an `alarm` row and
            escalates (`investigate`), and nothing is retrained.
        outcome_window: Batches after which a prediction has matured.
        epsilon: Smallest propensity, also the audit-slice rate; 0 requires
            complete labels.
        bypass_rate: Declared share of cases that skip the model.
        loss_bound: Upper bound of a loss.
        safe_default: Action for cases the model must not answer.
        reply_timeout: Batches after which an unanswered escalation closes
            with the safe default.
        min_matured: Fewest labelled cases decided by the model (not bypassed)
            that a matured batch needs to be used.
        challenger_budget: Matured batches a challenger gets.
        coverage_target: Nominal coverage of the conformal sets, or None.
        novelty_bound: Largest tolerated share of out-of-support cases.
        high_impact_actions: Decision values the charter classes high impact.
        tighten_step: Amount `tau` rises by on a rollback proposal (the
            result is capped at 1).
        initial_tau: Starting threshold; None escalates everything until
            `Loop.calibrate`.
        scope: Model version hash and stream id that attestations name.
    """

    reference_risk: float
    tolerance: float = 0.05
    alpha_alarm: float = 0.05
    delta_promo: float = 0.0
    alpha_promo: str = "1/20"
    alpha_rb: float = 0.05
    delta_rb: float = 0.01
    r_star: float = 0.05
    delta: float = 0.1
    c_max: float = 0.2
    alarm_gates_retrain: bool = False
    outcome_window: int = 1
    epsilon: float = 0.0
    bypass_rate: float = 0.0
    loss_bound: float = 1.0
    safe_default: str = "defer"
    reply_timeout: int = 100
    min_matured: int = 30
    challenger_budget: int = 200
    coverage_target: float | None = None
    novelty_bound: float = 0.05
    high_impact_actions: tuple[str, ...] = ()
    tighten_step: float = 0.05
    initial_tau: float | None = None
    scope: str = "stream"

    def __post_init__(self) -> None:
        """Checks the ranges of the fields.

        Raises:
            ValueError: If a probability-like field is outside (0, 1), epsilon
                or c_max is out of range, the loss bound is not positive or the
                window negative, the reference risk plus tolerance is outside
                (0, scale), or the safe default or scope is blank.
        """
        for name in ("alpha_alarm", "alpha_rb", "r_star", "delta"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise errors.InputError(
                    f"{name} must be strictly between 0 and 1"
                )
        if not 0.0 <= self.epsilon < 1.0 or not 0.0 <= self.c_max <= 1.0:
            raise errors.InputError(
                "epsilon must be in [0, 1) and c_max in [0, 1]"
            )
        if self.loss_bound <= 0.0 or self.outcome_window < 0:
            raise errors.InputError("loss_bound must be positive, window >= 0")
        if not 0.0 < self.reference_risk + self.tolerance < self.scale:
            raise errors.InputError(
                "reference_risk + tolerance must be in (0, scale)"
            )
        if not self.safe_default.strip() or not self.scope.strip():
            raise errors.InputError("safe_default and scope are required")

    @property
    def alpha(self) -> float:
        """`alpha_promo` as a float."""
        return float(Fraction(self.alpha_promo))

    @property
    def scale(self) -> float:
        """Upper bound of a (propensity-weighted) batch mean loss."""
        if self.epsilon > 0.0:
            return self.loss_bound / self.epsilon
        return self.loss_bound

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record.

        Returns:
            The policy's fields as a dict, with the high-impact actions as a
            list.
        """
        record = dataclasses.asdict(self)
        record["high_impact_actions"] = list(self.high_impact_actions)
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Preregistration":
        """Rebuilds a policy from `to_dict`.

        Args:
            record: A mapping as produced by `to_dict`.

        Returns:
            The policy with the same field values.
        """
        values = dict(record)
        values["high_impact_actions"] = tuple(
            values.get("high_impact_actions", ())
        )
        return cls(**values)


@dataclasses.dataclass(frozen=True)
class Escalation:
    """A question waiting for a person.

    Attributes:
        id: Escalation id (``esc-<row>``).
        question: The question, JSON-ready.
        question_hash: Its hash; a reply must carry it.
        evidence: References to the evidence.
        proposed_default: What happens if nobody answers.
    """

    id: str
    question: Mapping[str, Any]
    question_hash: str
    evidence: tuple[str, ...]
    proposed_default: str


@dataclasses.dataclass(frozen=True)
class Observation:
    """What `observe` decided for a batch.

    Attributes:
        batch: Batch index.
        routing: Per-case routing.
        actions: Per case: the decision (answered), `escalate` or the safe
            default.
    """

    batch: int
    routing: escalation.Routing
    actions: tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class Outcome:
    """What a `step` found and did.

    Attributes:
        status: `ok`, `halted` or `insufficient_labels`.
        events: Ledger rows written by this step other than observation
            summaries, as (kind, reason); the reason is the row's `reason`
            or `rule`, else an empty string.
        escalation: The newest escalation opened by this step, if any.
        mechanism: The label mechanism of the newest batch that had enough
            labels (unchanged if none had); `unverified` means no retrain,
            promotion or rollback rests on this stream.
    """

    status: str
    events: tuple[tuple[str, str], ...]
    escalation: Escalation | None
    mechanism: labels.Mechanism


@dataclasses.dataclass(frozen=True)
class Consequence:
    """What recording a reply means for the dependent decision.

    Attributes:
        tier: `attested` when the person allowed the action; `unverified`
            when the reply rejects or cannot decide (the safe default
            applies).
        decision: The person's decision.
        safe_default_applies: Whether the proposed default is taken.
    """

    tier: qualification.Tier
    decision: deployment_ledger.Reply
    safe_default_applies: bool


@dataclasses.dataclass(frozen=True)
class Hooks:
    """Callbacks that act in the world; all optional.

    Attributes:
        retrain: Trains and stages a challenger and returns its name, or
            None if it staged none.
        open_incident: Called with the reason when the loop halts, after the
            loop has written its own `incident` ledger row
            (`Loop.record_incident`); the loop needs no hook for that.
    """

    retrain: Callable[[], str | None] | None = None
    open_incident: Callable[[str], None] | None = None


def jsonify(value: Any) -> Any:
    """Converts numpy scalars and arrays inside a structure for JSON.

    Args:
        value: A scalar, array, dict, list or tuple, possibly nested.

    Returns:
        The structure with numpy arrays as lists, numpy scalars as Python
        values, tuples as lists, dict keys as strings and non-finite Python
        floats (also those inside arrays) as None. A non-finite numpy scalar
        is returned as the Python float, not None.
    """
    if isinstance(value, numpy.ndarray):
        return [jsonify(item) for item in value.tolist()]
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonify(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonify(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def amend(ledger: deployment_ledger.DeploymentLedger) -> Mapping[str, Any]:
    """Returns the ledger's policy with every recorded amendment applied.

    Only amendment rows carry `policy_amendment`; other observe summaries
    have none and change nothing.

    Args:
        ledger: The verified deployment ledger.

    Returns:
        The header policy updated by each amendment in ledger order.

    Raises:
        ArtifactError: If an amendment is not a mapping. `dict.update`
            applied a list of pairs as if it were one, so malformed rows
            changed the policy without a word (ERR-23).
    """
    effective = dict(ledger.header["policy"])
    for row in ledger.rows:
        if row["kind"] != deployment_ledger.OBSERVE_SUMMARY:
            continue
        change = row["body"].get("policy_amendment", {})
        if not isinstance(change, Mapping):
            raise errors.ArtifactError(
                f"ledger row {row['seq']} has a policy amendment that is not "
                f"a mapping: {type(change).__name__}"
            )
        effective.update(change)
    return effective
