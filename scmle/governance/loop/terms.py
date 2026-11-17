"""The loop's pre-registered policy, its records and shared helpers."""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
from fractions import Fraction
import math
from typing import Any
from typing import ClassVar

import numpy

from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.contracts import records
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
    "target": -1,
    "delta": -1,
    "escalation": -1,
    "alarm": -1,
    "rollback.alpha": -1,
    "tolerance": -1,
    "rollback.delta": -1,
    "promotion.delta": 1,
    "coverage": 1,
}
GROUPS = {
    "promotion": {"alpha_promo": "alpha", "delta_promo": "delta"},
    "rollback": {"alpha_rb": "alpha", "delta_rb": "delta"},
}
"""Version-1 keys moved into the `promotion` and `rollback` objects."""
DEFAULTS: dict[str, Mapping[str, Any]] = {
    "promotion": {"alpha": "1/20", "delta": 0.0},
    "rollback": {"alpha": 0.05, "delta": 0.01},
}
"""Default `promotion` and `rollback`; a stored one may name only some."""


class RefusalError(errors.ArtifactError):
    """A reply was not recorded, and why."""


MIGRATIONS = (
    records.nest(
        {
            "reference_risk": "reference",
            "alpha_alarm": "alarm",
            "r_star": "target",
            "c_max": "escalation",
            "alarm_gates_retrain": "retrain",
            "outcome_window": "maturity",
            "bypass_rate": "bypass",
            "loss_bound": "loss",
            "safe_default": "fallback",
            "reply_timeout": "timeout",
            "min_matured": "matured",
            "challenger_budget": "budget",
            "coverage_target": "coverage",
            "novelty_bound": "novelty",
            "high_impact_actions": "critical",
            "tighten_step": "step",
            "initial_tau": "tau",
        },
        groups=GROUPS,
    ),
)
"""The loop policy's version steps, oldest first."""
VERSION = records.ORIGIN + len(MIGRATIONS)


@dataclasses.dataclass(frozen=True)
class Preregistration:
    """Pre-registered inputs of the loop (changed only by `Loop.amend_policy`).

    Risks, tolerances and margins are in loss units, alphas and ``delta``
    are probabilities, and times are counted in batches.

    Attributes:
        reference: Reference risk `r_ref` of the bundle.
        tolerance: `Delta`, the rise over `r_ref` that is worth an alarm.
        alarm: Level of the L1 alarm.
        promotion: `alpha`, the FDR level of e-LOND over challengers, as
            ``"p/q"``, and `delta`, the margin the challenger must beat the
            champion by.
        rollback: `alpha`, the level of the rollback process, and `delta`,
            the margin by which the new champion must be worse.
        target: Target selective risk `r_star`.
        delta: Confidence ``1 - delta`` of the threshold.
        escalation: Largest escalated share of a batch (`c_max`).
        retrain: Whether an L1 alarm may trigger a retrain. Set it True only
            if the loop certificate for the detection delay (a rise of
            `2 * tolerance` found within the charter's `D_max` with
            probability at least 0.9) passed for these settings; otherwise an
            alarm records an `alarm` row and escalates (`investigate`), and
            nothing is retrained.
        maturity: Batches after which a prediction has matured.
        epsilon: Smallest propensity, also the audit-slice rate; 0 requires
            complete labels.
        bypass: Declared share of cases that skip the model.
        loss: Upper bound of a loss.
        fallback: Action for cases the model must not answer.
        timeout: Batches after which an unanswered escalation closes with
            the fallback.
        matured: Fewest labelled cases decided by the model (not bypassed)
            that a matured batch needs to be used.
        budget: Matured batches a challenger gets.
        coverage: Nominal coverage of the conformal sets, or None.
        novelty: Largest tolerated share of out-of-support cases.
        critical: Decision values the charter classes high impact.
        step: Amount `tau` rises by on a rollback proposal (the result is
            capped at 1).
        tau: Starting threshold; None escalates everything until
            `Loop.calibrate`.
        scope: Model version hash and stream id that attestations name.
        migrations: The policy's version steps, oldest first.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = MIGRATIONS

    reference: float
    tolerance: float = 0.05
    alarm: float = 0.05
    promotion: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: dict(DEFAULTS["promotion"])
    )
    rollback: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: dict(DEFAULTS["rollback"])
    )
    target: float = 0.05
    delta: float = 0.1
    escalation: float = 0.2
    retrain: bool = False
    maturity: int = 1
    epsilon: float = 0.0
    bypass: float = 0.0
    loss: float = 1.0
    fallback: str = "defer"
    timeout: int = 100
    matured: int = 30
    budget: int = 200
    coverage: float | None = None
    novelty: float = 0.05
    critical: tuple[str, ...] = ()
    step: float = 0.05
    tau: float | None = None
    scope: str = "stream"

    def __post_init__(self) -> None:
        """Checks the ranges of the fields.

        Raises:
            ValueError: If a probability-like field is outside (0, 1), epsilon
                or the escalated share is out of range, the loss bound is not
                positive or the maturity negative, the reference risk plus
                tolerance is outside (0, scale), the promotion or rollback
                lacks its alpha or delta, or the fallback or scope is blank.
        """
        for group in GROUPS:
            if set(getattr(self, group)) != {"alpha", "delta"}:
                raise errors.InputError(f"{group} needs alpha and delta only")
        levels = {
            "alarm": self.alarm,
            "rollback.alpha": self.rollback["alpha"],
            "target": self.target,
            "delta": self.delta,
        }
        for name, value in levels.items():
            if not 0.0 < value < 1.0:
                raise errors.InputError(
                    f"{name} must be strictly between 0 and 1"
                )
        if not 0.0 <= self.epsilon < 1.0 or not 0.0 <= self.escalation <= 1.0:
            raise errors.InputError(
                "epsilon must be in [0, 1) and escalation in [0, 1]"
            )
        if self.loss <= 0.0 or self.maturity < 0:
            raise errors.InputError("loss must be positive, maturity >= 0")
        if not 0.0 < self.reference + self.tolerance < self.scale:
            raise errors.InputError(
                "reference + tolerance must be in (0, scale)"
            )
        if not self.fallback.strip() or not self.scope.strip():
            raise errors.InputError("fallback and scope are required")

    @property
    def alpha(self) -> float:
        """The promotion `alpha` as a float."""
        return float(Fraction(self.promotion["alpha"]))

    @property
    def scale(self) -> float:
        """Upper bound of a (propensity-weighted) batch mean loss."""
        if self.epsilon > 0.0:
            return self.loss / self.epsilon
        return self.loss

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record.

        Returns:
            The policy's fields as a dict at the current version, with the
            high-impact actions as a list.
        """
        record = dataclasses.asdict(self)
        record["promotion"] = dict(self.promotion)
        record["rollback"] = dict(self.rollback)
        record["critical"] = list(self.critical)
        return {records.KEY: VERSION, **record}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Preregistration":
        """Rebuilds a policy from `to_dict`, of this or the first version.

        Args:
            record: A mapping as produced by `to_dict`; one without a
                `schema_version` is version 1 and is migrated.

        Returns:
            The policy with the same field values.

        Raises:
            ValueError: If the version is unknown.
        """
        values = upgrade(record)
        del values[records.KEY]
        for group, defaults in DEFAULTS.items():
            if group in values:
                values[group] = {**defaults, **values[group]}
        values["critical"] = tuple(values.get("critical", ()))
        return cls(**values)


def upgrade(record: Mapping[str, Any]) -> records.Layout:
    """Moves a stored policy, or an amendment of one, to the current version.

    A layout without `schema_version` is version 1; its keys are renamed and
    its promotion and rollback keys nested (only those present, so a
    version-1 amendment migrates too; an absent group comes out empty).

    Args:
        record: The stored policy or amendment.

    Returns:
        The same content at the current version, with `schema_version`.

    Raises:
        ValueError: If the version is not a positive integer or is newer
            than this build.
    """
    values = records.load("policy", MIGRATIONS, record)
    for group in GROUPS:
        if not values.get(group, True):
            del values[group]
    return values


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
    have none and change nothing. The header and every amendment are moved
    to the current version first (a ledger may mix versions), and an
    amendment of the promotion or rollback replaces only the keys it names.

    Args:
        ledger: The verified deployment ledger.

    Returns:
        The header policy updated by each amendment in ledger order, at the
        current version.

    Raises:
        ArtifactError: If an amendment is not a mapping. `dict.update`
            applied a list of pairs as if it were one, so malformed rows
            changed the policy without a word (ERR-23).
    """
    effective = upgrade(ledger.header["policy"])
    for row in ledger.rows:
        if row["kind"] != deployment_ledger.OBSERVE_SUMMARY:
            continue
        change = row["body"].get("policy_amendment", {})
        if not isinstance(change, Mapping):
            raise errors.ArtifactError(
                f"ledger row {row['seq']} has a policy amendment that is not "
                f"a mapping: {type(change).__name__}"
            )
        for key, value in upgrade(change).items():
            effective[key] = (
                {**effective[key], **value} if key in GROUPS else value
            )
    return effective
