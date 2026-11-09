"""The closed loop: drift, retrain, gated promotion and asking a person.

    loop = Loop.open(deploy_dir, charter, policy)
    routing = loop.observe(batch, predictions, decisions)
    loop.mature(losses)                    # outcomes arrive, with delay
    result = loop.step()                   # alarms, proposals, escalations
    pending = loop.pending_escalations()
    loop.reply(pending[0].id, attestation)  # a person's signed reply

What the loop does by itself, and how each rule behaves:

* L1. A betting alarm on the matured, propensity-weighted mean loss
  (`RiskAlarm`): false alarm probability at most ``alpha_alarm`` over all time
  for the first alarm of an episode. On an alarm with a verified label
  mechanism it records an `alarm` row. If ``alarm_gates_retrain`` is set it
  then records a `retrain` request (and calls the `retrain` hook, if given,
  which must train and stage a challenger and return its name); otherwise it
  escalates for investigation and retrains nothing. When the naive
  (unweighted) alarm crosses on an unverified mechanism it halts and
  escalates; it never retrains.
* L2. Each challenger is registered in the ledger before it is scored, scored
  by an e-process on paired losses, and decided in registration order by
  e-LOND. A crossing becomes a `promotion_proposal`; the signed approval of
  the registry is still needed to move the champion alias.
* L3. After a promotion, a reverse e-process (`watch_rollback`) on the new
  champion against the previous one can create a `rollback_proposal` and
  tightens `tau`.
* E1. Cases below the calibrated threshold, and any case beyond the model's
  reach, are escalated. If the selective risk of answered cases is evidenced
  above ``r*`` the loop halts: every case is escalated until a recalibration
  passes. Escalations beyond ``c_max`` get the safe default action.
* E2. System-level triggers write an `escalation` row with the question, the
  evidence and the proposed default before anything waits.

What it cannot do: see `scmle.governance.never_list`. The decision on the
champion alias, the sealed sets, the charter and the other listed actions
needs a person's attestation; `Loop.reply` records it. Key custody for that
signature is a deployment requirement the software cannot enforce.

State. The ledger holds what happened; `loop_state.json` holds the monitors'
running state and the batches awaiting maturity. State is saved after every
public call and carries the ledger head: if they disagree at `open`, the loop
starts halted rather than guess.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import datetime
from fractions import Fraction
import json
import math
from pathlib import Path
import types
from typing import Any

import numpy
import pandas

from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.contracts import signing
from scmle.governance import deployment_ledger
from scmle.governance import escalation
from scmle.governance import never_list
from scmle.monitoring import labels
from scmle.monitoring import sequential_tests

STATE_FILE = "loop_state.json"
RISK_COVERAGE_FILE = "risk_coverage.json"
LABEL_MECHANISM_FILE = "label_mechanism.json"
STATE_VERSION = 1
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


class ReplyRefused(errors.ArtifactError):
    """A reply was not recorded, and why."""


@dataclasses.dataclass(frozen=True)
class LoopPolicy:
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
                raise ValueError(f"{name} must be strictly between 0 and 1")
        if not 0.0 <= self.epsilon < 1.0 or not 0.0 <= self.c_max <= 1.0:
            raise ValueError("epsilon must be in [0, 1) and c_max in [0, 1]")
        if self.loss_bound <= 0.0 or self.outcome_window < 0:
            raise ValueError("loss_bound must be positive, window >= 0")
        if not 0.0 < self.reference_risk + self.tolerance < self.scale:
            raise ValueError("reference_risk + tolerance must be in (0, scale)")
        if not self.safe_default.strip() or not self.scope.strip():
            raise ValueError("safe_default and scope are required")

    @property
    def promo_level(self) -> float:
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
    def from_dict(cls, record: Mapping[str, Any]) -> "LoopPolicy":
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
class StepResult:
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
class ReplyOutcome:
    """What recording a reply means for the dependent decision.

    Attributes:
        tier: `attested` when the person allowed the action; `unverified`
            when the reply rejects or cannot decide (the safe default
            applies).
        decision: The person's decision.
        safe_default_applies: Whether the proposed default is taken.
    """

    tier: qualification.Tier
    decision: deployment_ledger.ReplyDecision
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


def json_ready(value: Any) -> Any:
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
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class Loop:
    """One deployment's closed loop (see the module docstring)."""

    def __init__(
        self,
        directory: Path,
        ledger: deployment_ledger.DeploymentLedger,
        policy: LoopPolicy,
        guard: never_list.Guard,
        hooks: Hooks,
        verification_keys: Mapping[str, str | bytes],
        today: Callable[[], datetime.date],
    ) -> None:
        """Builds a loop from parts that `Loop.open` assembles.

        Hooks the guard's ``on_write`` callback to `save_state` and creates
        the monitors. It does not load the saved state; `Loop.open` does.

        Args:
            directory: Directory of the ledger and loop state files.
            ledger: The deployment ledger.
            policy: The pre-registered policy.
            guard: The never-list guard.
            hooks: Optional callbacks.
            verification_keys: Map from key id to the key that verifies a
                person's attestation.
            today: Clock, replaceable in tests.
        """
        self.directory = directory
        self.ledger = ledger
        self.policy = policy
        self.guard = guard
        self.hooks = hooks
        self.verification_keys = dict(verification_keys)
        self.today = today
        self.batch = 0
        self.tau = (
            math.inf if policy.initial_tau is None else policy.initial_tau
        )
        self.pending_batches: list[dict[str, Any]] = []
        self.escalation_batch: dict[str, int] = {}
        self.consecutive_rollbacks = 0
        self.stopped: dict[str, float] = {}
        self.queue: list[str] = []
        self.active: dict[str, dict[str, Any]] = {}
        self.rollback_watch: dict[str, Any] | None = None
        self.mechanism = labels.Mechanism.COMPLETE
        self.halt_reason: str | None = None
        self.halted_by_state = False
        self.naive_flag = False
        self.calibrated = False
        self.feedback: dict[str, Any] | None = None
        self.guard.on_write = self.save_state
        self.build_monitors()

    # ------------------------------------------------------------------ open

    @classmethod
    def open(
        cls,
        deploy_dir: str | Path,
        charter: Any,
        policy: LoopPolicy,
        *,
        hooks: Hooks | None = None,
        verification_keys: Mapping[str, str | bytes] | None = None,
        system_keys: Sequence[str | bytes] = (),
        today: Callable[[], datetime.date] = datetime.date.today,
        single_use_attestations: bool = False,
    ) -> "Loop":
        """Opens the loop of a deployment, creating its ledger on first use.

        Creates the directory if needed, loads the saved state (halting if it
        does not match the ledger) and saves the state again.

        Args:
            deploy_dir: Directory of `deployment_ledger.jsonl` and
                `loop_state.json`.
            charter: Anything with a `charter_hash` (the analysis charter).
            policy: The pre-registered inputs.
            hooks: Optional callbacks.
            verification_keys: Map from key id (`key_fingerprint`) to the
                key that verifies a person's attestation. The system must not
                hold the signing side of these keys.
            system_keys: Keys the system itself holds; attestations signed
                with them (or with `SCMLE_SIGNING_KEY`) are refused. Their
                fingerprints are recorded only when the ledger is created; for
                an existing ledger the ones it already records apply and
                this argument has no effect.
            today: Clock, replaceable in tests.
            single_use_attestations: Spend an attestation on the first action
                it allows (an `attestation_used` row) instead of leaving it
                valid until it expires.

        Returns:
            The open loop, with its saved state loaded.

        Raises:
            DeploymentLedgerError: If an existing ledger fails verification
                or belongs to another charter or policy.
        """
        directory = Path(deploy_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / deployment_ledger.FILE_NAME
        fingerprints = [
            deployment_ledger.key_fingerprint(k) for k in system_keys
        ]
        environment = signing.resolve_key(None)
        if environment is not None:
            fingerprints.append(deployment_ledger.key_fingerprint(environment))
        keys = dict(verification_keys or {})
        if path.exists():
            ledger = deployment_ledger.DeploymentLedger.load(path)
            header = ledger.header
            if header["charter_hash"] != charter.charter_hash:
                raise deployment_ledger.DeploymentLedgerError(
                    "ledger belongs to a different charter"
                )
            effective = dict(header["policy"])
            for row in ledger.rows:
                body = row["body"]
                if row["kind"] == deployment_ledger.OBSERVE_SUMMARY:
                    effective.update(body.get("policy_amendment", {}))
            if effective != json_ready(policy.to_dict()):
                raise deployment_ledger.DeploymentLedgerError(
                    "ledger was opened with a different policy"
                )
        else:
            ledger = deployment_ledger.DeploymentLedger.create(
                path,
                charter_hash=charter.charter_hash,
                policy=json_ready(policy.to_dict()),
                alpha_promo=policy.alpha_promo,
                system_keys=sorted(set(fingerprints)),
            )
        guard = never_list.Guard(
            ledger, today, keys, single_use=single_use_attestations
        )
        loop = cls(
            directory, ledger, policy, guard, hooks or Hooks(), keys, today
        )
        loop.load_state()
        loop.save_state()
        return loop

    @classmethod
    def resume(
        cls,
        deploy_dir: str | Path,
        *,
        hooks: Hooks | None = None,
        verification_keys: Mapping[str, str | bytes] | None = None,
        system_keys: Sequence[str | bytes] = (),
        today: Callable[[], datetime.date] = datetime.date.today,
        single_use_attestations: bool = False,
    ) -> "Loop":
        """Opens an existing deployment from what its ledger records.

        The charter hash and the policy (with its recorded amendments) are
        read from the ledger, so a command line needs only the directory. A
        missing ledger, or one that fails verification, is refused by the
        ledger with a `DeploymentLedgerError`. `system_keys` has no effect
        here, because the ledger already exists (see `open`).

        Args:
            deploy_dir: Directory of the deployment ledger.
            hooks: Optional callbacks.
            verification_keys: Map from key id to the key that verifies a
                person's attestation.
            system_keys: Keys the system itself holds; ignored, as the
                ledger records its own.
            today: Clock, replaceable in tests.
            single_use_attestations: Spend an attestation on the first action
                it allows.

        Returns:
            The open loop.
        """
        path = Path(deploy_dir) / deployment_ledger.FILE_NAME
        ledger = deployment_ledger.DeploymentLedger.load(path)
        effective = dict(ledger.header["policy"])
        for row in ledger.rows:
            if row["kind"] == deployment_ledger.OBSERVE_SUMMARY:
                effective.update(row["body"].get("policy_amendment", {}))
        policy = LoopPolicy.from_dict(effective)
        charter = types.SimpleNamespace(
            charter_hash=ledger.header["charter_hash"]
        )
        return cls.open(
            deploy_dir,
            charter,
            policy,
            hooks=hooks,
            verification_keys=verification_keys,
            system_keys=system_keys,
            today=today,
            single_use_attestations=single_use_attestations,
        )

    def build_monitors(self) -> None:
        """Creates fresh monitors from the policy."""
        policy = self.policy
        self.alarm = sequential_tests.RiskAlarm(
            policy.reference_risk,
            policy.tolerance,
            policy.alpha_alarm,
            policy.scale,
        )
        self.naive_alarm = sequential_tests.RiskAlarm(
            policy.reference_risk,
            policy.tolerance,
            policy.alpha_alarm,
            policy.loss_bound,
        )
        self.selective = sequential_tests.RiskAlarm(
            policy.r_star, 0.0, policy.delta, policy.scale
        )
        self.coverage_test: sequential_tests.MixtureTest | None = (
            None
            if policy.coverage_target is None
            else sequential_tests.MixtureTest(
                policy.coverage_target,
                sequential_tests.LESS,
                policy.alpha_alarm,
            )
        )
        self.novelty_test = sequential_tests.MixtureTest(
            min(max(policy.novelty_bound, 1e-6), 1 - 1e-6),
            sequential_tests.GREATER,
            policy.alpha_alarm,
        )

    # ----------------------------------------------------------------- state

    @property
    def state_path(self) -> Path:
        """The state file."""
        return self.directory / STATE_FILE

    def save_state(self) -> None:
        """Writes the running state with the ledger head it belongs to."""
        challengers = {
            name: {
                "process": item["process"].to_dict(),
                "batches": item["batches"],
            }
            for name, item in self.active.items()
        }
        state = {
            "version": STATE_VERSION,
            "ledger_head": self.ledger.head,
            "batch": self.batch,
            "tau": None if math.isinf(self.tau) else self.tau,
            "mechanism": self.mechanism.value,
            "halt_reason": self.halt_reason,
            "naive_flag": self.naive_flag,
            "calibrated": self.calibrated,
            "feedback": self.feedback,
            "consecutive_rollbacks": self.consecutive_rollbacks,
            "escalation_batch": self.escalation_batch,
            "queue": self.queue,
            "stopped": self.stopped,
            "alarm": self.alarm.to_dict(),
            "naive_alarm": self.naive_alarm.to_dict(),
            "selective": self.selective.to_dict(),
            "coverage_test": (
                None
                if self.coverage_test is None
                else self.coverage_test.to_dict()
            ),
            "novelty_test": self.novelty_test.to_dict(),
            "active": challengers,
            "rollback_watch": (
                None
                if self.rollback_watch is None
                else {
                    "previous": self.rollback_watch["previous"],
                    "process": self.rollback_watch["process"].to_dict(),
                }
            ),
            "pending_batches": self.pending_batches,
        }
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(json_ready(state), sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(self.state_path)

    def load_state(self) -> None:
        """Reads the state file; halts if it does not match the ledger."""
        if not self.state_path.exists():
            self.halted_by_state = self.ledger.halted
            return
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state["ledger_head"] != self.ledger.head:
            self.halted_by_state = True
            self.halt_reason = HALT_STATE
            return
        self.batch = state["batch"]
        self.tau = math.inf if state["tau"] is None else state["tau"]
        self.mechanism = labels.Mechanism(state["mechanism"])
        self.halt_reason = state["halt_reason"]
        self.naive_flag = state["naive_flag"]
        self.calibrated = state["calibrated"]
        self.feedback = state.get("feedback")
        self.consecutive_rollbacks = state["consecutive_rollbacks"]
        self.escalation_batch = dict(state["escalation_batch"])
        self.queue = list(state["queue"])
        self.stopped = dict(state["stopped"])
        self.alarm = sequential_tests.RiskAlarm.from_dict(state["alarm"])
        self.naive_alarm = sequential_tests.RiskAlarm.from_dict(
            state["naive_alarm"]
        )
        self.selective = sequential_tests.RiskAlarm.from_dict(
            state["selective"]
        )
        if state["coverage_test"] is not None:
            self.coverage_test = sequential_tests.MixtureTest.from_dict(
                state["coverage_test"]
            )
        self.novelty_test = sequential_tests.MixtureTest.from_dict(
            state["novelty_test"]
        )
        self.active = {
            name: {
                "process": sequential_tests.PairedEProcess.from_dict(
                    item["process"]
                ),
                "batches": item["batches"],
            }
            for name, item in state["active"].items()
        }
        watch = state["rollback_watch"]
        if watch is not None:
            self.rollback_watch = {
                "previous": watch["previous"],
                "process": sequential_tests.PairedEProcess.from_dict(
                    watch["process"]
                ),
            }
        self.pending_batches = state["pending_batches"]
        self.halted_by_state = self.ledger.halted

    @property
    def halted(self) -> bool:
        """Whether every case is escalated."""
        return self.ledger.halted or self.halted_by_state

    # --------------------------------------------------------------- helpers

    def verify_ledger(self) -> bool:
        """Re-checks the ledger file; enters a halt (in memory) if it fails.

        Returns:
            True if the ledger verifies; False if it fails, in which case the
            loop is halted in memory.
        """
        try:
            self.ledger.verify()
        except deployment_ledger.DeploymentLedgerError:
            self.halted_by_state = True
            self.halt_reason = HALT_LEDGER
            return False
        return True

    def pending_escalations(self) -> list[Escalation]:
        """Lists the escalations with no reply and no timeout, oldest first.

        Returns:
            The unanswered, un-timed-out escalations, oldest first.
        """
        return [
            Escalation(
                id=str(row["id"]),
                question=row["question"],
                question_hash=str(row["question_hash"]),
                evidence=tuple(row["evidence"]),
                proposed_default=str(row["proposed_default"]),
            )
            for row in self.ledger.pending()
        ]

    def escalate(
        self,
        action: never_list.NeverAction | str,
        detail: Mapping[str, Any],
        evidence: tuple[str, ...] = (),
        proposed_default: str | None = None,
    ) -> Escalation | None:
        """Writes an escalation row for a system-level question.

        Args:
            action: The never-list action or other question name.
            detail: Facts that identify the question.
            evidence: Names of the evidence files the person should read.
            proposed_default: What happens if nobody answers; the policy's
                safe default if None.

        Returns:
            The new escalation, or None if an identical question is already
            pending.
        """
        name = (
            action.value
            if isinstance(action, never_list.NeverAction)
            else action
        )
        question = {
            "action": name,
            "detail": json_ready(dict(detail)),
            "scope": self.policy.scope,
        }
        if self.guard.pending_for(question) is not None:
            return None
        identity = f"esc-{len(self.ledger.rows)}"
        self.ledger.commit(
            deployment_ledger.ESCALATION,
            {
                "phase": deployment_ledger.PHASE_OPEN,
                "id": identity,
                "question": question,
                "question_hash": deployment_ledger.question_hash(question),
                "evidence": list(evidence),
                "proposed_default": proposed_default
                or self.policy.safe_default,
            },
        )
        self.escalation_batch[identity] = self.batch
        return next(e for e in self.pending_escalations() if e.id == identity)

    def enter_halt(self, reason: str, detail: Mapping[str, Any]) -> None:
        """Writes a halt row and an incident row, then calls the incident hook.

        Does nothing if the ledger is already halted. Otherwise it commits a
        halt row, records `reason` as the halt reason, writes the default
        `incident` row and calls the ``open_incident`` hook, if one was given.

        Args:
            reason: Why the loop halts.
            detail: Facts recorded with the halt.
        """
        if self.ledger.halted:
            return
        self.ledger.commit(
            deployment_ledger.HALT,
            {
                "phase": deployment_ledger.ENTER,
                "reason": reason,
                "batch": self.batch,
                "detail": json_ready(dict(detail)),
            },
        )
        self.halt_reason = reason
        self.record_incident(reason)
        if self.hooks.open_incident is not None:
            self.hooks.open_incident(reason)

    def record_incident(self, reason: str) -> None:
        """Writes an `incident` ledger row with status ``open``.

        It records that a person must look; it is not the SLO incident
        workflow of `scmle.monitoring.incidents`, which needs an SLO
        evaluation that a halt does not have.

        Args:
            reason: Why the incident is opened.
        """
        self.ledger.commit(
            deployment_ledger.INCIDENT,
            {"status": "open", "reason": reason, "batch": self.batch},
        )

    # --------------------------------------------------------------- observe

    def observe(
        self,
        batch: pandas.DataFrame,
        predictions: Sequence[Any],
        decisions: Sequence[Any],
    ) -> Observation:
        """Routes a batch of served cases and logs what was decided.

        Verifies the ledger first (a failure halts the loop in memory), then
        keeps the batch pending until its outcomes mature, commits an
        observation summary and advances the batch counter.

        Args:
            batch: One row per case. Optional columns: `case_id`; `set_size`
                (size of the conformal set; absent means a singleton);
                `score` (use instead of `set_size`); `novel` and
                `under_covered` (bool flags); `propensity` (logged probability
                that the case's label will be observed); `audit` (case is in
                the uniform audit slice); `bypass` (case skipped the model).
            predictions: The model's output per case.
            decisions: The action the model would take per case.

        Returns:
            The routing and the action taken per case. While halted, every
            case is escalated. Cases beyond `c_max` get the safe default.

        Raises:
            ValueError: If the batch is empty or the predictions or decisions
                do not have one entry per row.
        """
        size = len(batch)
        if len(predictions) != size or len(decisions) != size or size == 0:
            raise ValueError("batch, predictions and decisions must align")
        self.verify_ledger()
        frame = batch.reset_index(drop=True)
        ids = (
            [str(v) for v in frame["case_id"]]
            if "case_id" in frame
            else [f"b{self.batch}-{i}" for i in range(size)]
        )
        novel = self.flag(frame, "novel")
        under = self.flag(frame, "under_covered")
        if "score" in frame:
            score = numpy.asarray(frame["score"], dtype=float)
        else:
            sizes = (
                frame["set_size"] if "set_size" in frame else numpy.ones(size)
            )
            score = escalation.set_size_score(sizes, novel)
        high = numpy.isin(
            numpy.asarray(decisions, dtype=object),
            list(self.policy.high_impact_actions),
        )
        blocked = (novel | under) & high
        routing = escalation.route(
            score, self.tau, self.policy.c_max, self.halted, blocked
        )
        actions = tuple(
            decisions[i]
            if routing.status[i] == escalation.ANSWERED
            else (
                self.policy.safe_default
                if routing.status[i] == escalation.SAFE_DEFAULT
                else "escalate"
            )
            for i in range(size)
        )
        record = {
            "index": self.batch,
            "ids": ids,
            "score": score,
            "status": routing.status.tolist(),
            "propensity": (
                numpy.asarray(frame["propensity"], dtype=float)
                if "propensity" in frame
                else None
            ),
            "audit": self.flag(frame, "audit"),
            "bypass": self.flag(frame, "bypass"),
            "novel": novel,
            "labels": {},
            "covered": {},
            "challenger_losses": {},
        }
        self.pending_batches.append(json_ready(record))
        self.novelty_test.update(float(novel.mean()))
        self.ledger.commit(
            deployment_ledger.OBSERVE_SUMMARY,
            {
                "batch": self.batch,
                "cases": size,
                "answered": int(routing.answered.sum()),
                "escalated": int(
                    (routing.status == escalation.ESCALATED).sum()
                ),
                "safe_default": routing.backlog,
                "blocked_high_impact": int(blocked.sum()),
                "tau": None if math.isinf(self.tau) else self.tau,
                "halted": self.halted,
                "novel_rate": float(novel.mean()),
            },
        )
        self.batch += 1
        self.save_state()
        return Observation(self.batch - 1, routing, actions)

    @staticmethod
    def flag(frame: pandas.DataFrame, name: str) -> numpy.ndarray:
        """Returns a boolean column of the batch, or all False.

        Args:
            frame: The batch.
            name: Column name.

        Returns:
            The column as a boolean array, or all False if it is absent.
        """
        if name in frame:
            return numpy.asarray(frame[name], dtype=bool)
        return numpy.zeros(len(frame), dtype=bool)

    # ---------------------------------------------------------------- mature

    def mature(
        self,
        losses: Mapping[str, float] | pandas.Series,
        covered: Mapping[str, float] | pandas.Series | None = None,
        challengers: Mapping[str, Mapping[str, float] | pandas.Series]
        | None = None,
    ) -> int:
        """Stores outcomes that arrived; `step` uses them once the window ends.

        Values for ids that match no pending batch are ignored; a repeated id
        overwrites the earlier value.

        Args:
            losses: Loss in ``[0, loss_bound]`` of the deployed model, by
                case id.
            covered: 1 if the case's outcome was inside its conformal set,
                by case id.
            challengers: For each registered challenger (and the previous
                champion under a rollback watch), its loss by case id on the
                same cases.

        Returns:
            Number of entries of `losses` that matched a pending batch.
        """
        index: dict[str, dict[str, Any]] = {}
        for record in self.pending_batches:
            for identity in record["ids"]:
                index[identity] = record
        matched = 0
        for identity, value in dict(losses).items():
            found = index.get(str(identity))
            if found is not None:
                found["labels"][str(identity)] = float(value)
                matched += 1
        for identity, value in dict(covered or {}).items():
            found = index.get(str(identity))
            if found is not None:
                found["covered"][str(identity)] = float(value)
        for name, mapping in (challengers or {}).items():
            for identity, value in dict(mapping).items():
                found = index.get(str(identity))
                if found is not None:
                    store = found["challenger_losses"].setdefault(name, {})
                    store[str(identity)] = float(value)
        self.save_state()
        return matched

    # ------------------------------------------------------------ challengers

    def register_challenger(self, name: str) -> int:
        """Registers a challenger in the ledger before it is scored.

        Args:
            name: Name of the challenger.

        Returns:
            The number of challengers now queued, including this one.
        """
        self.ledger.commit(
            deployment_ledger.CHALLENGER_RESULT,
            {
                "name": name,
                "phase": deployment_ledger.REGISTERED,
                "batch": self.batch,
            },
        )
        policy = self.policy
        self.active[name] = {
            "process": sequential_tests.PairedEProcess(
                policy.delta_promo,
                policy.promo_level,
                policy.scale,
            ),
            "batches": 0,
        }
        self.queue.append(name)
        self.save_state()
        return len(self.queue)

    def watch_rollback(self, previous: str) -> None:
        """Starts the reverse e-process after a promotion.

        Args:
            previous: Name under which the previous champion's losses are
                passed to `mature`.
        """
        policy = self.policy
        self.rollback_watch = {
            "previous": previous,
            "process": sequential_tests.PairedEProcess(
                policy.delta_rb, policy.alpha_rb, policy.scale
            ),
        }
        self.save_state()

    # ------------------------------------------------------------------ step

    def estimate(
        self,
        record: dict[str, Any],
        losses: numpy.ndarray,
        observed: numpy.ndarray,
        mask: numpy.ndarray | None = None,
    ) -> labels.BatchEstimate:
        """Estimates the Horvitz-Thompson mean loss of a batch or a subset.

        Args:
            record: The stored batch, with its propensities and audit flags.
            losses: The loss of each case.
            observed: Whether each case's label is observed.
            mask: Boolean per case, True for those to include; all if None.

        Returns:
            The estimate of the mean loss with its label mechanism. Its mean
            is None if the mechanism is `unverified`.
        """
        keep = numpy.ones(len(observed), dtype=bool) if mask is None else mask
        propensity = record["propensity"]
        audit = record["audit"]
        return labels.estimate_batch(
            losses[keep],
            observed[keep],
            None
            if propensity is None
            else numpy.asarray(propensity, float)[keep],
            None if audit is None else numpy.asarray(audit, bool)[keep],
            self.policy.epsilon,
            self.policy.loss_bound,
        )

    def estimate_signed(
        self,
        record: dict[str, Any],
        values: numpy.ndarray,
        observed: numpy.ndarray,
        mask: numpy.ndarray,
    ) -> labels.BatchEstimate:
        """Like `estimate`, for values in ``[-loss_bound, loss_bound]``.

        Args:
            record: The stored batch, with its propensities and audit flags.
            values: The signed value of each case.
            observed: Whether each case's label is observed.
            mask: Boolean per case, True for those to include.

        Returns:
            The estimate of the mean value with its label mechanism. Its mean
            is None if the mechanism is `unverified`.
        """
        propensity = record["propensity"]
        audit = record["audit"]
        return labels.estimate_batch(
            values[mask],
            observed[mask],
            None
            if propensity is None
            else numpy.asarray(propensity, float)[mask],
            None if audit is None else numpy.asarray(audit, bool)[mask],
            self.policy.epsilon,
            self.policy.loss_bound,
            signed=True,
        )

    def step(self) -> StepResult:
        """Processes matured batches and applies the decision map.

        A batch is processed once ``outcome_window`` batches have passed since
        it was observed; younger batches stay pending. If the ledger fails
        verification the step stops at once and writes nothing to the ledger.

        Returns:
            The result. Its status is `halted` if the loop is halted (also
            when the ledger fails verification, with the single event
            ``("halt", reason)``), else `ok` if at least one matured batch
            had enough labels, else `insufficient_labels`. Its events, newest
            escalation and mechanism are described by `StepResult`.
        """
        if not self.verify_ledger():
            self.save_state()
            return StepResult(
                HALTED, (("halt", HALT_LEDGER),), None, self.mechanism
            )
        first_row = len(self.ledger.rows)
        now = self.batch - 1
        keep: list[dict[str, Any]] = []
        used = False
        for record in self.pending_batches:
            window = self.policy.outcome_window
            if labels.is_matured(record["index"], now, window):
                used |= self.process_batch(record)
            else:
                keep.append(record)
        self.pending_batches = keep
        newest = self.decide()
        newest = self.time_out() or newest
        events = tuple(
            (
                row["kind"],
                str(row["body"].get("reason", row["body"].get("rule", ""))),
            )
            for row in self.ledger.rows[first_row:]
            if row["kind"] != deployment_ledger.OBSERVE_SUMMARY
        )
        self.write_label_mechanism()
        self.save_state()
        status = (
            HALTED if self.halted else (OK if used else INSUFFICIENT_LABELS)
        )
        return StepResult(status, events, newest, self.mechanism)

    def process_batch(self, record: dict[str, Any]) -> bool:
        """Feeds one matured batch to the monitors; False if unusable.

        When the label mechanism permits action, the batch feeds the loss
        alarm, the selective-risk alarm (answered cases), the coverage test
        (if the batch has coverage values) and the challenger e-processes;
        otherwise it feeds only the naive alarm, which sets a flag on
        crossing. Records the batch's label mechanism on the loop. A
        ``ValueError`` from the estimator (a labelled loss outside
        ``[0, loss_bound]``) propagates.

        Args:
            record: The stored batch with its matured labels.

        Returns:
            False if fewer than ``min_matured`` cases decided by the model
            (not bypassed) have a label, in which case nothing is fed;
            otherwise True.

        Raises:
            ScmleError: If the estimate has no mean although the mechanism
                permits action.
        """
        ids = record["ids"]
        stored = record["labels"]
        observed = numpy.array([i in stored for i in ids])
        losses = numpy.array(
            [stored.get(i, math.nan) for i in ids], dtype=float
        )
        bypass = numpy.asarray(record["bypass"], dtype=bool)
        status = numpy.asarray(record["status"])
        model_cases = ~bypass
        if int(observed[model_cases].sum()) < self.policy.min_matured:
            return False
        estimate = self.estimate(record, losses, observed, model_cases)
        self.mechanism = estimate.mechanism
        naive = float(numpy.mean(losses[observed & model_cases]))
        crossed = self.naive_alarm.update(naive)
        if crossed and not labels.permits_action(estimate.mechanism):
            self.naive_flag = True
        if not labels.permits_action(estimate.mechanism):
            return True
        if estimate.mean is None:
            raise errors.ScmleError("estimate has no mean")
        self.alarm.update(estimate.mean)
        answered = (status == escalation.ANSWERED) & model_cases
        if int(observed[answered].sum()) > 0:
            selective = self.estimate(record, losses, observed, answered)
            if selective.mean is not None:
                self.selective.update(selective.mean)
        if self.coverage_test is not None and record["covered"]:
            values = list(record["covered"].values())
            self.coverage_test.update(float(numpy.mean(values)))
        self.update_challengers(record, ids, losses, observed, model_cases)
        return True

    def update_challengers(
        self,
        record: dict[str, Any],
        ids: list[str],
        losses: numpy.ndarray,
        observed: numpy.ndarray,
        model_cases: numpy.ndarray,
    ) -> None:
        """Feeds the e-processes with Horvitz-Thompson paired differences.

        Only challengers registered with the loop, and the previous champion
        under a rollback watch, are fed. A challenger is skipped for this
        batch if it lacks a loss for any labelled model case or the estimate
        has no mean. The batch mean is clamped to the policy's scale, and an
        active challenger's matured-batch count rises by one.

        Args:
            record: The stored batch with its challenger losses.
            ids: Case identifiers of the batch.
            losses: The served model's loss per case.
            observed: Whether each case's label is observed.
            model_cases: Cases that went through the model.
        """
        needed = observed & model_cases
        for name, other in record["challenger_losses"].items():
            holder: Any
            if name in self.active:
                holder = self.active[name]
            elif (
                self.rollback_watch is not None
                and name == self.rollback_watch["previous"]
            ):
                holder = self.rollback_watch
            else:
                continue
            paired = numpy.array([i in other for i in ids])
            if not paired[needed].all():
                continue
            second = numpy.array(
                [other.get(i, math.nan) for i in ids], dtype=float
            )
            gap = numpy.where(needed, losses - second, 0.0)
            estimate = self.estimate_signed(record, gap, observed, model_cases)
            if estimate.mean is None:
                continue
            scale = self.policy.scale
            holder["process"].update_gap(min(max(estimate.mean, -scale), scale))
            if "batches" in holder:
                holder["batches"] += 1

    # -------------------------------------------------------------- decisions

    def decide(self) -> Escalation | None:
        """Applies the decision map to what the monitors now say.

        Acts on a naive alarm under an unverified mechanism (halt and
        escalate), an L1 alarm (alarm row, then an investigation escalation or
        a retrain request, by ``alarm_gates_retrain``), a selective-risk alarm
        (halt and escalate) and a crossed coverage or novelty test
        (escalate). Each crossed monitor is replaced by a fresh one. When the
        mechanism permits action it also decides rollback proposals and the
        challenger queue.

        Returns:
            The newest escalation opened by the decisions, or None if none was
            opened.
        """
        newest: Escalation | None = None
        batch = self.batch - 1
        verified = labels.permits_action(self.mechanism)
        if self.naive_flag and not verified:
            self.naive_flag = False
            self.ledger.commit(
                deployment_ledger.ALARM,
                {"rule": "L1_naive", "reason": HALT_UNVERIFIED, "batch": batch},
            )
            self.enter_halt(HALT_UNVERIFIED, {"batch": batch})
            newest = (
                self.escalate(
                    "l1_alarm_unverified_label_mechanism",
                    {"batch": batch, "mechanism": self.mechanism.value},
                )
                or newest
            )
        if verified and bool(self.alarm.test.crossed):
            self.ledger.commit(
                deployment_ledger.ALARM,
                {
                    "rule": "L1",
                    "reason": "mean loss above reference plus tolerance",
                    "batch": batch,
                    "e_value": float(self.alarm.test.e_value),
                    "mechanism": self.mechanism.value,
                },
            )
            if not self.policy.alarm_gates_retrain:
                newest = (
                    self.escalate(
                        "l1_alarm_investigate",
                        {"batch": batch, "reason": "alarm is not certified"},
                    )
                    or newest
                )
            else:
                self.retrain_challenger(batch)
            self.alarm = sequential_tests.RiskAlarm(
                self.policy.reference_risk,
                self.policy.tolerance,
                self.policy.alpha_alarm,
                self.policy.scale,
            )
        if verified and bool(self.selective.test.crossed):
            self.ledger.commit(
                deployment_ledger.ALARM,
                {
                    "rule": "selective_risk",
                    "reason": HALT_SELECTIVE,
                    "batch": batch,
                },
            )
            self.enter_halt(HALT_SELECTIVE, {"batch": batch})
            newest = (
                self.escalate("halt_selective_risk", {"batch": batch}) or newest
            )
            self.selective = sequential_tests.RiskAlarm(
                self.policy.r_star, 0.0, self.policy.delta, self.policy.scale
            )
        if self.coverage_test is not None and bool(self.coverage_test.crossed):
            newest = (
                self.escalate("coverage_below_nominal", {"batch": batch})
                or newest
            )
            self.coverage_test = sequential_tests.MixtureTest(
                self.policy.coverage_target or 0.9,
                sequential_tests.LESS,
                self.policy.alpha_alarm,
            )
        if bool(self.novelty_test.crossed):
            newest = (
                self.escalate(
                    "out_of_support_rate_above_bound", {"batch": batch}
                )
                or newest
            )
            self.novelty_test = sequential_tests.MixtureTest(
                min(max(self.policy.novelty_bound, 1e-6), 1 - 1e-6),
                sequential_tests.GREATER,
                self.policy.alpha_alarm,
            )
        if verified:
            newest = self.decide_rollback() or newest
            newest = self.decide_queue() or newest
        return newest

    def retrain_challenger(self, batch: int) -> None:
        """Records a retrain request and registers the hook's challenger.

        If a challenger is already queued, the hook is not called and the
        request is recorded as skipped. Otherwise the ``retrain`` hook, if
        any, is called. A returned name not yet registered in the ledger is
        registered as a challenger.

        Args:
            batch: Index of the batch whose alarm triggered the request.
        """
        name = None
        skipped = None
        if self.queue:
            skipped = "a challenger is already pending"
        elif self.hooks.retrain is not None:
            name = self.hooks.retrain()
        body = {"reason": "L1 alarm", "batch": batch, "challenger": name}
        if skipped is not None:
            body["skipped"] = skipped
        self.ledger.commit(deployment_ledger.RETRAIN, body)
        if name is not None and name not in self.ledger.registered:
            self.register_challenger(name)

    def decide_rollback(self) -> Escalation | None:
        """Turns a crossed reverse process into a rollback proposal.

        If the watch has crossed, commits the proposal, raises a finite
        `tau` by ``tighten_step`` (capped at 1; an infinite `tau` is kept) and
        ends the watch.

        Returns:
            The escalation opened for a second consecutive rollback proposal,
            or None if there is none, no proposal was made or an identical
            question is already pending.
        """
        watch = self.rollback_watch
        if watch is None or not watch["process"].crossed:
            return None
        self.ledger.commit(
            deployment_ledger.ROLLBACK_PROPOSAL,
            {
                "previous": watch["previous"],
                "e_value": watch["process"].e_value,
                "batch": self.batch - 1,
                "tau_before": None if math.isinf(self.tau) else self.tau,
            },
        )
        if not math.isinf(self.tau):
            self.tau = min(1.0, self.tau + self.policy.tighten_step)
        self.rollback_watch = None
        self.consecutive_rollbacks += 1
        if self.consecutive_rollbacks >= 2:
            return self.escalate(
                "two_consecutive_rollbacks", {"batch": self.batch - 1}
            )
        return None

    def decide_queue(self) -> Escalation | None:
        """Decides challengers in registration order with e-LOND.

        Decides the first queued challenger once its evidence reaches the
        current e-LOND level or it has used ``challenger_budget`` matured
        batches, then the next, and stops at the first that is neither. A
        rejection writes a promotion proposal.

        Returns:
            The newest escalation opened for a challenger that used its budget
            without enough evidence, or None.
        """
        newest: Escalation | None = None
        sequence = self.ledger.require_promotion()
        while self.queue:
            name = self.queue[0]
            item = self.active[name]
            level = sequence.rule.level(sequence.state)
            value = min(item["process"].e_value, 1e300)
            reached = Fraction(value) * level >= 1
            exhausted = item["batches"] >= self.policy.challenger_budget
            if not (reached or exhausted):
                break
            step = sequence.rule.step(
                sequence.state,
                deployment_ledger.sequential_tests.online_fdr.Evidence(e=value),
            )
            self.ledger.commit(
                deployment_ledger.CHALLENGER_RESULT,
                {
                    "name": name,
                    "phase": deployment_ledger.DECIDED,
                    "e_value": value,
                    "level": str(step.level),
                    "rejected": step.rejected,
                    "wealth_after": str(step.wealth_after),
                    "matured_batches": item["batches"],
                    "mechanism": self.mechanism.value,
                },
            )
            self.queue.pop(0)
            del self.active[name]
            if step.rejected:
                self.consecutive_rollbacks = 0
                self.ledger.commit(
                    deployment_ledger.PROMOTION_PROPOSAL,
                    {
                        "name": name,
                        "e_value": value,
                        "level": str(step.level),
                        "wealth_after": str(step.wealth_after),
                        "matured_batches": item["batches"],
                        "needs": "signed registry approval and attestation",
                    },
                )
            elif not reached:
                newest = (
                    self.escalate(
                        "challenger_evidence_insufficient",
                        {"name": name, "matured_batches": item["batches"]},
                    )
                    or newest
                )
        return newest

    def time_out(self) -> Escalation | None:
        """Closes escalations unanswered for ``reply_timeout`` batches.

        Each is closed with a timeout row that applies its own proposed
        default (the safe default unless the escalation proposed another).
        An escalation the loop has no opening batch for counts from the
        current batch.

        Returns:
            Always None; the closed escalations are recorded in the ledger.
        """
        for row in list(self.ledger.pending()):
            identity = str(row["id"])
            opened = self.escalation_batch.setdefault(identity, self.batch)
            if self.batch - opened >= self.policy.reply_timeout:
                self.ledger.commit(
                    deployment_ledger.ESCALATION,
                    {
                        "phase": deployment_ledger.PHASE_TIMEOUT,
                        "id": identity,
                        "applied": row["proposed_default"],
                        "tier": qualification.Tier.UNVERIFIED.value,
                    },
                )
        return None

    # ------------------------------------------------------------ calibration

    def calibrate(
        self,
        score: Any,
        loss: Any,
        weights: Any = None,
        confidence: numpy.ndarray | None = None,
    ) -> escalation.ThresholdResult:
        """Selects `tau` on matured calibration cases and applies it.

        The first calibration, and any raise of `tau`, is free. Lowering `tau`
        after the first calibration relaxes the rule and needs an attestation
        (the relax-threshold action; without one the guard raises
        `AttestationRequired` and `tau` is unchanged), except as the
        recalibration that ends a halt, which must itself pass the
        fixed-sequence test (`tau` finite). While the ledger stays halted,
        `tau` is not changed. Every call also clears the in-memory halt that
        a failed ledger verification or a state mismatch sets.

        Args:
            score: The escalation score of each calibration case.
            loss: The loss of each case.
            weights: Optional Horvitz-Thompson weights ``1 / propensity``.
            confidence: Not used.

        Returns:
            The result of the threshold selection. It is written to
            `risk_coverage.json` first, then to the ledger: as the halt
            release when it ends a halt, otherwise as an observation summary.
        """
        result = escalation.select_threshold(
            score, loss, self.policy.r_star, self.policy.delta, weights
        )
        curve = escalation.risk_coverage(score, loss, weights)
        (self.directory / RISK_COVERAGE_FILE).write_text(
            json.dumps(
                json_ready(
                    {"curve": curve.to_dict(), "threshold": result.to_dict()}
                )
            ),
            encoding="utf-8",
            newline="\n",
        )
        released = False
        if self.ledger.halted and not result.everything_escalates:
            self.ledger.commit(
                deployment_ledger.HALT,
                {
                    "phase": deployment_ledger.RELEASE,
                    "recalibration": json_ready(result.to_dict()),
                    "batch": self.batch,
                },
            )
            self.halt_reason = None
            released = True
        elif result.tau < self.tau and self.calibrated:
            self.guard.require(
                never_list.NeverAction.RELAX_THRESHOLD,
                {"tau_from": self.tau, "tau_to": result.tau},
                self.policy.scope,
            )
        if not released:
            self.ledger.commit(
                deployment_ledger.OBSERVE_SUMMARY,
                {
                    "calibration": json_ready(result.to_dict()),
                    "batch": self.batch,
                },
            )
        if not self.ledger.halted or released:
            self.tau = result.tau
        self.calibrated = True
        self.halted_by_state = (
            self.halted_by_state and released is False and False
        )
        self.save_state()
        return result

    def check_feedback(
        self,
        bypass_features: pandas.DataFrame,
        model_features: pandas.DataFrame,
    ) -> qualification.Check:
        """Tests whether the model's decisions shift the next-period features.

        Runs `labels.bypass_test` on the next-period features of the
        randomized bypass slice against those of model-decided cases, records
        the result in the ledger and in `label_mechanism.json`, and escalates
        if a feedback loop is detected. The test is not run by `step`: the
        next-period features are the caller's to supply.

        Args:
            bypass_features: Next-period features of the bypass slice.
            model_features: Next-period features of model-decided cases.

        Returns:
            The bypass test's result, as a check with its statistic and
            p-value.

        Raises:
            ValueError: If the policy declares no bypass slice.
        """
        if self.policy.bypass_rate <= 0.0:
            raise ValueError(
                "no bypass slice is declared (policy.bypass_rate is 0): "
                "feedback cannot be tested"
            )
        check = labels.bypass_test(
            bypass_features, model_features, seed=len(self.ledger.rows)
        )
        verdict = labels.feedback_verdict(self.policy.bypass_rate, check)
        self.feedback = {
            "verdict": verdict,
            "p_value": check.p_value,
            "statistic": check.statistic,
            "n_bypass": len(bypass_features),
            "n_model": len(model_features),
            "batch": self.batch,
        }
        self.ledger.commit(
            deployment_ledger.OBSERVE_SUMMARY,
            {"feedback_test": json_ready(self.feedback)},
        )
        if verdict == "loop_detected":
            self.escalate(
                "feedback_loop_detected",
                {"p_value": check.p_value, "batch": self.batch},
                proposed_default=self.policy.safe_default,
            )
        self.write_label_mechanism()
        self.save_state()
        return check

    def write_label_mechanism(self) -> None:
        """Writes `label_mechanism.json`."""
        record = {
            "mechanism": self.mechanism.value,
            "actions_permitted": labels.permits_action(self.mechanism),
            "epsilon": self.policy.epsilon,
            "bypass_rate": self.policy.bypass_rate,
            "feedback": (
                labels.UNVERIFIED
                if self.policy.bypass_rate <= 0.0
                else (
                    "bypass slice declared"
                    if self.feedback is None
                    else self.feedback["verdict"]
                )
            ),
            "feedback_test": self.feedback,
            "feedback_note": (
                labels.FEEDBACK_UNVERIFIED
                if self.policy.bypass_rate <= 0.0
                else ""
            ),
            "outcome_window": self.policy.outcome_window,
        }
        (self.directory / LABEL_MECHANISM_FILE).write_text(
            json.dumps(record, sort_keys=True), encoding="utf-8", newline="\n"
        )

    # ------------------------------------------------------------------ reply

    def reply(
        self,
        escalation_id: str,
        attestation: deployment_ledger.Attestation,
        key: str | bytes | None = None,
    ) -> ReplyOutcome:
        """Records a person's signed reply to a pending escalation.

        On success the attestation is committed as a ledger row, which closes
        the escalation.

        Args:
            escalation_id: The escalation answered.
            attestation: The signed reply.
            key: Key that verifies the signature; defaults to the one
                registered for the reply's `key_id` at `open`.

        Returns:
            The outcome: tier `attested` if the person approved or overrode,
            else `unverified` with the safe default applying; also the
            decision itself.

        Raises:
            ReplyRefused: If the escalation is not pending, the reply is
                bound to another escalation, question or scope, was signed
                with a key the system holds (or is verified by one), no
                verification key is known for its key id, it does not verify,
                or it is expired or dated in the future. A malformed date
                raises the ledger's `DeploymentLedgerError` instead.
        """
        pending = {e.id: e for e in self.pending_escalations()}
        target = pending.get(escalation_id)
        if target is None:
            raise ReplyRefused(f"{escalation_id!r} is not a pending escalation")
        if attestation.escalation_id != escalation_id:
            raise ReplyRefused("the reply answers a different escalation")
        if attestation.question_hash != target.question_hash:
            raise ReplyRefused("the reply is bound to a different question")
        if attestation.scope != target.question["scope"]:
            raise ReplyRefused("the reply names a different scope")
        if attestation.key_id in self.ledger.system_keys:
            raise ReplyRefused(
                "the reply is signed with a key the system holds"
            )
        verifier = (
            key
            if key is not None
            else self.verification_keys.get(attestation.key_id)
        )
        if verifier is None:
            raise ReplyRefused("no verification key for the reply's key id")
        if (
            deployment_ledger.key_fingerprint(verifier)
            in self.ledger.system_keys
        ):
            raise ReplyRefused("the verification key is one the system holds")
        if not attestation.verify(verifier):
            raise ReplyRefused("the reply's signature does not verify")
        today = self.today()
        issued = deployment_ledger.parse_date(attestation.date, "date")
        expires = deployment_ledger.parse_date(attestation.expires, "expires")
        if not issued <= today <= expires:
            raise ReplyRefused("the reply is expired or dated in the future")
        self.ledger.commit(deployment_ledger.ATTESTATION, attestation.to_dict())
        self.save_state()
        allowed = attestation.decision in never_list.ALLOWING
        return ReplyOutcome(
            qualification.Tier.ATTESTED
            if allowed
            else qualification.Tier.UNVERIFIED,
            attestation.decision,
            not allowed,
        )

    # -------------------------------------------------------------- never list

    def perform(
        self,
        action: never_list.NeverAction,
        detail: Mapping[str, Any],
        do: Callable[[], Any],
        proposed_default: str = "action_not_performed",
    ) -> Any:
        """Runs `do()` only with a valid attestation for this exact action.

        Without one the guard raises `AttestationRequired` and `do` is not
        called; an `escalation` row is written unless an identical question
        is already pending. The loop state is saved either way.

        Args:
            action: The never-list action.
            detail: Facts that identify this instance of the action.
            do: The action to perform.
            proposed_default: What happens by default while the escalation is
                pending.

        Returns:
            Whatever `do` returns.
        """
        try:
            return self.guard.perform(
                action, detail, self.policy.scope, do, proposed_default
            )
        finally:
            self.save_state()

    def guard_registry(self, registry: Any) -> None:
        """Routes every champion move of `registry` through this guard.

        `promote_challenger`, `check_rollback` and `register_run(promote=True)`
        of a `LifecycleManager` over this registry then need an attestation,
        as `move_champion` does. Moving the alias by editing the registry's
        files is not stopped.

        Args:
            registry: The model registry to guard.
        """
        registry.attach_guard(self.guard, self.policy.scope)

    def move_champion(
        self,
        registry: Any,
        model: str,
        version: int,
        approval: Any,
        signing_key: str | bytes | None = None,
    ) -> Any:
        """Promotes a version to champion, with an attestation for that move.

        The registry still checks its own signed approval underneath.

        Args:
            registry: The model registry.
            model: Model name.
            version: Version to make champion.
            approval: The registry approval of that version.
            signing_key: Key that verifies the approval.

        Returns:
            Whatever the registry's `promote` returns.
        """

        def move() -> Any:
            return registry.promote(
                model, version, approval=approval, signing_key=signing_key
            )

        if getattr(registry, "guard", None) is self.guard:
            try:
                return move()
            finally:
                self.save_state()
        return self.perform(
            never_list.NeverAction.MOVE_CHAMPION,
            {"model": model, "version": version},
            move,
        )

    def execute_rollback(
        self, do: Callable[[], Any], detail: Mapping[str, Any]
    ) -> Any:
        """Runs a rollback, with an attestation for the move-champion action.

        Args:
            do: The rollback to perform.
            detail: Facts that identify it.

        Returns:
            Whatever `do` returns.
        """
        return self.perform(
            never_list.NeverAction.MOVE_CHAMPION, {"rollback": dict(detail)}, do
        )

    def reopen_sealed(
        self, do: Callable[[], Any], detail: Mapping[str, Any]
    ) -> Any:
        """Runs `do()` that reopens `C` or `F` or touches a sealed set.

        It runs only with an attestation for that exact detail.

        Args:
            do: The action to perform.
            detail: Facts that identify it.

        Returns:
            Whatever `do` returns.
        """
        return self.perform(
            never_list.NeverAction.REOPEN_SEALED, dict(detail), do
        )

    def rewrite_ledger(
        self, do: Callable[[], Any], detail: Mapping[str, Any]
    ) -> Any:
        """Runs `do()` that edits, deletes, reorders or truncates ledger rows.

        It runs only with an attestation for that exact detail.

        Args:
            do: The action to perform.
            detail: Facts that identify it.

        Returns:
            Whatever `do` returns.
        """
        return self.perform(
            never_list.NeverAction.EDIT_LEDGER, dict(detail), do
        )

    def act_on_unverified(self, kind: str, do: Callable[[], Any]) -> Any:
        """Runs `do()` that retrains, promotes or rolls back unverified.

        It runs only with an attestation naming the kind of action.

        Args:
            kind: The kind of action.
            do: The action to perform.

        Returns:
            Whatever `do` returns.
        """
        return self.perform(
            never_list.NeverAction.ACT_ON_UNVERIFIED, {"kind": kind}, do
        )

    def silence_alert(self, name: str, do: Callable[[], Any]) -> Any:
        """Runs `do()` that suppresses, silences or reroutes an alert.

        It runs only with an attestation naming the alert.

        Args:
            name: Name of the alert.
            do: The action to perform.

        Returns:
            Whatever `do` returns.
        """
        return self.perform(
            never_list.NeverAction.SUPPRESS_ALERT, {"alert": name}, do
        )

    def change_label_policy(
        self, change: Mapping[str, Any], do: Callable[[], Any]
    ) -> Any:
        """Runs `do()` that changes the labelling or randomization policy.

        It runs only with an attestation for exactly this change.

        Args:
            change: The change, as recorded.
            do: The action to perform.

        Returns:
            Whatever `do` returns.
        """
        return self.perform(
            never_list.NeverAction.CHANGE_LABEL_POLICY, dict(change), do
        )

    def delete_or_send(self, what: str, do: Callable[[], Any]) -> Any:
        """Runs `do()` that deletes data, models or artifacts, or sends out.

        It runs only with an attestation naming what is deleted or sent.

        Args:
            what: What is deleted or sent.
            do: The action to perform.

        Returns:
            Whatever `do` returns.
        """
        return self.perform(
            never_list.NeverAction.DELETE_OR_EXTERNAL, {"what": what}, do
        )

    def amend_policy(self, **changes: Any) -> LoopPolicy:
        """Changes the policy: tightening is free, anything else is gated.

        A field in ``NUMERIC_ORIENTATION`` (for example ``r_star`` or
        ``tolerance``) moved to a stricter or equal value needs no
        attestation. A looser value needs one (the relax-threshold action),
        and so does any other field, or setting such a field to None (the
        change-charter action: safe defaults, high-impact actions, scope).
        The change is recorded in the ledger and restarts the monitors with
        fresh state. A gated change without an attestation is refused with
        `AttestationRequired`, and nothing is changed.

        Args:
            **changes: New values of policy fields, by field name.

        Returns:
            The updated policy.

        Raises:
            ValueError: For an unknown field, or a new value the policy's
                range checks reject.
        """
        names = {f.name for f in dataclasses.fields(LoopPolicy)}
        unknown = set(changes) - names
        if unknown:
            raise ValueError(f"unknown policy fields: {sorted(unknown)}")
        for field, new in changes.items():
            old = getattr(self.policy, field)
            orientation = NUMERIC_ORIENTATION.get(field)
            if orientation is not None and new is not None:
                if (new - old) * orientation >= 0:
                    continue
                action = never_list.NeverAction.RELAX_THRESHOLD
            else:
                action = never_list.NeverAction.CHANGE_CHARTER
            self.guard.require(
                action,
                {
                    "field": field,
                    "from": json_ready(old),
                    "to": json_ready(new),
                },
                self.policy.scope,
            )
        updated = dataclasses.replace(self.policy, **changes)
        self.ledger.commit(
            deployment_ledger.OBSERVE_SUMMARY,
            {
                "policy_amendment": json_ready(changes),
                "batch": self.batch,
            },
        )
        self.policy = updated
        self.build_monitors()
        self.save_state()
        return updated
