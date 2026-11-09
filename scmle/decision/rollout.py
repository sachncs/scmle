"""The promote, hold and rollback rule.

For a challenger against the champion, with `Delta` the challenger's
advantage on the primary metric (in the metric's favourable direction):

    promote   when, at one and the same observation, the e-process of
              `H0_primary: Delta <= delta_min` and the e-process of
              `H0_guard_g: harm_g >= margin_g` for every guardrail `g` are at
              least `1 / level`, the sample-ratio check has not rejected
              and the experiment's level was assigned from the decision
              budget.
    hold      otherwise, until `n_max` units, then `hold_final`.
    hold_invalid  when the sample-ratio check rejects: no claim is made.

Error guarantee (intersection-union). The promotion event is contained in the
event that the e-process of any one component ever reaches `1 / level`. If
the challenger is not worth `delta_min`, or is harmful on some guardrail,
that component's null is true, and by Ville's inequality its e-process
reaches `1 / level` with probability at most `level`, for every stopping rule
including a look after every unit. No correction across components is
needed. Across experiments the levels follow the discount-only rule of
`DecisionBudget`, so the family-wise probability of a false promotion is at
most `alpha_promote` by the union bound.

After a promotion the `RollbackMonitor` tests `H0_rb: Delta_new_vs_old >=
-tau` on fresh randomized traffic and rolls back when its e-process reaches
`1 / alpha_rollback`; `P(rollback | no worse than tau) <= alpha_rollback`
over the monitoring lifetime. The converse error is described by the
detection delay, which the certificates measure.

The pseudo-outcomes are centred by the running mean of earlier outcomes
(`anytime.running_center`), which leaves their mean unchanged and cuts the
variance the e-process has to overcome.

Assumptions, stated in every record: randomized assignment with the recorded
probability, no interference between units, and a stationary effect (under
drift the confidence sequence covers the running average of `Delta_t`). A
promotion is a recommendation: the independent signed approval of
`LifecycleManager` is unchanged and still required.
"""

from collections.abc import Mapping
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy
from numpy.typing import ArrayLike
from numpy.typing import NDArray

import scmle.data.charter.decision
from scmle.decision import anytime
from scmle.decision import design

HOLD = "hold"
PROMOTE = "promote"
HOLD_FINAL = "hold_final"
HOLD_INVALID = "hold_invalid"
ROLLBACK = "rollback"
ASSUMPTIONS = (
    "randomized assignment with the recorded probability",
    "no interference between units",
    "stationary effect over the window (else the running average)",
    "missing outcomes handled by a declared rule (none applied here)",
)


@dataclasses.dataclass(frozen=True, slots=True)
class Decision:
    """The state of a rollout after the latest observations.

    Attributes:
        status: One of `hold`, `promote`, `hold_final`, `hold_invalid`.
        units: Units observed (at the decision, when one was taken).
        level: The experiment's level from the decision budget.
        log_e: Log e-value of the primary and of each guardrail at the end
            of the last batch.
        lower_bound: Time-uniform lower confidence bound on the primary
            advantage at level `level`, in metric units.
        challenger_units: Units assigned to the challenger.
        tier: `confirmed` once the rule promoted with the sample-ratio check
            passing, otherwise `unverified`.
        assumptions: The conditions the guarantee needs.
    """

    status: str
    units: int
    level: float
    log_e: dict[str, float]
    lower_bound: float
    challenger_units: int
    tier: str
    assumptions: tuple[str, ...] = ASSUMPTIONS

    def to_dict(self) -> dict[str, Any]:
        """Returns the JSON-ready record.

        Returns:
            Every field of the decision as a mapping.
        """
        return dataclasses.asdict(self)


def metric_stream(
    plan: scmle.data.charter.decision.RolloutPlan,
    guard: scmle.data.charter.decision.Guardrail | None,
) -> anytime.Stream:
    """The centred pseudo-outcome stream and null of the primary or a guardrail.

    A higher-is-worse guardrail tests whether the harm reaches its margin (a
    `less` test); a lower-is-worse one negates its values and does the same.

    Args:
        plan: The rollout plan giving the assignment probability, the
            primary metric range and the smallest worthwhile improvement.
        guard: The guardrail, or None for the primary metric.

    Returns:
        The stream on which the e-process for that metric is run, with its
        null in rescaled units.
    """
    q = plan.assignment_prob
    if guard is None:
        return anytime.centered_stream(
            q, plan.delta_min / (plan.high - plan.low)
        )
    margin = guard.margin / (guard.high - guard.low)
    return anytime.centered_stream(q, margin, "less")


def metric_values(
    plan: scmle.data.charter.decision.RolloutPlan,
    guard: scmle.data.charter.decision.Guardrail | None,
    arm: NDArray[numpy.float64],
    outcome: ArrayLike,
    carry: tuple[float, int] = (0.0, 0),
) -> tuple[NDArray[numpy.float64], tuple[float, int]]:
    """Centred pseudo-outcomes of one metric and the updated centre state.

    The centre of each unit is the shrunken mean of the outcomes seen before
    it, so the values stay a valid stream however the batches are cut.

    Args:
        plan: The rollout plan giving the assignment probability and the
            primary metric range.
        guard: The guardrail, or None for the primary metric.
        arm: 1 for the challenger, 0 for the champion, per unit.
        outcome: Raw metric value per unit, aligned with `arm`.
        carry: The `(sum, count)` of rescaled outcomes seen in earlier
            batches.

    Returns:
        The pseudo-outcomes (negated for a lower-is-worse guardrail), and
        the updated `(sum, count)` that includes this batch.
    """
    low, high = (
        (plan.low, plan.high) if guard is None else (guard.low, guard.high)
    )
    y = (numpy.asarray(outcome, dtype=float) - low) / (high - low)
    z = anytime.pseudo_outcomes(
        arm, y, plan.assignment_prob, anytime.running_center(y, carry=carry)
    )
    if guard is not None and not guard.higher_is_worse:
        z = -z
    return z, (carry[0] + float(y.sum()), carry[1] + len(y))


class Rollout:
    """One experiment: a challenger against the champion.

    The level is supplied by the caller, assigned from the `DecisionBudget`
    before any traffic; nothing in this class draws on the budget.

    Attributes:
        plan: The rollout plan.
        experiment_id: Identity of the experiment.
        level: The level assigned to this experiment.
        status: One of `hold`, `promote`, `hold_final`, `hold_invalid`.
        units: Units observed so far.
        challenger_units: Units assigned to the challenger so far.
        done_at: The unit at which the status stopped being `hold`, or 0
            while the experiment is still running.
    """

    def __init__(
        self,
        plan: scmle.data.charter.decision.RolloutPlan,
        experiment_id: str,
        level: float,
    ) -> None:
        """Starts an experiment at the level the budget assigned.

        Args:
            plan: The rollout plan.
            experiment_id: Identity of the experiment.
            level: The level the decision budget assigned, strictly between
                0 and 1.

        Raises:
            ValueError: If `level` is not strictly between 0 and 1.
        """
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between 0 and 1")
        self.plan = plan
        self.experiment_id = experiment_id
        self.level = level
        self.target = -math.log(level)
        names = [plan.primary, *(g.name for g in plan.guardrails)]
        self.names = names
        self.guards: list[scmle.data.charter.decision.Guardrail | None] = [
            None,
            *plan.guardrails,
        ]
        self.processes = {
            name: anytime.EProcess(metric_stream(plan, guard))
            for name, guard in zip(names, self.guards, strict=True)
        }
        self.carry: dict[str, tuple[float, int]] = dict.fromkeys(
            names, (0.0, 0)
        )
        self.primary_z: list[NDArray[numpy.float64]] = []
        self.units = 0
        self.challenger_units = 0
        self.status = HOLD
        self.done_at = 0

    def observe(
        self,
        arm: ArrayLike,
        primary: ArrayLike,
        guardrails: Mapping[str, ArrayLike] | None = None,
    ) -> Decision:
        """Adds a batch of units in order and returns the decision so far.

        Every unit is a look: the promotion and the sample-ratio check are
        evaluated at each observation in the batch, so batch size does not
        change the error guarantee. After a decision, later units are
        ignored.

        Args:
            arm: 1 for the challenger, 0 for the champion, per unit.
            primary: Raw primary metric value per unit, aligned with `arm`.
            guardrails: Raw value per unit for each guardrail, keyed by
                guardrail name.

        Returns:
            The decision after this batch. Units beyond `n_max` are dropped.

        Raises:
            ValueError: If a guardrail of the rollout plan has no values in
                `guardrails`. Invalid arms or outcomes are rejected with a
                `ValueError` by `anytime.pseudo_outcomes`.
        """
        if self.status != HOLD:
            return self.decision()
        a = numpy.asarray(arm, dtype=float)
        room = self.plan.n_max - self.units
        a = a[:room]
        supplied = dict(guardrails or {})
        missing = [
            g.name for g in self.plan.guardrails if g.name not in supplied
        ]
        if missing:
            raise ValueError(f"missing guardrail outcomes: {missing}")
        paths = []
        for name, guard in zip(self.names, self.guards, strict=True):
            raw = primary if guard is None else supplied[name]
            values, self.carry[name] = metric_values(
                self.plan,
                guard,
                a,
                numpy.asarray(raw, dtype=float)[:room],
                self.carry[name],
            )
            self.processes[name].update(values)
            paths.append(self.processes[name].path)
            if guard is None:
                self.primary_z.append(values)
        counts = self.challenger_units + numpy.cumsum(a)
        totals = self.units + numpy.arange(1, len(a) + 1)
        invalid = design.srm_log_bayes_factor(
            counts, totals, self.plan.assignment_prob
        ) >= -math.log(self.plan.srm_level)
        crossed = (numpy.vstack(paths) >= self.target).all(axis=0)
        first_bad = int(numpy.argmax(invalid)) if invalid.any() else None
        first_win = int(numpy.argmax(crossed)) if crossed.any() else None
        if first_bad is not None and (
            first_win is None or first_bad <= first_win
        ):
            self.status, self.done_at = HOLD_INVALID, self.units + first_bad + 1
        elif first_win is not None:
            self.status, self.done_at = PROMOTE, self.units + first_win + 1
        self.units += len(a)
        self.challenger_units += int(a.sum())
        if self.status == HOLD and self.units >= self.plan.n_max:
            self.status, self.done_at = HOLD_FINAL, self.units
        return self.decision()

    def decision(self) -> Decision:
        """The decision record for the current state.

        Returns:
            The status, units, level, log e-values and lower confidence
            bound on the primary advantage (in metric units; the negative
            metric range when there are no units). After a promotion or an
            invalid hold, the bound uses only the units up to the decision.
        """
        scale = self.plan.high - self.plan.low
        stream = metric_stream(self.plan, None)
        z = (
            numpy.concatenate(self.primary_z)
            if self.primary_z
            else numpy.zeros(0)
        )
        if self.done_at and self.status in (PROMOTE, HOLD_INVALID):
            z = z[: self.done_at]
        bound = (
            anytime.confidence_bound(z, stream, self.level) * scale
            if z.size
            else -scale
        )
        return Decision(
            status=self.status,
            units=self.done_at or self.units,
            level=self.level,
            log_e={n: p.log_e for n, p in self.processes.items()},
            lower_bound=bound,
            challenger_units=self.challenger_units,
            tier="confirmed" if self.status == PROMOTE else "unverified",
        )


class RollbackMonitor:
    """Monitors a promoted champion on fresh randomized traffic.

    The previous champion is held out at probability `1 - new_prob`. The
    monitor rolls back at the first observation where the e-process of
    `H_0: Delta_new_vs_old >= -tolerance` reaches `1 / alpha_rollback`.

    Attributes:
        status: `hold` or `rollback`.
        units: Units observed so far.
        done_at: The unit at which the monitor rolled back, or 0.
    """

    def __init__(
        self,
        tolerance: float,
        alpha_rollback: float,
        new_prob: float = 0.5,
        low: float = 0.0,
        high: float = 1.0,
    ) -> None:
        """Starts a monitor for one promoted champion.

        Args:
            tolerance: Largest drop of the new champion against the old one
                that is accepted, in metric units.
            alpha_rollback: Level of the monitor: the e-process triggers a
                rollback once it reaches `1 / alpha_rollback`, and a champion
                no worse than `tolerance` is rolled back with probability at
                most this value.
            new_prob: Probability that a unit is served the new champion.
            low: Smallest value of the metric.
            high: Largest value of the metric.
        """
        scale = high - low
        self.low, self.high = low, high
        self.new_prob = new_prob
        self.target = -math.log(alpha_rollback)
        self.stream = anytime.centered_stream(
            new_prob, -tolerance / scale, "less"
        )
        self.carry = (0.0, 0)
        self.process = anytime.EProcess(self.stream)
        self.alpha = alpha_rollback
        self.units = 0
        self.status = HOLD
        self.done_at = 0

    def observe(self, arm: ArrayLike, outcome: ArrayLike) -> str:
        """Adds a batch (arm 1 is the new champion); returns the status.

        Args:
            arm: 1 for the new champion, 0 for the previous one, per unit.
            outcome: Raw metric value per unit, aligned with `arm`.

        Returns:
            `hold`, or `rollback` once the e-process has reached
            `1 / alpha_rollback`; a rolled-back monitor ignores further
            batches.
        """
        if self.status != HOLD:
            return self.status
        y = (numpy.asarray(outcome, dtype=float) - self.low) / (
            self.high - self.low
        )
        z = anytime.pseudo_outcomes(
            numpy.asarray(arm, dtype=float),
            y,
            self.new_prob,
            anytime.running_center(y, carry=self.carry),
        )
        self.carry = (self.carry[0] + float(y.sum()), self.carry[1] + len(y))
        self.process.update(z)
        hit = self.process.path >= self.target
        if hit.any():
            self.status = ROLLBACK
            self.done_at = self.units + int(numpy.argmax(hit)) + 1
        self.units += len(z)
        return self.status


class DecisionLog:
    """Hash-chained rows of the rollout decisions.

    The pipeline writes it as `rollout_decisions.jsonl`. Each row carries the
    SHA-256 of the previous row, so a replay reproduces the chain bit for bit
    and `verify` fails if a row is edited, reordered or removed from before
    the last row. Rows dropped from the end of the chain are not detected.

    Attributes:
        rows: The records in order, each with `seq`, `kind`, `body`,
            `previous` and `hash`.
    """

    def __init__(self) -> None:
        """Starts an empty chain."""
        self.rows: list[dict[str, Any]] = []

    def append(self, kind: str, body: Mapping[str, Any]) -> dict[str, Any]:
        """Adds a row linked to the previous one.

        Args:
            kind: Label of the record.
            body: The record's content; must be JSON-serialisable.

        Returns:
            The new row, including its sequence number, the previous hash
            and its own hash.
        """
        previous = self.rows[-1]["hash"] if self.rows else "0" * 64
        record = {
            "seq": len(self.rows),
            "kind": kind,
            "body": dict(body),
            "previous": previous,
        }
        record["hash"] = row_hash(record)
        self.rows.append(record)
        return record

    def verify(self) -> bool:
        """Whether every row links to the one before and hashes correctly.

        Returns:
            True if sequence numbers, links and hashes are all consistent
            (also for an empty chain), otherwise False.
        """
        previous = "0" * 64
        for index, row in enumerate(self.rows):
            if row["seq"] != index or row["previous"] != previous:
                return False
            if row["hash"] != row_hash(row):
                return False
            previous = row["hash"]
        return True

    def write(self, path: str | Path) -> None:
        """Writes the rows as JSON lines.

        Args:
            path: File to write, replacing any existing content.
        """
        text = "".join(
            json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n"
            for r in self.rows
        )
        Path(path).write_text(text, encoding="utf-8", newline="\n")

    @classmethod
    def read(cls, path: str | Path) -> "DecisionLog":
        """Reads rows back without verifying the chain; see `verify`.

        Args:
            path: A file of JSON lines written by `write`.

        Returns:
            A log holding the rows as read, without checking them.
        """
        log = cls()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            log.rows.append(json.loads(line))
        return log


def row_hash(record: Mapping[str, Any]) -> str:
    """SHA-256 of a row without its own hash.

    Args:
        record: A row; its `hash` entry, if any, is left out.

    Returns:
        The hexadecimal digest of the canonical JSON of the rest of the row.
    """
    body = {k: v for k, v in record.items() if k != "hash"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
