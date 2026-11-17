"""Intake: routing served batches and feeding matured outcomes."""

from collections.abc import Mapping
from collections.abc import Sequence
import math
from typing import Any

import numpy
import pandas

from scmle.contracts import errors
from scmle.governance import deployment_ledger
from scmle.governance import escalation
from scmle.governance.loop import contest
from scmle.governance.loop import terms
from scmle.monitoring import labels


class Intake(contest.Contest):
    """Routing each served batch and feeding its outcomes once matured."""

    def observe(
        self,
        batch: pandas.DataFrame,
        predictions: Sequence[Any],
        decisions: Sequence[Any],
    ) -> terms.Observation:
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
            raise errors.InputError(
                "batch, predictions and decisions must align"
            )
        self.recheck()
        frame = batch.reset_index(drop=True)
        ids = (
            [str(v) for v in frame["case_id"]]
            if "case_id" in frame
            else [f"b{self.batch}-{i}" for i in range(size)]
        )
        novel, score, blocked, routing = self.route(frame, decisions)
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
        self.pend(frame, ids, score, routing, novel)
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
        return terms.Observation(self.batch - 1, routing, actions)

    def pend(
        self,
        frame: pandas.DataFrame,
        ids: list[str],
        score: numpy.ndarray,
        routing: escalation.Routing,
        novel: numpy.ndarray,
    ) -> None:
        """Keeps a routed batch pending until its outcomes mature.

        Args:
            frame: The batch, indexed from 0.
            ids: Case identifiers.
            score: The escalation score of each case.
            routing: The batch's routing.
            novel: Novelty flag of each case.
        """
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
        self.pending_batches.append(terms.jsonify(record))

    def route(
        self, frame: pandas.DataFrame, decisions: Sequence[Any]
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, escalation.Routing]:
        """Scores a batch's cases and routes them by `tau`.

        A case is blocked when it is novel or under-covered and its decision
        is high impact.

        Args:
            frame: The batch, indexed from 0.
            decisions: The action the model would take per case.

        Returns:
            The novelty flags, the escalation scores, the blocked flags and
            the routing.
        """
        novel = self.flag(frame, "novel")
        under = self.flag(frame, "under_covered")
        if "score" in frame:
            score = numpy.asarray(frame["score"], dtype=float)
        else:
            sizes = (
                frame["set_size"]
                if "set_size" in frame
                else numpy.ones(len(frame))
            )
            score = escalation.ambiguity(sizes, novel)
        high = numpy.isin(
            numpy.asarray(decisions, dtype=object),
            list(self.policy.high_impact_actions),
        )
        blocked = (novel | under) & high
        routing = escalation.route(
            score, self.tau, self.policy.c_max, self.halted, blocked
        )
        return novel, score, blocked, routing

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

    def feed(self, record: dict[str, Any]) -> bool:
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
        if crossed and not labels.actionable(estimate.mechanism):
            self.naive_flag = True
        if not labels.actionable(estimate.mechanism):
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
        self.challenge(record, ids, losses, observed, model_cases)
        return True
