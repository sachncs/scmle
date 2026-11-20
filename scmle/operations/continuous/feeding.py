"""Feeding the loop: labels and outcomes become losses, runs are recorded."""

from collections.abc import Callable
from collections.abc import Mapping
import json
import pathlib
from typing import Any

import pandas

from scmle import durability
from scmle.contracts import errors
from scmle.governance import deployment_ledger
import scmle.governance.loop
from scmle.operations.continuous import policies
from scmle.operations.continuous import records


class Feeding:
    """The evaluation's state, the feeding of the loop and the history.

    Attributes:
        loop: The closed loop of the deployment.
        directory: Where history, retained outcomes and datasets live.
        policy: The retraining policy.
        retrain: Hook that trains and stages a challenger.
        clock: Returns the current time in epoch seconds.
        history: The hash-chained training history.
        alarms: The loop alarms already acted on.
        proposed: The batch of the last retrain proposal, if any.
    """

    loop: scmle.governance.loop.Loop
    directory: pathlib.Path
    policy: policies.Policy
    retrain: Callable[[], str | None] | None
    clock: Callable[[], float]
    history: records.History
    alarms: int
    proposed: int | None

    def retain(self, kind: str, values: Mapping[str, Any]) -> None:
        """Keeps raw labels or outcomes with the batch they arrived at.

        Args:
            kind: The kind of values, for example ``business_outcomes``.
            values: The raw values by case id.
        """
        line = json.dumps(
            {
                "at": round(self.clock(), 3),
                "batch": self.loop.batch,
                "kind": kind,
                "values": {str(k): v for k, v in values.items()},
            },
            sort_keys=True,
            default=str,
        )
        durability.append(self.directory / records.LABELS_FILE, line + "\n")

    def absorb(
        self,
        predictions: Mapping[str, Any],
        labels: Mapping[str, Any],
        challengers: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> int:
        """Delayed labels become 0/1 losses for the loop.

        Args:
            predictions: The serving model's prediction by case id.
            labels: The true label by case id, as it arrives.
            challengers: For each challenger, its prediction by case id on
                the same cases.

        Returns:
            Cases that matched a pending batch.
        """
        losses = {
            str(case): 0.0 if predictions[case] == truth else 1.0
            for case, truth in labels.items()
            if case in predictions
        }
        paired: dict[str, dict[str, float]] = {}
        for name, theirs in (challengers or {}).items():
            paired[name] = {
                str(case): 0.0 if theirs[case] == truth else 1.0
                for case, truth in labels.items()
                if case in theirs
            }
        self.retain("labels", labels)
        return self.loop.mature(losses, challengers=paired or None)

    def feed(
        self,
        outcomes: Mapping[str, float],
        cap: float,
        challengers: Mapping[str, Mapping[str, float]] | None = None,
    ) -> int:
        """Business outcomes (a cost, higher is worse) become losses.

        The loss is ``min(outcome / cap, 1)``; ``cap`` is the outcome that is
        as bad as it gets for the purpose of the loop and is declared by the
        owner of the metric, not inferred.

        Args:
            outcomes: The outcome of each case, by case id.
            cap: The outcome that counts as the worst possible loss.
            challengers: The outcomes the challengers would have had, by
                challenger name and case id.

        Returns:
            The number of cases the loop matched to earlier batches.

        Raises:
            ValueError: If ``cap`` is not positive or an outcome is negative.
        """
        if cap <= 0 or any(v < 0 for v in outcomes.values()):
            raise errors.InputError(
                "cap must be positive and outcomes not negative"
            )
        losses = {str(k): min(v / cap, 1.0) for k, v in outcomes.items()}
        paired = {
            name: {str(k): min(v / cap, 1.0) for k, v in theirs.items()}
            for name, theirs in (challengers or {}).items()
        }
        self.retain("business_outcomes", outcomes)
        return self.loop.mature(losses, challengers=paired or None)

    def advance(self) -> scmle.governance.loop.Outcome:
        """Steps the loop and writes new proposals to the history.

        Returns:
            The loop's step result; new promotion proposals are also written to
            the history.
        """
        before = len(self.loop.ledger.rows)
        result = self.loop.step()
        for row in self.loop.ledger.rows[before:]:
            if row["kind"] == deployment_ledger.PROMOTION_PROPOSAL:
                self.history.record(
                    "promotion_proposal", self.clock(), **row["body"]
                )
        return result

    def record(
        self,
        name: str,
        frame: pandas.DataFrame,
        metrics: Mapping[str, float],
        reason: str,
        preserve: bool = True,
    ) -> dict[str, Any]:
        """Records a training run and keeps the exact dataset.

        Args:
            name: Model or challenger name.
            frame: The training data as used.
            metrics: Evaluation metrics of the run, recorded as given.
            reason: Why it was trained.
            preserve: Whether to write a copy of the dataset (as its
                canonical CSV, unless one with the same version already
                exists) under the datasets folder of the directory.

        Returns:
            The history row.
        """
        version = records.stamp(frame)
        if preserve:
            path = (
                self.directory
                / records.DATASET_DIRECTORY
                / (version.replace(":", "-") + ".csv")
            )
            if not path.exists():
                durability.write(path, records.serialize(frame))
        return self.history.record(
            "training",
            self.clock(),
            model=name,
            dataset_version=version,
            rows=len(frame),
            columns=sorted(frame.columns),
            metrics=dict(metrics),
            reason=reason,
            preserved=preserve,
        )

    def nominations(self) -> list[str]:
        """Challengers the loop's e-process proposed, oldest first.

        Returns:
            The names of the challengers the loop proposed for promotion, in
            order of proposal.
        """
        return [
            str(r["body"]["name"])
            for r in self.loop.ledger.rows
            if r["kind"] == deployment_ledger.PROMOTION_PROPOSAL
        ]
