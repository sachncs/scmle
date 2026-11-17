"""Experiment plan and trial result contracts."""

from collections.abc import Mapping
import dataclasses
import enum
import json
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.contracts import records

DEFAULT_STRATEGY = "default"


class Termination(enum.StrEnum):
    """Terminal status of a trial."""

    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"
    PRUNED = "pruned"


class Edict(enum.StrEnum):
    """What the planner did with a technique."""

    INCLUDE = "include"
    REJECT = "reject"
    DEFER = "defer"


@dataclasses.dataclass(frozen=True, slots=True)
class Decision:
    """One auditable planner decision.

    Attributes:
        technique: Model or technique the decision is about.
        action: Whether it was included, rejected or deferred.
        reason: Why.
    """

    technique: str
    action: Edict
    reason: str

    def to_dict(self) -> dict[str, str]:
        """Returns a JSON-serializable mapping."""
        return {
            "technique": self.technique,
            "action": str(self.action),
            "reason": self.reason,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Plan:
    """Declarative, auditable description of the proposed workflow.

    Attributes:
        identifier: Deterministic identifier.
        task: Resolved task type value.
        metric: Primary metric name.
        validation: Resolved split strategy name.
        candidates: Models that will be trained, in execution order.
        budget: Wall-clock optimization budget, in seconds.
        decisions: Included / rejected / deferred techniques with reasons.
        warnings: Risks surfaced during planning.
        migrations: Version steps of the stored plan.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "plan_id": "identifier",
                "validation_strategy": "validation",
                "candidate_models": "candidates",
                "optimization_budget_seconds": "budget",
            }
        ),
    )

    identifier: str
    task: str
    metric: str
    validation: str
    candidates: tuple[str, ...]
    budget: int
    decisions: tuple[Decision, ...] = ()
    warnings: tuple[str, ...] = ()

    def summary(self) -> str:
        """Returns a human-readable multi-line summary."""
        lines = [
            (
                f"Plan {self.identifier}: {self.task}, "
                f"metric={self.metric}, validation={self.validation}, "
                f"budget={self.budget}s"
            ),
            "Candidates: " + ", ".join(self.candidates),
        ]
        lines += [
            f"  [{d.action}] {d.technique}: {d.reason}" for d in self.decisions
        ]
        lines += [f"  WARNING: {warning}" for warning in self.warnings]
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Self:
        """Rebuilds a plan from `to_dict` output.

        A stored plan of an older schema version is migrated first. Every
        key written by `to_dict` is required; a missing key raises
        `KeyError`.

        Args:
            record: A mapping as produced by `to_dict`, at any supported
                schema version.

        Returns:
            The plan.
        """
        record = records.load(cls.__name__, cls.migrations, record)
        return cls(
            identifier=record["identifier"],
            task=record["task"],
            metric=record["metric"],
            validation=record["validation"],
            candidates=tuple(record["candidates"]),
            budget=record["budget"],
            decisions=tuple(
                Decision(d["technique"], Edict(d["action"]), d["reason"])
                for d in record["decisions"]
            ),
            warnings=tuple(record["warnings"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        return {
            "schema_version": records.ORIGIN + len(self.migrations),
            "identifier": self.identifier,
            "task": self.task,
            "metric": self.metric,
            "validation": self.validation,
            "candidates": list(self.candidates),
            "budget": self.budget,
            "decisions": [d.to_dict() for d in self.decisions],
            "warnings": list(self.warnings),
        }


@dataclasses.dataclass(slots=True)
class Trial:
    """Outcome of evaluating one complete candidate pipeline.

    Attributes:
        trial_id: Identifier within the run.
        model_name: Candidate model name.
        status: Terminal status.
        metric: Metric name the scores refer to.
        score_mean: Mean cross-validation score (natural units).
        score_std: Standard deviation across folds.
        fold_scores: Per-fold scores.
        fit_seconds: Wall-clock cost.
        error: Failure or skip reason, empty on success.
        failure: The failure's type, message, traceback and causes
            (`errors.Failure.dump`), empty unless the trial raised.
        params: Hyperparameter overrides (empty for default configuration).
        strategy: Data strategy the candidate was built with.
        observed_access: What each fold's fit and scoring actually received:
            per fold, one record for the fit rows and one for the evaluated
            rows, each holding the row count and a digest of the row ids.
            The ledger checks these observations against the split manifest.
        evaluation_config: How the candidate was evaluated (metric, number
            of folds, seed, whether weights were used), so a trial is
            reproducible from its own record.
        runtime: Runtime requirements of the candidate (Python version and
            the libraries its model needs).
        train_scores: Per-fold scores on the rows each model was fitted on,
            for the fit-health check (a bounded sample of those rows).
    """

    trial_id: str
    model_name: str
    status: Termination
    metric: str = ""
    score_mean: float = float("nan")
    score_std: float = float("nan")
    fold_scores: list[float] = dataclasses.field(default_factory=list)
    fit_seconds: float = 0.0
    error: str = ""
    failure: Mapping[str, object] = dataclasses.field(default_factory=dict)
    params: dict[str, Any] = dataclasses.field(default_factory=dict)
    strategy: str = DEFAULT_STRATEGY
    observed_access: list[dict[str, Any]] = dataclasses.field(
        default_factory=list
    )
    evaluation_config: dict[str, Any] = dataclasses.field(default_factory=dict)
    runtime: dict[str, Any] = dataclasses.field(default_factory=dict)
    train_scores: list[float] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping.

        `failure` is written as JSON text (empty when there is none), so a
        leaderboard of trials stays a flat table that Parquet can store.
        """
        record = dataclasses.asdict(self)
        record["status"] = str(self.status)
        record["failure"] = (
            json.dumps(dict(self.failure), sort_keys=True)
            if self.failure
            else ""
        )
        return record


@dataclasses.dataclass(frozen=True, slots=True)
class Precedent:
    """What is remembered about a past run, for meta-learning.

    Attributes:
        run_id: Run identifier.
        dataset_id: Dataset identity (content derived).
        data_hash: Hash of the dataset contents.
        task: Task type value.
        metric: Optimized metric name.
        fingerprint: Numeric dataset fingerprint.
        selected_model: Winning model name.
        selected_strategy: Winning data strategy.
        cv_score: Cross-validation score of the winner.
    """

    run_id: str
    dataset_id: str
    data_hash: str
    task: str
    metric: str
    fingerprint: dict[str, float]
    selected_model: str
    selected_strategy: str
    cv_score: float
