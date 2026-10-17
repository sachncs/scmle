"""Lifecycle automation: monitor, retrain a challenger, compare, promote."""

import dataclasses
import hashlib
import json
import pathlib
from typing import Any

import pandas as pd

import mml
from mml.api import results
from mml.contracts import errors
from mml.governance import registry as registry_module
from mml.monitoring import champion
from mml.monitoring import monitor as monitor_module

HOLDOUT_FRACTION = 0.3
STATE_SUFFIX = ".lifecycle.json"


@dataclasses.dataclass(frozen=True)
class RetrainPolicy:
    """When a drifting model may be retrained, so checks do not pile up.

    Attributes:
        min_rows: Labeled rows needed to train a challenger and compare it.
        cooldown_rows: After an attempt, fresh rows that must arrive before
            the next one (a rejected challenger is not retried on every
            check).
    """

    min_rows: int = 200
    cooldown_rows: int = 500


def fingerprint(frame: pd.DataFrame) -> str:
    """Short digest of the rows, to recognise data already acted on."""
    hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return hashlib.sha256(hashes.tobytes()).hexdigest()[:16]


@dataclasses.dataclass(frozen=True)
class LifecycleOutcome:
    """What one lifecycle review concluded and did.

    Attributes:
        monitoring: Monitoring report for the current champion.
        retrained: Whether a challenger was trained.
        comparison: Champion-challenger comparison, if a challenger exists.
        promoted_version: New champion version, if promotion happened.
        notes: Human-readable explanation.
        skipped: Why a recommended retrain did not run (`already_handled`,
            `cooldown` or `too_few_rows`), if it did not.
    """

    monitoring: monitor_module.MonitoringReport
    retrained: bool
    comparison: champion.Comparison | None
    promoted_version: int | None
    notes: list[str]
    skipped: str | None = None


class LifecycleManager:
    """Keeps a registered model fresh and safe to serve."""

    def __init__(
        self,
        registry: registry_module.ModelRegistryStore,
        model: str,
        work_dir: str,
        policy: RetrainPolicy | None = None,
    ) -> None:
        """Initializes the manager.

        Args:
            registry: Model registry.
            model: Registered model name.
            work_dir: Directory for challenger training runs.
            policy: When a recommended retrain may actually run.
        """
        self.registry = registry
        self.model = model
        self.work_dir = work_dir
        self.policy = policy or RetrainPolicy()

    @property
    def state_path(self) -> pathlib.Path:
        """Where the manager remembers its last attempt."""
        return self.registry.root / f"{self.model}{STATE_SUFFIX}"

    def read_state(self) -> dict[str, Any]:
        """The last attempt's data digest and the rows seen since."""
        if not self.state_path.exists():
            return {"last_digest": None, "rows_since_attempt": 0}
        return json.loads(self.state_path.read_text())

    def write_state(self, state: dict[str, Any]) -> None:
        """Saves the state atomically."""
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state))
        temporary.replace(self.state_path)

    def skip_reason(
        self, state: dict[str, Any], fresh: pd.DataFrame
    ) -> str | None:
        """Why a recommended retrain should not run now, if any reason holds."""
        if fingerprint(fresh) == state["last_digest"]:
            return "already_handled"
        if len(fresh) < self.policy.min_rows:
            return "too_few_rows"
        attempted = state["last_digest"] is not None
        pending = state["rows_since_attempt"] + len(fresh)
        if attempted and pending < self.policy.cooldown_rows:
            return "cooldown"
        return None

    def register_run(
        self, result: results.RunResult, promote: bool = True
    ) -> registry_module.VersionRecord:
        """Registers a finished run and optionally makes it the champion.

        Args:
            result: A completed run.
            promote: Point the champion alias at it.

        Returns:
            The registered version.
        """
        bundle_dir = result.save(result.run_dir / "export")
        record = self.registry.register(
            self.model,
            bundle_dir,
            metrics=result.evaluation["test_metrics"],
            config=result.plan.to_dict() | {"project": read_run_config(result)},
            run_id=result.run_id,
            trusted=True,  # exported moments ago by this very process
        )
        if promote:
            self.registry.promote(self.model, record.version)
        return record

    def review(
        self,
        fresh: pd.DataFrame,
        target: str,
        retrain: bool = True,
        project_options: dict[str, Any] | None = None,
    ) -> LifecycleOutcome:
        """Monitors the champion on fresh labeled data and acts on advice.

        If monitoring recommends retraining, a challenger is trained on part
        of the fresh data and compared with the champion on the held-out
        remainder; it is promoted only when it is clearly better.

        Args:
            fresh: Recent served rows including the true target column.
            target: Name of the target column.
            retrain: Allow training and promotion.
            project_options: Extra `MLProject` options for the challenger.

        Returns:
            The outcome.
        """
        current = self.registry.load(self.model)
        features = fresh.drop(columns=[target])
        report = monitor_module.ModelMonitor(current).assess(
            features, fresh[target]
        )
        notes = list(report.recommendation.reasons)
        state = self.read_state()
        self.registry.audit(
            "lifecycle.review",
            model=self.model,
            advice=report.recommendation.action,
            reasons=notes,
            rows=len(fresh),
        )
        if report.recommendation.action != "retrain" or not retrain:
            self.note_rows(state, fresh)
            return LifecycleOutcome(
                report, False, None, None, notes or ["healthy"]
            )
        skip = self.skip_reason(state, fresh)
        if skip is not None:
            self.registry.audit(
                "lifecycle.retrain_skipped",
                model=self.model,
                reason=skip,
                rows=len(fresh),
            )
            self.note_rows(state, fresh)
            notes.append(f"retraining skipped: {skip}")
            return LifecycleOutcome(
                report, False, None, None, notes, skipped=skip
            )
        digest = fingerprint(fresh)
        self.registry.audit(
            "lifecycle.retrain_started",
            model=self.model,
            rows=len(fresh),
            data_digest=digest,
            reasons=notes,
        )
        self.write_state({"last_digest": digest, "rows_since_attempt": 0})
        shuffled = fresh.sample(frac=1.0, random_state=0).reset_index(drop=True)
        cut = int(len(shuffled) * (1 - HOLDOUT_FRACTION))
        train, holdout = shuffled.iloc[:cut], shuffled.iloc[cut:]
        options = dict(project_options or {})
        options.setdefault("task", current.schema.task)
        options.setdefault("metric", current.info.metric)
        challenger_run = mml.MLProject(
            train, target, output_dir=self.work_dir, **options
        ).run()
        comparison = champion.ChampionChallenger(
            current, challenger_run.bundle
        ).compare(holdout.drop(columns=[target]), holdout[target])
        evidence = dataclasses.asdict(comparison)
        if comparison.decision != "promote":
            self.registry.audit(
                "lifecycle.challenger_rejected",
                model=self.model,
                champion_version=self.registry.resolve(self.model).version,
                **evidence,
            )
            notes.append("challenger was not clearly better; champion kept")
            return LifecycleOutcome(report, True, comparison, None, notes)
        record = self.register_run(challenger_run, promote=True)
        self.registry.audit(
            "lifecycle.challenger_promoted",
            model=self.model,
            version=record.version,
            **evidence,
        )
        notes.append(f"promoted challenger as v{record.version}")
        return LifecycleOutcome(report, True, comparison, record.version, notes)

    def note_rows(self, state: dict[str, Any], fresh: pd.DataFrame) -> None:
        """Counts fresh rows toward the cooldown after an attempt."""
        if (
            state["last_digest"] is None
            or fingerprint(fresh) == (state["last_digest"])
        ):
            return
        state["rows_since_attempt"] += len(fresh)
        self.write_state(state)


def read_run_config(result: results.RunResult) -> dict[str, Any]:
    """Reads the saved project configuration of a run."""
    path = result.run_dir / "config.json"
    if not path.exists():
        raise errors.ArtifactError("run has no config.json")
    return json.loads(path.read_text())
