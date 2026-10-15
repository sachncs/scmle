"""Lifecycle automation: monitor, retrain a challenger, compare, promote."""

import dataclasses
import json
from typing import Any

import pandas as pd

import mml
from mml.api import results
from mml.contracts import errors
from mml.deployment import registry as registry_module
from mml.monitoring import champion
from mml.monitoring import monitor as monitor_module

HOLDOUT_FRACTION = 0.3


@dataclasses.dataclass(frozen=True)
class LifecycleOutcome:
    """What one lifecycle review concluded and did.

    Attributes:
        monitoring: Monitoring report for the current champion.
        retrained: Whether a challenger was trained.
        comparison: Champion-challenger comparison, if a challenger exists.
        promoted_version: New champion version, if promotion happened.
        notes: Human-readable explanation.
    """

    monitoring: monitor_module.MonitoringReport
    retrained: bool
    comparison: champion.Comparison | None
    promoted_version: int | None
    notes: list[str]


class LifecycleManager:
    """Keeps a registered model fresh and safe to serve."""

    def __init__(
        self,
        registry: registry_module.ModelRegistryStore,
        model: str,
        work_dir: str,
    ) -> None:
        """Initializes the manager.

        Args:
            registry: Model registry.
            model: Registered model name.
            work_dir: Directory for challenger training runs.
        """
        self.registry = registry
        self.model = model
        self.work_dir = work_dir

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
        if report.recommendation.action != "retrain" or not retrain:
            return LifecycleOutcome(
                report, False, None, None, notes or ["healthy"]
            )
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
        if comparison.decision != "promote":
            notes.append("challenger was not clearly better; champion kept")
            return LifecycleOutcome(report, True, comparison, None, notes)
        record = self.register_run(challenger_run, promote=True)
        notes.append(f"promoted challenger as v{record.version}")
        return LifecycleOutcome(report, True, comparison, record.version, notes)


def read_run_config(result: results.RunResult) -> dict[str, Any]:
    """Reads the saved project configuration of a run."""
    path = result.run_dir / "config.json"
    if not path.exists():
        raise errors.ArtifactError("run has no config.json")
    return json.loads(path.read_text())
