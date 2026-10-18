"""MLflow tracking adapter (`tracking` extra).

Maps scml's run / trial / lineage records onto MLflow runs: one parent run
per scml run and one child run per trial, so the MLflow UI and registry can
be used without changing how runs are executed.
"""

import json
import pathlib
from typing import Any

from scml.contracts import errors
from scml.contracts import experiments
from scml.experimentation import stores

RUN_TAG = "scml.run_id"
LINEAGE_ARTIFACTS = frozenset({"config_json", "artifacts"})
PARENT_TAG = "mlflow.parentRunId"
TERMINAL = {
    "COMPLETED": "FINISHED",
    "FAILED": "FAILED",
    "CANCELLED": "KILLED",
    "BLOCKED": "FAILED",
}


class MlflowExperimentStore(stores.ExperimentStore):
    """Experiment store backed by an MLflow tracking server or database."""

    def __init__(
        self, tracking_uri: str | None = None, experiment: str = "scml"
    ) -> None:
        """Connects to MLflow.

        Args:
            tracking_uri: MLflow tracking URI (for example
                `sqlite:///mlflow.db` or an http server); None uses the
                library default.
            experiment: Experiment name (created if needed).

        Raises:
            ConfigError: If MLflow is not installed.
        """
        try:
            import mlflow
            from mlflow import tracking as mlflow_tracking
        except ImportError as error:
            raise errors.ConfigError(
                "MlflowExperimentStore needs the 'tracking' extra (mlflow)"
            ) from error
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow_tracking.MlflowClient(tracking_uri)
        existing = self.client.get_experiment_by_name(experiment)
        self.experiment_id = (
            existing.experiment_id
            if existing is not None
            else self.client.create_experiment(
                experiment,
                artifact_location=self.artifact_location(tracking_uri),
            )
        )
        self.parents: dict[str, str] = {}

    @staticmethod
    def artifact_location(tracking_uri: str | None) -> str | None:
        """Keeps artifacts beside a SQLite store instead of the working dir.

        Args:
            tracking_uri: The MLflow tracking URI.

        Returns:
            A directory URI next to the database file, or None to use the
            MLflow default (servers and file stores decide for themselves).
        """
        prefix = "sqlite:///"
        if not tracking_uri or not tracking_uri.startswith(prefix):
            return None
        database = pathlib.Path(tracking_uri[len(prefix) :])
        return (
            (database.parent / f"{database.stem}-artifacts").resolve().as_uri()
        )

    def find_parent(self, run_id: str) -> str | None:
        """Finds the MLflow run of a scml run, across processes."""
        if run_id in self.parents:
            return self.parents[run_id]
        found = self.client.search_runs(
            [self.experiment_id],
            f"tags.`{RUN_TAG}` = '{run_id}' and tags.`scml.kind` = 'run'",
        )
        if found:
            self.parents[run_id] = found[0].info.run_id
            return self.parents[run_id]
        return None

    def start_run(
        self, run_id: str, config_hash: str, run_dir: str, started_at: str
    ) -> None:
        """Creates the parent run, or reuses it when resuming."""
        parent = self.find_parent(run_id)
        if parent is None:
            created = self.client.create_run(
                self.experiment_id,
                run_name=run_id,
                tags={
                    RUN_TAG: run_id,
                    "scml.kind": "run",
                    "scml.config_hash": config_hash,
                    "scml.run_dir": run_dir,
                    "scml.started_at": started_at,
                },
            )
            parent = created.info.run_id
            self.parents[run_id] = parent
        self.client.set_tag(parent, "scml.state", "CREATED")

    def update_run(self, run_id: str, **fields: Any) -> None:
        """Maps fields to tags, metrics and run status."""
        parent = self.find_parent(run_id)
        if parent is None:
            return
        for name, value in fields.items():
            if name == "cv_score" and value is not None:
                self.client.log_metric(parent, "cv_score", float(value))
            elif name == "test_metrics":
                metrics = (
                    value if isinstance(value, dict) else json.loads(value)
                )
                for key, number in metrics.items():
                    if number == number:  # Skip NaN.
                        self.client.log_metric(
                            parent, f"test.{key}", float(number)
                        )
            elif name in LINEAGE_ARTIFACTS:
                text = (
                    json.dumps(value) if isinstance(value, dict) else str(value)
                )
                self.client.log_text(parent, text, f"{name}.json")
            else:
                text = (
                    json.dumps(value) if isinstance(value, dict) else str(value)
                )
                self.client.set_tag(parent, f"scml.{name}", text)
        state = fields.get("state")
        if state in TERMINAL:
            self.client.set_terminated(parent, TERMINAL[state])

    def record_trial(self, run_id: str, trial: experiments.TrialResult) -> None:
        """Logs a trial as a child run."""
        parent = self.find_parent(run_id)
        if parent is None:
            return
        child = self.client.create_run(
            self.experiment_id,
            run_name=trial.trial_id,
            tags={
                PARENT_TAG: parent,
                RUN_TAG: run_id,
                "scml.kind": "trial",
                "scml.trial_id": trial.trial_id,
                "scml.model": trial.model_name,
                "scml.strategy": trial.strategy,
                "scml.status": str(trial.status),
                "scml.error": trial.error[:250],
            },
        )
        child_id = child.info.run_id
        self.client.log_param(child_id, "model", trial.model_name)
        self.client.log_param(child_id, "strategy", trial.strategy)
        for key, value in trial.params.items():
            self.client.log_param(child_id, f"param.{key}", str(value)[:250])
        if trial.score_mean == trial.score_mean:
            self.client.log_metric(
                child_id, trial.metric or "score", trial.score_mean
            )
            self.client.log_metric(child_id, "score_std", trial.score_std)
        self.client.log_metric(child_id, "fit_seconds", trial.fit_seconds)
        self.client.set_terminated(
            child_id,
            "FINISHED" if str(trial.status) in ("ok", "pruned") else "FAILED",
        )

    def runs(self) -> list[dict[str, Any]]:
        """Lists scml parent runs, newest first."""
        found = self.client.search_runs(
            [self.experiment_id],
            "tags.`scml.kind` = 'run'",
            order_by=["attributes.start_time DESC"],
        )
        return [
            {
                "run_id": r.data.tags[RUN_TAG],
                "state": r.data.tags.get("scml.state", ""),
                "mlflow_run_id": r.info.run_id,
                **r.data.metrics,
            }
            for r in found
        ]

    def trials(self, run_id: str) -> list[dict[str, Any]]:
        """Lists the trials of a run in trial-id order."""
        found = self.client.search_runs(
            [self.experiment_id],
            f"tags.`{RUN_TAG}` = '{run_id}' and tags.`scml.kind` = 'trial'",
        )
        rows = [
            {
                "trial_id": r.data.tags["scml.trial_id"],
                "model_name": r.data.tags["scml.model"],
                "strategy": r.data.tags["scml.strategy"],
                "status": r.data.tags["scml.status"],
                **r.data.metrics,
            }
            for r in found
        ]
        return sorted(rows, key=lambda row: row["trial_id"])

    def history(self) -> list[experiments.HistoricalRun]:
        """Returns completed, fingerprinted runs for meta-learning."""
        history = []
        for run in self.client.search_runs(
            [self.experiment_id],
            "tags.`scml.kind` = 'run' and tags.`scml.state` = 'COMPLETED'",
        ):
            tags = run.data.tags
            if (
                "scml.fingerprint" not in tags
                or "scml.selected_model" not in tags
            ):
                continue
            history.append(
                experiments.HistoricalRun(
                    run_id=tags[RUN_TAG],
                    dataset_id=tags.get("scml.dataset_id", ""),
                    data_hash=tags.get("scml.data_hash", ""),
                    task=tags.get("scml.task", ""),
                    metric=tags.get("scml.metric", ""),
                    fingerprint=json.loads(tags["scml.fingerprint"]),
                    selected_model=tags["scml.selected_model"],
                    selected_strategy=tags.get(
                        "scml.selected_strategy", "default"
                    ),
                    cv_score=run.data.metrics.get("cv_score", float("nan")),
                )
            )
        return history
