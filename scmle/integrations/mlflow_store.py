"""MLflow tracking adapter (`tracking` extra).

Maps scmle's run / trial / lineage records onto MLflow runs: one parent run
per scmle run and one child run per trial, so the MLflow UI and registry can
be used without changing how runs are executed.
"""

import json
import pathlib
from typing import Any

from scmle.contracts import errors
from scmle.contracts import experiments
from scmle.experimentation import stores

RUN_TAG = "scmle.run_id"
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
        self, tracking_uri: str | None = None, experiment: str = "scmle"
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
            from mlflow import tracking
        except ImportError as error:
            raise errors.ConfigError(
                "MlflowExperimentStore needs the 'tracking' extra (mlflow)"
            ) from error
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.client = tracking.MlflowClient(tracking_uri)
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
        """Finds the MLflow run of a scmle run, across processes.

        The answer is cached after the first lookup; otherwise MLflow is
        searched for a run tagged with the scmle run id and kind `run`.

        Args:
            run_id: Identifier of the scmle run.

        Returns:
            The MLflow run id of the parent run, or None if no such run exists.
        """
        if run_id in self.parents:
            return self.parents[run_id]
        found = self.client.search_runs(
            [self.experiment_id],
            f"tags.`{RUN_TAG}` = '{run_id}' and tags.`scmle.kind` = 'run'",
        )
        if found:
            self.parents[run_id] = found[0].info.run_id
            return self.parents[run_id]
        return None

    def start_run(
        self, run_id: str, config_hash: str, run_dir: str, started_at: str
    ) -> None:
        """Creates the parent run, or reuses it when resuming.

        The parent run's `scmle.state` tag is set to `CREATED` in either case.

        Args:
            run_id: Identifier of the scmle run.
            config_hash: Hash of the run configuration, stored as a tag.
            run_dir: Run directory, stored as a tag.
            started_at: Start timestamp, stored as a tag.
        """
        parent = self.find_parent(run_id)
        if parent is None:
            created = self.client.create_run(
                self.experiment_id,
                run_name=run_id,
                tags={
                    RUN_TAG: run_id,
                    "scmle.kind": "run",
                    "scmle.config_hash": config_hash,
                    "scmle.run_dir": run_dir,
                    "scmle.started_at": started_at,
                },
            )
            parent = created.info.run_id
            self.parents[run_id] = parent
        self.client.set_tag(parent, "scmle.state", "CREATED")

    def update_run(self, run_id: str, **fields: Any) -> None:
        """Maps fields to tags, metrics and run status.

        `cv_score` and each `test_metrics` entry (non-NaN) become metrics; the
        fields in `LINEAGE_ARTIFACTS` are logged as JSON text artifacts; any
        other field becomes a `scmle.<name>` tag. A `state` field in `TERMINAL`
        also ends the MLflow run with the mapped status. Nothing happens if the
        run is unknown.

        Args:
            run_id: Identifier of the scmle run.
            **fields: Run fields to record. `test_metrics` may be a dict or a
                JSON string.
        """
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
                self.client.set_tag(parent, f"scmle.{name}", text)
        state = fields.get("state")
        if state in TERMINAL:
            self.client.set_terminated(parent, TERMINAL[state])

    def record_trial(self, run_id: str, trial: experiments.Trial) -> None:
        """Logs a trial as a child run of the parent run.

        Does nothing if the parent run is unknown. The child run is ended as
        `FINISHED` when the trial status is `ok` or `pruned`, otherwise
        `FAILED`.

        Args:
            run_id: Identifier of the scmle run the trial belongs to.
            trial: The trial result to log.
        """
        parent = self.find_parent(run_id)
        if parent is None:
            return
        child = self.client.create_run(
            self.experiment_id,
            run_name=trial.trial_id,
            tags={
                PARENT_TAG: parent,
                RUN_TAG: run_id,
                "scmle.kind": "trial",
                "scmle.trial_id": trial.trial_id,
                "scmle.model": trial.model_name,
                "scmle.strategy": trial.strategy,
                "scmle.status": str(trial.status),
                "scmle.error": trial.error[:250],
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
        """Lists scmle parent runs, newest first.

        Returns:
            One dict per run with `run_id`, `state`, `mlflow_run_id` and the
            run's logged metrics.
        """
        found = self.client.search_runs(
            [self.experiment_id],
            "tags.`scmle.kind` = 'run'",
            order_by=["attributes.start_time DESC"],
        )
        return [
            {
                "run_id": r.data.tags[RUN_TAG],
                "state": r.data.tags.get("scmle.state", ""),
                "mlflow_run_id": r.info.run_id,
                **r.data.metrics,
            }
            for r in found
        ]

    def trials(self, run_id: str) -> list[dict[str, Any]]:
        """Lists the trials of a run in trial-id order.

        Args:
            run_id: Identifier of the scmle run.

        Returns:
            One dict per trial with `trial_id`, `model_name`, `strategy`,
            `status` and the trial's logged metrics, sorted by `trial_id`.
        """
        found = self.client.search_runs(
            [self.experiment_id],
            f"tags.`{RUN_TAG}` = '{run_id}' and tags.`scmle.kind` = 'trial'",
        )
        rows = [
            {
                "trial_id": r.data.tags["scmle.trial_id"],
                "model_name": r.data.tags["scmle.model"],
                "strategy": r.data.tags["scmle.strategy"],
                "status": r.data.tags["scmle.status"],
                **r.data.metrics,
            }
            for r in found
        ]
        return sorted(rows, key=lambda row: row["trial_id"])

    def history(self) -> list[experiments.HistoricalRun]:
        """Returns completed, fingerprinted runs for meta-learning.

        Returns:
            One record per run whose state is `COMPLETED` and which carries both
            a dataset fingerprint and a selected model. A missing score is NaN.
        """
        history = []
        for run in self.client.search_runs(
            [self.experiment_id],
            "tags.`scmle.kind` = 'run' and tags.`scmle.state` = 'COMPLETED'",
        ):
            tags = run.data.tags
            if (
                "scmle.fingerprint" not in tags
                or "scmle.selected_model" not in tags
            ):
                continue
            history.append(
                experiments.HistoricalRun(
                    run_id=tags[RUN_TAG],
                    dataset_id=tags.get("scmle.dataset_id", ""),
                    data_hash=tags.get("scmle.data_hash", ""),
                    task=tags.get("scmle.task", ""),
                    metric=tags.get("scmle.metric", ""),
                    fingerprint=json.loads(tags["scmle.fingerprint"]),
                    selected_model=tags["scmle.selected_model"],
                    selected_strategy=tags.get(
                        "scmle.selected_strategy", "default"
                    ),
                    cv_score=run.data.metrics.get("cv_score", float("nan")),
                )
            )
        return history
