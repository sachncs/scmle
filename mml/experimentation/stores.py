"""Experiment stores.

`ExperimentStore` is the tracking contract; `SqliteExperimentStore` is the
zero-service local backend, and `NullExperimentStore` is the do-nothing
implementation that lets callers skip None checks. External trackers (for
example MLflow) implement the same interface.
"""

import abc
from collections.abc import Iterator
import contextlib
import json
import pathlib
import sqlite3
from typing import Any

from mml.contracts import events
from mml.contracts import experiments

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    config_hash TEXT NOT NULL,
    dataset_id TEXT,
    data_hash TEXT,
    split_fingerprint TEXT,
    state TEXT NOT NULL,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    selected_model TEXT,
    metric TEXT,
    cv_score REAL,
    test_metrics TEXT,
    run_dir TEXT,
    fingerprint TEXT,
    task TEXT,
    selected_strategy TEXT,
    config_json TEXT,
    artifacts TEXT
);
CREATE TABLE IF NOT EXISTS trials (
    run_id TEXT NOT NULL,
    trial_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    status TEXT NOT NULL,
    metric TEXT,
    score_mean REAL,
    score_std REAL,
    fit_seconds REAL,
    params TEXT,
    fold_scores TEXT,
    error TEXT,
    strategy TEXT,
    PRIMARY KEY (run_id, trial_id)
);
"""


class ExperimentStore(abc.ABC):
    """Persists runs, trials and lineage."""

    @abc.abstractmethod
    def start_run(
        self, run_id: str, config_hash: str, run_dir: str, started_at: str
    ) -> None:
        """Registers (or re-registers, on resume) a run."""

    @abc.abstractmethod
    def update_run(self, run_id: str, **fields: Any) -> None:
        """Updates run columns (state, dataset_id, selected_model, ...)."""

    @abc.abstractmethod
    def record_trial(self, run_id: str, trial: experiments.TrialResult) -> None:
        """Inserts or replaces a trial."""

    @abc.abstractmethod
    def runs(self) -> list[dict[str, Any]]:
        """Returns all runs, newest first."""

    @abc.abstractmethod
    def trials(self, run_id: str) -> list[dict[str, Any]]:
        """Returns a run's trials in insertion order."""

    @abc.abstractmethod
    def history(self) -> list[experiments.HistoricalRun]:
        """Returns completed runs that carry a dataset fingerprint."""

    def listen(self, run_id: str, event: events.Event) -> None:
        """Mirrors state-change events into the run's `state` column.

        Args:
            run_id: Run the event belongs to.
            event: An emitted event; non-state events are ignored.
        """
        if event.kind.startswith("state."):
            self.update_run(run_id, state=event.kind.removeprefix("state."))


class NullExperimentStore(ExperimentStore):
    """Tracking disabled: every operation is a no-op."""

    def start_run(
        self, run_id: str, config_hash: str, run_dir: str, started_at: str
    ) -> None:
        """Ignores the run."""

    def update_run(self, run_id: str, **fields: Any) -> None:
        """Ignores the update."""

    def record_trial(self, run_id: str, trial: experiments.TrialResult) -> None:
        """Ignores the trial."""

    def runs(self) -> list[dict[str, Any]]:
        """Returns nothing."""
        return []

    def trials(self, run_id: str) -> list[dict[str, Any]]:
        """Returns nothing."""
        return []

    def history(self) -> list[experiments.HistoricalRun]:
        """Returns nothing."""
        return []


class SqliteExperimentStore(ExperimentStore):
    """Local SQLite tracking database (one file, no services)."""

    COLUMNS = frozenset(
        {
            "dataset_id",
            "data_hash",
            "split_fingerprint",
            "state",
            "selected_model",
            "selected_strategy",
            "metric",
            "cv_score",
            "test_metrics",
            "fingerprint",
            "task",
            "config_json",
            "artifacts",
        }
    )

    def __init__(self, path: str | pathlib.Path) -> None:
        """Opens (creating if needed) the database.

        Args:
            path: Database file path; parent directories are created.
        """
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.session() as connection:
            connection.executescript(SCHEMA)
            self.migrate(connection)

    def migrate(self, connection: sqlite3.Connection) -> None:
        """Adds columns introduced after a database was first created."""
        existing = {
            row["name"] for row in connection.execute("PRAGMA table_info(runs)")
        }
        for column in (
            "fingerprint",
            "task",
            "selected_strategy",
            "config_json",
            "artifacts",
        ):
            if column not in existing:
                connection.execute(f"ALTER TABLE runs ADD COLUMN {column} TEXT")
        trial_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(trials)")
        }
        if "strategy" not in trial_columns:
            connection.execute("ALTER TABLE trials ADD COLUMN strategy TEXT")

    @contextlib.contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:
        """Opens a transaction and always closes the connection.

        `with sqlite3.connect(...)` only commits or rolls back; it never
        closes, which leaks a connection per call.
        """
        connection = self.connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def connect(self) -> sqlite3.Connection:
        """Opens a short-lived connection with row access by name."""
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def start_run(
        self, run_id: str, config_hash: str, run_dir: str, started_at: str
    ) -> None:
        """Inserts the run, or refreshes it when resuming."""
        with self.session() as connection:
            connection.execute(
                "INSERT INTO runs (run_id, config_hash, state, started_at, "
                "updated_at, run_dir) VALUES (?, ?, 'CREATED', ?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET state='CREATED', "
                "updated_at=excluded.updated_at",
                (run_id, config_hash, started_at, started_at, run_dir),
            )

    def update_run(self, run_id: str, **fields: Any) -> None:
        """Updates whitelisted columns; dict values are stored as JSON.

        Raises:
            ValueError: For columns outside the schema.
        """
        unknown = set(fields) - self.COLUMNS
        if unknown:
            raise ValueError(f"unknown run columns: {sorted(unknown)}")
        values = {
            k: json.dumps(v) if isinstance(v, dict) else v
            for k, v in fields.items()
        }
        assignments = ", ".join(f"{column} = ?" for column in values)
        with self.session() as connection:
            # Column names come from the COLUMNS whitelist checked above and
            # every value is bound as a parameter, so this is not injectable.
            connection.execute(
                f"UPDATE runs SET {assignments}, updated_at = "  # nosec B608
                "strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE run_id = ?",
                (*values.values(), run_id),
            )

    def record_trial(self, run_id: str, trial: experiments.TrialResult) -> None:
        """Upserts a trial."""
        with self.session() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO trials (run_id, trial_id, model_name, "
                "status, metric, score_mean, score_std, fit_seconds, params, "
                "fold_scores, error, strategy) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    trial.trial_id,
                    trial.model_name,
                    str(trial.status),
                    trial.metric,
                    trial.score_mean,
                    trial.score_std,
                    trial.fit_seconds,
                    json.dumps(trial.params, default=str),
                    json.dumps(trial.fold_scores),
                    trial.error,
                    trial.strategy,
                ),
            )

    def runs(self) -> list[dict[str, Any]]:
        """Returns runs newest first."""
        with self.session() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY started_at DESC, run_id DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def trials(self, run_id: str) -> list[dict[str, Any]]:
        """Returns a run's trials ordered by trial id."""
        with self.session() as connection:
            rows = connection.execute(
                "SELECT * FROM trials WHERE run_id = ? ORDER BY trial_id",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def history(self) -> list[experiments.HistoricalRun]:
        """Returns completed, fingerprinted runs, oldest first."""
        with self.session() as connection:
            rows = connection.execute(
                "SELECT * FROM runs WHERE state = 'COMPLETED' AND "
                "fingerprint IS NOT NULL AND selected_model IS NOT NULL "
                "ORDER BY started_at, run_id"
            ).fetchall()
        return [
            experiments.HistoricalRun(
                run_id=row["run_id"],
                dataset_id=row["dataset_id"] or "",
                data_hash=row["data_hash"] or "",
                task=row["task"] or "",
                metric=row["metric"] or "",
                fingerprint=json.loads(row["fingerprint"]),
                selected_model=row["selected_model"],
                selected_strategy=row["selected_strategy"] or "default",
                cv_score=row["cv_score"]
                if row["cv_score"] is not None
                else float("nan"),
            )
            for row in rows
        ]
