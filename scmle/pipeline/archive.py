"""Stage records: what a run directory shows of the stages that finished.

After each stage, once the stage's own files are written and its approval
gates have passed, the engine writes `stages/NN-name.json` through the
atomic writer: the record is written last, so a process killed at any point
leaves either no record or a whole one. Each record carries its own digest,
the configuration hash, the stage's seconds and the run's elapsed seconds
when it finished. After the run reaches COMPLETED the marker `complete.json`
is written last of all; a run without it is not complete, whatever other
files it holds (CON-3).

Resume reads the records back (`Directory.open`): an edited record, or one
written under another configuration or format, is refused, and the largest
elapsed time is carried into the new attempt, so the time budget is not
restarted (CON-4). Before the records are read the custody journal is asked
(`Journal.admit`): a run whose test set was opened is refused before any
stage runs again, so the exam is never reopened.
"""

import hashlib
import json
import pathlib
import time
from typing import Self

from scmle import durability
from scmle.assurance.custody import journal
from scmle.contracts import errors
from scmle.contracts import events
from scmle.contracts import experiments
from scmle.contracts import records
from scmle.contracts import tasks
from scmle.runtime import state

SCHEMA_VERSION = 1
FOLDER = "stages"
MARKER = "complete.json"


def digest(body: records.Layout) -> str:
    """Hashes a record body as canonical JSON.

    Args:
        body: The record without its `digest`.

    Returns:
        The hex SHA-256 digest.
    """
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def saved(directory: pathlib.Path) -> tasks.Config:
    """Reads the configuration a run was created with.

    Args:
        directory: The run directory.

    Returns:
        The specification in its `config.json`.
    """
    text = (directory / "config.json").read_text(encoding="utf-8")
    return tasks.Config.model_validate_json(text)


class Directory:
    """The stage records, trial checkpoints and completion marker of a run.

    Attributes:
        directory: The run directory.
        config: The configuration hash every record carries.
        started: Monotonic time the run's elapsed seconds count from; earlier
            than now by the seconds a resumed run already spent.
    """

    def __init__(
        self, directory: pathlib.Path, config: str, spent: float = 0.0
    ) -> None:
        """Initializes the records of a run.

        Args:
            directory: The run directory.
            config: The configuration hash.
            spent: Seconds earlier attempts of the run already spent.
        """
        self.directory = directory
        self.config = config
        self.started = time.monotonic() - spent

    @classmethod
    def open(cls, directory: pathlib.Path, spec: tasks.Config) -> Self:
        """The records of an interrupted run, carrying the seconds it spent.

        Args:
            directory: The run directory.
            spec: The specification the resumed run uses.

        Returns:
            Records whose clock starts the elapsed seconds already spent.

        Raises:
            ConfigError: If the saved configuration differs from `spec`, or
                the run is complete.
            BlockedError: If the custody journal refuses the resume (it is
                missing or edited, or the test set was opened:
                `Journal.admit`), or a record was edited, or was written under
                another configuration or record format.
        """
        config = spec.config_hash()
        stored = saved(directory).config_hash()
        if stored != config:
            raise errors.ConfigError(
                "cannot resume: configuration differs from the saved run",
                {"saved": stored, "current": config},
            )
        current = cls(directory, config)
        if current.complete():
            raise errors.ConfigError("run already completed; nothing to resume")
        journal.Journal.locate(directory).admit()
        found = current.read()
        spent = max((float(r["elapsed"]) for r in found), default=0.0)
        return cls(directory, config, spent)

    def plan(self) -> experiments.Plan | None:
        """The run's saved execution plan, applied as the approved plan.

        Returns:
            The plan in `execution_plan.json`, or None when there is none.
        """
        path = self.directory / "execution_plan.json"
        if not path.exists():
            return None
        return experiments.Plan.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )

    def path(self, index: int, name: str) -> pathlib.Path:
        """The record file of one stage.

        Args:
            index: The stage's position in the run, from zero.
            name: The stage's name.

        Returns:
            `stages/NN-name.json` in the run directory.
        """
        return self.directory / FOLDER / f"{index:02d}-{name.lower()}.json"

    def elapsed(self) -> float:
        """Seconds the run has spent, earlier attempts included."""
        return time.monotonic() - self.started

    def record(
        self, index: int, name: str, progress: state.Progress, seconds: float
    ) -> records.Layout:
        """Writes one stage's record, atomically.

        Args:
            index: The stage's position in the run, from zero.
            name: The stage's name.
            progress: The lifecycle state the stage ran in.
            seconds: Seconds the stage took.

        Returns:
            The record written.
        """
        body: records.Layout = {
            records.KEY: SCHEMA_VERSION,
            "stage": name,
            "index": index,
            "state": progress.value,
            "config": self.config,
            "seconds": seconds,
            "elapsed": self.elapsed(),
            "time": events.stamp(),
        }
        body["digest"] = digest(body)
        text = json.dumps(body, indent=2, sort_keys=True)
        durability.write_atomically(self.path(index, name), text.encode())
        return body

    def read(self) -> list[records.Layout]:
        """Reads and verifies every stage record.

        Returns:
            The records in stage order; empty when no stage finished.

        Raises:
            BlockedError: If a record is unreadable, was edited, or was
                written under another configuration or record format.
        """
        found = []
        for path in sorted((self.directory / FOLDER).glob("*.json")):
            try:
                body = json.loads(path.read_text(encoding="utf-8"))
                stated = body.pop("digest")
            except (ValueError, KeyError, AttributeError) as error:
                raise errors.BlockedError(
                    f"stage record {path} is unreadable: {error}",
                    {"record": str(path)},
                ) from error
            if body.get(records.KEY) != SCHEMA_VERSION:
                raise errors.BlockedError(
                    f"stage record {path} has format "
                    f"{body.get(records.KEY)!r}, not {SCHEMA_VERSION}; "
                    "start a new run",
                    {"record": str(path)},
                )
            if digest(body) != stated or body.get("config") != self.config:
                raise errors.BlockedError(
                    f"stage record {path} was edited or belongs to another "
                    "configuration; the run cannot be resumed",
                    {"record": str(path)},
                )
            found.append(body)
        return found

    def finish(self) -> None:
        """Writes the completion marker, after everything else."""
        body: records.Layout = {
            records.KEY: SCHEMA_VERSION,
            "config": self.config,
            "elapsed": self.elapsed(),
            "time": events.stamp(),
        }
        body["digest"] = digest(body)
        text = json.dumps(body, indent=2, sort_keys=True)
        durability.write_atomically(self.directory / MARKER, text.encode())

    def complete(self) -> bool:
        """Whether the run wrote its completion marker."""
        return (self.directory / MARKER).is_file()

    def trials(self) -> dict[str, experiments.Trial]:
        """Reads the run's completed trial checkpoints.

        Returns:
            The stored trial results keyed by trial id, read from the run's
            `trials/trial_*.json` files in file-name order.
        """
        found: dict[str, experiments.Trial] = {}
        for path in sorted((self.directory / "trials").glob("trial_*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            record["status"] = experiments.Termination(record["status"])
            found[record["trial_id"]] = experiments.Trial(**record)
        return found

    def recoverable(self) -> list[str]:
        """Lists the trial checkpoints a failed or interrupted run can resume.

        Returns:
            The sorted names (without extension) of the trial checkpoint
            files in the run, empty if there is no trials directory.
        """
        folder = self.directory / "trials"
        if not folder.exists():
            return []
        return sorted(path.stem for path in folder.glob("trial_*.json"))
