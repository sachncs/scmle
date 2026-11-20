"""Stage records: what a run directory shows of the stages that finished.

After each stage, once the stage's own files are written and its approval
gates have passed, the engine writes the stage's product (`products`) to
`stages/NN-name.product` when it is kept, then `stages/NN-name.json`
through the atomic writer: the record is written last, so a process killed
at any point leaves either no record or a whole one. Each record carries its
own digest, the configuration hash, the stage's seconds, the run's elapsed
seconds when it finished, the digest of its product (`product`) and the
digests of the products of every stage it needs, directly or not
(`needs`). After the run reaches COMPLETED the marker `complete.json` is
written last of all; a run without it is not complete, whatever other files
it holds (CON-3).

Resume reads the records back (`Directory.open`): an edited record, or one
written under another configuration or format, is refused, and the largest
elapsed time is carried into the new attempt, so the time budget is not
restarted (CON-4). Before the records are read the custody journal is asked
(`Journal.admit`): a run whose test set was opened is refused before any
stage runs again, so the exam is never reopened. A stage whose record names
a kept product that loads, and the same digests of what it needs as this
attempt made, is not run again: its product is applied (`restore`). Any
other stage runs again and its record is replaced.
"""

import enum
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
from scmle.pipeline import products

SCHEMA_VERSION = 1
FOLDER = "stages"
SUFFIX = ".product"
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
        finished: Verified records (`keep`), by file name; a stage restores
            its product only from one of these.
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
        self.finished: dict[str, records.Layout] = {}

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
        opened = cls(directory, config, spent)
        opened.keep(found)
        return opened

    def keep(self, found: list[records.Layout]) -> None:
        """Remembers verified records, from which stages may be restored.

        Args:
            found: Records `read` verified.
        """
        self.finished = {
            self.path(int(r["index"]), str(r["stage"])).name: r for r in found
        }

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

    def path(
        self, index: int, name: str, suffix: str = ".json"
    ) -> pathlib.Path:
        """The record file of one stage, or with `SUFFIX` its product.

        Args:
            index: The stage's position in the run, from zero.
            name: The stage's name.
            suffix: `.json` for the record, `SUFFIX` for the product.

        Returns:
            `stages/NN-name.json` (or `.product`) in the run directory.
        """
        return self.directory / FOLDER / f"{index:02d}-{name.lower()}{suffix}"

    def elapsed(self) -> float:
        """Seconds the run has spent, earlier attempts included."""
        return time.monotonic() - self.started

    def record(
        self,
        index: int,
        name: str,
        progress: enum.StrEnum,
        seconds: float,
        made: products.Product | None = None,
        needs: dict[str, str | None] | None = None,
    ) -> records.Layout:
        """Writes one stage's product, then its record, atomically.

        Args:
            index: The stage's position in the run, from zero.
            name: The stage's name.
            progress: The lifecycle state the stage ran in.
            seconds: Seconds the stage took.
            made: What the stage made, or None for a stage that declares
                nothing.
            needs: The product digests of the stages it needs, by name.

        Returns:
            The record written.
        """
        product = None
        if made is not None:
            product = made.save(self.path(index, name, SUFFIX))
        body: records.Layout = {
            records.KEY: SCHEMA_VERSION,
            "stage": name,
            "index": index,
            "state": progress.value,
            "config": self.config,
            "seconds": seconds,
            "elapsed": self.elapsed(),
            "time": events.stamp(),
            "product": product,
            "needs": dict(needs or {}),
        }
        body["digest"] = digest(body)
        text = json.dumps(body, indent=2, sort_keys=True)
        durability.write(self.path(index, name), text.encode())
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

    def restore(
        self,
        index: int,
        name: str,
        makes: type[products.Product] | None,
        needs: dict[str, str | None],
    ) -> tuple[products.Product, str] | None:
        """Loads a finished stage's product, if the stage need not run again.

        Args:
            index: The stage's position in the run, from zero.
            name: The stage's name.
            makes: The stage's product type, or None.
            needs: The product digests this attempt has for the stages it
                needs, by name.

        Returns:
            The product and its digest, or None when the stage must run: it
            has no record verified when the run was opened, or no product
            type, a product it needs could not be digested or differs from
            the recorded one, or its own product is derived, missing,
            unauthenticated or not the recorded one.
        """
        body = self.finished.get(self.path(index, name).name)
        if makes is None or body is None:
            return None
        product = body.get("product")
        if (
            body.get("stage") != name
            or body.get("needs") != needs
            or not isinstance(product, str)
            or not all(isinstance(value, str) for value in needs.values())
        ):
            return None
        made = makes.load(self.path(index, name, SUFFIX), product)
        return None if made is None else (made, product)

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
        durability.write(self.directory / MARKER, text.encode())

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
            stored = json.loads(path.read_text(encoding="utf-8"))
            trial = experiments.Trial.load(stored)
            found[trial.identifier] = trial
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
