"""Split records and the partitioner interface."""

import abc
import dataclasses
import hashlib
import math
from typing import Any
from typing import ClassVar

import numpy
import pandas

from scmle.contracts import errors
from scmle.contracts import records
from scmle.contracts import tasks
from scmle.data.splitting.splitters.lockbox import Lockbox

MIN_STRATUM_ROWS = 2


# One-sided error rate at which a class's test rows must carry evidence.
EVIDENCE_ALPHA = 0.05
# Fewest test rows of each class a classification split may hold. With all
# `m` test rows of a class predicted right, the one-sided exact upper bound
# at `EVIDENCE_ALPHA` of the error rate on that class is
# `1 - EVIDENCE_ALPHA ** (1 / m)`; it falls below one half, a coin, only
# from `m >= ln(EVIDENCE_ALPHA) / ln(1 / 2)` on, so 5. With fewer, even a
# perfect score on the class cannot be told from guessing; with none, ROC
# AUC is undefined and F1 scores a single class.
MIN_TEST_CLASS_ROWS = math.ceil(math.log(EVIDENCE_ALPHA) / math.log(0.5))


MIN_FOLDS = 2


Fold = tuple[numpy.ndarray, numpy.ndarray]


MIN_ROLLING_TRAIN = 10


# Strategies that draw rows as exchangeable; the splitter counts the test
# rows that repeat a development record (`Splitter.disclose`).
DUPLICATE_AWARE = frozenset({"random", "stratified", "stratified_events"})


MIN_ROLLING_FOLDS = 2


@dataclasses.dataclass(frozen=True, slots=True)
class Manifest:
    """Immutable record of how rows were partitioned.

    Attributes:
        strategy: Name of the strategy that produced the split.
        seed: Random seed used.
        training: Row ids of the development partition.
        test: Row ids of the locked test partition.
        folds: Cross-validation folds as positions within the development
            partition.
        notes: Adjustments made while splitting (for example fewer folds).
        entities: For a temporal or rolling split, the declared
            `validation.entities` (`known` or `unseen`); None otherwise.
    """

    strategy: str
    seed: int
    training: tuple[int, ...]
    test: tuple[int, ...]
    folds: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    notes: tuple[str, ...] = ()
    entities: str | None = None

    KEYS: ClassVar[records.Migration] = records.Migration(
        renames={"train_ids": "training", "test_ids": "test"}
    )
    """Version-1 keys to the written ones; `downgrade` restores version 1."""

    @property
    def fingerprint(self) -> str:
        """The first 16 hex digits of a SHA-256 hash of the entire split.

        The hash covers the strategy name, seed, train and test ids and folds,
        so equal splits have equal fingerprints.
        """
        payload = repr(
            (
                self.strategy,
                self.seed,
                self.training,
                self.test,
                self.folds,
            )
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Converts the manifest to a JSON-serializable mapping.

        Returns:
            A dict with the strategy, seed, fingerprint, row counts, the train
            and test ids as lists under `training` and `test`, the folds as
            `{"train": ..., "val": ...}` dicts of positions, the notes as a
            list and, for a temporal or rolling split, the declared
            `entities`.
        """
        declared = {} if self.entities is None else {"entities": self.entities}
        return self.KEYS.upgrade(
            declared
            | {
                "strategy": self.strategy,
                "seed": self.seed,
                "fingerprint": self.fingerprint,
                "n_train": len(self.training),
                "n_test": len(self.test),
                "train_ids": list(self.training),
                "test_ids": list(self.test),
                "folds": [
                    {"train": list(train), "val": list(val)}
                    for train, val in self.folds
                ],
                "notes": list(self.notes),
            }
        )


@dataclasses.dataclass(slots=True)
class Partition:
    """Raw output of a split strategy.

    Attributes:
        development: Row ids available for model development.
        test: Row ids reserved for the final test.
        folds: Cross-validation folds over positions in `development`.
        notes: Adjustments made while splitting.
    """

    development: numpy.ndarray
    test: numpy.ndarray
    folds: list[Fold]
    notes: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class Split:
    """Everything produced by splitting a dataset.

    Attributes:
        manifest: Immutable split record.
        features: Development features (target removed).
        target: Development labels.
        test: The sealed test partition.
    """

    manifest: Manifest
    features: pandas.DataFrame
    target: pandas.Series
    test: Lockbox


def tuplify(values: numpy.ndarray) -> tuple[int, ...]:
    """Converts an index array to a tuple of Python ints.

    Args:
        values: Row ids or positions.

    Returns:
        The same values, in order, as plain Python ints.
    """
    return tuple(int(v) for v in values)


class Partitioner(abc.ABC):
    """One validation protocol (holdout + cross-validation folds).

    Attributes:
        name: Strategy name used in configuration.
        scope: Task types the strategy can serve.
        data: The data section of the project specification.
        validation: The validation section of the project specification.
        task: The resolved task type.
    """

    name: ClassVar[str]
    scope: ClassVar[frozenset[tasks.Task]] = frozenset(tasks.Task)

    def __init__(self, spec: tasks.Config, task: tasks.Task) -> None:
        """Initializes the strategy.

        The task is checked by `check`, so a `BlockedError` from it
        propagates when the strategy cannot serve the task.

        Args:
            spec: Project specification.
            task: Resolved task type.
        """
        self.data = spec.data
        self.validation = spec.validation
        self.task = task
        self.check()

    def check(self) -> None:
        """Rejects tasks outside `scope`.

        Raises:
            BlockedError: If the strategy cannot serve the task.
        """
        if self.task not in self.scope:
            raise errors.BlockedError(
                f"{self.name} validation does not support {self.task.value}"
            )

    @abc.abstractmethod
    def partition(self, frame: pandas.DataFrame) -> Partition:
        """Splits the frame into development, test and folds.

        Args:
            frame: The full dataset including the target column.

        Returns:
            The development ids, test ids, folds and notes.

        Raises:
            BlockedError: If a leakage-free split is impossible.
        """
