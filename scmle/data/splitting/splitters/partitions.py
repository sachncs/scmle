"""Split records, the sealed test set and the partitioner interface."""

import abc
import dataclasses
import hashlib
import secrets
from typing import Any
from typing import ClassVar

import numpy
import pandas

from scmle.contracts import errors
from scmle.contracts import tasks

MIN_STRATUM_ROWS = 2


MIN_FOLDS = 2


Fold = tuple[numpy.ndarray, numpy.ndarray]


MIN_ROLLING_TRAIN = 10


DUPLICATE_AWARE = frozenset({"random", "stratified", "stratified_events"})


MIN_ROLLING_FOLDS = 2


@dataclasses.dataclass(frozen=True, slots=True)
class Manifest:
    """Immutable record of how rows were partitioned.

    Attributes:
        strategy: Name of the strategy that produced the split.
        seed: Random seed used.
        train_ids: Row ids of the development partition.
        test_ids: Row ids of the locked test partition.
        folds: Cross-validation folds as positions within the development
            partition.
        notes: Adjustments made while splitting (for example fewer folds).
    """

    strategy: str
    seed: int
    train_ids: tuple[int, ...]
    test_ids: tuple[int, ...]
    folds: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    notes: tuple[str, ...] = ()

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
                self.train_ids,
                self.test_ids,
                self.folds,
            )
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Converts the manifest to a JSON-serializable mapping.

        Returns:
            A dict with the strategy, seed, fingerprint, row counts, the train
            and test ids as lists, the folds as `{"train": ..., "val": ...}`
            dicts of positions, and the notes as a list.
        """
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "fingerprint": self.fingerprint,
            "n_train": len(self.train_ids),
            "n_test": len(self.test_ids),
            "train_ids": list(self.train_ids),
            "test_ids": list(self.test_ids),
            "folds": [
                {"train": list(train), "val": list(val)}
                for train, val in self.folds
            ],
            "notes": list(self.notes),
        }


@dataclasses.dataclass(slots=True)
class Partition:
    """Raw output of a split strategy.

    Attributes:
        dev_ids: Row ids available for model development.
        test_ids: Row ids reserved for the final test.
        folds: Cross-validation folds over positions in `dev_ids`.
        notes: Adjustments made while splitting.
    """

    dev_ids: numpy.ndarray
    test_ids: numpy.ndarray
    folds: list[Fold]
    notes: list[str] = dataclasses.field(default_factory=list)


class Lockbox(abc.ABC):
    """The test partition: unavailable to search, released once.

    Attributes:
        ids: Row ids of the test rows.
        opened: Whether the rows were released.
    """

    ids: tuple[int, ...]
    opened: bool

    @abc.abstractmethod
    def mint(self) -> str:
        """Returns the release token; call only once selection is frozen.

        Returns:
            The token that `open` requires.
        """

    @abc.abstractmethod
    def bind(self, binding: str) -> None:
        """Fixes what the single release is for; call before `open`.

        After a binding is set, `open` releases the rows only to a caller
        that presents the same binding. A set can be bound once.

        Args:
            binding: Digest of the selection the release is meant to test.

        Raises:
            LeakageError: If a binding is already set or the rows were
                released.
        """

    @abc.abstractmethod
    def open(
        self, token: str, binding: str | None = None
    ) -> tuple[pandas.DataFrame, pandas.Series]:
        """Releases the test data a single time.

        Args:
            token: The token from `mint`.
            binding: The binding fixed by `bind`; required once one is set.

        Returns:
            Test features and labels.

        Raises:
            LeakageError: On a wrong token, a repeated release or a binding
                that differs from the one set.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Releases any resource held for the lock."""

    def __len__(self) -> int:
        """Returns the number of test rows.

        Returns:
            The length of `ids`.
        """
        return len(self.ids)


@dataclasses.dataclass(slots=True)
class ResidentTestSet(Lockbox):
    """Test rows kept in this process, behind a one-time token.

    Python cannot hide attributes, so the lock is a protocol: trial code only
    receives development data, and `open` enforces a one-time, token-gated
    release that is recorded. The sealed variant moves the rows to a separate
    process (see `scmle.assurance.custody.vault`).

    Attributes:
        features: Test features.
        target: Test labels.
        ids: Row ids of the test rows.
        token: Secret issued by `mint` (excluded from repr).
        opened: Whether the set has been released.
        binding: Digest the release is bound to, or None.
    """

    features: pandas.DataFrame = dataclasses.field(repr=False)
    target: pandas.Series = dataclasses.field(repr=False)
    ids: tuple[int, ...]
    token: str = dataclasses.field(
        default_factory=lambda: secrets.token_hex(8), repr=False
    )
    opened: bool = False
    binding: str | None = None

    def mint(self) -> str:
        """Returns the release token; call only once selection is frozen.

        Returns:
            The token that `open` requires.
        """
        return self.token

    def bind(self, binding: str) -> None:
        """Fixes what the single release is for.

        Args:
            binding: Digest of the selection the release is meant to test.

        Raises:
            LeakageError: If a binding is already set or the rows were
                released.
        """
        if self.binding is not None or self.opened:
            raise errors.LeakageError("test set is already bound or used")
        self.binding = binding

    def open(
        self, token: str, binding: str | None = None
    ) -> tuple[pandas.DataFrame, pandas.Series]:
        """Releases the test data a single time.

        Args:
            token: The token from `mint`.
            binding: The binding fixed by `bind`, which must come first.

        Returns:
            Test features and labels.

        Raises:
            LeakageError: On a wrong token, a repeated release, a set that was
                never bound, or a binding that differs from the one set. The
                token and binding are compared in constant time (GATE-3).
        """
        if not secrets.compare_digest(token.encode(), self.token.encode()):
            raise errors.LeakageError("test set is locked: invalid token")
        if self.opened:
            raise errors.LeakageError(
                "test set was already used for final evaluation"
            )
        if self.binding is None:
            raise errors.LeakageError(
                "test set is locked: bind it to the frozen selection first"
            )
        if binding is None or not secrets.compare_digest(
            binding.encode(), self.binding.encode()
        ):
            raise errors.LeakageError("test set is locked: binding mismatch")
        self.opened = True
        return self.features, self.target

    def close(self) -> None:
        """Nothing to release: the rows live in this process."""


@dataclasses.dataclass(slots=True)
class Split:
    """Everything produced by splitting a dataset.

    Attributes:
        manifest: Immutable split record.
        dev_features: Development features (target removed).
        dev_target: Development labels.
        locked_test: The sealed test partition.
    """

    manifest: Manifest
    dev_features: pandas.DataFrame
    dev_target: pandas.Series
    locked_test: Lockbox


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
        supported_tasks: Task types the strategy can serve.
        data: The data section of the project specification.
        validation: The validation section of the project specification.
        task: The resolved task type.
    """

    name: ClassVar[str]
    supported_tasks: ClassVar[frozenset[tasks.Task]] = frozenset(tasks.Task)

    def __init__(self, spec: tasks.Config, task: tasks.Task) -> None:
        """Initializes the strategy.

        The task is checked by `check_task`, so a `BlockedError` from it
        propagates when the strategy cannot serve the task.

        Args:
            spec: Project specification.
            task: Resolved task type.
        """
        self.data = spec.data
        self.validation = spec.validation
        self.task = task
        self.check_task()

    def check_task(self) -> None:
        """Rejects tasks outside `supported_tasks`.

        Raises:
            BlockedError: If the strategy cannot serve the task.
        """
        if self.task not in self.supported_tasks:
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
