"""Leakage-aware splitting: strategies, immutable manifest, sealed test set.

Row identifiers are positional indices into the ingested frame. Each
`SplitStrategy` subclass owns one validation protocol; `SplitManager` selects
and applies it. The test partition lives behind `LockedTestSet`, which opens
exactly once, with the token the orchestrator issues after selection.
"""

import abc
import dataclasses
import hashlib
import secrets
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn import model_selection

from mml.contracts import errors
from mml.contracts import tasks

Fold = tuple[np.ndarray, np.ndarray]


@dataclasses.dataclass(frozen=True, slots=True)
class SplitManifest:
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
        """Deterministic hash of the entire split."""
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

    def to_dict(self) -> dict:
        """Returns a JSON-serializable mapping."""
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

    dev_ids: np.ndarray
    test_ids: np.ndarray
    folds: list[Fold]
    notes: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class LockedTestSet:
    """Holds the test partition; unavailable to search, openable once.

    Python cannot hide attributes, so the lock is a protocol: trial code only
    receives development data, and `open` enforces a one-time, token-gated
    release that is recorded.

    Attributes:
        features: Test features.
        target: Test labels.
        ids: Row ids of the test rows.
        token: Secret issued by `issue_token` (excluded from repr).
        opened: Whether the set has been released.
    """

    features: pd.DataFrame = dataclasses.field(repr=False)
    target: pd.Series = dataclasses.field(repr=False)
    ids: tuple[int, ...]
    token: str = dataclasses.field(
        default_factory=lambda: secrets.token_hex(8), repr=False
    )
    opened: bool = False

    def issue_token(self) -> str:
        """Returns the release token; call only once selection is frozen."""
        return self.token

    def open(self, token: str) -> tuple[pd.DataFrame, pd.Series]:
        """Releases the test data a single time.

        Args:
            token: The token from `issue_token`.

        Returns:
            Test features and labels.

        Raises:
            LeakageError: On a wrong token or a repeated release.
        """
        if token != self.token:
            raise errors.LeakageError("test set is locked: invalid token")
        if self.opened:
            raise errors.LeakageError(
                "test set was already used for final evaluation"
            )
        self.opened = True
        return self.features, self.target

    def __len__(self) -> int:
        """Number of test rows."""
        return len(self.ids)


@dataclasses.dataclass(slots=True)
class SplitResult:
    """Everything produced by splitting a dataset.

    Attributes:
        manifest: Immutable split record.
        dev_features: Development features (target removed).
        dev_target: Development labels.
        locked_test: The sealed test partition.
    """

    manifest: SplitManifest
    dev_features: pd.DataFrame
    dev_target: pd.Series
    locked_test: LockedTestSet


def to_ids(values: np.ndarray) -> tuple[int, ...]:
    """Converts an index array to a tuple of Python ints."""
    return tuple(int(v) for v in values)


class SplitStrategy(abc.ABC):
    """One validation protocol (holdout + cross-validation folds).

    Attributes:
        name: Strategy name used in configuration.
    """

    name: ClassVar[str]
    supported_tasks: ClassVar[frozenset[tasks.TaskType]] = frozenset(
        tasks.TaskType
    )

    def __init__(self, spec: tasks.ProjectSpec, task: tasks.TaskType) -> None:
        """Initializes the strategy.

        Args:
            spec: Project specification.
            task: Resolved task type.

        Raises:
            BlockedError: If the strategy cannot serve the task.
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
    def partition(self, frame: pd.DataFrame) -> Partition:
        """Splits the frame into development, test and folds.

        Args:
            frame: The full dataset including the target column.

        Returns:
            The partition.

        Raises:
            BlockedError: If a leakage-free split is impossible.
        """


class RandomSplit(SplitStrategy):
    """Shuffled holdout and K-fold; the default for regression."""

    name = "random"

    def stratify_labels(self, frame: pd.DataFrame) -> pd.Series | None:
        """Labels to stratify the holdout on; none for plain random."""
        return None

    def cross_validation_folds(
        self, frame: pd.DataFrame, dev_ids: np.ndarray, notes: list[str]
    ) -> list[Fold]:
        """Builds shuffled K-fold splits over positions in `dev_ids`."""
        splitter = model_selection.KFold(
            n_splits=self.validation.folds,
            shuffle=True,
            random_state=self.validation.seed,
        )
        return list(splitter.split(np.arange(len(dev_ids))))

    def partition(self, frame: pd.DataFrame) -> Partition:
        """Random holdout followed by K-fold on the development rows."""
        dev_ids, test_ids = model_selection.train_test_split(
            np.arange(len(frame)),
            test_size=self.validation.test_fraction,
            random_state=self.validation.seed,
            stratify=self.stratify_labels(frame),
        )
        dev_ids, test_ids = np.sort(dev_ids), np.sort(test_ids)
        notes: list[str] = []
        folds = self.cross_validation_folds(frame, dev_ids, notes)
        return Partition(dev_ids, test_ids, folds, notes)


class StratifiedSplit(RandomSplit):
    """Class-proportion preserving split for classification."""

    name = "stratified"
    supported_tasks = frozenset(
        {tasks.TaskType.BINARY, tasks.TaskType.MULTICLASS}
    )

    def stratify_labels(self, frame: pd.DataFrame) -> pd.Series:
        """Stratifies on the target; every class needs two examples."""
        labels = frame[self.data.target]
        if (labels.value_counts() < 2).any():
            raise errors.BlockedError(
                "a class has fewer than 2 examples: cannot stratify"
            )
        return labels

    def cross_validation_folds(
        self, frame: pd.DataFrame, dev_ids: np.ndarray, notes: list[str]
    ) -> list[Fold]:
        """Stratified K-fold, shrinking K to the rarest class if needed."""
        labels = frame[self.data.target].iloc[dev_ids]
        rarest = int(labels.value_counts().min())
        if rarest < 2:
            raise errors.BlockedError(
                "insufficient minority-class examples after test split"
            )
        folds = self.validation.folds
        if rarest < folds:
            notes.append(
                f"folds reduced {folds}->{rarest} (minority class size)"
            )
            folds = rarest
        splitter = model_selection.StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=self.validation.seed
        )
        return list(splitter.split(np.arange(len(dev_ids)), labels))


class GroupSplit(SplitStrategy):
    """Keeps every group wholly inside one partition."""

    name = "group"

    def partition(self, frame: pd.DataFrame) -> Partition:
        """Group-disjoint holdout and group K-fold."""
        groups = frame[self.data.group_column].to_numpy()
        if pd.isna(groups).any():
            raise errors.BlockedError(
                f"group column '{self.data.group_column}' has missing values"
            )
        folds = self.validation.folds
        if len(set(groups)) < folds + 2:
            raise errors.BlockedError(
                f"need at least {folds + 2} groups for group validation"
            )
        holdout = model_selection.GroupShuffleSplit(
            n_splits=1,
            test_size=self.validation.test_fraction,
            random_state=self.validation.seed,
        )
        dev_ids, test_ids = next(
            holdout.split(np.arange(len(frame)), groups=groups)
        )
        dev_groups = groups[dev_ids]
        folds = min(folds, len(set(dev_groups)))
        notes: list[str] = []
        if not self.task.is_classification:
            splitter = model_selection.GroupKFold(n_splits=folds)
            return Partition(
                dev_ids,
                test_ids,
                list(
                    splitter.split(np.arange(len(dev_ids)), groups=dev_groups)
                ),
                notes,
            )
        labels = frame[self.data.target].to_numpy()[dev_ids]
        chosen = self.class_complete_folds(labels, dev_groups, folds, notes)
        return Partition(dev_ids, test_ids, chosen, notes)

    def class_complete_folds(
        self,
        labels: np.ndarray,
        groups: np.ndarray,
        folds: int,
        notes: list[str],
    ) -> list[Fold]:
        """Builds group folds that each contain every class.

        Every class must appear in every training and validation part, since
        metrics such as ROC-AUC are undefined otherwise. Tries the requested
        number of folds, then fewer.

        Raises:
            BlockedError: If no group-disjoint split keeps all classes in
                every fold.
        """
        classes = set(labels.tolist())
        positions = np.arange(len(labels))
        for count in range(folds, 1, -1):
            splitter = model_selection.StratifiedGroupKFold(
                n_splits=count,
                shuffle=True,
                random_state=self.validation.seed,
            )
            candidate = list(splitter.split(positions, labels, groups))
            if all(
                set(labels[fit].tolist()) == classes
                and set(labels[held].tolist()) == classes
                for fit, held in candidate
            ):
                if count != folds:
                    notes.append(
                        f"group folds reduced {folds}->{count} so every fold "
                        "contains every class"
                    )
                return candidate
        raise errors.BlockedError(
            "no group-disjoint split keeps every class in every fold; the "
            "classes are concentrated in too few groups"
        )


class TemporalSplit(SplitStrategy):
    """Trains on the past and evaluates on the future."""

    name = "temporal"

    def partition(self, frame: pd.DataFrame) -> Partition:
        """Chronological holdout and expanding-window folds."""
        column = self.data.time_column
        times = pd.to_datetime(frame[column], errors="coerce")
        if times.isna().any():
            raise errors.BlockedError(
                f"time column '{column}' has unparseable/missing values: "
                "ordering ambiguous"
            )
        order = np.argsort(times.to_numpy(), kind="stable")
        rows = len(frame)
        test_rows = max(1, round(rows * self.validation.test_fraction))
        embargo = self.validation.embargo_rows
        dev_ids = order[: rows - test_rows - embargo]
        test_ids = order[rows - test_rows :]
        folds = self.validation.folds
        if len(dev_ids) < folds + 1 + embargo:
            raise errors.BlockedError(
                "too few rows for temporal cross-validation"
            )
        splitter = model_selection.TimeSeriesSplit(n_splits=folds, gap=embargo)
        notes = []
        if embargo:
            notes.append(
                f"embargo of {embargo} rows between development and test and "
                "between each training window and its validation window"
            )
        return Partition(
            dev_ids,
            test_ids,
            list(splitter.split(np.arange(len(dev_ids)))),
            notes,
        )


STRATEGIES: dict[str, type[SplitStrategy]] = {
    strategy.name: strategy
    for strategy in (RandomSplit, StratifiedSplit, GroupSplit, TemporalSplit)
}


def resolve_strategy_name(spec: tasks.ProjectSpec, task: tasks.TaskType) -> str:
    """Chooses the split strategy for a project.

    Args:
        spec: Project specification.
        task: Resolved task type.

    Returns:
        The explicit strategy, or for `auto` the one matching the data
        structure: temporal, then group, then stratified / random.
    """
    requested = spec.validation.strategy
    if requested != "auto":
        return requested
    if spec.data.time_column:
        return "temporal"
    if spec.data.group_column:
        return "group"
    return "stratified" if task.is_classification else "random"


class SplitManager:
    """Selects and applies the validation protocol for a project."""

    def __init__(self, spec: tasks.ProjectSpec, task: tasks.TaskType) -> None:
        """Initializes the manager.

        Args:
            spec: Project specification.
            task: Resolved task type.
        """
        self.spec = spec
        self.task = task
        self.strategy_name = resolve_strategy_name(spec, task)

    def split(self, frame: pd.DataFrame) -> SplitResult:
        """Splits the dataset.

        Args:
            frame: The full dataset including the target column.

        Returns:
            Manifest, development data and the sealed test set.
        """
        strategy = STRATEGIES[self.strategy_name](self.spec, self.task)
        part = strategy.partition(frame)
        manifest = SplitManifest(
            strategy=self.strategy_name,
            seed=self.spec.validation.seed,
            train_ids=to_ids(part.dev_ids),
            test_ids=to_ids(part.test_ids),
            folds=tuple((to_ids(a), to_ids(b)) for a, b in part.folds),
            notes=tuple(part.notes),
        )
        target_name = self.spec.data.target
        features = frame.drop(columns=[target_name])
        target = frame[target_name]
        locked = LockedTestSet(
            features.iloc[part.test_ids].reset_index(drop=True),
            target.iloc[part.test_ids].reset_index(drop=True),
            manifest.test_ids,
        )
        return SplitResult(
            manifest,
            features.iloc[part.dev_ids].reset_index(drop=True),
            target.iloc[part.dev_ids].reset_index(drop=True),
            locked,
        )
