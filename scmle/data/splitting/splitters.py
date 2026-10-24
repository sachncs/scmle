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

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.readers import streaming
from scmle.data.splitting import duplicates
from scmle.data.splitting import iterative

Fold = tuple[np.ndarray, np.ndarray]
MIN_ROLLING_TRAIN = 10
DUPLICATE_AWARE = frozenset({"random", "stratified", "stratified_events"})
MIN_ROLLING_FOLDS = 2


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


class LockedTestSet(abc.ABC):
    """The test partition: unavailable to search, released once.

    Attributes:
        ids: Row ids of the test rows.
        opened: Whether the rows were released.
    """

    ids: tuple[int, ...]
    opened: bool

    @abc.abstractmethod
    def issue_token(self) -> str:
        """Returns the release token; call only once selection is frozen."""

    @abc.abstractmethod
    def open(self, token: str) -> tuple[pd.DataFrame, pd.Series]:
        """Releases the test data a single time.

        Args:
            token: The token from `issue_token`.

        Returns:
            Test features and labels.

        Raises:
            LeakageError: On a wrong token or a repeated release.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Releases any resource held for the lock."""

    def __len__(self) -> int:
        """Number of test rows."""
        return len(self.ids)


@dataclasses.dataclass(slots=True)
class ResidentTestSet(LockedTestSet):
    """Test rows kept in this process, behind a one-time token.

    Python cannot hide attributes, so the lock is a protocol: trial code only
    receives development data, and `open` enforces a one-time, token-gated
    release that is recorded. The sealed variant moves the rows to a separate
    process (see `scmle.governance.vault`).

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
        """Releases the test data a single time."""
        if token != self.token:
            raise errors.LeakageError("test set is locked: invalid token")
        if self.opened:
            raise errors.LeakageError(
                "test set was already used for final evaluation"
            )
        self.opened = True
        return self.features, self.target

    def close(self) -> None:
        """Nothing to release: the rows live in this process."""


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


class EventStratifiedSplit(RandomSplit):
    """Keeps the share of observed events equal across splits (survival)."""

    name = "stratified_events"
    supported_tasks = frozenset({tasks.TaskType.SURVIVAL})

    def events(self, frame: pd.DataFrame) -> pd.Series:
        """The 0/1 event column."""
        return frame[self.data.event_column]

    def stratify_labels(self, frame: pd.DataFrame) -> pd.Series:
        """Stratifies the holdout on the event indicator."""
        events = self.events(frame)
        if (events.value_counts() < 2).any():
            raise errors.BlockedError(
                "events or censored rows number fewer than 2: cannot stratify"
            )
        return events

    def cross_validation_folds(
        self, frame: pd.DataFrame, dev_ids: np.ndarray, notes: list[str]
    ) -> list[Fold]:
        """Stratified K-fold on events, so every fold has events to rank."""
        events = self.events(frame).iloc[dev_ids]
        rarest = int(events.value_counts().min())
        if rarest < 2:
            raise errors.BlockedError(
                "too few events (or censored rows) after the test split"
            )
        folds = self.validation.folds
        if rarest < folds:
            notes.append(f"folds reduced {folds}->{rarest} (rarest outcome)")
            folds = rarest
        splitter = model_selection.StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=self.validation.seed
        )
        return list(splitter.split(np.arange(len(dev_ids)), events))


class MultilabelStratifiedSplit(RandomSplit):
    """Keeps every label's share steady across splits (multi-label)."""

    name = "multilabel_stratified"
    supported_tasks = frozenset({tasks.TaskType.MULTILABEL})

    def label_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        """The 0/1 matrix of all label columns."""
        return frame[self.data.targets].to_numpy().astype(int)

    def partition(self, frame: pd.DataFrame) -> Partition:
        """Iteratively stratified holdout, then folds over development."""
        labels = self.label_matrix(frame)
        share = self.validation.test_fraction
        side = iterative.iterative_assignment(
            labels, [1.0 - share, share], self.validation.seed
        )
        dev_ids = np.flatnonzero(side == 0)
        test_ids = np.flatnonzero(side == 1)
        notes: list[str] = []
        folds = self.cross_validation_folds(frame, dev_ids, notes)
        return Partition(dev_ids, test_ids, folds, notes)

    def cross_validation_folds(
        self, frame: pd.DataFrame, dev_ids: np.ndarray, notes: list[str]
    ) -> list[Fold]:
        """Iteratively stratified folds over the development rows."""
        count = self.validation.folds
        assignment = iterative.iterative_assignment(
            self.label_matrix(frame)[dev_ids],
            [1.0 / count] * count,
            self.validation.seed,
        )
        positions = np.arange(len(dev_ids))
        folds = [
            (positions[assignment != k], positions[assignment == k])
            for k in range(count)
        ]
        missing = [
            column
            for column in self.data.targets
            if any(
                frame[column].to_numpy()[dev_ids[fold[1]]].sum() == 0
                for fold in folds
            )
        ]
        if missing:
            notes.append(
                f"labels too rare to appear in every fold: {sorted(missing)}"
            )
        return folds


class TreatmentStratifiedSplit(RandomSplit):
    """Keeps treatment and outcome shares equal across splits (uplift)."""

    name = "treatment_stratified"
    supported_tasks = frozenset({tasks.TaskType.UPLIFT})

    def cells(self, frame: pd.DataFrame) -> pd.Series:
        """Treatment arm combined with a binary outcome (arm alone else)."""
        arm = frame[self.data.treatment_column].astype(int)
        outcome = frame[self.data.target]
        if set(pd.unique(outcome.dropna())) <= {0, 1}:
            return arm * 2 + outcome.astype(int)
        return arm

    def stratify_labels(self, frame: pd.DataFrame) -> pd.Series:
        """Stratifies the holdout on the arm / outcome cell."""
        cells = self.cells(frame)
        if (cells.value_counts() < 2).any():
            raise errors.BlockedError(
                "a treatment / outcome combination has fewer than 2 rows"
            )
        return cells

    def cross_validation_folds(
        self, frame: pd.DataFrame, dev_ids: np.ndarray, notes: list[str]
    ) -> list[Fold]:
        """Stratified K-fold on the cells, shrinking K for rare ones."""
        cells = self.cells(frame).iloc[dev_ids]
        rarest = int(cells.value_counts().min())
        folds = min(self.validation.folds, rarest)
        if folds < 2:
            raise errors.BlockedError(
                "too few rows in a treatment / outcome combination"
            )
        if folds < self.validation.folds:
            notes.append(f"folds reduced {self.validation.folds}->{folds}")
        splitter = model_selection.StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=self.validation.seed
        )
        return list(splitter.split(np.arange(len(dev_ids)), cells))


class PresetSplit(RandomSplit):
    """Uses a split that was decided while streaming the file.

    The `split_side` column says which rows are locked (1) and which are
    development (0); development rows are then cross-validated at random
    (stratified for classification).
    """

    name = "preset"

    def partition(self, frame: pd.DataFrame) -> Partition:
        """Reads the sides from the column; folds over the development rows."""
        side = frame[streaming.SPLIT_COLUMN].to_numpy()
        dev_ids = np.flatnonzero(side == streaming.DEVELOPMENT)
        test_ids = np.flatnonzero(side == streaming.LOCKED)
        notes = ["split decided by hashing row positions while streaming"]
        if self.task.is_classification:
            labels = frame[self.data.target].iloc[dev_ids]
            folds = min(self.validation.folds, int(labels.value_counts().min()))
            splitter = model_selection.StratifiedKFold(
                n_splits=max(folds, 2),
                shuffle=True,
                random_state=self.validation.seed,
            )
            parts = list(splitter.split(np.arange(len(dev_ids)), labels))
        else:
            parts = self.cross_validation_folds(frame, dev_ids, notes)
        return Partition(dev_ids, test_ids, parts, notes)


class GroupSplit(SplitStrategy):
    """Keeps every group wholly inside one partition."""

    name = "group"

    def partition(self, frame: pd.DataFrame) -> Partition:
        """Group-disjoint holdout and group K-fold."""
        column = self.data.group_column or self.data.query_column
        groups = frame[column].to_numpy()
        if pd.isna(groups).any():
            raise errors.BlockedError(
                f"group column '{column}' has missing values"
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

    def embargo(self, frame: pd.DataFrame) -> int:
        """Rows dropped between training and later windows.

        A forecast row's target looks `horizon` steps ahead, so the rows just
        before a window would otherwise train on the window's own values; the
        embargo covers `horizon` time steps of every series.
        """
        configured = self.validation.embargo_rows
        horizon = self.data.horizon
        if horizon is None:
            return configured
        series = (
            frame[self.data.group_column].nunique()
            if self.data.group_column
            else 1
        )
        return max(configured, horizon * series)

    def chronological_holdout(
        self, frame: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Orders rows in time and holds out the latest for the final test.

        Returns:
            Development row positions (oldest first, an embargo before the
            test window removed) and test row positions.

        Raises:
            BlockedError: If the time column cannot order the rows.
        """
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
        embargo = self.embargo(frame)
        return order[: rows - test_rows - embargo], order[rows - test_rows :]

    def partition(self, frame: pd.DataFrame) -> Partition:
        """Chronological holdout and expanding-window folds."""
        dev_ids, test_ids = self.chronological_holdout(frame)
        embargo = self.embargo(frame)
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
        notes.append(
            "folds are listed latest window first, so cheap first-rung "
            "evaluations use the largest training window"
        )
        windows = list(splitter.split(np.arange(len(dev_ids))))
        return Partition(dev_ids, test_ids, windows[::-1], notes)


class RollingSplit(TemporalSplit):
    """Rolling-origin evaluation: slide forward, validate on what follows.

    Each fold trains on the rows before a cut-off (at most `window_rows` of
    them when set, otherwise all earlier rows), skips `embargo_rows`, and
    validates on the next `horizon_rows`. Successive folds move the cut-off
    back by one horizon, so every validation window is genuinely later than
    its training window and the folds cover the most recent history.
    """

    name = "rolling"

    def partition(self, frame: pd.DataFrame) -> Partition:
        """Chronological holdout and sliding training/validation windows."""
        dev_ids, test_ids = self.chronological_holdout(frame)
        validation = self.validation
        rows = len(dev_ids)
        horizon = validation.horizon_rows or max(
            1, rows // (validation.folds + 1)
        )
        windows = []
        for fold in range(validation.folds):
            held_end = rows - fold * horizon
            held_start = held_end - horizon
            train_end = held_start - validation.embargo_rows
            train_start = (
                0
                if validation.window_rows is None
                else max(0, train_end - validation.window_rows)
            )
            if train_end - train_start < MIN_ROLLING_TRAIN or held_start < 0:
                break
            windows.append(
                (
                    np.arange(train_start, train_end),
                    np.arange(held_start, held_end),
                )
            )
        if len(windows) < MIN_ROLLING_FOLDS:
            raise errors.BlockedError(
                f"only {len(windows)} rolling window(s) fit; shorten "
                "horizon_rows, window_rows or embargo_rows, or add data"
            )
        notes = [
            f"rolling origin: {len(windows)} windows, horizon {horizon} rows, "
            + (
                f"training window {validation.window_rows} rows"
                if validation.window_rows
                else "expanding training window"
            ),
            "folds are listed latest window first",
        ]
        return Partition(dev_ids, test_ids, windows, notes)


STRATEGIES: dict[str, type[SplitStrategy]] = {
    strategy.name: strategy
    for strategy in (
        RandomSplit,
        StratifiedSplit,
        EventStratifiedSplit,
        MultilabelStratifiedSplit,
        TreatmentStratifiedSplit,
        PresetSplit,
        GroupSplit,
        TemporalSplit,
        RollingSplit,
    )
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
    if task is tasks.TaskType.SURVIVAL:
        return "stratified_events"
    if task is tasks.TaskType.RANKING:
        return "group"
    if task is tasks.TaskType.MULTILABEL:
        return "multilabel_stratified"
    if task is tasks.TaskType.UPLIFT:
        return "treatment_stratified"
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

    def feature_columns(self, frame: pd.DataFrame) -> list[str]:
        """Columns that describe a record (everything but the roles)."""
        data = self.spec.data
        roles = {
            data.target,
            data.treatment_column,
            *data.targets,
            data.group_column,
            data.time_column,
            data.weight_column,
            data.event_column,
            *data.id_columns,
            *data.excluded_columns,
            *data.unavailable_at_prediction,
        }
        return [c for c in frame.columns if c not in roles]

    def keep_duplicates_together(
        self, frame: pd.DataFrame, part: Partition
    ) -> Partition:
        """Moves copies of one record to the same side of every split."""
        groups = duplicates.duplicate_groups(frame, self.feature_columns(frame))
        if (groups == duplicates.NOT_DUPLICATED).all():
            return part
        dev, test, folds, count = duplicates.keep_together(
            part.dev_ids, part.test_ids, part.folds, groups
        )
        covered = int((groups != duplicates.NOT_DUPLICATED).sum())
        notes = [
            *part.notes,
            f"{count} groups of duplicate rows ({covered} rows) kept on one "
            "side of every split",
        ]
        return Partition(dev, test, folds, notes)

    def split(self, frame: pd.DataFrame) -> SplitResult:
        """Splits the dataset.

        Args:
            frame: The full dataset including the target column.

        Returns:
            Manifest, development data and the sealed test set.
        """
        name = (
            PresetSplit.name
            if streaming.SPLIT_COLUMN in frame.columns
            else self.strategy_name
        )
        strategy = STRATEGIES[name](self.spec, self.task)
        part = strategy.partition(frame)
        if name in DUPLICATE_AWARE:
            part = self.keep_duplicates_together(frame, part)
        manifest = SplitManifest(
            strategy=name,
            seed=self.spec.validation.seed,
            train_ids=to_ids(part.dev_ids),
            test_ids=to_ids(part.test_ids),
            folds=tuple((to_ids(a), to_ids(b)) for a, b in part.folds),
            notes=tuple(part.notes),
        )
        target_name = self.spec.data.target
        features = frame.drop(columns=[target_name])
        target = frame[target_name]
        locked = ResidentTestSet(
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
