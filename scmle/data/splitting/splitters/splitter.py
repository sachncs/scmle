"""The strategy registry and the splitter that applies it."""

import numpy
import pandas

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.readers import streaming
from scmle.data.splitting import duplicates
from scmle.data.splitting.splitters import lockbox
from scmle.data.splitting.splitters import splits
from scmle.data.splitting.splitters import temporal
from scmle.data.splitting.splitters.partitions import DUPLICATE_AWARE
from scmle.data.splitting.splitters.partitions import Manifest
from scmle.data.splitting.splitters.partitions import MIN_STRATUM_ROWS
from scmle.data.splitting.splitters.partitions import MIN_TEST_CLASS_ROWS
from scmle.data.splitting.splitters.partitions import Partition
from scmle.data.splitting.splitters.partitions import Partitioner
from scmle.data.splitting.splitters.partitions import Split
from scmle.data.splitting.splitters.partitions import tuplify

STRATEGIES: dict[str, type[Partitioner]] = {
    strategy.name: strategy
    for strategy in (
        splits.Random,
        splits.Stratified,
        splits.Events,
        splits.Multilabel,
        splits.Treatment,
        splits.Preset,
        splits.Group,
        temporal.Chronology,
        temporal.Rolling,
    )
}


TASK_STRATEGIES = {
    tasks.Task.SURVIVAL: "stratified_events",
    tasks.Task.RANKING: "group",
    tasks.Task.MULTILABEL: "multilabel_stratified",
    tasks.Task.UPLIFT: "treatment_stratified",
}


def scheme(spec: tasks.Config, task: tasks.Task) -> str:
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
    if task in TASK_STRATEGIES:
        return TASK_STRATEGIES[task]
    return "stratified" if task.is_classification else "random"


class Splitter:
    """Selects and applies the validation protocol for a project.

    Attributes:
        spec: Project specification.
        task: Resolved task type.
        strategy_name: Name of the strategy chosen for the project.
    """

    def __init__(self, spec: tasks.Config, task: tasks.Task) -> None:
        """Initializes the manager.

        Args:
            spec: Project specification.
            task: Resolved task type.
        """
        self.spec = spec
        self.task = task
        self.strategy_name = scheme(spec, task)

    def feature_columns(self, frame: pandas.DataFrame) -> list[str]:
        """Finds the columns that describe a record.

        Args:
            frame: The full dataset.

        Returns:
            The frame's column names, in order, other than those with a role:
            target, treatment, baseline, propensity, logged action, targets,
            group, time, weight, event, id, excluded and unavailable at
            prediction.
        """
        data = self.spec.data
        roles = {
            data.target,
            data.treatment_column,
            data.baseline_column,
            data.propensity_column,
            data.logged_action_column,
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

    def disclose(self, frame: pandas.DataFrame, part: Partition) -> Partition:
        """Counts the test rows that repeat a development record.

        The random and stratified strategies treat rows as exchangeable
        draws from the population the model will serve, as scikit-learn's
        (stratified) k-fold and the OpenML splits do. A repeated record is
        then a draw like any other: a future case can repeat a past one at
        the rate the data shows, and the held-out score is an unbiased
        estimate of that mix only when repeats may fall on both sides.
        Keeping them together would estimate the error on never-seen
        records instead and steer selection away from what the model will
        meet. Copies that belong to one entity are not exchangeable; the
        user declares them with `data.group_column`, and a charter run's
        custody carve keeps copies of one record together. The share is
        recorded so a memorising score stays visible.

        Args:
            frame: The full dataset.
            part: The partition, unchanged.

        Returns:
            `part` itself when no test row repeats a development record;
            otherwise `part` with a note counting those rows.
        """
        groups = duplicates.duplicate_groups(frame, self.feature_columns(frame))
        seen = set(groups[part.dev_ids].tolist()) - {duplicates.NOT_DUPLICATED}
        repeated = int(numpy.isin(groups[part.test_ids], list(seen)).sum())
        if not repeated:
            return part
        notes = [
            *part.notes,
            (
                f"{repeated} of {len(part.test_ids)} test rows repeat a "
                "development record; rows are treated as exchangeable "
                "draws (set data.group_column if copies belong to one "
                "entity)"
            ),
        ]
        return Partition(part.dev_ids, part.test_ids, part.folds, notes)

    def reserve(
        self,
        frame: pandas.DataFrame,
        strategy: splits.Stratified,
        part: Partition,
    ) -> Partition:
        """Moves development rows of a rare class into the test rows.

        A proportional holdout gives a class of `c` rows about
        `c * test_fraction` test rows, none at all for a small class. Each
        class gets at least `MIN_TEST_CLASS_ROWS` test rows when it can keep
        `MIN_STRATUM_ROWS` development rows for its folds, and otherwise as
        many as leave it those; the rows moved are drawn with the validation
        seed and the folds are drawn again. A class too small for the
        minimum is then refused (`support`).

        Args:
            frame: The full dataset including the target column.
            strategy: The stratified strategy that drew `part`.
            part: Its partition.

        Returns:
            `part` itself when no class needs rows; otherwise the partition
            with the moved rows in the test ids and new folds.
        """
        labels = frame[self.spec.data.target].to_numpy()
        rng = numpy.random.default_rng(self.spec.validation.seed)
        dev, test = part.dev_ids, part.test_ids
        moved = []
        for label in sorted(set(labels[dev].tolist()), key=str):
            pool = dev[labels[dev] == label]
            held = int((labels[test] == label).sum())
            wanted = min(
                MIN_TEST_CLASS_ROWS, held + len(pool) - MIN_STRATUM_ROWS
            )
            if wanted > held:
                moved.append(rng.choice(pool, wanted - held, replace=False))
        if not moved:
            return part
        taken = numpy.concatenate(moved)
        dev, test = numpy.setdiff1d(dev, taken), numpy.union1d(test, taken)
        notes: list[str] = []
        folds = strategy.cross_validation_folds(frame, dev, notes)
        return Partition(dev, test, folds, notes)

    def support(self, frame: pandas.DataFrame, part: Partition) -> None:
        """Refuses test rows that cannot score every class of the task.

        Binary and multiclass metrics are read per class: with no test row
        of a class ROC AUC is undefined and F1 scores a single class, and
        with fewer than `MIN_TEST_CLASS_ROWS` even a perfect score on a
        class cannot be told from guessing (see the constant). Every class
        of the target must reach that count among the test rows.

        Args:
            frame: The full dataset including the target column.
            part: The partition the strategy drew.

        Raises:
            BlockedError: If a class has fewer than `MIN_TEST_CLASS_ROWS`
                test rows; `details["short"]` maps each such class to its
                count.
        """
        if self.task not in (tasks.Task.BINARY, tasks.Task.MULTICLASS):
            return
        labels = frame[self.spec.data.target].dropna()
        held = frame[self.spec.data.target].iloc[part.test_ids].value_counts()
        short = {
            str(label): int(held.get(label, 0))
            for label in sorted(labels.unique(), key=str)
            if held.get(label, 0) < MIN_TEST_CLASS_ROWS
        }
        if short:
            raise errors.BlockedError(
                f"the test rows hold fewer than {MIN_TEST_CLASS_ROWS} rows of "
                f"class(es) {sorted(short)} ({short}), too few to score the "
                "task's metrics on them; provide more rows of those classes, "
                "raise validation.test or use the stratified strategy (a "
                f"class needs {MIN_TEST_CLASS_ROWS + MIN_STRATUM_ROWS} rows "
                "to reach the test minimum and keep two for training)",
                {"short": short, "minimum": MIN_TEST_CLASS_ROWS},
            )

    def split(self, frame: pandas.DataFrame) -> Split:
        """Splits the dataset.

        Errors from the chosen strategy propagate: a `BlockedError` when it
        cannot serve the task or split the data without leakage, and a
        `DataError` when a stratified holdout cannot hold every class.

        Args:
            frame: The full dataset including the target column.

        Returns:
            Manifest, development data and the sealed test set.
        """
        name = (
            splits.Preset.name
            if streaming.SPLIT_COLUMN in frame.columns
            else self.strategy_name
        )
        strategy = STRATEGIES[name](self.spec, self.task)
        part = strategy.partition(frame)
        if isinstance(strategy, splits.Stratified):
            part = self.reserve(frame, strategy, part)
        self.support(frame, part)
        if name in DUPLICATE_AWARE:
            part = self.disclose(frame, part)
        manifest = Manifest(
            strategy=name,
            seed=self.spec.validation.seed,
            train_ids=tuplify(part.dev_ids),
            test_ids=tuplify(part.test_ids),
            folds=tuple((tuplify(a), tuplify(b)) for a, b in part.folds),
            notes=tuple(part.notes),
            entities=(
                self.spec.validation.entities
                if isinstance(strategy, temporal.Chronology)
                else None
            ),
        )
        target_name = self.spec.data.target
        features = frame.drop(columns=[target_name])
        target = frame[target_name]
        locked = lockbox.Resident(
            features.iloc[part.test_ids].reset_index(drop=True),
            target.iloc[part.test_ids].reset_index(drop=True),
            manifest.test_ids,
        )
        return Split(
            manifest,
            features.iloc[part.dev_ids].reset_index(drop=True),
            target.iloc[part.dev_ids].reset_index(drop=True),
            locked,
        )
