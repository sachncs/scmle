"""The strategy registry and the splitter that applies it."""

import pandas

from scmle.contracts import tasks
from scmle.data.readers import streaming
from scmle.data.splitting import duplicates
from scmle.data.splitting.splitters import lockbox
from scmle.data.splitting.splitters import splits
from scmle.data.splitting.splitters.partitions import DUPLICATE_AWARE
from scmle.data.splitting.splitters.partitions import Manifest
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
        splits.Chronology,
        splits.Rolling,
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

    def gather(self, frame: pandas.DataFrame, part: Partition) -> Partition:
        """Moves copies of one record to the same side of every split.

        Args:
            frame: The full dataset.
            part: The partition to adjust.

        Returns:
            `part` itself when no rows are duplicated; otherwise a new
            partition with each duplicate group wholly in development or
            test, and a note counting the groups and rows kept together.
        """
        groups = duplicates.duplicate_groups(frame, self.feature_columns(frame))
        if (groups == duplicates.NOT_DUPLICATED).all():
            return part
        dev, test, folds, count = duplicates.cohere(
            part.dev_ids, part.test_ids, part.folds, groups
        )
        covered = int((groups != duplicates.NOT_DUPLICATED).sum())
        notes = [
            *part.notes,
            (
                f"{count} groups of duplicate rows ({covered} rows) kept on "
                "one side of every split"
            ),
        ]
        return Partition(dev, test, folds, notes)

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
        if name in DUPLICATE_AWARE:
            part = self.gather(frame, part)
        manifest = Manifest(
            strategy=name,
            seed=self.spec.validation.seed,
            train_ids=tuplify(part.dev_ids),
            test_ids=tuplify(part.test_ids),
            folds=tuple((tuplify(a), tuplify(b)) for a, b in part.folds),
            notes=tuple(part.notes),
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
