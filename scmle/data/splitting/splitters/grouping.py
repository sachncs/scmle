"""The group-aware partitioner."""

import numpy
import pandas
from sklearn import model_selection

from scmle.contracts import errors
from scmle.data.splitting.splitters.partitions import Fold
from scmle.data.splitting.splitters.partitions import Partition
from scmle.data.splitting.splitters.partitions import Partitioner


class GroupSplit(Partitioner):
    """Keeps every group wholly inside one partition."""

    name = "group"

    def partition(self, frame: pandas.DataFrame) -> Partition:
        """Group-disjoint holdout and group K-fold.

        Args:
            frame: The full dataset including the target column.

        Returns:
            The development and test ids, group-disjoint folds over
            development positions and any notes. Classification folds each
            contain every class.

        Raises:
            BlockedError: If the group column has missing values, there are
                fewer than `folds + 2` groups, or (for classification) no
                group-disjoint folds keep every class.
        """
        column = self.data.group_column or self.data.query_column
        groups = frame[column].to_numpy()
        if pandas.isna(groups).any():
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
            holdout.split(numpy.arange(len(frame)), groups=groups)
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
                    splitter.split(
                        numpy.arange(len(dev_ids)), groups=dev_groups
                    )
                ),
                notes,
            )
        labels = frame[self.data.target].to_numpy()[dev_ids]
        chosen = self.class_complete_folds(labels, dev_groups, folds, notes)
        return Partition(dev_ids, test_ids, chosen, notes)

    def class_complete_folds(
        self,
        labels: numpy.ndarray,
        groups: numpy.ndarray,
        folds: int,
        notes: list[str],
    ) -> list[Fold]:
        """Builds group folds that each contain every class.

        Every class must appear in every training and validation part, since
        metrics such as ROC-AUC are undefined otherwise. Tries the requested
        number of folds, then fewer.

        Args:
            labels: Class label of each development row.
            groups: Group of each development row.
            folds: The requested number of folds.
            notes: Adjustments made while splitting; gets a note when fewer
                folds are used.

        Returns:
            One `(train, validation)` pair of position arrays for each fold,
            with groups kept disjoint between the two.

        Raises:
            BlockedError: If no group-disjoint split keeps all classes in
                every fold.
        """
        classes = set(labels.tolist())
        positions = numpy.arange(len(labels))
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
