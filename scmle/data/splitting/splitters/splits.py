"""The split strategies: one `Partitioner` per validation protocol.

Each class is one way to hold out test rows and fold the development rows,
named for what it preserves, so a call site reads `splits.Group`; the
time-ordered strategies are in `temporal`. `Splitter` selects one by its
configured `name`.
"""

import math

import numpy
import pandas
from sklearn import model_selection

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.readers import streaming
from scmle.data.splitting import iterative
from scmle.data.splitting.splitters.partitions import Fold
from scmle.data.splitting.splitters.partitions import MIN_FOLDS
from scmle.data.splitting.splitters.partitions import MIN_STRATUM_ROWS
from scmle.data.splitting.splitters.partitions import Partition
from scmle.data.splitting.splitters.partitions import Partitioner


class Random(Partitioner):
    """Shuffled holdout and K-fold; the default for regression."""

    name = "random"

    def stratifier(self, frame: pandas.DataFrame) -> pandas.Series | None:  # noqa: ARG002 -- hook that subclasses override and read the frame in
        """Labels to stratify the holdout on; none for plain random.

        Args:
            frame: The full dataset including the target column.

        Returns:
            None, so the holdout is not stratified. Subclasses return one
            label per row of the frame.
        """
        return None

    def fold(
        self,
        frame: pandas.DataFrame,  # noqa: ARG002 -- hook that subclasses override and read the frame in
        dev_ids: numpy.ndarray,
        notes: list[str],  # noqa: ARG002 -- hook that subclasses override and append notes to
    ) -> list[Fold]:
        """Builds shuffled K-fold splits over positions in `dev_ids`.

        Args:
            frame: The full dataset including the target column.
            dev_ids: Row ids of the development partition.
            notes: Adjustments made while splitting; subclasses append to it
                when they change the number of folds.

        Returns:
            One `(train, validation)` pair of position arrays into `dev_ids`
            for each of the configured number of folds.
        """
        splitter = model_selection.KFold(
            n_splits=self.validation.folds,
            shuffle=True,
            random_state=self.validation.seed,
        )
        return list(splitter.split(numpy.arange(len(dev_ids))))

    def require(self, rows: int, labels: pandas.Series) -> None:
        """Stops with advice when the held-out part cannot hold every class.

        A stratified holdout needs at least one test row per class, and the
        development part needs at least as many rows as classes. A table too
        small for that fails inside scikit-learn with a message about
        `test_size`; this says what to do instead.

        Args:
            rows: Number of rows in the dataset.
            labels: The class label of each row.

        Raises:
            DataError: If `rows` is too few for the classes and the fraction.
        """
        classes = int(labels.nunique())
        held_out = math.ceil(rows * self.validation.test_fraction)
        if held_out < classes or rows - held_out < classes:
            needed = math.ceil(classes / self.validation.test_fraction)
            raise errors.DataError(
                f"{rows} rows are too few to hold out "
                f"{self.validation.test_fraction:.0%} for testing and still "
                f"have every one of the {classes} classes in the test rows "
                f"and the training rows (the test part would have "
                f"{held_out} rows). Provide at least {needed} rows, or "
                "lower the number of classes, or raise the test fraction."
            )

    def partition(self, frame: pandas.DataFrame) -> Partition:
        """Random holdout followed by K-fold on the development rows.

        A `DataError` from `require` propagates when a
        stratified holdout cannot hold every class, and subclasses may raise
        a `BlockedError` when they cannot stratify or fold the rows.

        Args:
            frame: The full dataset including the target column.

        Returns:
            The sorted development and test ids, the folds over development
            positions and any notes.
        """
        labels = self.stratifier(frame)
        if labels is not None:
            self.require(len(frame), labels)
        dev_ids, test_ids = model_selection.train_test_split(
            numpy.arange(len(frame)),
            test_size=self.validation.test_fraction,
            random_state=self.validation.seed,
            stratify=labels,
        )
        dev_ids, test_ids = numpy.sort(dev_ids), numpy.sort(test_ids)
        notes: list[str] = []
        folds = self.fold(frame, dev_ids, notes)
        return Partition(dev_ids, test_ids, folds, notes)


class Stratified(Random):
    """Class-proportion preserving split for classification."""

    name = "stratified"
    scope = frozenset({tasks.Task.BINARY, tasks.Task.MULTICLASS})

    def stratifier(self, frame: pandas.DataFrame) -> pandas.Series:
        """Stratifies on the target; every class needs two examples.

        Args:
            frame: The full dataset including the target column.

        Returns:
            The target column.

        Raises:
            BlockedError: If a class has fewer than two examples.
        """
        labels = frame[self.data.target]
        if (labels.value_counts() < MIN_STRATUM_ROWS).any():
            raise errors.BlockedError(
                "a class has fewer than 2 examples: cannot stratify"
            )
        return labels

    def fold(
        self, frame: pandas.DataFrame, dev_ids: numpy.ndarray, notes: list[str]
    ) -> list[Fold]:
        """Stratified K-fold, shrinking K to the rarest class if needed.

        Args:
            frame: The full dataset including the target column.
            dev_ids: Row ids of the development partition.
            notes: Adjustments made while splitting; gets a note when the
                number of folds is reduced.

        Returns:
            One `(train, validation)` pair of position arrays into `dev_ids`
            for each fold.

        Raises:
            BlockedError: If a class has fewer than two development rows.
        """
        labels = frame[self.data.target].iloc[dev_ids]
        rarest = int(labels.value_counts().min())
        if rarest < MIN_STRATUM_ROWS:
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
        return list(splitter.split(numpy.arange(len(dev_ids)), labels))


class Events(Random):
    """Keeps the share of observed events equal across splits (survival)."""

    name = "stratified_events"
    scope = frozenset({tasks.Task.SURVIVAL})

    def events(self, frame: pandas.DataFrame) -> pandas.Series:
        """Selects the 0/1 event column.

        Args:
            frame: The full dataset.

        Returns:
            The event column, one value per row.
        """
        return frame[self.data.event_column]

    def stratifier(self, frame: pandas.DataFrame) -> pandas.Series:
        """Stratifies the holdout on the event indicator.

        Args:
            frame: The full dataset.

        Returns:
            The event column.

        Raises:
            BlockedError: If events or censored rows number fewer than two.
        """
        events = self.events(frame)
        if (events.value_counts() < MIN_STRATUM_ROWS).any():
            raise errors.BlockedError(
                "events or censored rows number fewer than 2: cannot stratify"
            )
        return events

    def fold(
        self, frame: pandas.DataFrame, dev_ids: numpy.ndarray, notes: list[str]
    ) -> list[Fold]:
        """Stratified K-fold on events, so every fold has events to rank.

        Args:
            frame: The full dataset.
            dev_ids: Row ids of the development partition.
            notes: Adjustments made while splitting; gets a note when the
                number of folds is reduced.

        Returns:
            One `(train, validation)` pair of position arrays into `dev_ids`
            for each fold.

        Raises:
            BlockedError: If events or censored rows number fewer than two
                among the development rows.
        """
        events = self.events(frame).iloc[dev_ids]
        rarest = int(events.value_counts().min())
        if rarest < MIN_STRATUM_ROWS:
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
        return list(splitter.split(numpy.arange(len(dev_ids)), events))


class Multilabel(Random):
    """Keeps every label's share steady across splits (multi-label)."""

    name = "multilabel_stratified"
    scope = frozenset({tasks.Task.MULTILABEL})

    def stack(self, frame: pandas.DataFrame) -> numpy.ndarray:
        """Collects all label columns into one matrix.

        Args:
            frame: The full dataset.

        Returns:
            An integer matrix with one row per frame row and one column per
            label, in the order of the configured targets.
        """
        return frame[self.data.targets].to_numpy().astype(int)

    def partition(self, frame: pandas.DataFrame) -> Partition:
        """Iteratively stratified holdout, then folds over development.

        Args:
            frame: The full dataset.

        Returns:
            The development and test ids, the folds over development
            positions and any notes.
        """
        labels = self.stack(frame)
        share = self.validation.test_fraction
        side = iterative.assign(
            labels, [1.0 - share, share], self.validation.seed
        )
        dev_ids = numpy.flatnonzero(side == 0)
        test_ids = numpy.flatnonzero(side == 1)
        notes: list[str] = []
        folds = self.fold(frame, dev_ids, notes)
        return Partition(dev_ids, test_ids, folds, notes)

    def fold(
        self, frame: pandas.DataFrame, dev_ids: numpy.ndarray, notes: list[str]
    ) -> list[Fold]:
        """Iteratively stratified folds over the development rows.

        Args:
            frame: The full dataset.
            dev_ids: Row ids of the development partition.
            notes: Adjustments made while splitting; gets a note naming the
                labels that are absent from some validation fold.

        Returns:
            One `(train, validation)` pair of position arrays into `dev_ids`
            for each configured fold.
        """
        count = self.validation.folds
        assignment = iterative.assign(
            self.stack(frame)[dev_ids],
            [1.0 / count] * count,
            self.validation.seed,
        )
        positions = numpy.arange(len(dev_ids))
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


class Treatment(Random):
    """Keeps treatment and outcome shares equal across splits (uplift)."""

    name = "treatment_stratified"
    scope = frozenset({tasks.Task.UPLIFT})

    def cells(self, frame: pandas.DataFrame) -> pandas.Series:
        """Combines the treatment arm with a binary outcome.

        Args:
            frame: The full dataset.

        Returns:
            For a 0/1 outcome, `2 * arm + outcome` per row; otherwise the
            treatment arm alone.
        """
        arm = frame[self.data.treatment_column].astype(int)
        outcome = frame[self.data.target]
        if set(pandas.unique(outcome.dropna())) <= {0, 1}:
            return arm * 2 + outcome.astype(int)
        return arm

    def stratifier(self, frame: pandas.DataFrame) -> pandas.Series:
        """Stratifies the holdout on the arm / outcome cell.

        Args:
            frame: The full dataset.

        Returns:
            The arm / outcome cell of each row.

        Raises:
            BlockedError: If a cell has fewer than two rows.
        """
        cells = self.cells(frame)
        if (cells.value_counts() < MIN_STRATUM_ROWS).any():
            raise errors.BlockedError(
                "a treatment / outcome combination has fewer than 2 rows"
            )
        return cells

    def fold(
        self, frame: pandas.DataFrame, dev_ids: numpy.ndarray, notes: list[str]
    ) -> list[Fold]:
        """Stratified K-fold on the cells, shrinking K for rare ones.

        Args:
            frame: The full dataset.
            dev_ids: Row ids of the development partition.
            notes: Adjustments made while splitting; gets a note when the
                number of folds is reduced.

        Returns:
            One `(train, validation)` pair of position arrays into `dev_ids`
            for each fold.

        Raises:
            BlockedError: If the rarest cell leaves fewer than two folds.
        """
        cells = self.cells(frame).iloc[dev_ids]
        rarest = int(cells.value_counts().min())
        folds = min(self.validation.folds, rarest)
        if folds < MIN_FOLDS:
            raise errors.BlockedError(
                "too few rows in a treatment / outcome combination"
            )
        if folds < self.validation.folds:
            notes.append(f"folds reduced {self.validation.folds}->{folds}")
        splitter = model_selection.StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=self.validation.seed
        )
        return list(splitter.split(numpy.arange(len(dev_ids)), cells))


class Preset(Random):
    """Uses a split that was decided while streaming the file.

    The `split_side` column says which rows are locked (1) and which are
    development (0); development rows are then cross-validated at random
    (stratified for classification).
    """

    name = "preset"

    def partition(self, frame: pandas.DataFrame) -> Partition:
        """Reads the sides from the column; folds over the development rows.

        Args:
            frame: The full dataset, with the streaming split column.

        Returns:
            The development and test ids taken from the split column, folds
            over development positions and a note that the split was decided
            while streaming. Classification uses stratified folds, with at
            least two, otherwise shuffled folds.
        """
        side = frame[streaming.SPLIT_COLUMN].to_numpy()
        dev_ids = numpy.flatnonzero(side == streaming.DEVELOPMENT)
        test_ids = numpy.flatnonzero(side == streaming.LOCKED)
        notes = ["split decided by hashing row positions while streaming"]
        if self.task.classification:
            labels = frame[self.data.target].iloc[dev_ids]
            folds = min(self.validation.folds, int(labels.value_counts().min()))
            splitter = model_selection.StratifiedKFold(
                n_splits=max(folds, 2),
                shuffle=True,
                random_state=self.validation.seed,
            )
            parts = list(splitter.split(numpy.arange(len(dev_ids)), labels))
        else:
            parts = self.fold(frame, dev_ids, notes)
        return Partition(dev_ids, test_ids, parts, notes)


class Group(Partitioner):
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
        if not self.task.classification:
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
        chosen = self.fold(labels, dev_groups, folds, notes)
        return Partition(dev_ids, test_ids, chosen, notes)

    def fold(
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
