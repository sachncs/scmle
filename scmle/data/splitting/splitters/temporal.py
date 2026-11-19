"""The time-ordered split strategies: train on the past, test on the future.

`Chronology` holds out the latest rows and folds the development rows in
expanding windows; `Rolling` slides fixed windows forward. Both keep tied
timestamps, and entities declared unseen, on one side of every split.
`Succession` splits whole groups in the order they occur, each group on one
side of the test split and of every fold.
"""

from collections.abc import Hashable

import numpy
import pandas
from sklearn import model_selection

from scmle.contracts import errors
from scmle.data.splitting.splitters.partitions import Fold
from scmle.data.splitting.splitters.partitions import MIN_FOLDS
from scmle.data.splitting.splitters.partitions import MIN_ROLLING_FOLDS
from scmle.data.splitting.splitters.partitions import MIN_ROLLING_TRAIN
from scmle.data.splitting.splitters.partitions import Partition
from scmle.data.splitting.splitters.partitions import Partitioner


def untie(times: numpy.ndarray, folds: list[Fold], least: int) -> list[Fold]:
    """Keeps every fold's training rows strictly before its validation rows.

    A time-ordered fold cut by row count can put rows of one timestamp in
    both its training and its validation part; those training rows are
    dropped, so no timestamp sits on both sides.

    Args:
        times: Timestamp of each development position.
        folds: `(train, validation)` position arrays.
        least: Fewest training rows a fold may keep.

    Returns:
        The folds in order, each training part cut to rows earlier than its
        validation part's first timestamp; a fold left with fewer than
        `least` training rows is dropped.
    """
    kept = []
    for fit, held in folds:
        trimmed = fit[times[fit] < times[held].min()]
        if len(trimmed) >= least:
            kept.append((trimmed, held))
    return kept


def apart(
    entities: numpy.ndarray, folds: list[Fold], least: int, seed: int
) -> list[Fold]:
    """Keeps every entity on one side of every fold.

    The development entities are dealt at random into one team per fold;
    fold `k` validates on team `k` alone and trains on the other teams, so
    no entity is in both parts of a fold and each is validated in one fold.

    Args:
        entities: Entity of each development position.
        folds: `(train, validation)` position arrays.
        least: Fewest training rows a fold may keep.
        seed: Seed of the deal.

    Returns:
        The folds in order, each cut to its team's validation rows and the
        other teams' training rows; a fold left with fewer than `least`
        training rows or no validation row is dropped.
    """
    if not folds:
        return []
    names = sorted(set(entities.tolist()), key=str)
    order = numpy.random.default_rng(seed).permutation(len(names))
    team = {names[i]: rank % len(folds) for rank, i in enumerate(order)}
    labels = numpy.array([team[e] for e in entities.tolist()])
    kept = []
    for index, (fit, held) in enumerate(folds):
        trimmed = fit[labels[fit] != index]
        chosen = held[labels[held] == index]
        if len(trimmed) >= least and len(chosen):
            kept.append((trimmed, chosen))
    return kept


class Chronology(Partitioner):
    """Trains on the past and evaluates on the future.

    With a `group_column`, `validation.entities` decides whether one entity
    may sit on both sides: `known` (a forecast of the entities trained on)
    lets its past train and its future be held out; `unseen` keeps every
    entity wholly on one side of the test split and of each fold.
    """

    name = "temporal"

    def embargo(self, frame: pandas.DataFrame) -> int:
        """Rows dropped between training and later windows.

        A forecast row's target looks `horizon` steps ahead, so the rows just
        before a window would otherwise train on the window's own values; the
        embargo covers `horizon` time steps of every series.

        Args:
            frame: The full dataset.

        Returns:
            The configured embargo rows when no horizon is set; otherwise the
            larger of that and the horizon times the number of series (the
            groups, or one).
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

    def times(self, frame: pandas.DataFrame) -> numpy.ndarray:
        """Parses the time column.

        Args:
            frame: The full dataset.

        Returns:
            One timestamp per row of the frame.

        Raises:
            BlockedError: If a value cannot be parsed or is missing.
        """
        column = self.data.time_column
        parsed = pandas.to_datetime(frame[column], errors="coerce")
        if parsed.isna().any():
            raise errors.BlockedError(
                f"time column '{column}' has unparseable/missing values: "
                "ordering ambiguous"
            )
        return parsed.to_numpy()

    def entities(self, frame: pandas.DataFrame) -> numpy.ndarray | None:
        """The entity of each row when the task predicts unseen entities.

        Args:
            frame: The full dataset.

        Returns:
            The `group_column` values when `validation.entities` is
            `unseen`; None for a forecast of known entities, whose past may
            train and future be held out.

        Raises:
            BlockedError: If the entity column has missing values or fewer
                than `folds + 2` entities (one team per fold, the held-out
                entities and at least one to train on).
        """
        column = self.data.group_column
        if self.validation.entities != "unseen" or not column:
            return None
        values = frame[column].to_numpy()
        if pandas.isna(values).any():
            raise errors.BlockedError(
                f"group column '{column}' has missing values"
            )
        least = self.validation.folds + 2
        if len(set(values.tolist())) < least:
            raise errors.BlockedError(
                f"validation.entities 'unseen' needs at least {least} "
                f"entities in '{column}' to keep each on one side of every "
                "split"
            )
        return values

    def remark(self) -> list[str]:
        """States the entity declaration the split followed.

        Returns:
            A note when an entity column is declared; empty otherwise.
        """
        column = self.data.group_column
        if not column:
            return []
        if self.validation.entities == "unseen":
            text = (
                f"entities declared unseen: every entity of '{column}' sits "
                "wholly on one side of the test split and of every fold; "
                "rows that would put an entity on both sides are left out"
            )
        else:
            text = (
                f"entities declared known: an entity of '{column}' may train "
                "on its past and be validated on its future; declare "
                "validation.entities 'unseen' to predict new entities"
            )
        return [text]

    def chronological_holdout(
        self, frame: pandas.DataFrame
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Orders rows in time and holds out the latest for the final test.

        The test window starts at a timestamp, not inside one: when the
        planned cut falls inside a run of rows with one timestamp, the whole
        run goes to the test set, so no timestamp sits on both sides.

        Args:
            frame: The full dataset.

        With entities declared unseen, a seeded `test_fraction` of the
        entities is held out: the test rows are their rows in the test
        window and the development rows are the other entities' rows
        before it.

        Returns:
            Development row positions (oldest first, an embargo before the
            test window removed) and test row positions.

        Raises:
            BlockedError: If the time column cannot order the rows, every
                row before the planned cut shares the test window's first
                timestamp (no earlier timestamp to develop on), or with
                entities declared unseen, either side is left empty.
        """
        times = self.times(frame)
        order = numpy.argsort(times, kind="stable")
        ordered = times[order]
        rows = len(frame)
        test_rows = max(1, round(rows * self.validation.test_fraction))
        start = int(
            numpy.searchsorted(ordered, ordered[rows - test_rows], "left")
        )
        if start == 0:
            raise errors.BlockedError(
                f"time column '{self.data.time_column}' has no timestamp "
                "before the test window: every development row would tie "
                "with a test row"
            )
        embargo = self.embargo(frame)
        dev_ids, test_ids = order[: max(start - embargo, 0)], order[start:]
        entities = self.entities(frame)
        if entities is None:
            return dev_ids, test_ids
        names = sorted(set(entities.tolist()), key=str)
        count = max(1, round(len(names) * self.validation.test_fraction))
        rng = numpy.random.default_rng(self.validation.seed)
        held = [names[i] for i in rng.choice(len(names), count, replace=False)]
        dev_ids = dev_ids[~numpy.isin(entities[dev_ids], held)]
        test_ids = test_ids[numpy.isin(entities[test_ids], held)]
        if not len(dev_ids) or not len(test_ids):
            raise errors.BlockedError(
                "validation.entities 'unseen': the held-out entities have no "
                "rows in the test window, or the others none before it"
            )
        return dev_ids, test_ids

    def partition(self, frame: pandas.DataFrame) -> Partition:
        """Chronological holdout and expanding-window folds.

        Args:
            frame: The full dataset.

        Returns:
            The development and test positions, the folds listed latest
            window first, and notes on the embargo and fold order.

        Raises:
            BlockedError: If the time column cannot order the rows or there
                are too few development rows for the folds and embargo.
        """
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
        notes += self.remark()
        windows = untie(
            self.times(frame)[dev_ids],
            list(splitter.split(numpy.arange(len(dev_ids)))),
            1,
        )
        entities = self.entities(frame)
        if entities is not None:
            windows = apart(entities[dev_ids], windows, 1, self.validation.seed)
        if len(windows) < MIN_FOLDS:
            raise errors.BlockedError(
                f"only {len(windows)} temporal fold(s) keep tied timestamps "
                "and declared unseen entities on one side; add data or use "
                "fewer folds"
            )
        return Partition(dev_ids, test_ids, windows[::-1], notes)


class Rolling(Chronology):
    """Rolling-origin evaluation: slide forward, validate on what follows.

    Each fold trains on the rows before a cut-off (at most `window_rows` of
    them when set, otherwise all earlier rows), skips `embargo_rows`, and
    validates on the next `horizon_rows`. Successive folds move the cut-off
    back by one horizon, so every validation window is genuinely later than
    its training window and the folds cover the most recent history.
    """

    name = "rolling"

    def partition(self, frame: pandas.DataFrame) -> Partition:
        """Chronological holdout and sliding training/validation windows.

        Args:
            frame: The full dataset.

        Returns:
            The development and test positions, the rolling windows listed
            latest first as `(train, validation)` position arrays, and notes
            describing them.

        Raises:
            BlockedError: If the time column cannot order the rows or fewer
                than two windows fit.
        """
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
                    numpy.arange(train_start, train_end),
                    numpy.arange(held_start, held_end),
                )
            )
        windows = untie(self.times(frame)[dev_ids], windows, MIN_ROLLING_TRAIN)
        entities = self.entities(frame)
        if entities is not None:
            windows = apart(
                entities[dev_ids],
                windows,
                MIN_ROLLING_TRAIN,
                self.validation.seed,
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
            *self.remark(),
        ]
        return Partition(dev_ids, test_ids, windows, notes)


class Succession(Chronology):
    """Grouped temporal split: whole groups, taken in the order they occur.

    A group (a batch, a session, a store's week) is ordered by its first
    timestamp, then its last, then its name, and sits wholly on one side of
    the test split and of every fold. The latest groups are the test set;
    the development groups are cut into `folds + 1` runs in that order and
    fold `k` validates on run `k` after training on the runs before it, as
    scikit-learn's `TimeSeriesSplit` does with rows. A training row is kept
    only when it is earlier than every row it is checked against, and the
    embargo then drops the latest training rows, so time order holds even
    where groups overlap in time.
    """

    name = "grouped_temporal"

    def labels(self, frame: pandas.DataFrame) -> numpy.ndarray:
        """The group of each row.

        Args:
            frame: The full dataset.

        Returns:
            The `group_column` values.

        Raises:
            BlockedError: If the group or time column is not declared, or
                the group column has missing values.
        """
        column = self.data.group_column
        if not column or not self.data.time_column:
            raise errors.BlockedError(
                "grouped_temporal validation needs data.group_column and "
                "data.time_column"
            )
        values = frame[column].to_numpy()
        if pandas.isna(values).any():
            raise errors.BlockedError(
                f"group column '{column}' has missing values"
            )
        return values

    def spans(
        self, times: numpy.ndarray, labels: numpy.ndarray
    ) -> pandas.DataFrame:
        """Each group's first and last timestamp.

        Args:
            times: Timestamp of each row.
            labels: Group of each row.

        Returns:
            One row per group, indexed by group, with columns `min` and
            `max`.
        """
        rows = pandas.DataFrame({"group": labels, "time": times})
        return rows.groupby("group", sort=False)["time"].agg(["min", "max"])

    def sequence(
        self, times: numpy.ndarray, labels: numpy.ndarray
    ) -> list[Hashable]:
        """The groups in the order they occur.

        Args:
            times: Timestamp of each row.
            labels: Group of each row.

        Returns:
            The group names by first timestamp, then last, then name.
        """
        bounds = self.spans(times, labels)
        bounds["name"] = [str(name) for name in bounds.index]
        ordered = bounds.sort_values(["min", "max", "name"], kind="stable")
        return list(ordered.index)

    def concurrency(self, frame: pandas.DataFrame) -> int:
        """The most groups running at one time.

        Two groups run at once when their spans from first to last
        timestamp meet. The count is greatest at some group's first
        timestamp: the groups started by then less those already ended.

        Args:
            frame: The full dataset.

        Returns:
            The largest number of groups whose spans hold one timestamp.

        Raises:
            BlockedError: If the group or time column is missing or
                unusable.
        """
        bounds = self.spans(self.times(frame), self.labels(frame))
        starts = numpy.sort(bounds["min"].to_numpy())
        ends = numpy.sort(bounds["max"].to_numpy())
        started = numpy.searchsorted(starts, starts, "right")
        ended = numpy.searchsorted(ends, starts, "left")
        return int(numpy.max(started - ended))

    def embargo(self, frame: pandas.DataFrame) -> int:
        """Rows dropped before the test window and each validation window.

        A forecast row's target looks `horizon` steps ahead. Groups that
        follow one another are one series cut into pieces, so `horizon`
        time steps cover `horizon` rows of each group running at that time,
        not of every group.

        Args:
            frame: The full dataset.

        Returns:
            The configured embargo rows when no horizon is set; otherwise
            the larger of that and the horizon times `concurrency`.

        Raises:
            BlockedError: If the group or time column is missing or
                unusable.
        """
        configured = self.validation.embargo_rows
        horizon = self.data.horizon
        if horizon is None:
            return configured
        return max(configured, horizon * self.concurrency(frame))

    def window(
        self,
        clock: numpy.ndarray,
        fit: numpy.ndarray,
        held: numpy.ndarray,
        embargo: int,
    ) -> numpy.ndarray:
        """Training positions strictly before the checked ones, less embargo.

        Args:
            clock: Timestamp of each position, ascending.
            fit: Candidate training positions, ascending.
            held: Positions checked against.
            embargo: Latest training rows to drop.

        Returns:
            The candidates earlier than every checked position, without the
            latest `embargo` of them.
        """
        earlier: numpy.ndarray = fit[clock[fit] < clock[held].min()]
        return earlier[: max(len(earlier) - embargo, 0)]

    def partition(self, frame: pandas.DataFrame) -> Partition:
        """Group holdout at the end of time and expanding group-run folds.

        Args:
            frame: The full dataset.

        Returns:
            The development positions (oldest first), the test positions,
            the folds latest window first, and notes.

        Raises:
            BlockedError: If the columns are missing or unparseable, there
                are fewer than `round(groups * test_fraction) + folds + 1`
                groups, or fewer than `MIN_FOLDS` folds keep a training row.
        """
        labels = self.labels(frame)
        times = self.times(frame)
        names = self.sequence(times, labels)
        folds = self.validation.folds
        count = max(1, round(len(names) * self.validation.test_fraction))
        least = count + folds + 1
        if len(names) < least:
            raise errors.BlockedError(
                f"grouped_temporal validation needs at least {least} groups "
                f"({count} to test, one run per fold and one to train first); "
                f"'{self.data.group_column}' has {len(names)}"
            )
        kept = names[:-count]
        tested = pandas.Series(labels).isin(names[-count:]).to_numpy()
        test_ids = numpy.flatnonzero(tested)
        test_ids = test_ids[numpy.argsort(times[test_ids], kind="stable")]
        rest = numpy.flatnonzero(~tested)
        rest = rest[numpy.argsort(times[rest], kind="stable")]
        embargo = self.embargo(frame)
        dev_ids = self.window(times, rest, test_ids, embargo)
        clock, local = times[dev_ids], pandas.Series(labels[dev_ids])
        everything = numpy.arange(len(dev_ids))
        runs = numpy.array_split(numpy.arange(len(kept)), folds + 1)
        windows = []
        for index in range(1, folds + 1):
            before = [kept[i] for run in runs[:index] for i in run]
            held = everything[local.isin([kept[i] for i in runs[index]])]
            if not len(held):
                continue
            fit = self.window(
                clock, everything[local.isin(before)], held, embargo
            )
            if len(fit):
                windows.append((fit, held))
        if len(windows) < MIN_FOLDS:
            raise errors.BlockedError(
                f"only {len(windows)} grouped temporal fold(s) keep a training "
                "row earlier than their validation groups; the groups overlap "
                "in time (use temporal with validation.entities 'unseen' for "
                "series that run side by side) or add groups"
            )
        summary = (
            f"grouped temporal: {len(names)} groups of "
            f"'{self.data.group_column}' in order of first timestamp; the "
            f"latest {count} are the test set and {len(windows)} folds "
            "validate later runs of groups than they train on"
        )
        notes = [summary, "folds are listed latest window first"]
        overlap = int((times[rest] >= times[test_ids].min()).sum())
        if overlap:
            notes.append(
                f"{overlap} row(s) of development groups fall at or after the "
                "test window's first timestamp and are left out"
            )
        if embargo:
            notes.append(
                f"embargo of {embargo} rows before the test window and before "
                "each validation window"
            )
        return Partition(dev_ids, test_ids, windows[::-1], notes)
