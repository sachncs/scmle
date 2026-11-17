"""Chronological holdout and rolling-origin partitioners."""

import numpy
import pandas
from sklearn import model_selection

from scmle.contracts import errors
from scmle.data.splitting.splitters.partitions import MIN_ROLLING_FOLDS
from scmle.data.splitting.splitters.partitions import MIN_ROLLING_TRAIN
from scmle.data.splitting.splitters.partitions import Partition
from scmle.data.splitting.splitters.partitions import Partitioner


class Chronology(Partitioner):
    """Trains on the past and evaluates on the future."""

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

    def chronological_holdout(
        self, frame: pandas.DataFrame
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Orders rows in time and holds out the latest for the final test.

        Args:
            frame: The full dataset.

        Returns:
            Development row positions (oldest first, an embargo before the
            test window removed) and test row positions.

        Raises:
            BlockedError: If the time column cannot order the rows.
        """
        column = self.data.time_column
        times = pandas.to_datetime(frame[column], errors="coerce")
        if times.isna().any():
            raise errors.BlockedError(
                f"time column '{column}' has unparseable/missing values: "
                "ordering ambiguous"
            )
        order = numpy.argsort(times.to_numpy(), kind="stable")
        rows = len(frame)
        test_rows = max(1, round(rows * self.validation.test_fraction))
        embargo = self.embargo(frame)
        return order[: rows - test_rows - embargo], order[rows - test_rows :]

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
        windows = list(splitter.split(numpy.arange(len(dev_ids))))
        return Partition(dev_ids, test_ids, windows[::-1], notes)


class RollingSplit(Chronology):
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
