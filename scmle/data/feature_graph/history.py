"""Series features computed with stored history."""

import numpy
import pandas

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.feature_graph.registry import lookup
from scmle.data.feature_graph.registry import roots
from scmle.data.feature_graph.sequences import Sequential
from scmle.data.validation import columns


class Temporal:
    """Computes ordered (time-series) features on a frame.

    Features use only the current and earlier rows of the same group, ordered
    by the time column, so they are available when a prediction is made.
    """

    __module__ = "scmle.data.feature_graph"

    def __init__(
        self,
        nodes: list[tasks.Node],
        group_column: str | None,
        time_column: str | None,
        history: pandas.DataFrame | None = None,
    ) -> None:
        """Initializes the builder.

        Args:
            nodes: Ordered nodes, in dependency order.
            group_column: Entity column (None = one series).
            time_column: Column ordering the rows.
            history: Earlier rows per entity (see `retain`), used to
                fill features of rows whose own batch lacks the past.
        """
        self.nodes = nodes
        self.group = group_column
        self.timestamp = time_column
        self.history = history

    @property
    def historical(self) -> list[str]:
        """The columns a history frame holds.

        These are the group and time columns (when set) and the raw inputs of
        the nodes, without duplicates.
        """
        derived = {node.name for node in self.nodes}
        raw = roots(self.nodes, derived)
        names = [self.group, self.timestamp, *raw]
        return list(dict.fromkeys(c for c in names if c))

    @property
    def depth(self) -> int | None:
        """The number of earlier rows to keep per entity.

        This is the sum of the nodes' look-back distances, or None when any
        node needs all earlier rows.

        Raises:
            ConfigError: If a node is not an ordered operation.
        """
        total = 0
        for node in self.nodes:
            op = lookup(node.op)
            if not isinstance(op, Sequential):
                raise errors.ConfigError(f"{node.op} is not an ordered op")
            needed = op.lookback(op.resolved(node))
            if needed is None:
                return None
            total += needed
        return total

    def identify(self, frame: pandas.DataFrame) -> numpy.ndarray:
        """Computes the entity key of every row.

        Args:
            frame: The rows.

        Returns:
            One key per row, as strings; all zeros without a group column.
        """
        if self.group:
            return frame[self.group].astype(str).to_numpy()
        return numpy.zeros(len(frame), dtype=int)

    def retain(self, frame: pandas.DataFrame) -> pandas.DataFrame:
        """Keeps the rows later predictions can still need.

        Args:
            frame: Rows with at least the history columns.

        Returns:
            The most recent rows per entity in time order, enough for every
            ordered node (all rows if some node looks back without limit).
            Rows with an unparseable time are dropped. A node that is not an
            ordered operation makes `depth` raise `ConfigError`.
        """
        columns = self.historical
        subset = frame[columns].copy()
        times = pandas.to_datetime(subset[self.timestamp], errors="coerce")
        subset = subset[times.notna().to_numpy()]
        times = times[times.notna()]
        keys = self.identify(subset)
        order = numpy.lexsort(
            (numpy.arange(len(subset)), times.to_numpy(), keys)
        )
        ordered = subset.iloc[order].reset_index(drop=True)
        ordered_keys = keys[order]
        rows = self.depth
        if rows is None or ordered.empty:
            return ordered
        rank_from_end = (
            pandas.Series(numpy.arange(len(ordered)))
            .groupby(ordered_keys)
            .transform(lambda s: s.max() - s)
        )
        return ordered[(rank_from_end < rows).to_numpy()].reset_index(drop=True)

    def apply(self, frame: pandas.DataFrame) -> pandas.DataFrame:
        """Appends the ordered features to a frame.

        Args:
            frame: The rows to extend.

        Returns:
            A copy of `frame` with one column per node; `frame` itself when
            there are no nodes. Errors from `compute` or
            `backfill` propagate: `DataError` when the time
            column is missing, `ConfigError` when it has unparseable values
            or a node is not an ordered operation.
        """
        if not self.nodes:
            return frame
        if self.history is None or self.history.empty:
            return self.compute(frame)
        return self.backfill(frame)

    def backfill(self, frame: pandas.DataFrame) -> pandas.DataFrame:
        """Appends the features, adding stored history before the batch.

        Only history strictly earlier than an entity's first row in `frame`
        is used, so the batch's own rows always win on overlap.

        Args:
            frame: The rows to extend.

        Returns:
            A copy of `frame` with one column per node.

        Raises:
            DataError: If the time column is missing from the frame.
            ConfigError: If the time column has unparseable values, or a node
                is not an ordered operation.
        """
        if self.timestamp not in frame.columns:
            raise errors.DataError(
                f"time column '{self.timestamp}' is required to compute "
                "series features"
            )
        times = pandas.to_datetime(frame[self.timestamp], errors="coerce")
        if times.isna().any():
            raise errors.ConfigError(
                f"time column '{self.timestamp}' has unparseable values"
            )
        earliest = (
            pandas.Series(times.to_numpy()).groupby(self.identify(frame)).min()
        )
        history = self.history if self.history is not None else frame.iloc[:0]
        past_times = pandas.to_datetime(history[self.timestamp])
        limits = pandas.Series(self.identify(history)).map(earliest)
        keep = (past_times.to_numpy() < limits.to_numpy()) & limits.notna()
        past = history[numpy.asarray(keep)][
            [c for c in self.historical if c in frame.columns]
        ]
        combined = pandas.concat([past, frame], ignore_index=True)
        computed = self.compute(combined).iloc[len(past) :]
        result = frame.copy()
        for node in self.nodes:
            result[node.name] = computed[node.name].to_numpy()
        return result

    def compute(self, frame: pandas.DataFrame) -> pandas.DataFrame:
        """Appends the features using only the rows of `frame`.

        Args:
            frame: The rows to extend.

        Returns:
            A copy of `frame` with one column per node, in the original row
            order.

        Raises:
            DataError: If the time column is missing from the frame.
            ConfigError: If the time column has unparseable values, or a node
                is not an ordered operation.
        """
        if self.timestamp not in frame.columns:
            raise errors.DataError(
                f"time column '{self.timestamp}' is required to compute "
                "series features"
            )
        result = frame.copy()
        times = pandas.to_datetime(result[self.timestamp], errors="coerce")
        if times.isna().any():
            raise errors.ConfigError(
                f"time column '{self.timestamp}' has unparseable values"
            )
        groups = (
            result[self.group].astype(str).to_numpy()
            if self.group
            else numpy.zeros(len(result), dtype=int)
        )
        position = numpy.arange(len(result))
        order = numpy.lexsort((position, times.to_numpy(), groups))
        sorted_groups = groups[order]
        for node in self.nodes:
            values = columns.scrub(result[node.inputs[0]]).to_numpy()[order]
            grouped = pandas.Series(values).groupby(sorted_groups)
            op = lookup(node.op)
            if not isinstance(op, Sequential):
                raise errors.ConfigError(f"{node.op} is not an ordered op")
            computed = op.compute(grouped, op.resolved(node))
            restored = numpy.empty(len(result))
            restored[order] = numpy.asarray(computed, dtype=float)
            result[node.name] = restored
        return result
