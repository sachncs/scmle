"""The feature-transformation graph: operations, fitting and execution.

Two kinds of operation exist:

* **Row-wise** operations (`log1p`, `ratio`, `bin`, `datepart`, ...) run
  inside the candidate pipeline. Operations that need statistics (`clip`,
  `bin`, `standardize`) learn them in `fit`, i.e. from the training fold only.
* **Ordered** operations (`lag`, `diff`, `rolling_mean`, ...) need row order.
  They are computed per group in time order, using only the current and past
  rows (never future ones, and never the target), at ingestion and again at
  inference. A saved model carries the recent history it needs (the last few
  rows of every entity), so even a single new record gets filled-in lag and
  rolling features; history rows from the batch itself take precedence.
"""

import abc
from collections.abc import Mapping
from typing import Any
from typing import ClassVar
from typing import Self

import numpy as np
import pandas as pd
from sklearn import base

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.validation import columns

MAX_ARITY = 64
Inputs = list[pd.Series]
Params = Mapping[str, Any]


EPOCH = pd.Timestamp("1970-01-01")


class FeatureOp(abc.ABC):
    """One operation of the graph.

    Attributes:
        name: Name used in `FeatureNode.op`.
        min_inputs: Minimum number of inputs.
        max_inputs: Maximum number of inputs.
        ordered: Whether the operation needs row order (time series).
        defaults: Parameters and their defaults; unknown parameters are
            rejected.
    """

    name: ClassVar[str]
    min_inputs: ClassVar[int] = 1
    max_inputs: ClassVar[int] = 1
    ordered: ClassVar[bool] = False
    defaults: ClassVar[Mapping[str, Any]] = {}

    def validate(self, node: tasks.FeatureNode) -> None:
        """Checks arity and parameters.

        Raises:
            ConfigError: For a wrong input count or unknown parameter.
        """
        count = len(node.inputs)
        if not self.min_inputs <= count <= self.max_inputs:
            raise errors.ConfigError(
                f"node '{node.name}': {self.name} takes "
                f"{self.min_inputs}..{self.max_inputs} inputs, got {count}"
            )
        unknown = set(node.params) - set(self.defaults)
        if unknown:
            raise errors.ConfigError(
                f"node '{node.name}': unknown parameters {sorted(unknown)} "
                f"for {self.name}; allowed: {sorted(self.defaults)}"
            )

    def resolved(self, node: tasks.FeatureNode) -> dict[str, Any]:
        """Parameters with defaults applied."""
        return {**self.defaults, **node.params}

    def fit(self, inputs: Inputs, params: Params) -> Any:
        """Learns state from training data; default is stateless."""
        return None

    @abc.abstractmethod
    def apply(self, inputs: Inputs, params: Params, state: Any) -> pd.Series:
        """Computes the derived column."""


def numeric(series: pd.Series) -> pd.Series:
    """Coerces to float; unparseable and infinite values become NaN."""
    return columns.to_finite(series)


class Log1p(FeatureOp):
    """`log(1 + max(x, 0))`."""

    name = "log1p"

    def apply(self, inputs: Inputs, params: Params, state: Any) -> pd.Series:
        """Log-transforms the clipped input."""
        return np.log1p(numeric(inputs[0]).clip(lower=0))


class Sqrt(FeatureOp):
    """`sqrt(max(x, 0))`."""

    name = "sqrt"

    def apply(self, inputs: Inputs, params: Params, state: Any) -> pd.Series:
        """Square root of the clipped input."""
        return np.sqrt(numeric(inputs[0]).clip(lower=0))


class Square(FeatureOp):
    """`x ** 2`."""

    name = "square"

    def apply(self, inputs: Inputs, params: Params, state: Any) -> pd.Series:
        """Squares the input."""
        return numeric(inputs[0]) ** 2


class Absolute(FeatureOp):
    """`abs(x)`."""

    name = "abs"

    def apply(self, inputs: Inputs, params: Params, state: Any) -> pd.Series:
        """Absolute value."""
        return numeric(inputs[0]).abs()


class IsMissing(FeatureOp):
    """1 where the input is missing, else 0."""

    name = "is_missing"

    def apply(self, inputs: Inputs, params: Params, state: Any) -> pd.Series:
        """Missing-value indicator (works for any dtype)."""
        return inputs[0].isna().astype(float)


class Sum(FeatureOp):
    """Sum of two or more inputs."""

    name = "sum"
    min_inputs = 2
    max_inputs = MAX_ARITY

    def apply(self, inputs: Inputs, params: Params, state: Any) -> pd.Series:
        """Adds the inputs."""
        return sum(numeric(s) for s in inputs[1:]) + numeric(inputs[0])


class Difference(FeatureOp):
    """`a - b`."""

    name = "difference"
    min_inputs = 2
    max_inputs = 2

    def apply(self, inputs: Inputs, params: Params, state: Any) -> pd.Series:
        """Subtracts the second input from the first."""
        return numeric(inputs[0]) - numeric(inputs[1])


class Product(FeatureOp):
    """Product of two or more inputs."""

    name = "product"
    min_inputs = 2
    max_inputs = MAX_ARITY

    def apply(self, inputs: Inputs, params: Params, state: Any) -> pd.Series:
        """Multiplies the inputs."""
        result = numeric(inputs[0])
        for series in inputs[1:]:
            result = result * numeric(series)
        return result


class Ratio(FeatureOp):
    """`a / b`, NaN where b is zero."""

    name = "ratio"
    min_inputs = 2
    max_inputs = 2

    def apply(self, inputs: Inputs, params: Params, state: Any) -> pd.Series:
        """Divides the first input by the second."""
        denominator = numeric(inputs[1]).replace(0.0, np.nan)
        return numeric(inputs[0]) / denominator


class Clip(FeatureOp):
    """Winsorizes to quantiles learned on the training fold."""

    name = "clip"
    defaults = {"lower": 0.01, "upper": 0.99}

    def fit(self, inputs: Inputs, params: Params) -> Any:
        """Learns the clipping bounds from training data."""
        values = numeric(inputs[0]).dropna()
        if values.empty:
            return (-np.inf, np.inf)
        low, high = np.quantile(values, [params["lower"], params["upper"]])
        return float(low), float(high)

    def apply(self, inputs: Inputs, params: Params, state: Any) -> pd.Series:
        """Clips to the learned bounds."""
        return numeric(inputs[0]).clip(lower=state[0], upper=state[1])


class Standardize(FeatureOp):
    """Z-score using the training fold's mean and standard deviation."""

    name = "standardize"

    def fit(self, inputs: Inputs, params: Params) -> Any:
        """Learns mean and std from training data."""
        values = numeric(inputs[0]).dropna()
        spread = float(values.std()) if len(values) > 1 else 0.0
        return (float(values.mean()) if len(values) else 0.0, spread or 1.0)

    def apply(self, inputs: Inputs, params: Params, state: Any) -> pd.Series:
        """Standardizes with the learned statistics."""
        return (numeric(inputs[0]) - state[0]) / state[1]


class Bin(FeatureOp):
    """Quantile bins (ordinal codes) with edges learned on the training fold."""

    name = "bin"
    defaults = {"bins": 5}

    def fit(self, inputs: Inputs, params: Params) -> Any:
        """Learns interior quantile edges from training data."""
        values = numeric(inputs[0]).dropna().to_numpy()
        if values.size == 0:
            return np.array([])
        edges = np.quantile(
            values, np.linspace(0, 1, int(params["bins"]) + 1)[1:-1]
        )
        return np.unique(edges)

    def apply(self, inputs: Inputs, params: Params, state: Any) -> pd.Series:
        """Assigns each value to its bin (NaN stays NaN)."""
        values = numeric(inputs[0])
        codes = np.searchsorted(state, values.to_numpy(), side="right").astype(
            float
        )
        return pd.Series(
            np.where(values.isna(), np.nan, codes), index=values.index
        )


class DatePart(FeatureOp):
    """A component of a datetime column."""

    name = "datepart"
    defaults = {"part": "month"}
    parts: ClassVar[frozenset[str]] = frozenset(
        {
            "year",
            "quarter",
            "month",
            "day",
            "dayofweek",
            "dayofyear",
            "hour",
            "minute",
            "timestamp",
        }
    )

    def validate(self, node: tasks.FeatureNode) -> None:
        """Also checks the requested part."""
        super().validate(node)
        part = self.resolved(node)["part"]
        if part not in self.parts:
            raise errors.ConfigError(
                f"node '{node.name}': datepart part must be one of "
                f"{sorted(self.parts)}, got '{part}'"
            )

    def apply(self, inputs: Inputs, params: Params, state: Any) -> pd.Series:
        """Extracts the part; unparseable dates give NaN."""
        stamps = pd.to_datetime(inputs[0], errors="coerce")
        if params["part"] == "timestamp":
            return (stamps - EPOCH) / pd.Timedelta(days=1)
        return getattr(stamps.dt, params["part"]).astype(float)


class OrderedOp(FeatureOp):
    """An operation over time-ordered values within a group."""

    ordered = True

    def apply(self, inputs: Inputs, params: Params, state: Any) -> pd.Series:
        """Ordered operations run through `SeriesFeatureBuilder` only."""
        raise errors.ConfigError(f"{self.name} is an ordered operation")

    @abc.abstractmethod
    def compute(self, grouped: Any, params: Mapping[str, Any]) -> pd.Series:
        """Computes the feature from values already sorted by time per group."""

    @abc.abstractmethod
    def lookback(self, params: Mapping[str, Any]) -> int | None:
        """Earlier rows needed per entity (None = all of them)."""


class Lag(OrderedOp):
    """The value `k` rows earlier within the group."""

    name = "lag"
    defaults = {"k": 1}

    def validate(self, node: tasks.FeatureNode) -> None:
        """Requires a positive lag so no future value can leak."""
        super().validate(node)
        if int(self.resolved(node)["k"]) < 1:
            raise errors.ConfigError(f"node '{node.name}': lag k must be >= 1")

    def compute(self, grouped: Any, params: Params) -> pd.Series:
        """Shifts forward in time."""
        return grouped.shift(int(params["k"]))

    def lookback(self, params: Params) -> int | None:
        """The lag distance."""
        return int(params["k"])


class Diff(OrderedOp):
    """Change versus `k` rows earlier within the group."""

    name = "diff"
    defaults = {"k": 1}

    def validate(self, node: tasks.FeatureNode) -> None:
        """Requires a positive lag."""
        super().validate(node)
        if int(self.resolved(node)["k"]) < 1:
            raise errors.ConfigError(f"node '{node.name}': diff k must be >= 1")

    def compute(self, grouped: Any, params: Params) -> pd.Series:
        """Current value minus the lagged one."""
        return grouped.transform(lambda s: s - s.shift(int(params["k"])))

    def lookback(self, params: Params) -> int | None:
        """The lag distance."""
        return int(params["k"])


class RollingMean(OrderedOp):
    """Mean of the last `window` rows (including the current one)."""

    name = "rolling_mean"
    defaults = {"window": 3}

    def compute(self, grouped: Any, params: Params) -> pd.Series:
        """Trailing mean."""
        window = int(params["window"])
        return grouped.transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )

    def lookback(self, params: Params) -> int | None:
        """The window."""
        return int(params["window"])


class RollingStd(OrderedOp):
    """Standard deviation of the last `window` rows."""

    name = "rolling_std"
    defaults = {"window": 3}

    def compute(self, grouped: Any, params: Params) -> pd.Series:
        """Trailing standard deviation (needs two observations)."""
        window = int(params["window"])
        return grouped.transform(
            lambda s: s.rolling(window, min_periods=2).std()
        )

    def lookback(self, params: Params) -> int | None:
        """The window."""
        return int(params["window"])


class ExpandingMean(OrderedOp):
    """Mean of all rows up to and including the current one."""

    name = "expanding_mean"

    def compute(self, grouped: Any, params: Params) -> pd.Series:
        """Running mean."""
        return grouped.transform(lambda s: s.expanding().mean())

    def lookback(self, params: Params) -> int | None:
        """Every earlier row contributes."""
        return None


OPS: dict[str, FeatureOp] = {
    op.name: op
    for op in (
        Log1p(),
        Sqrt(),
        Square(),
        Absolute(),
        IsMissing(),
        Sum(),
        Difference(),
        Product(),
        Ratio(),
        Clip(),
        Standardize(),
        Bin(),
        DatePart(),
        Lag(),
        Diff(),
        RollingMean(),
        RollingStd(),
        ExpandingMean(),
    )
}


def get_op(name: str) -> FeatureOp:
    """Returns a registered operation.

    Raises:
        ConfigError: If the name is unknown.
    """
    if name not in OPS:
        raise errors.ConfigError(
            f"unknown feature operation '{name}'; available: {sorted(OPS)}"
        )
    return OPS[name]


def split_nodes(
    spec: tasks.FeatureGraphSpec,
) -> tuple[list[tasks.FeatureNode], list[tasks.FeatureNode]]:
    """Separates ordered (series) nodes from row-wise nodes."""
    ordered = [n for n in spec.nodes if get_op(n.op).ordered]
    rowwise = [n for n in spec.nodes if not get_op(n.op).ordered]
    return ordered, rowwise


def validate_graph(
    spec: tasks.FeatureGraphSpec,
    available: set[str],
    data: tasks.DataSpec,
) -> None:
    """Validates a graph against the dataset's columns and the data spec.

    Args:
        spec: The graph.
        available: Raw column names in the dataset.
        data: Column roles (target, group, time).

    Raises:
        ConfigError: For unknown operations, bad arity or parameters, missing
            inputs, use of the target, name clashes, ordered operations
            without a time column, or ordered nodes consuming row-wise ones.
    """
    known = set(available)
    ordered_names: set[str] = set()
    rowwise_names: set[str] = set()
    for node in spec.nodes:
        op = get_op(node.op)
        op.validate(node)
        if node.name in available:
            raise errors.ConfigError(
                f"node '{node.name}' clashes with an existing column"
            )
        if data.target in node.inputs:
            raise errors.ConfigError(
                f"node '{node.name}' uses the target as an input (leakage)"
            )
        missing = [c for c in node.inputs if c not in known]
        if missing:
            raise errors.ConfigError(
                f"node '{node.name}' uses unknown columns {missing}"
            )
        if op.ordered:
            if not data.time_column:
                raise errors.ConfigError(
                    f"node '{node.name}': {node.op} needs data.time_column"
                )
            if set(node.inputs) & rowwise_names:
                raise errors.ConfigError(
                    f"ordered node '{node.name}' cannot consume row-wise nodes"
                )
            ordered_names.add(node.name)
        else:
            rowwise_names.add(node.name)
        known.add(node.name)


def required_raw_inputs(
    nodes: list[tasks.FeatureNode], derived: set[str]
) -> list[str]:
    """Raw columns (not produced by the graph) that `nodes` read."""
    needed: list[str] = []
    for node in nodes:
        for column in node.inputs:
            if column not in derived and column not in needed:
                needed.append(column)
    return needed


class SeriesFeatureBuilder:
    """Computes ordered (time-series) features on a frame.

    Features use only the current and earlier rows of the same group, ordered
    by the time column, so they are available when a prediction is made.
    """

    def __init__(
        self,
        nodes: list[tasks.FeatureNode],
        group_column: str | None,
        time_column: str | None,
        history: pd.DataFrame | None = None,
    ) -> None:
        """Initializes the builder.

        Args:
            nodes: Ordered nodes, in dependency order.
            group_column: Entity column (None = one series).
            time_column: Column ordering the rows.
            history: Earlier rows per entity (see `build_history`), used to
                fill features of rows whose own batch lacks the past.
        """
        self.nodes = nodes
        self.group_column = group_column
        self.time_column = time_column
        self.history = history

    @property
    def history_columns(self) -> list[str]:
        """Columns a history frame holds."""
        derived = {node.name for node in self.nodes}
        raw = required_raw_inputs(self.nodes, derived)
        names = [self.group_column, self.time_column, *raw]
        return list(dict.fromkeys(c for c in names if c))

    @property
    def history_rows(self) -> int | None:
        """Earlier rows to keep per entity (None = all)."""
        total = 0
        for node in self.nodes:
            op = get_op(node.op)
            if not isinstance(op, OrderedOp):
                raise errors.ConfigError(f"{node.op} is not an ordered op")
            needed = op.lookback(op.resolved(node))
            if needed is None:
                return None
            total += needed
        return total

    def entity_keys(self, frame: pd.DataFrame) -> np.ndarray:
        """Entity key per row (one shared key without a group column)."""
        if self.group_column:
            return frame[self.group_column].astype(str).to_numpy()
        return np.zeros(len(frame), dtype=int)

    def build_history(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Keeps the rows later predictions can still need.

        Args:
            frame: Rows with at least the history columns.

        Returns:
            The most recent rows per entity in time order, enough for every
            ordered node (all rows if some node looks back without limit).
        """
        columns = self.history_columns
        subset = frame[columns].copy()
        times = pd.to_datetime(subset[self.time_column], errors="coerce")
        subset = subset[times.notna().to_numpy()]
        times = times[times.notna()]
        keys = self.entity_keys(subset)
        order = np.lexsort((np.arange(len(subset)), times.to_numpy(), keys))
        ordered = subset.iloc[order].reset_index(drop=True)
        ordered_keys = keys[order]
        rows = self.history_rows
        if rows is None or ordered.empty:
            return ordered
        rank_from_end = (
            pd.Series(np.arange(len(ordered)))
            .groupby(ordered_keys)
            .transform(lambda s: s.max() - s)
        )
        return ordered[(rank_from_end < rows).to_numpy()].reset_index(drop=True)

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Returns a copy of `frame` with the ordered features appended.

        Raises:
            ConfigError: If the time column cannot be parsed.
            DataError: If the time column is missing from the frame.
        """
        if not self.nodes:
            return frame
        if self.history is None or self.history.empty:
            return self.compute(frame)
        return self.compute_with_history(frame)

    def compute_with_history(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Appends the features, adding stored history before the batch.

        Only history strictly earlier than an entity's first row in `frame`
        is used, so the batch's own rows always win on overlap.
        """
        if self.time_column not in frame.columns:
            raise errors.DataError(
                f"time column '{self.time_column}' is required to compute "
                "series features"
            )
        times = pd.to_datetime(frame[self.time_column], errors="coerce")
        if times.isna().any():
            raise errors.ConfigError(
                f"time column '{self.time_column}' has unparseable values"
            )
        earliest = (
            pd.Series(times.to_numpy()).groupby(self.entity_keys(frame)).min()
        )
        history = self.history if self.history is not None else frame.iloc[:0]
        past_times = pd.to_datetime(history[self.time_column])
        limits = pd.Series(self.entity_keys(history)).map(earliest)
        keep = (past_times.to_numpy() < limits.to_numpy()) & limits.notna()
        past = history[np.asarray(keep)][
            [c for c in self.history_columns if c in frame.columns]
        ]
        combined = pd.concat([past, frame], ignore_index=True)
        computed = self.compute(combined).iloc[len(past) :]
        result = frame.copy()
        for node in self.nodes:
            result[node.name] = computed[node.name].to_numpy()
        return result

    def compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Appends the features using only the rows of `frame`."""
        if self.time_column not in frame.columns:
            raise errors.DataError(
                f"time column '{self.time_column}' is required to compute "
                "series features"
            )
        result = frame.copy()
        times = pd.to_datetime(result[self.time_column], errors="coerce")
        if times.isna().any():
            raise errors.ConfigError(
                f"time column '{self.time_column}' has unparseable values"
            )
        groups = (
            result[self.group_column].astype(str).to_numpy()
            if self.group_column
            else np.zeros(len(result), dtype=int)
        )
        position = np.arange(len(result))
        order = np.lexsort((position, times.to_numpy(), groups))
        sorted_groups = groups[order]
        for node in self.nodes:
            values = numeric(result[node.inputs[0]]).to_numpy()[order]
            grouped = pd.Series(values).groupby(sorted_groups)
            op = get_op(node.op)
            if not isinstance(op, OrderedOp):
                raise errors.ConfigError(f"{node.op} is not an ordered op")
            computed = op.compute(grouped, op.resolved(node))
            restored = np.empty(len(result))
            restored[order] = np.asarray(computed, dtype=float)
            result[node.name] = restored
        return result


class FeatureGraphTransformer(base.TransformerMixin, base.BaseEstimator):
    """Row-wise graph nodes as a fitted scikit-learn transformer.

    Statistics for `clip`, `bin` and `standardize` are learned in `fit`
    (the training fold) and stored, so the transform is identical at
    inference. Derived columns are appended to the input frame.

    Attributes:
        nodes: The row-wise nodes, in dependency order.
        states_: Learned state per node name.
    """

    def __init__(self, nodes: tuple[tasks.FeatureNode, ...] = ()) -> None:
        """Initializes the transformer.

        Args:
            nodes: Row-wise nodes in dependency order.
        """
        self.nodes = nodes

    def fit(self, features: Any, target: Any = None) -> Self:
        """Learns operation state by evaluating the graph on training data."""
        frame = pd.DataFrame(features).copy()
        self.states_: dict[str, Any] = {}
        for node in self.nodes:
            op = get_op(node.op)
            params = op.resolved(node)
            inputs = [frame[c] for c in node.inputs]
            self.states_[node.name] = op.fit(inputs, params)
            frame[node.name] = op.apply(inputs, params, self.states_[node.name])
        return self

    def transform(self, features: Any) -> pd.DataFrame:
        """Appends the derived columns using the learned state."""
        frame = pd.DataFrame(features).copy()
        for node in self.nodes:
            op = get_op(node.op)
            params = op.resolved(node)
            inputs = [frame[c] for c in node.inputs]
            frame[node.name] = op.apply(inputs, params, self.states_[node.name])
        return frame
