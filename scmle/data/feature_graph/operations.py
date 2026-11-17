"""The operation interface and the row-wise operations."""

import abc
from collections.abc import Mapping
from typing import Any
from typing import cast
from typing import ClassVar

import numpy
import pandas

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.validation import columns

MAX_ARITY = 64


Inputs = list[pandas.Series]


Params = Mapping[str, Any]


EPOCH = pandas.Timestamp("1970-01-01")


class Operation(abc.ABC):
    """One operation of the graph.

    Attributes:
        name: Name used in `Node.op`.
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

    def validate(self, node: tasks.Node) -> None:
        """Checks the node's input count and parameter names.

        Args:
            node: The node to check.

        Raises:
            ConfigError: If the input count is outside `min_inputs` to
                `max_inputs`, or a parameter is not in `defaults`.
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

    def resolved(self, node: tasks.Node) -> dict[str, Any]:
        """Applies the defaults to a node's parameters.

        Args:
            node: The node whose parameters are resolved.

        Returns:
            The defaults updated with the node's own parameters.
        """
        return {**self.defaults, **node.params}

    def fit(self, inputs: Inputs, params: Params) -> Any:  # noqa: ARG002 -- Operation interface
        """Learns state from training data; the default is stateless.

        Args:
            inputs: The input columns of the training fold.
            params: The resolved parameters.

        Returns:
            The learned state passed to `apply`; None when stateless.
        """
        return None

    @abc.abstractmethod
    def apply(
        self, inputs: Inputs, params: Params, state: Any
    ) -> pandas.Series:
        """Computes the derived column.

        Args:
            inputs: The input columns.
            params: The resolved parameters.
            state: The value returned by `fit`.

        Returns:
            The derived column, aligned with the inputs' index.
        """


class Log1p(Operation):
    """`log(1 + max(x, 0))`."""

    name = "log1p"

    def apply(
        self,
        inputs: Inputs,
        params: Params,  # noqa: ARG002 -- Operation interface
        state: Any,  # noqa: ARG002 -- Operation interface
    ) -> pandas.Series:
        """Log-transform the input after clipping it at zero.

        Args:
            inputs: One column.
            params: Unused.
            state: Unused.

        Returns:
            `log(1 + max(x, 0))` per row.
        """
        # A ufunc on a Series returns a Series; the numpy stubs say ndarray.
        return cast(
            "pandas.Series",
            numpy.log1p(columns.scrub(inputs[0]).clip(lower=0)),
        )


class Sqrt(Operation):
    """`sqrt(max(x, 0))`."""

    name = "sqrt"

    def apply(
        self,
        inputs: Inputs,
        params: Params,  # noqa: ARG002 -- Operation interface
        state: Any,  # noqa: ARG002 -- Operation interface
    ) -> pandas.Series:
        """Takes the square root of the input after clipping it at zero.

        Args:
            inputs: One column.
            params: Unused.
            state: Unused.

        Returns:
            `sqrt(max(x, 0))` per row.
        """
        return cast(
            "pandas.Series",
            numpy.sqrt(columns.scrub(inputs[0]).clip(lower=0)),
        )


class Square(Operation):
    """`x ** 2`."""

    name = "square"

    def apply(
        self,
        inputs: Inputs,
        params: Params,  # noqa: ARG002 -- Operation interface
        state: Any,  # noqa: ARG002 -- Operation interface
    ) -> pandas.Series:
        """Squares the input.

        Args:
            inputs: One column.
            params: Unused.
            state: Unused.

        Returns:
            `x ** 2` per row.
        """
        return columns.scrub(inputs[0]) ** 2


class Absolute(Operation):
    """`abs(x)`."""

    name = "abs"

    def apply(
        self,
        inputs: Inputs,
        params: Params,  # noqa: ARG002 -- Operation interface
        state: Any,  # noqa: ARG002 -- Operation interface
    ) -> pandas.Series:
        """Takes the absolute value of the input.

        Args:
            inputs: One column.
            params: Unused.
            state: Unused.

        Returns:
            `abs(x)` per row.
        """
        return columns.scrub(inputs[0]).abs()


class Missingness(Operation):
    """1 where the input is missing, else 0."""

    name = "is_missing"

    def apply(
        self,
        inputs: Inputs,
        params: Params,  # noqa: ARG002 -- Operation interface
        state: Any,  # noqa: ARG002 -- Operation interface
    ) -> pandas.Series:
        """Flags missing values; works for any dtype.

        Args:
            inputs: One column.
            params: Unused.
            state: Unused.

        Returns:
            1.0 where the input is missing, else 0.0.
        """
        return inputs[0].isna().astype(float)


class Sum(Operation):
    """Sum of two or more inputs."""

    name = "sum"
    min_inputs = 2
    max_inputs = MAX_ARITY

    def apply(
        self,
        inputs: Inputs,
        params: Params,  # noqa: ARG002 -- Operation interface
        state: Any,  # noqa: ARG002 -- Operation interface
    ) -> pandas.Series:
        """Adds the inputs.

        Args:
            inputs: Two or more columns.
            params: Unused.
            state: Unused.

        Returns:
            The row-wise sum; NaN where any input is NaN.
        """
        return sum(columns.scrub(s) for s in inputs[1:]) + columns.scrub(
            inputs[0]
        )


class Difference(Operation):
    """`a - b`."""

    name = "difference"
    min_inputs = 2
    max_inputs = 2

    def apply(
        self,
        inputs: Inputs,
        params: Params,  # noqa: ARG002 -- Operation interface
        state: Any,  # noqa: ARG002 -- Operation interface
    ) -> pandas.Series:
        """Subtracts the second input from the first.

        Args:
            inputs: Two columns.
            params: Unused.
            state: Unused.

        Returns:
            `a - b` per row.
        """
        return columns.scrub(inputs[0]) - columns.scrub(inputs[1])


class Product(Operation):
    """Product of two or more inputs."""

    name = "product"
    min_inputs = 2
    max_inputs = MAX_ARITY

    def apply(
        self,
        inputs: Inputs,
        params: Params,  # noqa: ARG002 -- Operation interface
        state: Any,  # noqa: ARG002 -- Operation interface
    ) -> pandas.Series:
        """Multiplies the inputs.

        Args:
            inputs: Two or more columns.
            params: Unused.
            state: Unused.

        Returns:
            The row-wise product; NaN where any input is NaN.
        """
        result = columns.scrub(inputs[0])
        for series in inputs[1:]:
            result = result * columns.scrub(series)
        return result


class Ratio(Operation):
    """`a / b`, NaN where b is zero."""

    name = "ratio"
    min_inputs = 2
    max_inputs = 2

    def apply(
        self,
        inputs: Inputs,
        params: Params,  # noqa: ARG002 -- Operation interface
        state: Any,  # noqa: ARG002 -- Operation interface
    ) -> pandas.Series:
        """Divides the first input by the second.

        Args:
            inputs: Two columns.
            params: Unused.
            state: Unused.

        Returns:
            `a / b` per row, NaN where `b` is zero.
        """
        denominator = columns.scrub(inputs[1]).replace(0.0, numpy.nan)
        return columns.scrub(inputs[0]) / denominator


class Clip(Operation):
    """Winsorizes to quantiles learned on the training fold."""

    name = "clip"
    defaults = {"lower": 0.01, "upper": 0.99}

    def fit(self, inputs: Inputs, params: Params) -> Any:
        """Learns the clipping bounds from training data.

        Args:
            inputs: One column.
            params: Parameters with the `lower` and `upper` quantile levels.

        Returns:
            The (low, high) quantile values, or (-inf, inf) when the column
            has no finite values.
        """
        values = columns.scrub(inputs[0]).dropna()
        if values.empty:
            return (-numpy.inf, numpy.inf)
        low, high = numpy.quantile(values, [params["lower"], params["upper"]])
        return float(low), float(high)

    def apply(
        self,
        inputs: Inputs,
        params: Params,  # noqa: ARG002 -- Operation interface
        state: Any,
    ) -> pandas.Series:
        """Clips to the learned bounds.

        Args:
            inputs: One column.
            params: Unused.
            state: The (low, high) bounds returned by `fit`.

        Returns:
            The column limited to the bounds.
        """
        return columns.scrub(inputs[0]).clip(lower=state[0], upper=state[1])


class Standardize(Operation):
    """Z-score using the training fold's mean and standard deviation."""

    name = "standardize"

    def fit(self, inputs: Inputs, params: Params) -> Any:  # noqa: ARG002 -- Operation interface
        """Learns the mean and standard deviation from training data.

        Args:
            inputs: One column.
            params: Unused.

        Returns:
            The (mean, std) pair; the std is 1.0 when it is zero or fewer than
            two values exist, and the mean is 0.0 for an empty column.
        """
        values = columns.scrub(inputs[0]).dropna()
        spread = float(values.std()) if len(values) > 1 else 0.0
        return (float(values.mean()) if len(values) else 0.0, spread or 1.0)

    def apply(
        self,
        inputs: Inputs,
        params: Params,  # noqa: ARG002 -- Operation interface
        state: Any,
    ) -> pandas.Series:
        """Standardizes with the learned statistics.

        Args:
            inputs: One column.
            params: Unused.
            state: The (mean, std) pair returned by `fit`.

        Returns:
            The z-score per row.
        """
        scores: pandas.Series = (columns.scrub(inputs[0]) - state[0]) / state[1]
        return scores


class Bin(Operation):
    """Quantile bins (ordinal codes) with edges learned on the training fold."""

    name = "bin"
    defaults = {"bins": 5}

    def fit(self, inputs: Inputs, params: Params) -> Any:
        """Learns the interior quantile edges from training data.

        Args:
            inputs: One column.
            params: Parameters with the number of `bins`.

        Returns:
            The sorted unique interior edges as an array; empty when the
            column has no finite values.
        """
        values = columns.scrub(inputs[0]).dropna().to_numpy()
        if values.size == 0:
            return numpy.array([])
        edges = numpy.quantile(
            values, numpy.linspace(0, 1, int(params["bins"]) + 1)[1:-1]
        )
        return numpy.unique(edges)

    def apply(
        self,
        inputs: Inputs,
        params: Params,  # noqa: ARG002 -- Operation interface
        state: Any,
    ) -> pandas.Series:
        """Assigns each value to its bin.

        Args:
            inputs: One column.
            params: Unused.
            state: The edges returned by `fit`.

        Returns:
            The ordinal bin code per row as a float; NaN stays NaN.
        """
        values = columns.scrub(inputs[0])
        codes = numpy.searchsorted(
            state, values.to_numpy(), side="right"
        ).astype(float)
        return pandas.Series(
            numpy.where(values.isna(), numpy.nan, codes), index=values.index
        )


class Calendar(Operation):
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

    def validate(self, node: tasks.Node) -> None:
        """Checks arity, parameters and the requested datetime part.

        Args:
            node: The node to check.

        Raises:
            ConfigError: If the arity or parameters are wrong, or `part` is
                not one of `parts`.
        """
        super().validate(node)
        part = self.resolved(node)["part"]
        if part not in self.parts:
            raise errors.ConfigError(
                f"node '{node.name}': datepart part must be one of "
                f"{sorted(self.parts)}, got '{part}'"
            )

    def apply(
        self,
        inputs: Inputs,
        params: Params,
        state: Any,  # noqa: ARG002 -- Operation interface
    ) -> pandas.Series:
        """Extracts the requested part of the datetime.

        Args:
            inputs: One column.
            params: Parameters with the `part` to extract.
            state: Unused.

        Returns:
            The part as a float per row (days since 1970-01-01 for
            `timestamp`); NaN where the date cannot be parsed.
        """
        stamps = pandas.to_datetime(inputs[0], errors="coerce")
        if params["part"] == "timestamp":
            days: pandas.Series = (stamps - EPOCH) / pandas.Timedelta(days=1)
            return days
        part: pandas.Series = getattr(stamps.dt, params["part"]).astype(float)
        return part
