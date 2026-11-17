"""Ordered operations computed per series in time order."""

import abc
from collections.abc import Mapping
from typing import Any

import pandas

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.feature_graph.operations import Inputs
from scmle.data.feature_graph.operations import Operation
from scmle.data.feature_graph.operations import Params


class Sequential(Operation):
    """An operation over time-ordered values within a group."""

    ordered = True

    def apply(
        self,
        inputs: Inputs,  # noqa: ARG002 -- Operation interface
        params: Params,  # noqa: ARG002 -- Operation interface
        state: Any,  # noqa: ARG002 -- Operation interface
    ) -> pandas.Series:
        """Rejects row-wise application of an ordered operation.

        Ordered operations run through `Temporal` only.

        Args:
            inputs: Unused.
            params: Unused.
            state: Unused.

        Raises:
            ConfigError: Always.
        """
        raise errors.ConfigError(f"{self.name} is an ordered operation")

    @abc.abstractmethod
    def compute(self, grouped: Any, params: Mapping[str, Any]) -> pandas.Series:
        """Computes the feature from values already sorted by time per group.

        Args:
            grouped: The values, sorted by time and grouped by entity.
            params: The resolved parameters.

        Returns:
            The feature, aligned with the sorted values.
        """

    @abc.abstractmethod
    def lookback(self, params: Mapping[str, Any]) -> int | None:
        """Reports how many earlier rows an entity needs.

        Args:
            params: The resolved parameters.

        Returns:
            The number of earlier rows needed, or None when all are needed.
        """


class Lag(Sequential):
    """The value `k` rows earlier within the group."""

    name = "lag"
    defaults = {"k": 1}

    def validate(self, node: tasks.Node) -> None:
        """Checks arity, parameters and that the lag is positive.

        A positive lag ensures no future value can leak.

        Args:
            node: The node to check.

        Raises:
            ConfigError: If the arity or parameters are wrong, or `k` is
                below 1.
        """
        super().validate(node)
        if int(self.resolved(node)["k"]) < 1:
            raise errors.ConfigError(f"node '{node.name}': lag k must be >= 1")

    def compute(self, grouped: Any, params: Params) -> pandas.Series:
        """Shifts the values forward in time.

        Args:
            grouped: The values, sorted by time and grouped by entity.
            params: Parameters with the lag `k`.

        Returns:
            The value `k` rows earlier, NaN at the start of each entity.
        """
        lagged: pandas.Series = grouped.shift(int(params["k"]))
        return lagged

    def lookback(self, params: Params) -> int | None:
        """Reports the lag distance.

        Args:
            params: Parameters with the lag `k`.

        Returns:
            The lag `k`, the number of earlier rows needed.
        """
        return int(params["k"])


class Diff(Sequential):
    """Change versus `k` rows earlier within the group."""

    name = "diff"
    defaults = {"k": 1}

    def validate(self, node: tasks.Node) -> None:
        """Checks arity, parameters and that the lag is positive.

        Args:
            node: The node to check.

        Raises:
            ConfigError: If the arity or parameters are wrong, or `k` is
                below 1.
        """
        super().validate(node)
        if int(self.resolved(node)["k"]) < 1:
            raise errors.ConfigError(f"node '{node.name}': diff k must be >= 1")

    def compute(self, grouped: Any, params: Params) -> pandas.Series:
        """Subtracts the lagged value from the current one.

        Args:
            grouped: The values, sorted by time and grouped by entity.
            params: Parameters with the lag `k`.

        Returns:
            The change versus `k` rows earlier, NaN at the start of each
            entity.
        """
        changed: pandas.Series = grouped.transform(
            lambda s: s - s.shift(int(params["k"]))
        )
        return changed

    def lookback(self, params: Params) -> int | None:
        """Reports the lag distance.

        Args:
            params: Parameters with the lag `k`.

        Returns:
            The lag `k`, the number of earlier rows needed.
        """
        return int(params["k"])


class Rolling(Sequential):
    """Mean of the last `window` rows (including the current one)."""

    name = "rolling_mean"
    defaults = {"window": 3}

    def compute(self, grouped: Any, params: Params) -> pandas.Series:
        """Computes the trailing mean.

        Args:
            grouped: The values, sorted by time and grouped by entity.
            params: Parameters with the `window` size.

        Returns:
            The mean of up to `window` rows ending at the current one.
        """
        window = int(params["window"])
        means: pandas.Series = grouped.transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        return means

    def lookback(self, params: Params) -> int | None:
        """Reports the window size.

        Args:
            params: Parameters with the `window` size.

        Returns:
            The window, the number of rows needed including the current one.
        """
        return int(params["window"])


class Volatility(Sequential):
    """Standard deviation of the last `window` rows."""

    name = "rolling_std"
    defaults = {"window": 3}

    def compute(self, grouped: Any, params: Params) -> pandas.Series:
        """Computes the trailing standard deviation.

        Args:
            grouped: The values, sorted by time and grouped by entity.
            params: Parameters with the `window` size.

        Returns:
            The standard deviation of up to `window` rows ending at the
            current one; NaN until two observations exist.
        """
        window = int(params["window"])
        deviations: pandas.Series = grouped.transform(
            lambda s: s.rolling(window, min_periods=2).std()
        )
        return deviations

    def lookback(self, params: Params) -> int | None:
        """Reports the window size.

        Args:
            params: Parameters with the `window` size.

        Returns:
            The window, the number of rows needed including the current one.
        """
        return int(params["window"])


class ExpandingMean(Sequential):
    """Mean of all rows up to and including the current one."""

    name = "expanding_mean"

    def compute(self, grouped: Any, params: Params) -> pandas.Series:  # noqa: ARG002 -- Sequential interface
        """Computes the running mean.

        Args:
            grouped: The values, sorted by time and grouped by entity.
            params: Unused.

        Returns:
            The mean of all rows up to and including the current one.
        """
        means: pandas.Series = grouped.transform(lambda s: s.expanding().mean())
        return means

    def lookback(self, params: Params) -> int | None:  # noqa: ARG002 -- Sequential interface
        """Reports that every earlier row contributes.

        Args:
            params: Unused.

        Returns:
            None, meaning all earlier rows are needed.
        """
        return None
