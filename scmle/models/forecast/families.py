"""Forecast model families over a panel: classical baselines and boosting.

Every family fits on a panel's history and forecasts the next `steps` rows
of each of its series, so a backtest treats them alike. `Seasonal` is the
baseline every forecast must beat (Hyndman and Athanasopoulos, Forecasting:
Principles and Practice, 3rd ed., section 5.2); `Smoothing` is simple
exponential smoothing (section 8.1); `Boosted` is one gradient boosting
model per step ahead, shared by every series (the direct strategy of
Taieb et al. 2012), on lag and rolling features scaled per series.
"""

import abc
from collections.abc import Hashable
from collections.abc import Mapping
from collections.abc import Sequence
import math
from typing import Any
from typing import ClassVar
from typing import Self

import numpy
import pandas
from scipy import optimize
from sklearn import ensemble

from scmle.contracts import errors
from scmle.models.forecast.panel import Panel

LAGS = (1, 2, 3, 7)
"""Default lags, in rows: the defaults of `data.forecast_lags` (the last
three values and the same weekday of daily data)."""

WINDOWS = (3, 7)
"""Default rolling windows, in rows: the defaults of `data.forecast_windows`."""

LEAF = 20
"""Fewest rows of a leaf: the default `min_samples_leaf` of scikit-learn's
`HistGradientBoostingRegressor`. A step is learnt only from at least two
leaves' worth of rows, the fewest that allow one split; with fewer the
regressor could only return the mean change."""

ALPHA = (1e-4, 1.0)
"""Bounds of the smoothing weight: the lower bound of R's `forecast::ets`
(Hyndman and Khandakar 2008) keeps the level from freezing, and 1 is the
naive forecast."""


class Forecaster(abc.ABC):
    """A family that forecasts every series of a panel.

    Attributes:
        name: Registry name of the family.
        steps: Rows ahead the fitted model forecasts (0 before `fit`).
    """

    name: ClassVar[str]
    steps: int = 0

    @abc.abstractmethod
    def fit(self, panel: Panel, steps: int) -> Self:
        """Learns from the panel's history.

        Args:
            panel: The history.
            steps: Rows ahead to forecast.

        Returns:
            This forecaster, fitted.
        """

    @abc.abstractmethod
    def predict(self) -> dict[Hashable, numpy.ndarray]:
        """Forecasts the `steps` rows after each series' last row.

        Returns:
            Entity to a float array of `steps` forecasts.
        """

    def check(self) -> None:
        """Refuses a forecast before a fit.

        Raises:
            OrderError: If `fit` has not run.
        """
        if not self.steps:
            raise errors.OrderError(f"{self.name}: call fit before predict")


class Seasonal(Forecaster):
    """Seasonal naive: each step repeats the value one season earlier.

    Attributes:
        period: Rows per season; None (or a series shorter than a season)
            repeats the last value, the naive forecast.
        history: The fitted series.
    """

    name = "seasonal_naive"

    def __init__(self, period: int | None = None) -> None:
        """Initializes the baseline.

        Args:
            period: Rows per season, or None for the naive forecast.
        """
        self.period = period
        self.history: dict[Hashable, numpy.ndarray] = {}

    def fit(self, panel: Panel, steps: int) -> Self:
        """Keeps the history.

        Args:
            panel: The history.
            steps: Rows ahead to forecast.

        Returns:
            This forecaster.
        """
        self.history, self.steps = panel.series(), steps
        return self

    def predict(self) -> dict[Hashable, numpy.ndarray]:
        """The last season (or value) repeated over the steps.

        Returns:
            Entity to its forecasts.
        """
        self.check()
        found = {}
        for key, values in self.history.items():
            span = self.period or 1
            span = span if len(values) >= span else 1
            found[key] = numpy.resize(values[-span:], self.steps)
        return found


def loss(alpha: float, values: numpy.ndarray) -> float:
    """Sum of squared one-step errors of simple exponential smoothing.

    Args:
        alpha: The smoothing weight.
        values: The series.

    Returns:
        The sum over rows `1..n-1` of the squared error of the level before
        that row, the level starting at the first value.
    """
    level, total = values[0], 0.0
    for value in values[1:]:
        miss = value - level
        total += miss * miss
        level += alpha * miss
    return float(total)


class Smoothing(Forecaster):
    """Simple exponential smoothing with its weight fitted per series.

    The weight minimises the one-step squared error over the history
    (bounded scalar search within `ALPHA`); the forecast is the final level
    for every step.

    Attributes:
        levels: The final level of each fitted series.
        weights: The fitted smoothing weight of each series.
    """

    name = "exponential_smoothing"

    def __init__(self) -> None:
        """Initializes the family."""
        self.levels: dict[Hashable, float] = {}
        self.weights: dict[Hashable, float] = {}

    def fit(self, panel: Panel, steps: int) -> Self:
        """Fits the weight and level of each series.

        Args:
            panel: The history.
            steps: Rows ahead to forecast.

        Returns:
            This forecaster.
        """
        self.levels, self.weights = {}, {}
        for key, values in panel.series().items():
            weight = 1.0
            if len(values) > 1 and numpy.ptp(values) > 0:
                found = optimize.minimize_scalar(
                    loss, bounds=ALPHA, args=(values,), method="bounded"
                )
                weight = float(found.x)
            level = values[0]
            for value in values[1:]:
                level += weight * (value - level)
            self.levels[key], self.weights[key] = float(level), weight
        self.steps = steps
        return self

    def predict(self) -> dict[Hashable, numpy.ndarray]:
        """The final level repeated over the steps.

        Returns:
            Entity to its forecasts.
        """
        self.check()
        return {k: numpy.full(self.steps, v) for k, v in self.levels.items()}


def scale(values: numpy.ndarray) -> float:
    """A series' typical one-row change, the unit its features are read in.

    Args:
        values: The series.

    Returns:
        The mean absolute one-row change; 1 when it is zero or undefined
        (a constant or one-row series, whose scaled features are all zero
        whatever the unit).
    """
    if len(values) < 2:  # noqa: PLR2004 -- a change needs two rows
        return 1.0
    change = float(numpy.abs(numpy.diff(values)).mean())
    return change if change > 0 else 1.0


class Boosted(Forecaster):
    """Direct multi-step gradient boosting on scaled lag features.

    For each step `h` one histogram gradient boosting regressor, shared by
    every series, learns `(y[t+h] - y[t]) / s` from features of row `t`:
    the level `y[t] / s`, each lag's change `(y[t-k+1] - y[t]) / s`, each
    window's mean change and spread over `s`, and with a period the
    seasonal-naive change for step `h`. `s` is the series' mean absolute
    one-row change, so series of any level share one model. A lag reaching
    before the series' start is missing, which the regressor handles.

    Attributes:
        lags: Lags of the value, in rows.
        windows: Rolling-window sizes, in rows.
        period: Rows per season, or None.
        seed: Random seed of the regressors.
        params: Overrides of the regressors' parameters.
        models: The fitted regressor of each step.
        origins: Each fitted series' values, scale and features.
    """

    name = "lag_boosting"

    def __init__(
        self,
        lags: Sequence[int] = LAGS,
        windows: Sequence[int] = WINDOWS,
        period: int | None = None,
        seed: int = 0,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        """Initializes the family.

        Args:
            lags: Lags of the value, in rows (each at least 1).
            windows: Rolling-window sizes, in rows (each at least 2).
            period: Rows per season, or None.
            seed: Random seed.
            params: `HistGradientBoostingRegressor` overrides.

        Raises:
            InputError: If a lag is below 1, a window below 2 or the period
                below 2.
        """
        if any(lag < 1 for lag in lags):
            raise errors.InputError(f"forecast lags must be at least 1: {lags}")
        if any(size < 2 for size in windows):  # noqa: PLR2004 -- a spread needs two rows
            raise errors.InputError(
                f"forecast windows must be at least 2: {windows}"
            )
        if period is not None and period < 2:  # noqa: PLR2004 -- a season spans two rows
            raise errors.InputError("a seasonal period must be at least 2")
        self.lags, self.windows = tuple(lags), tuple(windows)
        self.period, self.seed = period, seed
        self.params = dict(params or {})
        self.models: list[ensemble.HistGradientBoostingRegressor] = []
        self.origins: dict[
            Hashable, tuple[numpy.ndarray, float, numpy.ndarray]
        ] = {}

    def features(self, values: numpy.ndarray, unit: float) -> numpy.ndarray:
        """The step-independent features of every row of a series.

        Args:
            values: The series.
            unit: Its scale.

        Returns:
            A `(rows, features)` float array.
        """
        series = pandas.Series(values)
        columns = [values / unit]
        columns += [
            (series.shift(lag - 1).to_numpy() - values) / unit
            for lag in self.lags
            if lag > 1
        ]
        for size in self.windows:
            rolled = series.rolling(size, min_periods=size)
            columns.append((rolled.mean().to_numpy() - values) / unit)
            columns.append(rolled.std().to_numpy() / unit)
        return numpy.column_stack(columns)

    def season(
        self, values: numpy.ndarray, unit: float, step: int
    ) -> numpy.ndarray:
        """The seasonal-naive change for `step` at every row of a series.

        Args:
            values: The series.
            unit: Its scale.
            step: Rows ahead.

        Returns:
            A `(rows, 1)` array; missing where the season reaches before the
            series' start, and empty columns without a period.
        """
        if self.period is None:
            return numpy.empty((len(values), 0))
        lag = math.ceil(step / self.period) * self.period - step
        shifted = pandas.Series(values).shift(lag).to_numpy(float)
        change: numpy.ndarray = (shifted - values) / unit
        return change[:, None]

    def fit(self, panel: Panel, steps: int) -> Self:
        """Fits one regressor per step on every series' rows.

        Args:
            panel: The history.
            steps: Rows ahead to forecast.

        Returns:
            This forecaster.

        Raises:
            DataError: If a step has fewer than two leaves' worth of rows
                (`LEAF`) with a value that many rows later.
        """
        tables = {}
        for key, values in panel.series().items():
            unit = scale(values)
            tables[key] = (values, unit, self.features(values, unit))
        self.models = []
        for step in range(1, steps + 1):
            inputs, outputs = [], []
            for values, unit, base in tables.values():
                rows = len(values) - step
                if rows <= 0:
                    continue
                season = self.season(values, unit, step)
                inputs.append(numpy.hstack([base, season])[:rows])
                outputs.append((values[step:] - values[:rows]) / unit)
            least = 2 * int(self.params.get("min_samples_leaf", LEAF))
            count = sum(len(rows) for rows in outputs)
            if count < least:
                raise errors.DataError(
                    f"{self.name}: step {step} has {count} row(s) with a "
                    f"value that many rows later; it needs at least {least} "
                    "(two leaves) to learn from: add history or shorten the "
                    "horizon"
                )
            model = ensemble.HistGradientBoostingRegressor(
                random_state=self.seed, **self.params
            )
            model.fit(numpy.vstack(inputs), numpy.concatenate(outputs))
            self.models.append(model)
        self.origins = tables
        self.steps = steps
        return self

    def predict(self) -> dict[Hashable, numpy.ndarray]:
        """Forecasts from each series' last row.

        Returns:
            Entity to its forecasts.
        """
        self.check()
        found = {}
        for key, (values, unit, base) in self.origins.items():
            rows = [
                numpy.hstack([base[-1], self.season(values, unit, step)[-1]])
                for step in range(1, self.steps + 1)
            ]
            guesses = [
                model.predict(row[None, :])[0]
                for model, row in zip(self.models, rows, strict=True)
            ]
            found[key] = values[-1] + unit * numpy.asarray(guesses, float)
        return found
