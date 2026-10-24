"""Forecasting estimators over the multi-target machinery."""

from typing import Any
from typing import Self

import numpy as np
import pandas as pd

from scmle.models.multitarget import wrapper


class ForecastEstimator(wrapper.MultiOutputEstimator):
    """A multi-output pipeline that also knows which column is the series.

    Attributes:
        value_column: The series' value column (the current value at each
            row, which the baselines repeat).
    """

    def __init__(
        self,
        estimator: Any,
        target_columns: tuple[str, ...],
        value_column: str = "",
    ) -> None:
        """Initializes the wrapper."""
        super().__init__(estimator, target_columns)
        self.value_column = value_column


class LastValueForecaster(wrapper.MultiOutputEstimator):
    """The naive baseline: the next `horizon` values equal the current one.

    It reads the current value straight from the rows (no preprocessing), so
    it is exactly what a person would call "no change".

    Attributes:
        value_column: The series' value column.
        target_columns: The future-value targets, one per step.
    """

    def __init__(
        self,
        estimator: Any = None,
        target_columns: tuple[str, ...] = (),
        value_column: str = "",
    ) -> None:
        """Initializes the baseline."""
        super().__init__(estimator, target_columns)
        self.value_column = value_column

    def fit(self, features: Any, target: Any) -> Self:
        """Nothing to learn."""
        return self

    def predict(self, features: Any) -> np.ndarray:
        """The current value repeated for every step."""
        current = pd.to_numeric(
            pd.DataFrame(features)[self.value_column], errors="coerce"
        ).to_numpy(dtype=float)
        return np.tile(current[:, None], (1, len(self.target_columns)))
