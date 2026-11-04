"""Forecast metrics: error per horizon step, scaled by the no-change forecast.

The baseline every forecast must beat is "tomorrow looks like today". MASE
here divides the model's mean absolute error by that baseline's error on the
same rows, so a value below 1 means the model forecasts better than no change.

That is not the usual MASE, which divides by the in-sample error of the naive
forecast on the training history (`scmle.evaluation.proper_scores.mase`) and
is the only one comparable with published values. The same-rows score is
kept under the explicit kind `mase_same_rows`; `mase` stays an alias of it so
stored runs read the same. `mase_insample` takes the training-history scale
from `ForecastMetric.scale`.
"""

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from scmle.contracts import metrics as metric_contracts

SCALE_FLOOR = 1e-12


def truth_matrix(
    estimator: Any, features: pd.DataFrame, primary: pd.Series
) -> np.ndarray:
    """All future-value targets: the primary one plus the other columns."""
    others = list(estimator.target_columns[1:])
    return np.column_stack(
        [np.asarray(primary, dtype=float), features[others].to_numpy(float)]
    )


def last_value(
    estimator: Any, features: pd.DataFrame, steps: int
) -> np.ndarray:
    """The no-change forecast: the current value repeated `steps` times."""
    current = pd.to_numeric(
        features[estimator.value_column], errors="coerce"
    ).to_numpy(float)
    return np.tile(current[:, None], (1, steps))


def mean_absolute_error(truth: np.ndarray, guess: np.ndarray) -> float:
    """Average absolute miss over rows and horizon steps."""
    return float(np.abs(truth - guess).mean())


def mase(truth: np.ndarray, guess: np.ndarray, naive: np.ndarray) -> float:
    """Error relative to the no-change forecast (below 1 beats it)."""
    scale = mean_absolute_error(truth, naive)
    return mean_absolute_error(truth, guess) / max(scale, SCALE_FLOOR)


def smape(truth: np.ndarray, guess: np.ndarray) -> float:
    """Symmetric mean absolute percentage error, in [0, 2]."""
    gap = np.abs(truth - guess)
    size = (np.abs(truth) + np.abs(guess)) / 2
    return float(np.mean(gap / np.maximum(size, SCALE_FLOOR)))


@dataclasses.dataclass(frozen=True, slots=True)
class ForecastMetric(metric_contracts.Metric):
    """A forecasting metric over all horizon steps.

    Attributes:
        kind: `mase_same_rows` (alias `mase`), `mase_insample`, `mae` or
            `smape`.
        scale: In-sample naive mean absolute error for `mase_insample`
            (see `proper_scores.insample_scale`).
    """

    kind: str = "mase"
    scale: float | None = None

    def accepts_weights(self) -> bool:
        """These scores are unweighted."""
        return False

    def scorer(self, positive: Any = None) -> Any:
        """Builds a scorer `(estimator, features, first_step_target)`."""
        kind = self.kind
        sign = 1.0 if self.greater_is_better else -1.0
        scale = self.scale
        if kind == "mase_insample" and (scale is None or scale <= 0):
            raise ValueError("mase_insample needs a positive scale")

        def score(
            estimator: Any,
            features: pd.DataFrame,
            target: pd.Series,
            sample_weight: Any = None,
        ) -> float:
            truth = truth_matrix(estimator, features, target)
            guess = np.asarray(estimator.predict(features), dtype=float)
            if kind == "mae":
                return sign * mean_absolute_error(truth, guess)
            if kind == "smape":
                return sign * smape(truth, guess)
            if kind == "mase_insample":
                return (
                    sign
                    * mean_absolute_error(truth, guess)
                    / float(scale or 1.0)
                )
            naive = last_value(estimator, features, truth.shape[1])
            return sign * mase(truth, guess, naive)

        return score
