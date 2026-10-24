"""Cox regression with covariates that change over time.

Each person contributes several rows in counting-process form: a row says the
person was observed from `start` (exclusive) to `stop` (inclusive) with the
covariates in that row, and `event` is 1 when the event happened at `stop`.
A person is in the risk set at time `t` only while `start < t <= stop`, and
the covariate value used is the one in the row covering `t`, so no future
information leaks backward. Ties follow the Breslow convention.
"""

from typing import Any
from typing import Self

import numpy as np
from scipy import optimize

MAX_CELLS = 50_000_000


class TimeVaryingCox:
    """Penalised Cox model on (start, stop, event) rows.

    Attributes:
        penalty: Ridge strength on the coefficients.
        coefficients_: Fitted coefficients, one per feature column.
        moments_: Distinct event times.
        baseline_: Cumulative baseline hazard at each event time.
    """

    def __init__(self, penalty: float = 1.0) -> None:
        """Stores the settings."""
        self.penalty = penalty

    def fit(self, features: Any, start: Any, stop: Any, event: Any) -> Self:
        """Maximises the penalised partial likelihood.

        Raises:
            ValueError: If a row has `stop <= start`, there is no event, or
                the risk-set table would be too large.
        """
        matrix = np.asarray(features, dtype=float)
        begin = np.asarray(start, dtype=float)
        end = np.asarray(stop, dtype=float)
        seen = np.asarray(event, dtype=bool)
        if (end <= begin).any():
            raise ValueError("every row needs stop > start")
        moments = np.unique(end[seen])
        if len(moments) == 0:
            raise ValueError("no event was observed")
        if len(end) * len(moments) > MAX_CELLS:
            raise ValueError("too many rows and event times for one fit")
        at_risk = (begin[None, :] < moments[:, None]) & (
            moments[:, None] <= end[None, :]
        )
        deaths = (end[None, :] == moments[:, None]) & seen[None, :]
        count = deaths.sum(axis=1).astype(float)
        event_rows = matrix[seen]

        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            eta = matrix @ beta
            shift = eta.max()
            weight = np.exp(eta - shift)
            risk = at_risk @ weight
            mean = (at_risk @ (matrix * weight[:, None])) / risk[:, None]
            value = -(eta[seen].sum() - (count * (np.log(risk) + shift)).sum())
            gradient = -(
                event_rows.sum(axis=0) - (count[:, None] * mean).sum(axis=0)
            )
            return (
                value + 0.5 * self.penalty * float(beta @ beta),
                gradient + self.penalty * beta,
            )

        fitted = optimize.minimize(
            objective, np.zeros(matrix.shape[1]), jac=True, method="L-BFGS-B"
        )
        self.coefficients_ = fitted.x
        eta = matrix @ self.coefficients_
        self.scale_ = float(eta.max())
        weight = np.exp(eta - self.scale_)
        self.moments_ = moments
        self.baseline_ = np.cumsum(count / (at_risk @ weight))
        return self

    def predict(self, features: Any) -> np.ndarray:
        """Linear predictor (log relative hazard) for a covariate row."""
        return np.asarray(features, dtype=float) @ self.coefficients_
