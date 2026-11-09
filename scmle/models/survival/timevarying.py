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

import numpy
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
        """Stores the settings.

        Args:
            penalty: Ridge strength on the coefficients.
        """
        self.penalty = penalty

    def fit(self, features: Any, start: Any, stop: Any, event: Any) -> Self:
        """Maximises the penalised partial likelihood.

        Args:
            features: Covariate matrix, one row per observed interval.
            start: Exclusive start time of each interval.
            stop: Inclusive stop time of each interval.
            event: 1 where the event happened at `stop`, else 0.

        Returns:
            This fitted model.

        Raises:
            ValueError: If a row has `stop <= start`, there is no event, or
                the risk-set table would be too large.
        """
        matrix = numpy.asarray(features, dtype=float)
        begin = numpy.asarray(start, dtype=float)
        end = numpy.asarray(stop, dtype=float)
        seen = numpy.asarray(event, dtype=bool)
        if (end <= begin).any():
            raise ValueError("every row needs stop > start")
        moments = numpy.unique(end[seen])
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

        def objective(beta: numpy.ndarray) -> tuple[float, numpy.ndarray]:
            """Penalised negative log partial likelihood and its gradient.

            Args:
                beta: Coefficient vector.

            Returns:
                The objective value and its gradient with respect to `beta`.
            """
            eta = matrix @ beta
            shift = eta.max()
            weight = numpy.exp(eta - shift)
            risk = at_risk @ weight
            mean = (at_risk @ (matrix * weight[:, None])) / risk[:, None]
            value = -(
                eta[seen].sum() - (count * (numpy.log(risk) + shift)).sum()
            )
            gradient = -(
                event_rows.sum(axis=0) - (count[:, None] * mean).sum(axis=0)
            )
            return (
                value + 0.5 * self.penalty * float(beta @ beta),
                gradient + self.penalty * beta,
            )

        fitted = optimize.minimize(
            objective, numpy.zeros(matrix.shape[1]), jac=True, method="L-BFGS-B"
        )
        self.coefficients_ = fitted.x
        eta = matrix @ self.coefficients_
        self.scale_ = float(eta.max())
        weight = numpy.exp(eta - self.scale_)
        self.moments_ = moments
        self.baseline_ = numpy.cumsum(count / (at_risk @ weight))
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """Linear predictor (log relative hazard) for a covariate row.

        Args:
            features: Covariate matrix, one row per person.

        Returns:
            Array of shape `(n_rows,)` of linear predictors.
        """
        return numpy.asarray(features, dtype=float) @ self.coefficients_
