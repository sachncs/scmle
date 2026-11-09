"""Ordinal model adapters: regressors wrapped with learned thresholds."""

from collections.abc import Mapping
import types
from typing import Any

from sklearn import ensemble
from sklearn import linear_model

from scmle.contracts import models
from scmle.contracts import search
from scmle.contracts import tasks
from scmle.models.ordinal import estimators
from scmle.models.sklearn import classical

ORDINAL = frozenset({tasks.Task.ORDINAL})
LEVELS = "levels"


class Ordinal(classical.Learner):
    """Wraps a scikit-learn regressor as an ordinal classifier.

    The pipeline passes the number of levels as the `levels` parameter, so a
    training fold missing the top level still yields a full probability row.
    """

    conformance_params: Mapping[str, Any] | None = types.MappingProxyType(
        {LEVELS: 3}
    )

    def build(
        self,
        task: tasks.Task,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Builds the thresholded regressor around the configured class.

        Args:
            task: Resolved task type; the inner regressor is always built as a
                regression.
            seed: Random seed.
            params: Search-space overrides for the regressor, plus the optional
                `levels` entry giving the number of ordered levels.

        Returns:
            An unfitted ordinal estimator wrapping the regressor.
        """
        settings = dict(params or {})
        levels = settings.pop(LEVELS, None)
        inner = super().build(tasks.Task.REGRESSION, seed, settings)
        return estimators.CumulativeThresholdRegressor(inner, levels, seed)


ORDINAL_RIDGE = Ordinal(
    name="ordinal_ridge",
    family="linear",
    capabilities=models.Capabilities(
        ORDINAL, needs_scaling=True, cost=2, probabilities=True
    ),
    regressor=linear_model.Ridge,
    search_space=search.SearchSpace(
        (search.FloatParameter("alpha", 1e-3, 100.0, log=True),)
    ),
)

ORDINAL_FOREST = Ordinal(
    name="ordinal_random_forest",
    family="tree",
    capabilities=models.Capabilities(
        ORDINAL, cost=3, memory_factor=6.0, probabilities=True
    ),
    regressor=ensemble.RandomForestRegressor,
    search_space=search.SearchSpace(
        (
            search.IntParameter("n_estimators", 50, 300),
            search.IntParameter("min_samples_leaf", 1, 20, log=True),
        )
    ),
)

ORDINAL_BOOSTING = Ordinal(
    name="ordinal_hist_gradient_boosting",
    family="boosting",
    capabilities=models.Capabilities(
        ORDINAL,
        handles_missing=True,
        cost=3,
        memory_factor=3.0,
        probabilities=True,
    ),
    regressor=ensemble.HistGradientBoostingRegressor,
    search_space=search.SearchSpace(
        (
            search.FloatParameter("learning_rate", 0.02, 0.3, log=True),
            search.IntParameter("max_leaf_nodes", 8, 64),
        )
    ),
)

ORDINAL_ADAPTERS: tuple[models.Model, ...] = (
    ORDINAL_RIDGE,
    ORDINAL_FOREST,
    ORDINAL_BOOSTING,
)
"""Ordinal adapter instances, in registration order."""
