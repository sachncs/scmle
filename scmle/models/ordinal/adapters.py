"""Ordinal model adapters: regressors wrapped with learned thresholds."""

from collections.abc import Mapping
import types
from typing import Any
from typing import ClassVar

from sklearn import ensemble
from sklearn import linear_model

from scmle.contracts import models
from scmle.contracts import search
from scmle.contracts import tasks
from scmle.models.ordinal import estimators
from scmle.models.sklearn import classical

ORDINAL = frozenset({tasks.TaskType.ORDINAL})
LEVELS = "levels"


class OrdinalAdapter(classical.SklearnAdapter):
    """Wraps a scikit-learn regressor as an ordinal classifier.

    The pipeline passes the number of levels as the `levels` parameter, so a
    training fold missing the top level still yields a full probability row.
    """

    conformance_params: ClassVar[Mapping[str, Any] | None] = (
        types.MappingProxyType({LEVELS: 3})
    )

    def build(
        self,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Builds the thresholded regressor around the configured class."""
        settings = dict(params or {})
        levels = settings.pop(LEVELS, None)
        inner = super().build(tasks.TaskType.REGRESSION, seed, settings)
        return estimators.CumulativeThresholdRegressor(inner, levels, seed)


class OrdinalRidgeAdapter(OrdinalAdapter):
    """Linear latent score."""

    name = "ordinal_ridge"
    family = "linear"
    capabilities = models.Capabilities(
        ORDINAL, needs_scaling=True, cost=2, probabilities=True
    )
    regressor = linear_model.Ridge
    search_space = search.SearchSpace(
        (search.FloatParameter("alpha", 1e-3, 100.0, log=True),)
    )


class OrdinalForestAdapter(OrdinalAdapter):
    """Random-forest latent score."""

    name = "ordinal_random_forest"
    family = "tree"
    capabilities = models.Capabilities(
        ORDINAL, cost=3, memory_factor=6.0, probabilities=True
    )
    regressor = ensemble.RandomForestRegressor
    search_space = search.SearchSpace(
        (
            search.IntParameter("n_estimators", 50, 300),
            search.IntParameter("min_samples_leaf", 1, 20, log=True),
        )
    )


class OrdinalBoostingAdapter(OrdinalAdapter):
    """Gradient-boosted latent score."""

    name = "ordinal_hist_gradient_boosting"
    family = "boosting"
    capabilities = models.Capabilities(
        ORDINAL,
        handles_missing=True,
        cost=3,
        memory_factor=3.0,
        probabilities=True,
    )
    regressor = ensemble.HistGradientBoostingRegressor
    search_space = search.SearchSpace(
        (
            search.FloatParameter("learning_rate", 0.02, 0.3, log=True),
            search.IntParameter("max_leaf_nodes", 8, 64),
        )
    )


ORDINAL_ADAPTERS: tuple[type[models.ModelAdapter], ...] = (
    OrdinalRidgeAdapter,
    OrdinalForestAdapter,
    OrdinalBoostingAdapter,
)
