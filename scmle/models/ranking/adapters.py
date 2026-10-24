"""Ranking model adapters."""

from collections.abc import Mapping
from typing import Any

from scmle.contracts import models
from scmle.contracts import search
from scmle.contracts import tasks
from scmle.models.ranking import estimators

RANKING = frozenset({tasks.TaskType.RANKING})


class PointwiseAdapter(models.ModelAdapter):
    """Gradient boosting regression on the relevance grade."""

    name = "pointwise_ranker"
    family = "boosting"
    capabilities = models.Capabilities(
        RANKING, handles_missing=True, cost=3, memory_factor=3.0
    )

    def build(
        self,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Returns an unfitted pointwise ranker."""
        from sklearn import ensemble

        return estimators.PointwiseRanker(
            ensemble.HistGradientBoostingRegressor(
                random_state=seed, **dict(params or {})
            )
        )

    search_space = search.SearchSpace(
        (
            search.FloatParameter("learning_rate", 0.02, 0.3, log=True),
            search.IntParameter("max_leaf_nodes", 8, 64),
        )
    )


class LibraryRankerAdapter(models.ModelAdapter):
    """Base for LightGBM and XGBoost rankers.

    Attributes:
        library: Import name of the boosting library.
    """

    library = ""
    family = "boosting"
    capabilities = models.Capabilities(
        RANKING, handles_missing=True, cost=3, memory_factor=3.0
    )
    search_space = search.SearchSpace(
        (
            search.FloatParameter("learning_rate", 0.02, 0.3, log=True),
            search.IntParameter("max_leaves", 8, 64),
        )
    )

    def build(
        self,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Returns an unfitted boosted ranker."""
        return estimators.LibraryRanker(
            self.library, random_state=seed, **dict(params or {})
        )


class LightGbmRankerAdapter(LibraryRankerAdapter):
    """LightGBM LambdaRank."""

    name = "lightgbm_ranker"
    library = "lightgbm"
    requires = ("lightgbm",)


class XgboostRankerAdapter(LibraryRankerAdapter):
    """XGBoost NDCG ranker."""

    name = "xgboost_ranker"
    library = "xgboost"
    requires = ("xgboost",)


RANKING_ADAPTERS: tuple[type[models.ModelAdapter], ...] = (
    PointwiseAdapter,
    LightGbmRankerAdapter,
    XgboostRankerAdapter,
)
