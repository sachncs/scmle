"""Ranking model adapters."""

from collections.abc import Mapping
from typing import Any

from sklearn import ensemble

from scmle.contracts import models
from scmle.contracts import search
from scmle.contracts import tasks
from scmle.models.ranking import estimators

RANKING = frozenset({tasks.Task.RANKING})


class Pointwise(models.Model):
    """Gradient boosting regression on the relevance grade."""

    name = "pointwise_ranker"
    family = "boosting"
    capabilities = models.Capabilities(
        RANKING, handles_missing=True, cost=3, memory_factor=3.0
    )

    def build(
        self,
        task: tasks.Task,  # noqa: ARG002 -- required by Model.build
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Returns an unfitted pointwise ranker.

        Args:
            task: Resolved task type.
            seed: Random seed.
            params: Search-space overrides passed to the underlying model.

        Returns:
            A pointwise ranker around a histogram gradient-boosting regressor.
        """
        return estimators.Scorer(
            ensemble.HistGradientBoostingRegressor(
                random_state=seed, **dict(params or {})
            )
        )

    search_space = search.Space(
        (
            search.Real("learning_rate", 0.02, 0.3, log=True),
            search.Integer("max_leaf_nodes", 8, 64),
        )
    )


class Ranker(models.Model):
    """Base for LightGBM and XGBoost rankers.

    Attributes:
        library: Import name of the boosting library.
    """

    library = ""
    family = "boosting"
    capabilities = models.Capabilities(
        RANKING, handles_missing=True, cost=3, memory_factor=3.0
    )
    search_space = search.Space(
        (
            search.Real("learning_rate", 0.02, 0.3, log=True),
            search.Integer("max_leaves", 8, 64),
        )
    )

    def __init__(
        self,
        *,
        name: str | None = None,
        library: str | None = None,
        requires: tuple[str, ...] | None = None,
    ) -> None:
        """Sets the declared data on this instance.

        An argument left out keeps the class-level value, so a subclass that
        declares these names as class attributes keeps working.

        Args:
            name: Unique registry name.
            library: Import name of the boosting library.
            requires: Importable modules needed at build time.
        """
        given: dict[str, Any] = {
            "name": name,
            "library": library,
            "requires": requires,
        }
        for attribute, value in given.items():
            if value is not None:
                setattr(self, attribute, value)

    def build(
        self,
        task: tasks.Task,  # noqa: ARG002 -- required by Model.build
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Returns an unfitted boosted ranker.

        Args:
            task: Resolved task type.
            seed: Random seed.
            params: Search-space overrides passed to the underlying model.

        Returns:
            A ranker built on the adapter's boosting library.
        """
        return estimators.Library(
            self.library, random_state=seed, **dict(params or {})
        )


LIGHTGBM_RANKER = Ranker(
    name="lightgbm_ranker", library="lightgbm", requires=("lightgbm",)
)
"""LightGBM LambdaRank."""

XGBOOST_RANKER = Ranker(
    name="xgboost_ranker", library="xgboost", requires=("xgboost",)
)
"""XGBoost NDCG ranker."""


class Linear(models.Model):
    """A linear ranker learnt from preference pairs within each query."""

    name = "pairwise_linear_ranker"
    family = "linear"
    capabilities = models.Capabilities(RANKING, needs_scaling=True, cost=2)
    search_space = search.Space(
        (search.Real("penalty", 1e-3, 100.0, log=True),)
    )

    def build(
        self,
        task: tasks.Task,  # noqa: ARG002 -- required by Model.build
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Returns an unfitted pairwise linear ranker.

        Args:
            task: Resolved task type.
            seed: Seed of the pair sampling.
            params: Search-space overrides passed to the ranker.

        Returns:
            An unfitted pairwise ranker.
        """
        return estimators.Pairwise(random_state=seed, **dict(params or {}))


PAIRWISE = Linear()
"""Pairwise logistic (RankSVM-style) linear ranker: the linear family."""

RANKING_ADAPTERS: tuple[models.Model, ...] = (
    Pointwise(),
    LIGHTGBM_RANKER,
    XGBOOST_RANKER,
    PAIRWISE,
)
"""Ranking adapter instances, in registration order."""
