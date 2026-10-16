"""Bagging: variance reduction for unstable learners.

Each member is trained on a bootstrap sample of the rows (and optionally a
subset of the columns) and the members are averaged. This helps most for
learners whose predictions swing with small changes in the data, such as deep
trees and nearest neighbours.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import ClassVar

from sklearn import ensemble

from mml.contracts import models
from mml.contracts import search
from mml.contracts import tasks
from mml.models.sklearn import adapters

BAG_PARAMS = ("n_estimators", "max_samples", "max_features", "bootstrap")
DEFAULT_MEMBERS = 25
MAX_COST = 5
MEMORY_MULTIPLIER = 3


class BaggedAdapter(models.ModelAdapter):
    """Bags a base model family.

    Attributes:
        base: Adapter class of the model that is bagged.
    """

    family = "bagging"
    base: ClassVar[type[models.ModelAdapter]]
    search_space = search.SearchSpace(
        (
            search.IntParameter("n_estimators", 10, 60),
            search.FloatParameter("max_samples", 0.5, 1.0),
            search.FloatParameter("max_features", 0.5, 1.0),
        )
    )

    def __init_subclass__(cls, **options: Any) -> None:
        """Derives the capabilities from the base model.

        The ensemble costs more, needs dense input, and keeps whatever the
        base model can do about missing values and probabilities.
        """
        super().__init_subclass__(**options)
        inner = cls.base.capabilities
        cls.capabilities = dataclasses.replace(
            inner,
            sparse_input=False,
            cost=min(inner.cost + 1, MAX_COST),
            memory_factor=inner.memory_factor * MEMORY_MULTIPLIER,
        )

    def build(
        self,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Creates the bagging estimator around a fresh base model.

        Args:
            task: Resolved task type.
            seed: Random seed.
            params: Bagging parameters, and any override of the base model.

        Returns:
            An unfitted bagging ensemble.
        """
        given = dict(params or {})
        bag = {name: given.pop(name) for name in BAG_PARAMS if name in given}
        bag.setdefault("n_estimators", DEFAULT_MEMBERS)
        member = self.base().build(task, seed, given)
        bagger = (
            ensemble.BaggingClassifier
            if task.is_classification
            else ensemble.BaggingRegressor
        )
        return bagger(estimator=member, random_state=seed, n_jobs=1, **bag)


class BaggedTreeAdapter(BaggedAdapter):
    """Bagged deep decision trees."""

    name = "bagged_decision_tree"
    base = adapters.DecisionTreeAdapter


class BaggedKnnAdapter(BaggedAdapter):
    """Bagged nearest neighbours."""

    name = "bagged_knn"
    base = adapters.KnnAdapter


class BaggedLogisticAdapter(BaggedAdapter):
    """Bagged logistic regression."""

    name = "bagged_logistic_regression"
    base = adapters.LogisticRegressionAdapter


BAGGING_ADAPTERS: tuple[type[models.ModelAdapter], ...] = (
    BaggedTreeAdapter,
    BaggedKnnAdapter,
    BaggedLogisticAdapter,
)
