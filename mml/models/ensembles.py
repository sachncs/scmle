"""Stacking ensemble adapter.

A stack is assembled from the best configurations found during the search and
evaluated like any other candidate. scikit-learn's stacking cross-fits the
base models *inside* each training fold (out-of-fold meta-features), so the
meta-learner never sees predictions made on rows a base model was trained on.
"""

from collections.abc import Mapping
from typing import Any

from sklearn import ensemble
from sklearn import linear_model

from mml.contracts import errors
from mml.contracts import models
from mml.contracts import tasks

STACK_FOLDS = 3
MIN_MEMBERS = 2


class StackingAdapter(models.ModelAdapter):
    """Meta-model built from other registered models.

    Attributes:
        registry: Registry used to build the member estimators.
    """

    name = "stacking"
    family = "ensemble"
    capabilities = models.Capabilities(
        frozenset(tasks.TaskType),
        needs_scaling=True,
        standalone=False,
        cost=5,
    )

    conformance_params = {
        "members": [{"model": "decision_tree"}, {"model": "extra_trees"}]
    }

    def __init__(self, registry: models.AdapterResolver) -> None:
        """Initializes the adapter.

        Args:
            registry: Registry that resolves member model names.
        """
        self.registry = registry

    def build(
        self,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Builds a stacking estimator from `params["members"]`.

        Args:
            task: Resolved task type.
            seed: Random seed.
            params: Must contain `members`: a list of
                `{"model": name, "params": {...}}` entries.

        Returns:
            An unfitted stacking classifier or regressor.

        Raises:
            ConfigError: If fewer than two members are given.
        """
        members = list((params or {}).get("members", []))
        if len(members) < MIN_MEMBERS:
            raise errors.ConfigError("stacking needs at least two members")
        estimators = [
            (
                f"m{index}_{member['model']}",
                self.registry.get(member["model"]).build(
                    task, seed, member.get("params")
                ),
            )
            for index, member in enumerate(members)
        ]
        if task.is_classification:
            return ensemble.StackingClassifier(
                estimators,
                final_estimator=linear_model.LogisticRegression(max_iter=1000),
                cv=STACK_FOLDS,
                stack_method="auto",
            )
        return ensemble.StackingRegressor(
            estimators,
            final_estimator=linear_model.RidgeCV(),
            cv=STACK_FOLDS,
        )
