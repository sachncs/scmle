"""Stacking and blending ensemble adapters.

A stack is assembled from the best configurations found during the search and
evaluated like any other candidate. scikit-learn's stacking cross-fits the
base models *inside* each training fold (out-of-fold meta-features), so the
meta-learner never sees predictions made on rows a base model was trained on.
"""

import abc
from collections.abc import Mapping
from typing import Any
from typing import Self

import numpy as np
from sklearn import base
from sklearn import ensemble
from sklearn import linear_model

from mml.contracts import data as data_contracts
from mml.contracts import errors
from mml.contracts import models
from mml.contracts import tasks
from mml.data.curation import preprocessing
from mml.data.splitting import inner_cv

STACK_FOLDS = 3
MIN_MEMBERS = 2


class GreedyBlender(base.BaseEstimator, abc.ABC):
    """Meta-learner that averages members with weights found greedily.

    Caruana-style ensemble selection: starting from nothing, repeatedly add
    (with replacement) the member that most lowers the loss of the running
    average of out-of-fold predictions. The share of times a member is chosen
    is its weight. Weights are non-negative and sum to one, so the blend can
    never extrapolate beyond its members, and it cannot overfit as freely as
    a free-form meta-model.

    The input is the concatenation of every member's out-of-fold output: one
    column per member for regression and binary classification, one column
    per class and member for multiclass.

    Attributes:
        members: Number of members whose outputs are concatenated.
        rounds: Number of greedy additions.
        weights_: Weight per member, set by fit.
    """

    def __init__(self, members: int = 2, rounds: int = 25) -> None:
        """Stores the member count and the number of greedy rounds."""
        self.members = members
        self.rounds = rounds

    def outputs(self, meta: Any) -> np.ndarray:
        """Per-member outputs, shape (rows, members, columns)."""
        values = np.asarray(meta, dtype=float)
        return values.reshape(len(values), self.members, -1)

    @abc.abstractmethod
    def prepare(
        self, stacked: np.ndarray, target: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns the member outputs and the truth in comparable shapes."""

    @abc.abstractmethod
    def loss(self, blend: np.ndarray, truth: np.ndarray) -> float:
        """Loss of a blended prediction."""

    def fit(self, meta: Any, target: Any) -> Self:
        """Selects the blend weights on out-of-fold predictions."""
        stacked, truth = self.prepare(self.outputs(meta), target)
        counts = np.zeros(self.members)
        running = np.zeros_like(stacked[:, 0, :])
        for step in range(self.rounds):
            blends = (running[:, None, :] * step + stacked) / (step + 1)
            losses = [
                self.loss(blends[:, member, :], truth)
                for member in range(self.members)
            ]
            chosen = int(np.argmin(losses))
            counts[chosen] += 1
            running = blends[:, chosen, :]
        self.weights_ = counts / counts.sum()
        return self

    def blend(self, meta: Any) -> np.ndarray:
        """The weighted average of the members' outputs."""
        stacked = self.expand(self.outputs(meta))
        return np.tensordot(self.weights_, stacked.transpose(1, 0, 2), axes=1)

    def expand(self, stacked: np.ndarray) -> np.ndarray:
        """Hook for turning single probability columns into two (binary)."""
        return stacked


class ClassifierBlender(base.ClassifierMixin, GreedyBlender):
    """Greedy blend of members' class probabilities."""

    def expand(self, stacked: np.ndarray) -> np.ndarray:
        """Binary members give P(positive) only; add P(negative)."""
        if stacked.shape[2] == 1:
            return np.concatenate([1.0 - stacked, stacked], axis=2)
        return stacked

    def prepare(
        self, stacked: np.ndarray, target: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """One-hot truth and two-column probabilities for binary members."""
        self.classes_, encoded = np.unique(target, return_inverse=True)
        return self.expand(stacked), np.eye(len(self.classes_))[encoded]

    def loss(self, blend: np.ndarray, truth: np.ndarray) -> float:
        """Log loss."""
        clipped = np.clip(blend, 1e-12, 1.0)
        return float(-np.mean(np.sum(truth * np.log(clipped), axis=1)))

    def predict_proba(self, meta: Any) -> np.ndarray:
        """Blended class probabilities."""
        return self.blend(meta)

    def predict(self, meta: Any) -> np.ndarray:
        """Most probable class."""
        return self.classes_[self.blend(meta).argmax(axis=1)]


class RegressorBlender(base.RegressorMixin, GreedyBlender):
    """Greedy blend of members' predictions."""

    def prepare(
        self, stacked: np.ndarray, target: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Column-vector truth."""
        return stacked, np.asarray(target, dtype=float).reshape(-1, 1)

    def loss(self, blend: np.ndarray, truth: np.ndarray) -> float:
        """Squared error."""
        return float(np.mean((blend - truth) ** 2))

    def predict(self, meta: Any) -> np.ndarray:
        """The blended value."""
        return self.blend(meta)[:, 0]


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

    @staticmethod
    def members_of(params: Mapping[str, Any] | None) -> list[dict[str, Any]]:
        """Returns the member specifications.

        Raises:
            ConfigError: If fewer than two members are given.
        """
        members = list((params or {}).get("members", []))
        if len(members) < MIN_MEMBERS:
            raise errors.ConfigError("stacking needs at least two members")
        return members

    def assemble(
        self,
        plan: data_contracts.FeaturePlan,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator | None:
        """Builds the stack from complete member pipelines.

        Every member brings its own preprocessing, so the stack consumes the
        raw frame and its inner cross-validation can follow the group and
        time structure of the data.
        """
        members = self.members_of(params)
        estimators = [
            (
                f"m{index}_{member['model']}",
                preprocessing.build_pipeline(
                    plan,
                    self.registry.get(member["model"]),
                    task,
                    seed,
                    member.get("params"),
                ),
            )
            for index, member in enumerate(members)
        ]
        inner = inner_cv.for_plan(
            plan, task.is_classification, STACK_FOLDS, seed
        )
        meta = self.meta_learner(task, len(estimators))
        if task.is_classification:
            return ensemble.StackingClassifier(
                estimators, final_estimator=meta, cv=inner, stack_method="auto"
            )
        return ensemble.StackingRegressor(
            estimators, final_estimator=meta, cv=inner
        )

    def meta_learner(self, task: tasks.TaskType, members: int) -> Any:
        """The model that combines the members' out-of-fold predictions."""
        if task.is_classification:
            return linear_model.LogisticRegression(max_iter=1000)
        return linear_model.RidgeCV()

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
        members = self.members_of(params)
        estimators = [
            (
                f"m{index}_{member['model']}",
                self.registry.get(member["model"]).build(
                    task, seed, member.get("params")
                ),
            )
            for index, member in enumerate(members)
        ]
        meta = self.meta_learner(task, len(estimators))
        if task.is_classification:
            return ensemble.StackingClassifier(
                estimators,
                final_estimator=meta,
                cv=STACK_FOLDS,
                stack_method="auto",
            )
        return ensemble.StackingRegressor(
            estimators, final_estimator=meta, cv=STACK_FOLDS
        )


class BlendingAdapter(StackingAdapter):
    """Weighted average of members, with weights chosen greedily."""

    name = "blending"

    def meta_learner(self, task: tasks.TaskType, members: int) -> Any:
        """A greedy blender instead of a fitted linear model."""
        if task.is_classification:
            return ClassifierBlender(members)
        return RegressorBlender(members)
