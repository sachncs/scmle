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

import numpy
from sklearn import base
from sklearn import ensemble
from sklearn import linear_model

from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import models
from scmle.contracts import tasks
from scmle.data.curation import preprocessing
from scmle.data.splitting import inner_cv

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
        """Stores the member count and the number of greedy rounds.

        Args:
            members: Number of members whose outputs are concatenated.
            rounds: Number of greedy additions.
        """
        self.members = members
        self.rounds = rounds

    def outputs(self, meta: Any) -> numpy.ndarray:
        """Splits the concatenated meta-features by member.

        Args:
            meta: Concatenated member outputs, one row per sample.

        Returns:
            Float array of shape (rows, members, columns per member).
        """
        values = numpy.asarray(meta, dtype=float)
        return values.reshape(len(values), self.members, -1)

    @abc.abstractmethod
    def prepare(
        self, stacked: numpy.ndarray, target: Any
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Puts the member outputs and the truth in comparable shapes.

        Args:
            stacked: Member outputs, shape (rows, members, columns).
            target: True targets, one per row.

        Returns:
            The member outputs, shape (rows, members, columns), and the
            truth, shape (rows, columns), matching the blended columns.
        """

    @abc.abstractmethod
    def loss(self, blend: numpy.ndarray, truth: numpy.ndarray) -> float:
        """Scores a blended prediction against the truth.

        Args:
            blend: Blended prediction, shape (rows, columns).
            truth: True values in the shape returned by `prepare`.

        Returns:
            The loss; smaller is better.
        """

    def fit(self, meta: Any, target: Any) -> Self:
        """Selects the blend weights on out-of-fold predictions.

        Args:
            meta: Concatenated out-of-fold member outputs, one row per
                sample.
            target: True targets, one per row.

        Returns:
            This blender, with `weights_` set.
        """
        stacked, truth = self.prepare(self.outputs(meta), target)
        counts = numpy.zeros(self.members)
        running = numpy.zeros_like(stacked[:, 0, :])
        for step in range(self.rounds):
            blends = (running[:, None, :] * step + stacked) / (step + 1)
            losses = [
                self.loss(blends[:, member, :], truth)
                for member in range(self.members)
            ]
            chosen = int(numpy.argmin(losses))
            counts[chosen] += 1
            running = blends[:, chosen, :]
        self.weights_ = counts / counts.sum()
        return self

    def blend(self, meta: Any) -> numpy.ndarray:
        """Averages the members' outputs with the selected weights.

        Args:
            meta: Concatenated member outputs, one row per sample.

        Returns:
            Array of shape (rows, columns per member) holding the weighted
            average.
        """
        stacked = self.expand(self.outputs(meta))
        return numpy.tensordot(
            self.weights_, stacked.transpose(1, 0, 2), axes=1
        )

    def expand(self, stacked: numpy.ndarray) -> numpy.ndarray:
        """Hook for turning single probability columns into two (binary).

        Args:
            stacked: Member outputs, shape (rows, members, columns).

        Returns:
            The member outputs, unchanged in this base class.
        """
        return stacked


class ClassifierBlender(base.ClassifierMixin, GreedyBlender):
    """Greedy blend of members' class probabilities."""

    def expand(self, stacked: numpy.ndarray) -> numpy.ndarray:
        """Adds the negative-class column for binary members.

        Args:
            stacked: Member outputs, shape (rows, members, columns).

        Returns:
            For members that give only P(positive), outputs with columns
            (P(negative), P(positive)); otherwise `stacked` unchanged.
        """
        if stacked.shape[2] == 1:
            return numpy.concatenate([1.0 - stacked, stacked], axis=2)
        return stacked

    def prepare(
        self, stacked: numpy.ndarray, target: Any
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Encodes the truth one-hot and expands binary member outputs.

        Args:
            stacked: Member outputs, shape (rows, members, columns).
            target: True class labels, one per row.

        Returns:
            The expanded member outputs, and the one-hot truth with one
            column per class in `classes_` order. Sets `classes_`.
        """
        self.classes_, encoded = numpy.unique(target, return_inverse=True)
        return self.expand(stacked), numpy.eye(len(self.classes_))[encoded]

    def loss(self, blend: numpy.ndarray, truth: numpy.ndarray) -> float:
        """Computes the log loss.

        Args:
            blend: Blended class probabilities, shape (rows, classes).
            truth: One-hot truth, shape (rows, classes).

        Returns:
            Mean negative log probability of the true class.
        """
        clipped = numpy.clip(blend, 1e-12, 1.0)
        return float(-numpy.mean(numpy.sum(truth * numpy.log(clipped), axis=1)))

    def predict_proba(self, meta: Any) -> numpy.ndarray:
        """Predicts class probabilities.

        Args:
            meta: Concatenated member outputs, one row per sample.

        Returns:
            Array of shape (rows, classes) in `classes_` order.
        """
        return self.blend(meta)

    def predict(self, meta: Any) -> numpy.ndarray:
        """Predicts the most probable class.

        Args:
            meta: Concatenated member outputs, one row per sample.

        Returns:
            Array of class labels, one per row.
        """
        return self.classes_[self.blend(meta).argmax(axis=1)]


class RegressorBlender(base.RegressorMixin, GreedyBlender):
    """Greedy blend of members' predictions."""

    def prepare(
        self, stacked: numpy.ndarray, target: Any
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Reshapes the truth to a column vector.

        Args:
            stacked: Member outputs, shape (rows, members, columns).
            target: True values, one per row.

        Returns:
            The member outputs unchanged, and the truth as floats with shape
            (rows, 1).
        """
        return stacked, numpy.asarray(target, dtype=float).reshape(-1, 1)

    def loss(self, blend: numpy.ndarray, truth: numpy.ndarray) -> float:
        """Computes the mean squared error.

        Args:
            blend: Blended predictions, shape (rows, 1).
            truth: True values, shape (rows, 1).

        Returns:
            Mean squared difference.
        """
        return float(numpy.mean((blend - truth) ** 2))

    def predict(self, meta: Any) -> numpy.ndarray:
        """Predicts the blended value.

        Args:
            meta: Concatenated member outputs, one row per sample.

        Returns:
            One-dimensional array of blended predictions, one per row.
        """
        return self.blend(meta)[:, 0]


class StackingAdapter(models.ModelAdapter):
    """Meta-model built from other registered models.

    Attributes:
        registry: Registry used to build the member estimators.
    """

    name = "stacking"
    family = "ensemble"
    capabilities = models.Capabilities(
        tasks.STANDARD_TASKS,
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

        Args:
            params: Model parameters, which may contain `members`.

        Returns:
            The list of member entries, each `{"model": name, "params":
            {...}}`.

        Raises:
            ConfigError: If fewer than two members are given.
        """
        members = list((params or {}).get("members", []))
        if len(members) < MIN_MEMBERS:
            raise errors.ConfigError("stacking needs at least two members")
        return members

    def assemble(
        self,
        plan: data.FeaturePlan,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator | None:
        """Builds the stack from complete member pipelines.

        Every member brings its own preprocessing, so the stack consumes the
        raw frame and its inner cross-validation can follow the group and
        time structure of the data.

        Args:
            plan: Feature plan the member pipelines preprocess with.
            task: Resolved task type.
            seed: Random seed.
            params: Must contain `members`: a list of
                `{"model": name, "params": {...}}` entries.

        Returns:
            An unfitted stacking classifier or regressor over the member
            pipelines. Fewer than two members is a `ConfigError`.
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
        """Creates the model that combines the members' out-of-fold predictions.

        Args:
            task: Resolved task type.
            members: Number of members; unused by this implementation.

        Returns:
            An unfitted logistic regression for classification, otherwise a
            cross-validated ridge regression.
        """
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
            An unfitted stacking classifier or regressor. Fewer than two
            members is a `ConfigError`.
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
        """Creates a greedy blender instead of a fitted linear model.

        Args:
            task: Resolved task type.
            members: Number of members whose outputs are blended.

        Returns:
            An unfitted classifier blender for classification, otherwise a
            regressor blender.
        """
        if task.is_classification:
            return ClassifierBlender(members)
        return RegressorBlender(members)
