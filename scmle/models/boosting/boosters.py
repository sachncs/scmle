"""Optional gradient-boosting adapters (`boosting` extra).

Libraries are imported only when an estimator is built, so listing or planning
never pays for (or requires) them; without the extra the planner simply
records the adapter as rejected for a missing dependency.
"""

from collections.abc import Mapping
import importlib
from typing import Any

from scmle.contracts import errors
from scmle.contracts import models
from scmle.contracts import search
from scmle.contracts import tasks
from scmle.models import wrappers
from scmle.models.boosting import early_stopping

NO_PARAMS: dict[str, Any] = {}


class Booster(models.Model):
    """Adapter whose estimator classes live in an optional library.

    Configure it with keyword arguments, or subclass it and declare the same
    names as class attributes. An argument left out keeps the class-level
    value, so both styles work.

    Attributes:
        library: Importable top-level module name.
        classifier_name: Classifier class name inside the library.
        regressor_name: Regressor class name inside the library.
        needs_integer_labels: Wrap classifiers so arbitrary class labels
            work even though the library only accepts 0..K-1.
        flatten_predictions: Wrap classifiers whose `predict` returns column
            vectors so predictions are 1-D.
        fixed_params: Constructor parameters applied to every build.
        fit_param_names: Search-space entries passed to `fit`, not to the
            constructor.
        stopping_rounds: Early-stopping patience, or None to disable.
    """

    library: str
    needs_integer_labels: bool = False
    flatten_predictions: bool = False
    classifier_name: str
    regressor_name: str
    fixed_params: dict[str, Any] = NO_PARAMS
    fit_param_names: tuple[str, ...] = ()
    stopping_rounds: int | None = early_stopping.DEFAULT_ROUNDS

    def __init__(
        self,
        *,
        name: str | None = None,
        family: str | None = None,
        capabilities: models.Capabilities | None = None,
        requires: tuple[str, ...] | None = None,
        library: str | None = None,
        classifier_name: str | None = None,
        regressor_name: str | None = None,
        needs_integer_labels: bool | None = None,
        flatten_predictions: bool | None = None,
        fixed_params: dict[str, Any] | None = None,
        fit_param_names: tuple[str, ...] | None = None,
        stopping_rounds: int | None = None,
        search_space: search.SearchSpace | None = None,
        monotonic_param: str | None = None,
    ) -> None:
        """Sets the declared data on this instance.

        Args:
            name: Unique registry name.
            family: Coarse family used for reporting.
            capabilities: Declared capabilities.
            requires: Importable modules the adapter needs.
            library: Importable top-level module name.
            classifier_name: Classifier class name inside the library.
            regressor_name: Regressor class name inside the library.
            needs_integer_labels: Wrap classifiers so arbitrary class labels
                work even though the library only accepts 0..K-1.
            flatten_predictions: Wrap classifiers whose `predict` returns
                column vectors so predictions are 1-D.
            fixed_params: Constructor parameters applied to every build.
            fit_param_names: Search-space entries passed to `fit`.
            stopping_rounds: Early-stopping patience; left out keeps the
                default, which a subclass can replace at class level.
            search_space: Tunable hyperparameters.
            monotonic_param: Name of the estimator parameter that takes one
                monotonic sign per input column.
        """
        given: dict[str, Any] = {
            "name": name,
            "family": family,
            "capabilities": capabilities,
            "requires": requires,
            "library": library,
            "classifier_name": classifier_name,
            "regressor_name": regressor_name,
            "needs_integer_labels": needs_integer_labels,
            "flatten_predictions": flatten_predictions,
            "fixed_params": fixed_params,
            "fit_param_names": fit_param_names,
            "stopping_rounds": stopping_rounds,
            "search_space": search_space,
            "monotonic_param": monotonic_param,
        }
        for attribute, value in given.items():
            if value is not None:
                setattr(self, attribute, value)

    def build(
        self,
        task: tasks.Task,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Imports the library and instantiates the matching class.

        Args:
            task: Resolved task type.
            seed: Random seed.
            params: Search-space overrides applied over `fixed_params`.

        Returns:
            An unfitted estimator.

        Raises:
            ConfigError: If the library is not installed.
        """
        if not self.is_available():
            raise errors.ConfigError(
                f"{self.name}: install the 'boosting' extra ({self.library})"
            )
        module = importlib.import_module(self.library)
        class_name = (
            self.classifier_name
            if task.is_classification
            else self.regressor_name
        )
        given = dict(params or {})
        fit_params = {
            name: given.pop(name)
            for name in self.fit_param_names
            if name in given
        }
        kwargs: dict[str, Any] = {
            **self.fixed_params,
            "random_state": seed,
            **given,
        }
        estimator: Any = getattr(module, class_name)(**kwargs)
        if self.stopping_rounds:
            estimator = early_stopping.wrap(
                estimator, self.stopping_rounds, seed
            )
        if fit_params:
            wrapper = (
                wrappers.FitParamsClassifier
                if task.is_classification
                else wrappers.FitParamsRegressor
            )
            estimator = wrapper(estimator, fit_params)
        if task.is_classification and self.needs_integer_labels:
            return wrappers.LabelEncodedClassifier(estimator)
        if task.is_classification and self.flatten_predictions:
            return wrappers.FlatPredictionClassifier(estimator)
        return estimator


class LightGbm(Booster):
    """LightGBM, which takes native categoricals at fit time."""

    def categorical_params(self, indices: list[int]) -> dict[str, Any]:
        """Marks the integer-coded columns as categorical.

        Args:
            indices: Positions of the integer-coded categorical columns.

        Returns:
            Fit-time parameters naming those columns, as
            `{"categorical_feature": [...]}`.
        """
        return {"categorical_feature": list(indices)}


LIGHTGBM = LightGbm(
    name="lightgbm",
    family="boosting",
    library="lightgbm",
    requires=("lightgbm",),
    classifier_name="LGBMClassifier",
    regressor_name="LGBMRegressor",
    monotonic_param="monotone_constraints",
    fit_param_names=("categorical_feature",),
    capabilities=models.Capabilities(
        tasks.STANDARD_TASKS,
        sparse_input=True,
        handles_missing=True,
        native_categorical=True,
        early_stopping=True,
        cost=3,
        memory_factor=2.0,
    ),
    fixed_params={"verbose": -1, "subsample_freq": 1, "n_jobs": 1},
    search_space=search.SearchSpace(
        (
            search.IntParameter("n_estimators", 50, 400),
            search.FloatParameter("learning_rate", 0.01, 0.3, log=True),
            search.IntParameter("num_leaves", 8, 64),
            search.IntParameter("min_child_samples", 5, 60),
            search.FloatParameter("subsample", 0.6, 1.0),
            search.FloatParameter("colsample_bytree", 0.5, 1.0),
            search.FloatParameter("reg_lambda", 1e-3, 10.0, log=True),
        )
    ),
)

XGBOOST = Booster(
    name="xgboost",
    family="boosting",
    library="xgboost",
    requires=("xgboost",),
    needs_integer_labels=True,
    classifier_name="XGBClassifier",
    regressor_name="XGBRegressor",
    monotonic_param="monotone_constraints",
    capabilities=models.Capabilities(
        tasks.STANDARD_TASKS,
        sparse_input=True,
        handles_missing=True,
        native_categorical=False,
        early_stopping=True,
        cost=3,
        memory_factor=2.5,
    ),
    fixed_params={"verbosity": 0, "n_jobs": 1},
    search_space=search.SearchSpace(
        (
            search.IntParameter("n_estimators", 50, 400),
            search.FloatParameter("learning_rate", 0.01, 0.3, log=True),
            search.IntParameter("max_depth", 2, 10),
            search.IntParameter("min_child_weight", 1, 10),
            search.FloatParameter("subsample", 0.6, 1.0),
            search.FloatParameter("colsample_bytree", 0.5, 1.0),
            search.FloatParameter("reg_lambda", 1e-3, 10.0, log=True),
        )
    ),
)

CATBOOST = Booster(
    name="catboost",
    family="boosting",
    flatten_predictions=True,
    library="catboost",
    requires=("catboost",),
    classifier_name="CatBoostClassifier",
    regressor_name="CatBoostRegressor",
    capabilities=models.Capabilities(
        tasks.STANDARD_TASKS,
        sparse_input=True,
        handles_missing=True,
        native_categorical=False,
        early_stopping=True,
        gpu=False,
        cost=4,
        memory_factor=3.0,
    ),
    fixed_params={
        "verbose": 0,
        "allow_writing_files": False,
        "thread_count": 1,
    },
    search_space=search.SearchSpace(
        (
            search.IntParameter("iterations", 100, 500),
            search.IntParameter("depth", 3, 8),
            search.FloatParameter("learning_rate", 0.02, 0.3, log=True),
            search.FloatParameter("l2_leaf_reg", 1.0, 10.0, log=True),
        )
    ),
)

BOOSTING_ADAPTERS: tuple[models.Model, ...] = (LIGHTGBM, XGBOOST, CATBOOST)
