"""Optional gradient-boosting adapters (`boosting` extra).

Libraries are imported only when an estimator is built, so listing or planning
never pays for (or requires) them; without the extra the planner simply
records the adapter as rejected for a missing dependency.
"""

from collections.abc import Mapping
import importlib
from typing import Any
from typing import cast

from scmle.contracts import errors
from scmle.contracts import models
from scmle.contracts import search
from scmle.contracts import tasks
from scmle.models.boosting import early_stopping
from scmle.preparation import wrappers

NO_PARAMS: dict[str, Any] = {}


class Booster(models.Model):
    """Adapter whose estimator classes live in an optional library.

    Configure it with keyword arguments, or subclass it and declare the same
    names as class attributes. An argument left out keeps the class-level
    value, so both styles work.

    Attributes:
        library: Importable top-level module name.
        classifier: Classifier class name inside the library.
        regressor: Regressor class name inside the library.
        relabel: Wrap classifiers so arbitrary class labels
            work even though the library only accepts 0..K-1.
        predictions: Wrap classifiers whose `predict` returns column
            vectors so predictions are 1-D.
        fixed: Constructor parameters applied to every build.
        fitting: Search-space entries passed to `fit`, not to the
            constructor.
        rounds: Early-stopping patience, or None to disable.
    """

    library: str
    relabel: bool = False
    predictions: bool = False
    classifier: str
    regressor: str
    fixed: dict[str, Any] = NO_PARAMS
    fitting: tuple[str, ...] = ()
    rounds: int | None = early_stopping.DEFAULT_ROUNDS

    def __init__(
        self,
        *,
        name: str | None = None,
        family: str | None = None,
        capabilities: models.Capabilities | None = None,
        requires: tuple[str, ...] | None = None,
        library: str | None = None,
        classifier: str | None = None,
        regressor: str | None = None,
        relabel: bool | None = None,
        predictions: bool | None = None,
        fixed: dict[str, Any] | None = None,
        fitting: tuple[str, ...] | None = None,
        rounds: int | None = None,
        search_space: search.Space | None = None,
        monotonic_param: str | None = None,
    ) -> None:
        """Sets the declared data on this instance.

        Args:
            name: Unique registry name.
            family: Coarse family used for reporting.
            capabilities: Declared capabilities.
            requires: Importable modules the adapter needs.
            library: Importable top-level module name.
            classifier: Classifier class name inside the library.
            regressor: Regressor class name inside the library.
            relabel: Wrap classifiers so arbitrary class labels
                work even though the library only accepts 0..K-1.
            predictions: Wrap classifiers whose `predict` returns
                column vectors so predictions are 1-D.
            fixed: Constructor parameters applied to every build.
            fitting: Search-space entries passed to `fit`.
            rounds: Early-stopping patience; left out keeps the
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
            "classifier": classifier,
            "regressor": regressor,
            "relabel": relabel,
            "predictions": predictions,
            "fixed": fixed,
            "fitting": fitting,
            "rounds": rounds,
            "space": search_space,
            "param": monotonic_param,
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
            params: Search-space overrides applied over `fixed`.

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
        class_name = self.classifier if task.classification else self.regressor
        given = dict(params or {})
        fit_params = {
            name: given.pop(name) for name in self.fitting if name in given
        }
        kwargs: dict[str, Any] = {
            **self.fixed,
            "random_state": seed,
            **given,
        }
        estimator: Any = getattr(module, class_name)(**kwargs)
        if self.rounds:
            estimator = early_stopping.wrap(estimator, self.rounds, seed)
        if fit_params:
            wrapper = (
                wrappers.Forwarder if task.classification else wrappers.Curried
            )
            estimator = wrapper(estimator, fit_params)
        if task.classification and self.relabel:
            return wrappers.Encoded(estimator)
        if task.classification and self.predictions:
            return wrappers.Flattener(estimator)
        return cast("models.Estimator", estimator)


class Lightgbm(Booster):
    """LightGBM, which takes native categoricals at fit time."""

    def nominals(self, indices: list[int]) -> dict[str, Any]:
        """Marks the integer-coded columns as categorical.

        Args:
            indices: Positions of the integer-coded categorical columns.

        Returns:
            Fit-time parameters naming those columns, as
            `{"categorical_feature": [...]}`.
        """
        return {"categorical_feature": list(indices)}


LIGHTGBM = Lightgbm(
    name="lightgbm",
    family="boosting",
    library="lightgbm",
    requires=("lightgbm",),
    classifier="LGBMClassifier",
    regressor="LGBMRegressor",
    monotonic_param="monotone_constraints",
    fitting=("categorical_feature",),
    capabilities=models.Capabilities(
        tasks.STANDARD_TASKS,
        sparse=True,
        missing=True,
        categorical=True,
        early_stopping=True,
        cost=3,
        factor=2.0,
    ),
    fixed={"verbose": -1, "subsample_freq": 1, "n_jobs": 1},
    search_space=search.Space(
        (
            search.Integer("n_estimators", 50, 400),
            search.Real("learning_rate", 0.01, 0.3, log=True),
            search.Integer("num_leaves", 8, 64),
            search.Integer("min_child_samples", 5, 60),
            search.Real("subsample", 0.6, 1.0),
            search.Real("colsample_bytree", 0.5, 1.0),
            search.Real("reg_lambda", 1e-3, 10.0, log=True),
        )
    ),
)

XGBOOST = Booster(
    name="xgboost",
    family="boosting",
    library="xgboost",
    requires=("xgboost",),
    relabel=True,
    classifier="XGBClassifier",
    regressor="XGBRegressor",
    monotonic_param="monotone_constraints",
    capabilities=models.Capabilities(
        tasks.STANDARD_TASKS,
        sparse=True,
        missing=True,
        categorical=False,
        early_stopping=True,
        cost=3,
        factor=2.5,
    ),
    fixed={"verbosity": 0, "n_jobs": 1},
    search_space=search.Space(
        (
            search.Integer("n_estimators", 50, 400),
            search.Real("learning_rate", 0.01, 0.3, log=True),
            search.Integer("max_depth", 2, 10),
            search.Integer("min_child_weight", 1, 10),
            search.Real("subsample", 0.6, 1.0),
            search.Real("colsample_bytree", 0.5, 1.0),
            search.Real("reg_lambda", 1e-3, 10.0, log=True),
        )
    ),
)

CATBOOST = Booster(
    name="catboost",
    family="boosting",
    predictions=True,
    library="catboost",
    requires=("catboost",),
    classifier="CatBoostClassifier",
    regressor="CatBoostRegressor",
    capabilities=models.Capabilities(
        tasks.STANDARD_TASKS,
        sparse=True,
        missing=True,
        categorical=False,
        early_stopping=True,
        gpu=False,
        cost=4,
        factor=3.0,
    ),
    fixed={
        "verbose": 0,
        "allow_writing_files": False,
        "thread_count": 1,
    },
    search_space=search.Space(
        (
            search.Integer("iterations", 100, 500),
            search.Integer("depth", 3, 8),
            search.Real("learning_rate", 0.02, 0.3, log=True),
            search.Real("l2_leaf_reg", 1.0, 10.0, log=True),
        )
    ),
)

BOOSTING_ADAPTERS: tuple[models.Model, ...] = (LIGHTGBM, XGBOOST, CATBOOST)
