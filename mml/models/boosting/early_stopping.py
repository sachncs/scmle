"""Early stopping for boosting libraries, with an internal holdout.

Boosting adds trees until told to stop; without a stopping rule it keeps
fitting noise (over-training). The libraries stop on a validation set they
are handed at fit time, which a scikit-learn pipeline cannot route, so this
wrapper carves the holdout out of the training rows it receives and wires
the right stopping arguments for each library.
"""

from typing import Any
from typing import Self

import numpy as np
from sklearn import base
from sklearn import model_selection

DEFAULT_ROUNDS = 25
DEFAULT_FRACTION = 0.15
MIN_ROWS = 60


class HoldoutEarlyStopping(base.BaseEstimator):
    """Stops boosting when a held-out slice stops improving.

    Attributes:
        estimator: Unfitted LightGBM, XGBoost or CatBoost estimator.
        rounds: Rounds without improvement before stopping.
        fraction: Share of the training rows held out for stopping.
        random_state: Seed of the holdout split.
        best_iteration_: Boosting rounds kept, set by fit (None when the
            data was too small to hold anything out).
    """

    def __init__(
        self,
        estimator: base.BaseEstimator,
        rounds: int = DEFAULT_ROUNDS,
        fraction: float = DEFAULT_FRACTION,
        random_state: int = 0,
    ) -> None:
        """Stores the wrapped estimator and the stopping rule."""
        self.estimator = estimator
        self.rounds = rounds
        self.fraction = fraction
        self.random_state = random_state

    @property
    def library(self) -> str:
        """Top-level package of the wrapped estimator."""
        return type(self.estimator).__module__.split(".")[0]

    def fit(self, features: Any, target: Any, **fit_params: Any) -> Self:
        """Fits with early stopping on an internal holdout.

        Args:
            features: Training matrix.
            target: Training labels or values.
            **fit_params: Extra arguments for the wrapped `fit`.

        Returns:
            The fitted wrapper.
        """
        self.estimator_ = base.clone(self.estimator)
        self.best_iteration_: int | None = None
        if len(target) < MIN_ROWS:
            self.estimator_.fit(features, target, **fit_params)
            return self
        stratify = (
            target
            if base.is_classifier(self.estimator)
            and np.min(np.unique(target, return_counts=True)[1]) >= 2
            else None
        )
        train, held = model_selection.train_test_split(
            np.arange(len(target)),
            test_size=self.fraction,
            random_state=self.random_state,
            stratify=None if stratify is None else np.asarray(stratify),
        )
        arguments = self.stopping_arguments(features, target, held)
        self.estimator_.fit(
            features[train]
            if hasattr(features, "tocsr")
            else self.take(features, train),
            np.asarray(target)[train],
            **arguments,
            **fit_params,
        )
        self.best_iteration_ = self.recorded_iteration()
        return self

    @staticmethod
    def take(features: Any, rows: np.ndarray) -> Any:
        """Selects rows of a dense matrix or a frame."""
        if hasattr(features, "iloc"):
            return features.iloc[rows]
        return np.asarray(features)[rows]

    def stopping_arguments(
        self, features: Any, target: Any, held: np.ndarray
    ) -> dict[str, Any]:
        """Per-library `fit` arguments that enable stopping on the holdout."""
        validation = (
            features[held]
            if hasattr(features, "tocsr")
            else self.take(features, held),
            np.asarray(target)[held],
        )
        if self.library == "lightgbm":
            import lightgbm

            return {
                "eval_set": [validation],
                "callbacks": [
                    lightgbm.early_stopping(self.rounds, verbose=False)
                ],
            }
        if self.library == "xgboost":
            self.estimator_.set_params(early_stopping_rounds=self.rounds)
            return {"eval_set": [validation], "verbose": False}
        self.estimator_.set_params(early_stopping_rounds=self.rounds)
        return {"eval_set": validation}

    def recorded_iteration(self) -> int | None:
        """Rounds the library kept, if it reports them."""
        for name in ("best_iteration_", "best_iteration"):
            value = getattr(self.estimator_, name, None)
            if isinstance(value, int | np.integer):
                return int(value)
        getter = getattr(self.estimator_, "get_best_iteration", None)
        return int(getter()) if callable(getter) else None

    def predict(self, features: Any) -> np.ndarray:
        """Predicts with the stopped model."""
        return np.asarray(self.estimator_.predict(features))

    def predict_proba(self, features: Any) -> np.ndarray:
        """Class probabilities of the stopped model."""
        return np.asarray(self.estimator_.predict_proba(features))

    @property
    def classes_(self) -> np.ndarray:
        """Class labels of the fitted model."""
        return np.asarray(self.estimator_.classes_)
