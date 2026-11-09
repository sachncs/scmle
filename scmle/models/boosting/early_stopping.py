"""Early stopping for boosting libraries, with an internal holdout.

Boosting adds trees until told to stop; without a stopping rule it keeps
fitting noise (over-training). The libraries stop on a validation set they
are handed at fit time, which a scikit-learn pipeline cannot route, so this
wrapper carves the holdout out of the training rows it receives and wires
the right stopping arguments for each library.
"""

import inspect
from typing import Any
from typing import Self

import numpy
from sklearn import base
from sklearn import model_selection
from sklearn import preprocessing

DEFAULT_ROUNDS = 25
DEFAULT_FRACTION = 0.15
MIN_ROWS = 60
MIN_CLASS_COUNT_TO_STRATIFY = 2


class HoldoutEarlyStopping(base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
    """Stops boosting when a held-out slice stops improving.

    Attributes:
        estimator: Unfitted LightGBM, XGBoost or CatBoost estimator.
        rounds: Rounds without improvement before stopping.
        fraction: Share of the training rows held out for stopping.
        random_state: Seed of the holdout split.
        best_iteration_: Boosting rounds kept, set by fit. None when the
            data was too small to hold anything out (fewer than `MIN_ROWS`
            rows) or the library does not report it.
    """

    def __init__(
        self,
        estimator: base.BaseEstimator,
        rounds: int = DEFAULT_ROUNDS,
        fraction: float = DEFAULT_FRACTION,
        random_state: int = 0,
    ) -> None:
        """Stores the wrapped estimator and the stopping rule.

        Args:
            estimator: Unfitted LightGBM, XGBoost or CatBoost estimator.
            rounds: Rounds without improvement before stopping.
            fraction: Share of the training rows held out for stopping.
            random_state: Seed of the holdout split.
        """
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
        if base.is_classifier(self.estimator):
            self.encoder_ = preprocessing.LabelEncoder().fit(target)
            target = self.encoder_.transform(target)
        if len(target) < MIN_ROWS:
            self.estimator_.fit(features, target, **fit_params)
            return self
        stratify = (
            target
            if base.is_classifier(self.estimator)
            and numpy.min(numpy.unique(target, return_counts=True)[1])
            >= MIN_CLASS_COUNT_TO_STRATIFY
            else None
        )
        train, held = model_selection.train_test_split(
            numpy.arange(len(target)),
            test_size=self.fraction,
            random_state=self.random_state,
            stratify=None if stratify is None else numpy.asarray(stratify),
        )
        arguments = self.stopping_arguments(features, target, held)
        self.estimator_.fit(
            features[train]
            if hasattr(features, "tocsr")
            else self.take(features, train),
            numpy.asarray(target)[train],
            **arguments,
            **fit_params,
        )
        self.best_iteration_ = self.recorded_iteration()
        return self

    @staticmethod
    def take(features: Any, rows: numpy.ndarray) -> Any:
        """Selects rows of a dense matrix or a frame.

        Args:
            features: Dense matrix or pandas frame.
            rows: Integer positions of the rows to keep.

        Returns:
            The selected rows, as a frame when `features` is one and as a numpy
            array otherwise.
        """
        if hasattr(features, "iloc"):
            return features.iloc[rows]
        return numpy.asarray(features)[rows]

    def stopping_arguments(
        self, features: Any, target: Any, held: numpy.ndarray
    ) -> dict[str, Any]:
        """Builds the per-library `fit` arguments that enable stopping.

        For XGBoost and CatBoost this also sets `early_stopping_rounds` on the
        cloned estimator before returning.

        Args:
            features: Full training matrix.
            target: Encoded training labels or values.
            held: Integer positions of the rows held out as the validation set.

        Returns:
            Keyword arguments for the library's `fit`: an evaluation set built
            from the held-out rows, plus the stopping callback (LightGBM) or
            `verbose=False` (XGBoost).
        """
        validation = (
            features[held]
            if hasattr(features, "tocsr")
            else self.take(features, held),
            numpy.asarray(target)[held],
        )
        if self.library == "lightgbm":
            import lightgbm  # noqa: PLC0415 -- optional dependency

            stopper = [lightgbm.early_stopping(self.rounds, verbose=False)]
            if "eval_X" in inspect.signature(self.estimator_.fit).parameters:
                return {
                    "eval_X": validation[0],
                    "eval_y": validation[1],
                    "callbacks": stopper,
                }
            return {"eval_set": [validation], "callbacks": stopper}
        if self.library == "xgboost":
            self.estimator_.set_params(early_stopping_rounds=self.rounds)
            return {"eval_set": [validation], "verbose": False}
        self.estimator_.set_params(early_stopping_rounds=self.rounds)
        return {"eval_set": validation}

    def recorded_iteration(self) -> int | None:
        """Rounds the library kept, if it reports them.

        Returns:
            The best iteration read from `best_iteration_`, `best_iteration` or
            `get_best_iteration()` of the fitted estimator, or None when none of
            them is available.
        """
        for name in ("best_iteration_", "best_iteration"):
            value = getattr(self.estimator_, name, None)
            if isinstance(value, int | numpy.integer):
                return int(value)
        getter = getattr(self.estimator_, "get_best_iteration", None)
        return int(getter()) if callable(getter) else None

    def predict(self, features: Any) -> numpy.ndarray:
        """Predicts with the stopped model, in the original labels.

        Args:
            features: Matrix to predict for.

        Returns:
            One prediction per row; class labels are mapped back to the original
            labels for classifiers.
        """
        predicted = numpy.asarray(self.estimator_.predict(features))
        if hasattr(self, "encoder_"):
            decoded: numpy.ndarray = self.encoder_.inverse_transform(
                predicted.astype(int)
            )
            return decoded
        return predicted

    def predict_proba(self, features: Any) -> numpy.ndarray:
        """Class probabilities of the stopped model.

        Args:
            features: Matrix to predict for.

        Returns:
            Array of shape `(n_rows, n_classes)`, columns in `classes_` order.
        """
        return numpy.asarray(self.estimator_.predict_proba(features))


class EarlyStoppingClassifier(base.ClassifierMixin, HoldoutEarlyStopping):  # type: ignore[misc]  # scikit-learn ships no type information
    """Early-stopped boosting classifier."""

    @property
    def classes_(self) -> numpy.ndarray:
        """Class labels of the fitted model."""
        return numpy.asarray(self.encoder_.classes_)


class EarlyStoppingRegressor(base.RegressorMixin, HoldoutEarlyStopping):  # type: ignore[misc]  # scikit-learn ships no type information
    """Early-stopped boosting regressor."""


def wrap(
    estimator: base.BaseEstimator, rounds: int = DEFAULT_ROUNDS, seed: int = 0
) -> HoldoutEarlyStopping:
    """Wraps an estimator in the early-stopping wrapper of its kind.

    Args:
        estimator: Unfitted boosting classifier or regressor.
        rounds: Rounds without improvement before stopping.
        seed: Seed of the holdout split.

    Returns:
        A classifier or regressor wrapper, so scikit-learn scorers treat it
        as the right kind of estimator.
    """
    kind = (
        EarlyStoppingClassifier
        if base.is_classifier(estimator)
        else EarlyStoppingRegressor
    )
    return kind(estimator, rounds, random_state=seed)
