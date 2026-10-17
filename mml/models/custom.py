"""Register any scikit-learn-compatible estimator as a model family.

`register_estimator` builds an adapter around estimator classes, runs the
shared conformance suite on it, and adds it to the models every later project
considers. The classes must be importable by reference (defined at module
level), because trial processes receive the adapter by pickle.
"""

import dataclasses
from typing import Any

from sklearn import base

from mml.contracts import errors
from mml.contracts import models
from mml.contracts import search
from mml.contracts import tasks
from mml.models import conformance
from mml.models.sklearn import classical

CLASSIFICATION = classical.CLASSIFICATION
REGRESSION_ONLY = classical.REGRESSION_ONLY


@dataclasses.dataclass(frozen=True)
class EstimatorSpec:
    """Everything needed to rebuild a registered estimator's adapter.

    Attributes:
        name: Registry name.
        classifier: Classifier class, if the model does classification.
        regressor: Regressor class, if the model does regression.
        search_space: Tunable hyperparameters, if any.
        family: Coarse family for reports.
        needs_scaling: Benefits from standardized inputs.
        handles_missing: Handles missing values natively.
        sparse_input: Accepts sparse matrices.
        cost: Relative training cost from 1 to 5.
    """

    name: str
    classifier: type | None
    regressor: type | None
    search_space: search.SearchSpace | None = None
    family: str = "custom"
    needs_scaling: bool = False
    handles_missing: bool = False
    sparse_input: bool = False
    cost: int = 3


def rebuild(spec: EstimatorSpec) -> models.ModelAdapter:
    """Creates the adapter for `spec` (also how it is unpickled)."""
    supported: frozenset[tasks.TaskType] = frozenset()
    if spec.classifier is not None:
        supported |= CLASSIFICATION
    if spec.regressor is not None:
        supported |= REGRESSION_ONLY
    probabilities = spec.classifier is not None and hasattr(
        spec.classifier, "predict_proba"
    )
    members: dict[str, Any] = {
        "name": spec.name,
        "family": spec.family,
        "classifier": spec.classifier,
        "regressor": spec.regressor,
        "search_space": spec.search_space,
        "capabilities": models.Capabilities(
            supported,
            probabilities=probabilities,
            needs_scaling=spec.needs_scaling,
            handles_missing=spec.handles_missing,
            sparse_input=spec.sparse_input,
            cost=spec.cost,
        ),
        "spec": spec,
        "__reduce__": lambda self: (rebuild, (self.spec,)),
    }
    kind = type("CustomEstimatorAdapter", (classical.SklearnAdapter,), members)
    return kind()


def register_estimator(
    name: str,
    classifier: type[base.BaseEstimator] | None = None,
    regressor: type[base.BaseEstimator] | None = None,
    *,
    search_space: search.SearchSpace | None = None,
    family: str = "custom",
    needs_scaling: bool = False,
    handles_missing: bool = False,
    sparse_input: bool = False,
    cost: int = 3,
    replace: bool = False,
) -> models.ModelAdapter:
    """Makes an estimator class a candidate in every later project.

    Args:
        name: Public model name.
        classifier: Classifier class (unfitted, default-constructible).
        regressor: Regressor class.
        search_space: Hyperparameters to tune.
        family: Coarse family for reports.
        needs_scaling: Whether inputs should be standardized for it.
        handles_missing: Whether it accepts missing values itself.
        sparse_input: Whether it accepts sparse matrices.
        cost: Relative training cost, 1 (cheap) to 5.
        replace: Allow re-registering a name you registered before.

    Returns:
        The registered adapter.

    Raises:
        ConfigError: If neither class is given, the name belongs to a
            built-in model or an earlier registration, or the adapter fails
            the conformance suite (the message names the failed check).
    """
    from mml.models import registry  # Deferred: registry imports this module.

    if classifier is None and regressor is None:
        raise errors.ConfigError(
            f"{name}: give a classifier class, a regressor class, or both"
        )
    if name in registry.builtin_names():
        raise errors.ConfigError(f"'{name}' is a built-in model name")
    if name in registry.USER_ADAPTERS and not replace:
        raise errors.ConfigError(f"model '{name}' is already registered")
    adapter = rebuild(
        EstimatorSpec(
            name,
            classifier,
            regressor,
            search_space,
            family,
            needs_scaling,
            handles_missing,
            sparse_input,
            cost,
        )
    )
    conformance.check_conformance(adapter)
    registry.USER_ADAPTERS[name] = adapter
    return adapter


def unregister_estimator(name: str) -> None:
    """Removes a registered estimator (no-op if it is not registered)."""
    from mml.models import registry  # Deferred: registry imports this module.

    registry.USER_ADAPTERS.pop(name, None)
