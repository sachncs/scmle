"""Register any scikit-learn-compatible estimator as a model family.

`register` builds an adapter around estimator classes, runs the
shared conformance suite on it, and adds it to the models every later project
considers. The classes must be importable by reference (defined at module
level), because trial processes receive the adapter by pickle.
"""

import dataclasses
from typing import Any

from sklearn import base

from scmle.contracts import errors
from scmle.contracts import models
from scmle.contracts import search
from scmle.contracts import tasks
from scmle.models import conformance
from scmle.models.sklearn import classical
from scmle.runtime import plugins


@dataclasses.dataclass(frozen=True)
class Registration:
    """Everything needed to rebuild a registered estimator's adapter.

    Attributes:
        name: Registry name.
        classifier: Classifier class, if the model does classification.
        regressor: Regressor class, if the model does regression.
        space: Tunable hyperparameters, if any.
        family: Coarse family for reports.
        scaling: Benefits from standardized inputs.
        missing: Handles missing values natively.
        sparse: Accepts sparse matrices.
        cost: Relative training cost from 1 to 5.
    """

    name: str
    classifier: type | None
    regressor: type | None
    space: search.Space | None = None
    family: str = "custom"
    scaling: bool = False
    missing: bool = False
    sparse: bool = False
    cost: int = 3


def rebuild(spec: Registration) -> models.Model:
    """Creates the adapter for `spec` (also how it is unpickled).

    Args:
        spec: Description of the estimator to adapt.

    Returns:
        A new adapter whose capabilities follow `spec`: classification tasks
        if a classifier class is given, regression if a regressor class is
        given, and probabilities if the classifier has `predict_proba`.
    """
    supported: frozenset[tasks.Task] = frozenset()
    if spec.classifier is not None:
        supported |= classical.CLASSIFICATION
    if spec.regressor is not None:
        supported |= classical.REGRESSION_ONLY
    probabilities = spec.classifier is not None and hasattr(
        spec.classifier, "predict_proba"
    )
    members: dict[str, Any] = {
        "name": spec.name,
        "family": spec.family,
        "classifier": spec.classifier,
        "regressor": spec.regressor,
        "space": spec.space,
        "capabilities": models.Capabilities(
            supported,
            probabilities=probabilities,
            scaling=spec.scaling,
            missing=spec.missing,
            sparse=spec.sparse,
            cost=spec.cost,
        ),
        "spec": spec,
        "__reduce__": lambda self: (rebuild, (self.spec,)),
    }
    kind = type("Custom", (classical.Learner,), members)
    learner: models.Model = kind()
    return learner


def register(
    name: str,
    classifier: type[base.BaseEstimator] | None = None,
    regressor: type[base.BaseEstimator] | None = None,
    *,
    search_space: search.Space | None = None,
    family: str = "custom",
    needs_scaling: bool = False,
    handles_missing: bool = False,
    sparse_input: bool = False,
    cost: int = 3,
    replace: bool = False,
) -> models.Model:
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
    from scmle.models import registry  # noqa: PLC0415 -- circular: registry imports this module

    if classifier is None and regressor is None:
        raise errors.ConfigError(
            f"{name}: give a classifier class, a regressor class, or both"
        )
    if name in registry.inventory():
        raise errors.ConfigError(f"'{name}' is a built-in model name")
    if name in plugins.REGISTRY.entries(plugins.Kind.ESTIMATORS) and not (
        replace
    ):
        raise errors.ConfigError(f"model '{name}' is already registered")
    adapter = rebuild(
        Registration(
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
    conformance.comply(adapter)
    plugins.REGISTRY.enrol(  # Authoritative, atomic check.
        plugins.Kind.ESTIMATORS, adapter.name, adapter, replace
    )
    return adapter


def unregister(name: str) -> None:
    """Removes a registered estimator (no-op if it is not registered).

    Args:
        name: Public model name given to `register`.
    """
    plugins.REGISTRY.withdraw(plugins.Kind.ESTIMATORS, name)
