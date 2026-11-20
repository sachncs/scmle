"""Model registry: third-party adapters register without core edits.

User adapters live in the shared plugin registry
(`scmle.runtime.plugins.REGISTRY`, kind `estimators`). Plugins are
discovered lazily through the `scmle.models` entry-point group, and heavy
imports only happen when an adapter is actually built.
"""

import logging
from typing import Self
import warnings

from scmle.contracts import errors
from scmle.contracts import models
from scmle.contracts import tasks
from scmle.models import conformance
from scmle.models import ensembles
from scmle.models.boosting import boosters
from scmle.models.deeplearning import networks
import scmle.models.multitarget.adapters
import scmle.models.ordinal.adapters
import scmle.models.ranking.adapters
from scmle.models.sklearn import bagging
from scmle.models.sklearn import classical
import scmle.models.survival.adapters
import scmle.models.uplift.adapters
from scmle.runtime import plugins

LOGGER = logging.getLogger(__name__)


ENTRY_POINT_GROUP = plugins.GROUPS[plugins.Kind.ESTIMATORS]
USER_ADAPTERS: plugins.Shelf = plugins.REGISTRY.shelves[plugins.Kind.ESTIMATORS]
"""The shared registry's estimator shelf itself; read it through
`plugins.REGISTRY.entries` and change it through `plugins.REGISTRY.enrol`
and `withdraw`, which hold `plugins.REGISTRY.lock`.
"""


def deprecate(call: str, successor: str) -> None:
    """Warns that a registry call moved to the shared plugin registry.

    Args:
        call: The deprecated call.
        successor: The `plugins.REGISTRY` call that replaces it.
    """
    warnings.warn(
        f"registry.{call} is deprecated; use plugins.REGISTRY.{successor}"
        "(plugins.Kind.ESTIMATORS, ...) instead",
        DeprecationWarning,
        stacklevel=3,
    )


def extensions() -> dict[str, models.Model]:
    """Returns a consistent copy of the registered user adapters.

    Deprecated: use `plugins.REGISTRY.entries(plugins.Kind.ESTIMATORS)`.

    Returns:
        A new mapping of model name to adapter.
    """
    deprecate("extensions", "entries")
    return plugins.REGISTRY.entries(plugins.Kind.ESTIMATORS)


def adopt(adapter: models.Model, replace: bool = False) -> None:
    """Registers a user adapter atomically.

    Deprecated: use `plugins.REGISTRY.enrol(plugins.Kind.ESTIMATORS, ...)`.

    Args:
        adapter: The adapter to add.
        replace: Allow replacing an earlier registration of the same name.
    """
    deprecate("adopt", "enrol")
    plugins.REGISTRY.enrol(
        plugins.Kind.ESTIMATORS, adapter.name, adapter, replace
    )


def deregister(name: str) -> None:
    """Removes a user adapter atomically (no-op if it is not registered).

    Deprecated: use `plugins.REGISTRY.withdraw(plugins.Kind.ESTIMATORS, ...)`.

    Args:
        name: Name of the adapter to remove.
    """
    deprecate("deregister", "withdraw")
    plugins.REGISTRY.withdraw(plugins.Kind.ESTIMATORS, name)


def inventory() -> frozenset[str]:
    """Lists the built-in model names.

    Returns:
        The names of the built-in adapters, excluding user registrations.
    """
    return frozenset(Catalog.stock(include_user=False).adapters)


class Catalog:
    """Name-indexed collection of `Model` instances."""

    def __init__(self, adapters: tuple[models.Model, ...] = ()) -> None:
        """Initializes the registry.

        Args:
            adapters: Adapters to register immediately.
        """
        self.adapters: dict[str, models.Model] = {}
        self.failures: dict[str, str] = {}
        for adapter in adapters:
            self.register(adapter)

    @classmethod
    def stock(cls, include_user: bool = True) -> Self:
        """Creates a registry with the built-in adapters.

        These are scikit-learn, boosting and PyTorch families.

        Optional adapters register unconditionally; they are rejected at
        planning time when their library is not installed.

        Args:
            include_user: Also add adapters from `custom.register`
                (a snapshot taken once, so later changes do not leak in).

        Returns:
            A new registry holding the built-in adapters, the stacking and
            blending adapters, and the user adapters if requested.
        """
        classes = (
            *bagging.BAGGING_ADAPTERS,
            *scmle.models.survival.adapters.SURVIVAL_ADAPTERS,
        )
        registry = cls(
            (
                *classical.BUILTIN_ADAPTERS,
                *boosters.BOOSTING_ADAPTERS,
                *networks.DEEP_LEARNING_ADAPTERS,
                *(adapter() for adapter in classes),
                *scmle.models.ordinal.adapters.ORDINAL_ADAPTERS,
                *scmle.models.ranking.adapters.RANKING_ADAPTERS,
                *scmle.models.multitarget.adapters.MULTITARGET_ADAPTERS,
                *scmle.models.uplift.adapters.UPLIFT_ADAPTERS,
            )
        )
        registry.register(ensembles.Stacking(registry))
        registry.register(ensembles.Blending(registry))
        if include_user:
            shelf = plugins.REGISTRY.entries(plugins.Kind.ESTIMATORS)
            for adapter in shelf.values():
                registry.register(adapter, overwrite=True)
        return registry

    def register(self, adapter: models.Model, overwrite: bool = False) -> None:
        """Adds an adapter.

        Args:
            adapter: The adapter to add.
            overwrite: Replace an existing adapter of the same name.

        Raises:
            ConfigError: If the name is taken and `overwrite` is False.
        """
        if adapter.name in self.adapters and not overwrite:
            raise errors.ConfigError(
                f"model '{adapter.name}' already registered"
            )
        self.adapters[adapter.name] = adapter

    def unregister(self, name: str) -> None:
        """Removes an adapter if present.

        Args:
            name: Name of the adapter to remove; unknown names are ignored.
        """
        self.adapters.pop(name, None)

    def load(self) -> list[str]:
        """Registers adapters advertised through entry points.

        Discovery goes through the shared registry (`plugins.REGISTRY`), so
        a discovered adapter is also offered to later catalogs. Every plugin
        adapter must pass the shared conformance suite before it is
        registered, and may not take a built-in model's name (as
        `custom.register` refuses too). Reasons for failures are
        kept in `failures`.

        Returns:
            Names of plugins that failed to load or conform (entry-point
            names) or that shadow a built-in (adapter names); a broken
            plugin never breaks the framework.
        """
        found = plugins.REGISTRY.discover(
            plugins.Kind.ESTIMATORS, conformance.comply
        )
        self.failures = dict(found.failures)
        builtin = inventory()
        for adapter in found.items:
            if adapter.name in builtin:
                LOGGER.warning(
                    "model plugin %s shadows a built-in model", adapter.name
                )
                self.failures[adapter.name] = (
                    f"'{adapter.name}' is a built-in model name"
                )
                continue
            plugins.REGISTRY.enrol(
                plugins.Kind.ESTIMATORS, adapter.name, adapter, replace=True
            )
            self.register(adapter, overwrite=True)
        return list(self.failures)

    def get(self, name: str) -> models.Model:
        """Returns the adapter called `name`.

        Args:
            name: Registered model name.

        Returns:
            The adapter registered under `name`.

        Raises:
            ConfigError: If it is not registered.
        """
        if name not in self.adapters:
            raise errors.ConfigError(
                f"unknown model '{name}'; registered: {sorted(self.adapters)}"
            )
        return self.adapters[name]

    def candidates(
        self,
        task: tasks.Task,
        row_count: int | None = None,
        experimental_models: tuple[str, ...] = (),
    ) -> tuple[list[models.Model], dict[str, str]]:
        """Splits adapters into compatible ones and rejection reasons.

        Args:
            task: Resolved task type.
            row_count: Dataset size, if known.
            experimental_models: Explicit model names allowed to bypass the
                automatic-candidate boundary.

        Returns:
            Compatible adapters sorted by (cost, name), and a mapping of
            rejected adapter name to the reason.

        Raises:
            ConfigError: If an explicitly requested experimental model is not
                registered in this registry.
        """
        unknown = sorted(set(experimental_models) - set(self.adapters))
        if unknown:
            raise errors.ConfigError(
                "unknown experimental model(s): "
                f"{unknown}; registered: {sorted(self.adapters)}"
            )
        compatible: list[models.Model] = []
        rejected: dict[str, str] = {}
        ordered = sorted(
            self.adapters.values(),
            key=lambda a: (a.capabilities.cost, a.name),
        )
        for adapter in ordered:
            if not adapter.capabilities.standalone:
                continue
            if (
                not adapter.auto_candidate
                and adapter.name not in experimental_models
            ):
                rejected[adapter.name] = (
                    "model is experimental-only; add it to "
                    "models.experimental_models for explicit opt-in"
                )
                continue
            reason = adapter.objection(task, row_count)
            if reason is None:
                compatible.append(adapter)
            else:
                if adapter.name in experimental_models:
                    raise errors.ConfigError(
                        f"explicit experimental model '{adapter.name}' "
                        f"cannot be used: {reason}"
                    )
                rejected[adapter.name] = reason
        return compatible, rejected
