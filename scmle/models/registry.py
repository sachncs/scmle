"""Model registry: third-party adapters register without core edits.

Plugins are discovered lazily through the `scmle.models` entry-point group, and
heavy imports only happen when an adapter is actually built.
"""

from importlib import metadata
import threading
from typing import Any
from typing import Self

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

ENTRY_POINT_GROUP = "scmle.models"
USER_ADAPTERS: dict[str, models.Model] = {}
"""Adapters added with `register_estimator`, shared by later projects.

Change it only through `add_user_adapter` and `remove_user_adapter`, and read
it only through `user_adapters`; all three hold `USER_LOCK`.
"""
USER_LOCK = threading.RLock()
"""Serializes check-then-act changes to `USER_ADAPTERS`."""


def user_adapters() -> dict[str, models.Model]:
    """Returns a consistent copy of the registered user adapters.

    Returns:
        A new mapping of model name to adapter; changing it does not change
        the shared registrations.
    """
    with USER_LOCK:
        return dict(USER_ADAPTERS)


def add_user_adapter(adapter: models.Model, replace: bool = False) -> None:
    """Registers a user adapter atomically.

    Args:
        adapter: The adapter to add.
        replace: Allow replacing an earlier registration of the same name.

    Raises:
        ConfigError: If the name is taken and `replace` is False.
    """
    with USER_LOCK:
        if adapter.name in USER_ADAPTERS and not replace:
            raise errors.ConfigError(
                f"model '{adapter.name}' is already registered"
            )
        USER_ADAPTERS[adapter.name] = adapter


def remove_user_adapter(name: str) -> None:
    """Removes a user adapter atomically (no-op if it is not registered).

    Args:
        name: Name of the adapter to remove.
    """
    with USER_LOCK:
        USER_ADAPTERS.pop(name, None)


def builtin_names() -> frozenset[str]:
    """Lists the built-in model names.

    Returns:
        The names of the built-in adapters, excluding user registrations.
    """
    return frozenset(Catalog.with_builtins(include_user=False).adapters)


class Catalog:
    """Name-indexed collection of `Model` instances."""

    def __init__(self, adapters: tuple[models.Model, ...] = ()) -> None:
        """Initializes the registry.

        Args:
            adapters: Adapters to register immediately.
        """
        self.adapters: dict[str, models.Model] = {}
        self.plugin_failures: dict[str, str] = {}
        for adapter in adapters:
            self.register(adapter)

    @classmethod
    def with_builtins(cls, include_user: bool = True) -> Self:
        """Creates a registry with the built-in adapters.

        These are scikit-learn, boosting and PyTorch families.

        Optional adapters register unconditionally; they are rejected at
        planning time when their library is not installed.

        Args:
            include_user: Also add adapters from `register_estimator`
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
            for adapter in user_adapters().values():
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

    def load_plugins(self) -> list[str]:
        """Registers adapters advertised through entry points.

        Every plugin adapter must pass the shared conformance suite before it
        is registered. Reasons for failures are kept in `plugin_failures`.

        Returns:
            Names of plugins that failed to load or conform; a broken plugin
            never breaks the framework.
        """
        failed = []
        self.plugin_failures = {}
        for entry in metadata.entry_points(group=ENTRY_POINT_GROUP):
            try:
                loaded = entry.load()
                found: Any = loaded() if callable(loaded) else loaded
                items = found if isinstance(found, (list, tuple)) else [found]
                for adapter in items:
                    conformance.check_conformance(adapter)
                    self.register(adapter, overwrite=True)
            except Exception as error:  # noqa: BLE001 -- isolate faulty plugins
                failed.append(entry.name)
                self.plugin_failures[entry.name] = (
                    f"{type(error).__name__}: {error}"
                )
        return failed

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
            reason = adapter.rejection_reason(task, row_count)
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
