"""Model registry: third-party adapters register without core edits.

Plugins are discovered lazily through the `scml.models` entry-point group, and
heavy imports only happen when an adapter is actually built.
"""

from importlib import metadata
from typing import Self

from scml.contracts import errors
from scml.contracts import models
from scml.contracts import tasks
from scml.models import conformance
from scml.models import ensembles
from scml.models.boosting import boosters
from scml.models.deeplearning import networks
from scml.models.multitarget import adapters as multitarget_adapters
from scml.models.ordinal import adapters as ordinal_adapters
from scml.models.ranking import adapters as ranking_adapters
from scml.models.sklearn import bagging
from scml.models.sklearn import classical
from scml.models.survival import adapters as survival_adapters
from scml.models.uplift import adapters as uplift_adapters

ENTRY_POINT_GROUP = "scml.models"
USER_ADAPTERS: dict[str, models.ModelAdapter] = {}
"""Adapters added with `register_estimator`, shared by later projects."""


def builtin_names() -> frozenset[str]:
    """Names of the built-in model families."""
    saved = dict(USER_ADAPTERS)
    USER_ADAPTERS.clear()
    try:
        return frozenset(ModelRegistry.with_builtins().adapters)
    finally:
        USER_ADAPTERS.update(saved)


class ModelRegistry:
    """Name-indexed collection of `ModelAdapter` instances."""

    def __init__(self, adapters: tuple[models.ModelAdapter, ...] = ()) -> None:
        """Initializes the registry.

        Args:
            adapters: Adapters to register immediately.
        """
        self.adapters: dict[str, models.ModelAdapter] = {}
        self.plugin_failures: dict[str, str] = {}
        for adapter in adapters:
            self.register(adapter)

    @classmethod
    def with_builtins(cls) -> Self:
        """Creates a registry with the built-in adapters.

        These are scikit-learn, boosting and PyTorch families.

        Optional adapters register unconditionally; they are rejected at
        planning time when their library is not installed.
        """
        classes = (
            *classical.BUILTIN_ADAPTERS,
            *boosters.BOOSTING_ADAPTERS,
            *networks.DEEP_LEARNING_ADAPTERS,
            *bagging.BAGGING_ADAPTERS,
            *survival_adapters.SURVIVAL_ADAPTERS,
            *ordinal_adapters.ORDINAL_ADAPTERS,
            *ranking_adapters.RANKING_ADAPTERS,
            *multitarget_adapters.MULTITARGET_ADAPTERS,
            *uplift_adapters.UPLIFT_ADAPTERS,
        )
        registry = cls(tuple(adapter() for adapter in classes))
        registry.register(ensembles.StackingAdapter(registry))
        registry.register(ensembles.BlendingAdapter(registry))
        for adapter in USER_ADAPTERS.values():
            registry.register(adapter, overwrite=True)
        return registry

    def register(
        self, adapter: models.ModelAdapter, overwrite: bool = False
    ) -> None:
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
        """Removes an adapter if present."""
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
                found = loaded() if callable(loaded) else loaded
                items = found if isinstance(found, (list, tuple)) else [found]
                for adapter in items:
                    conformance.check_conformance(adapter)
                    self.register(adapter, overwrite=True)
            except Exception as error:  # Isolate faulty third-party code.
                failed.append(entry.name)
                self.plugin_failures[entry.name] = (
                    f"{type(error).__name__}: {error}"
                )
        return failed

    def get(self, name: str) -> models.ModelAdapter:
        """Returns the adapter called `name`.

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
        task: tasks.TaskType,
        row_count: int | None = None,
        experimental_models: tuple[str, ...] = (),
    ) -> tuple[list[models.ModelAdapter], dict[str, str]]:
        """Splits adapters into compatible ones and rejection reasons.

        Args:
            task: Resolved task type.
            row_count: Dataset size, if known.
            experimental_models: Explicit model names allowed to bypass the
                automatic-candidate boundary.

        Returns:
            Compatible adapters sorted by (cost, name), and a mapping of
            rejected adapter name to the reason.
        """
        compatible: list[models.ModelAdapter] = []
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
                rejected[adapter.name] = reason
        return compatible, rejected
