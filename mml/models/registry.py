"""Model registry: third-party adapters register without core edits.

Plugins are discovered lazily through the `mml.models` entry-point group, and
heavy imports only happen when an adapter is actually built.
"""

from importlib import metadata

from mml.contracts import errors
from mml.contracts import models
from mml.contracts import tasks
from mml.models import sklearn_adapters

ENTRY_POINT_GROUP = "mml.models"


class ModelRegistry:
    """Name-indexed collection of `ModelAdapter` instances."""

    def __init__(
        self, adapters: tuple[models.ModelAdapter, ...] = ()
    ) -> None:
        """Initializes the registry.

        Args:
            adapters: Adapters to register immediately.
        """
        self.adapters: dict[str, models.ModelAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    @classmethod
    def with_builtins(cls) -> "ModelRegistry":
        """Creates a registry holding the built-in scikit-learn adapters."""
        return cls(tuple(a() for a in sklearn_adapters.BUILTIN_ADAPTERS))

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
            raise errors.ConfigError(f"model '{adapter.name}' already registered")
        self.adapters[adapter.name] = adapter

    def unregister(self, name: str) -> None:
        """Removes an adapter if present."""
        self.adapters.pop(name, None)

    def load_plugins(self) -> list[str]:
        """Registers adapters advertised through entry points.

        Returns:
            Names of plugins that failed to load; a broken plugin never
            breaks the framework.
        """
        failed = []
        for entry in metadata.entry_points(group=ENTRY_POINT_GROUP):
            try:
                loaded = entry.load()
                found = loaded() if callable(loaded) else loaded
                items = found if isinstance(found, (list, tuple)) else [found]
                for adapter in items:
                    self.register(adapter, overwrite=True)
            except Exception:  # Isolate faulty third-party code.
                failed.append(entry.name)
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
        self, task: tasks.TaskType, row_count: int | None = None
    ) -> tuple[list[models.ModelAdapter], dict[str, str]]:
        """Splits adapters into compatible ones and rejection reasons.

        Args:
            task: Resolved task type.
            row_count: Dataset size, if known.

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
            reason = adapter.rejection_reason(task, row_count)
            if reason is None:
                compatible.append(adapter)
            else:
                rejected[adapter.name] = reason
        return compatible, rejected
