"""One registry through which every extension is enrolled and discovered.

Estimators, metrics, readers, detectors, stages and transformers are each a
`Kind`. A
user enrols an item under a name (`Registry.enrol`), and a package
advertises items through the entry-point group of their kind (`GROUPS`),
which `Registry.discover` loads, vets and reports, so a broken plugin never
breaks the framework. `REGISTRY` is the one shared instance; the per-kind
tables that read it (the model catalog, the metric catalog) stay where they
are and draw their extensions from here.

Estimators keep the `scmle.models` group they were advertised through
before this registry existed, so installed plugins keep loading.
"""

from collections.abc import Callable
import dataclasses
import enum
from importlib import metadata
import logging
import threading
from typing import Any
import warnings

from scmle.contracts import errors

LOGGER = logging.getLogger(__name__)


class Kind(enum.StrEnum):
    """What an extension is.

    Attributes:
        ESTIMATORS: Model adapters (`scmle.contracts.models.Model`).
        METRICS: Scoring metrics (`scmle.contracts.metrics.Metric`).
        READERS: Data readers.
        DETECTORS: Data-quality and leakage detectors.
        STAGES: Pipeline stages.
        TRANSFORMERS: Feature transformers.
    """

    ESTIMATORS = "estimators"
    METRICS = "metrics"
    READERS = "readers"
    DETECTORS = "detectors"
    STAGES = "stages"
    TRANSFORMERS = "transformers"

    @property
    def noun(self) -> str:
        """The singular, for messages ("model", "metric", ...).

        Returns:
            The word that names one item of the kind.
        """
        return "model" if self is Kind.ESTIMATORS else self.value[:-1]


Shelf = dict[str, Any]
"""Name to item, the items of one kind."""

GROUPS = {
    Kind.ESTIMATORS: "scmle.models",
    Kind.METRICS: "scmle.metrics",
    Kind.READERS: "scmle.readers",
    Kind.DETECTORS: "scmle.detectors",
    Kind.STAGES: "scmle.stages",
    Kind.TRANSFORMERS: "scmle.transformers",
}
"""The entry-point group each kind is advertised through."""


def resolve(name: Kind | str) -> Kind:
    """Reads a kind, accepting the plain strings of the previous release.

    Args:
        name: A `Kind`, or its value as a string (deprecated).

    Returns:
        The kind.
    """
    if isinstance(name, Kind):
        return name
    warnings.warn(
        f"plugins.REGISTRY takes a plugins.Kind; the string '{name}' is "
        "deprecated and will be refused in the next release",
        DeprecationWarning,
        stacklevel=3,
    )
    return Kind(name)


@dataclasses.dataclass(frozen=True, slots=True)
class Discovery:
    """What one entry-point group yielded.

    Attributes:
        items: The items that loaded and passed the vetting, in order.
        failures: Entry-point name to why it did not load or conform.
    """

    items: tuple[Any, ...]
    failures: dict[str, str]


class Registry:
    """Named extensions of every kind, safe to change from many threads.

    Attributes:
        lock: Serializes check-then-act changes.
        shelves: Kind to name to item; read through `entries` or `find`.
    """

    def __init__(self) -> None:
        """Starts with nothing enrolled."""
        self.lock = threading.RLock()
        self.shelves: dict[Kind, Shelf] = {kind: {} for kind in Kind}

    def enrol(
        self, kind: Kind | str, name: str, item: Any, replace: bool = False
    ) -> None:
        """Adds an item under a name, atomically.

        Args:
            kind: What the item is.
            name: The name it is found by.
            item: The extension.
            replace: Allow replacing an earlier item of the same name.

        Raises:
            ConfigError: If the name is taken and `replace` is False.
        """
        kind = resolve(kind)
        with self.lock:
            shelf = self.shelves[kind]
            if name in shelf and not replace:
                raise errors.ConfigError(
                    f"{kind.noun} '{name}' is already registered"
                )
            shelf[name] = item

    def withdraw(self, kind: Kind | str, name: str) -> None:
        """Removes an item (no-op if it is not enrolled).

        Args:
            kind: What the item is.
            name: Its name.
        """
        kind = resolve(kind)
        with self.lock:
            self.shelves[kind].pop(name, None)

    def entries(self, kind: Kind | str) -> Shelf:
        """A consistent copy of the items of one kind.

        Args:
            kind: The kind.

        Returns:
            A new mapping of name to item; changing it changes nothing here.
        """
        kind = resolve(kind)
        with self.lock:
            return dict(self.shelves[kind])

    def find(self, kind: Kind | str, name: str) -> Any:
        """Returns the item enrolled under a name.

        Args:
            kind: The kind.
            name: The name.

        Returns:
            The item.

        Raises:
            ConfigError: If nothing of that kind is enrolled under the name.
        """
        kind = resolve(kind)
        with self.lock:
            shelf = self.shelves[kind]
            if name not in shelf:
                raise errors.ConfigError(
                    f"unknown {kind.noun} '{name}'; registered: {sorted(shelf)}"
                )
            return shelf[name]

    def discover(
        self, kind: Kind | str, vet: Callable[[Any], object] | None = None
    ) -> Discovery:
        """Loads the items advertised through the kind's entry-point group.

        An entry point names an item, a list or tuple of items, or a
        callable returning either. Each item is passed to `vet`, which
        raises to refuse it. Loading enrols nothing: the caller decides.

        Args:
            kind: The kind, which names the group (`GROUPS`).
            vet: Raises if an item does not conform, or None.

        Returns:
            The items that loaded and conformed, and the reason each failing
            entry point was dropped.
        """
        kind = resolve(kind)
        items: list[Any] = []
        failures: dict[str, str] = {}
        for entry in metadata.entry_points(group=GROUPS[kind]):
            try:
                loaded = entry.load()
                found = loaded() if callable(loaded) else loaded
                batch = found if isinstance(found, (list, tuple)) else [found]
                for item in batch:
                    if vet is not None:
                        vet(item)
                items.extend(batch)
            except Exception as error:  # plugin boundary
                # handled: a broken plugin is reported, never raised
                LOGGER.warning(
                    "%s plugin %s failed to load",
                    kind.noun,
                    entry.name,
                    exc_info=True,
                )
                failures[entry.name] = errors.Failure.of(error).text
        return Discovery(tuple(items), failures)


REGISTRY = Registry()
"""The registry every package enrols into and reads from."""
