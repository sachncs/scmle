"""Model registry: third-party estimators register without core edits.

Entry-point group `mml.models` is scanned lazily; heavy imports only happen when a model
is actually built.
"""

from __future__ import annotations

import importlib.util
from importlib.metadata import entry_points

from mml.contracts.errors import ConfigError
from mml.contracts.models import ModelSpec
from mml.contracts.tasks import TaskType

_REGISTRY: dict[str, ModelSpec] = {}
_DISCOVERED = False


def register_model(spec: ModelSpec, *, overwrite: bool = False) -> None:
    if spec.name in _REGISTRY and not overwrite:
        raise ConfigError(f"model '{spec.name}' already registered")
    _REGISTRY[spec.name] = spec


def _discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    from mml.models import sklearn_models  # noqa: F401  (registers built-ins)

    for ep in entry_points(group="mml.models"):
        try:
            obj = ep.load()
            specs = obj() if callable(obj) else obj
            for s in specs if isinstance(specs, (list, tuple)) else [specs]:
                register_model(s, overwrite=True)
        except Exception:  # a broken plugin must not break the framework
            continue


def available(spec: ModelSpec) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in spec.requires)


def get_model(name: str) -> ModelSpec:
    _discover()
    if name not in _REGISTRY:
        raise ConfigError(f"unknown model '{name}'; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def candidates(task: TaskType, n_rows: int | None = None) -> tuple[list[ModelSpec], dict[str, str]]:
    """Compatible candidates plus {name: reason} for excluded ones."""
    _discover()
    ok: list[ModelSpec] = []
    rejected: dict[str, str] = {}
    for s in sorted(_REGISTRY.values(), key=lambda m: (m.capabilities.cost, m.name)):
        c = s.capabilities
        if task not in c.tasks:
            rejected[s.name] = f"does not support {task.value}"
        elif not available(s):
            rejected[s.name] = f"missing dependency: {', '.join(s.requires)}"
        elif n_rows is not None and c.max_rows is not None and n_rows > c.max_rows:
            rejected[s.name] = f"dataset too large ({n_rows} > {c.max_rows} rows)"
        else:
            ok.append(s)
    return ok, rejected
