"""Helpers shared by test modules."""

import mml
from mml.contracts import tasks

NO_TUNING = tasks.SearchSpec(top_models=0)
NO_STRATEGIES = tasks.StrategySpec(top_models=0)
NO_ENSEMBLE = tasks.EnsembleSpec(size=0)
FAST_MODELS = tasks.ModelSpec(exclude_families=["deep_learning"])


def make_project(data, target, **kwargs):
    """Builds a fast project: no search, strategies, ensembles or PyTorch."""
    kwargs.setdefault("search", NO_TUNING)
    kwargs.setdefault("strategies", NO_STRATEGIES)
    kwargs.setdefault("ensemble", NO_ENSEMBLE)
    kwargs.setdefault("models", FAST_MODELS)  # PyTorch trains slowly
    kwargs.setdefault("enforcement", "cooperative")  # fast, in-process
    return mml.MLProject(data, target, **kwargs)
