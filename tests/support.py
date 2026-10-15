"""Helpers shared by test modules."""

import mml
from mml.contracts import tasks

NO_TUNING = tasks.SearchSpec(top_models=0)
NO_STRATEGIES = tasks.StrategySpec(top_models=0)
NO_ENSEMBLE = tasks.EnsembleSpec(size=0)


def make_project(data, target, **kwargs):
    """Builds a project with search, strategies and ensembles off unless overridden."""
    kwargs.setdefault("search", NO_TUNING)
    kwargs.setdefault("strategies", NO_STRATEGIES)
    kwargs.setdefault("ensemble", NO_ENSEMBLE)
    return mml.MLProject(data, target, **kwargs)
