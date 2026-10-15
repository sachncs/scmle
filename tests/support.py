"""Helpers shared by test modules."""

import mml
from mml.contracts import tasks

NO_TUNING = tasks.SearchSpec(top_models=0)


def make_project(data, target, **kwargs):
    """Builds a project with hyperparameter search disabled unless overridden."""
    kwargs.setdefault("search", NO_TUNING)
    return mml.MLProject(data, target, **kwargs)
