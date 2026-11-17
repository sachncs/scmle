"""Helpers shared by test modules."""

import pathlib
import time

import pytest
from sklearn import dummy

import scmle.api.project
from scmle.contracts import errors
from scmle.contracts import models
from scmle.contracts import tasks

NO_TUNING = tasks.Search(top_models=0)
NO_STRATEGIES = tasks.Strategies(top_models=0)
NO_ENSEMBLE = tasks.Stacking(size=0)
FAST_MODELS = tasks.Candidates(
    exclude_families=["bagging", "deep_learning", "foundation"]
)


def interrupt(project):
    """Runs a project that crashes just before its test set opens.

    A completed run cannot stand in for an interrupted one: its test set was
    opened, and the custody journal refuses to resume it.

    Returns:
        The run directory, which `Project.resume` can continue.
    """
    testing = next(
        s for s in project.orchestrator.stages if s.state.value == "TESTING"
    )

    def crash(self, context):
        raise RuntimeError("simulated crash before the test set opens")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(testing), "execute", crash)
        with pytest.raises(errors.ScmleError) as caught:
            project.run()
    return pathlib.Path(caught.value.details["run_dir"])


def make_project(data, target, **kwargs):
    """Builds a fast project: no search, strategies, ensembles or PyTorch."""
    kwargs.setdefault("search", NO_TUNING)
    kwargs.setdefault("strategies", NO_STRATEGIES)
    kwargs.setdefault("ensemble", NO_ENSEMBLE)
    kwargs.setdefault("models", FAST_MODELS)  # PyTorch trains slowly
    kwargs.setdefault("enforcement", "cooperative")  # fast, in-process
    return scmle.api.project.Project(data, target, **kwargs)


class HungryClassifier(dummy.DummyClassifier):
    """A classifier whose `fit` grabs far more memory than any limit allows."""

    def fit(self, features, target):
        block = bytearray(1500 * 1024**2)
        block[::4096] = b"x" * len(block[::4096])
        time.sleep(60)
        return super().fit(features, target)


class HungryAdapter(models.Model):
    """Adapter for `HungryClassifier`."""

    name = "hungry"
    family = "test"
    capabilities = models.Capabilities(frozenset({tasks.Task.BINARY}), cost=2)

    def build(self, task, seed, params=None):
        return HungryClassifier()
