"""Guarantees of README.md that no other test enforced."""

import pathlib
import types

import numpy
import pandas
from sklearn import naive_bayes

from benchmarks import reliability
from scmle.assurance.admissibility import leakage
from scmle.assurance.admissibility import screen
from scmle.contracts import experiments
from scmle.contracts import models
from scmle.contracts import tasks
from scmle.data.profiling import profiler
from scmle.data.readers import tabular
import scmle.intelligence.planner
from scmle.models import custom
from scmle.models import registry

TITANIC = (
    pathlib.Path(__file__).resolve().parents[1]
    / "fixtures"
    / "titanic_extract.csv"
)


class Taskless(models.Model):
    """An adapter that declares no task, so conformance refuses it."""

    name = "taskless"
    family = "test"
    capabilities = models.Capabilities(frozenset())

    def build(self, task, seed, params=None):
        """Builds the estimator.

        Args:
            task: The learning task.
            seed: Random seed.
            params: Hyperparameters.

        Returns:
            A naive Bayes classifier.
        """
        return naive_bayes.GaussianNB()


def entry(name, adapter):
    """Builds a stand-in for an `importlib.metadata` entry point.

    Args:
        name: Entry-point name.
        adapter: What loading the entry point returns.

    Returns:
        An object with the `name` and `load` of an entry point.
    """
    return types.SimpleNamespace(name=name, load=lambda: adapter)


def test_the_plan_gives_a_reason_for_every_technique():
    """Enforces README.md:103."""
    rng = numpy.random.default_rng(0)
    frame = pandas.DataFrame(
        {"x": rng.normal(size=300), "y": rng.integers(0, 2, 300)}
    )
    spec = tasks.Config(
        data=tasks.Dataset(uri="m", target="y"),
        resources=tasks.Resources(time_budget_seconds=60),
    )
    profile = profiler.Profiler(spec).profile(
        frame, tabular.manifested(frame, "y"), tasks.Task.BINARY
    )
    plan = scmle.intelligence.planner.Planner(registry.Catalog.stock()).plan(
        scmle.intelligence.planner.Situation(spec, profile, tasks.Task.BINARY)
    )
    actions = {d.action for d in plan.decisions}
    assert actions <= set(experiments.Edict)
    assert {experiments.Edict.INCLUDE, experiments.Edict.REJECT} <= actions
    assert all(d.reason.strip() for d in plan.decisions)


def test_the_reliability_benchmark_checks_the_nine_named_properties():
    """Enforces README.md:148."""
    assert list(reliability.CHECKS) == [
        "determinism",
        "replay",
        "no_false_skill",
        "leak_detection",
        "bundle_roundtrip",
        "tamper_detection",
        "interval_coverage",
        "drift_detection",
        "failure_isolation",
    ]
    assert reliability.CHECKS["failure_isolation"].__doc__.startswith(
        "A candidate that always crashes does not fail the run."
    )


def test_titanic_boat_and_body_are_flagged_by_the_documented_mechanisms():
    """Enforces README.md:180."""
    frame = pandas.read_csv(TITANIC)
    report = screen.screen_table(
        frame,
        "survived",
        "classification",
        ["boat", "body"],
        leakage.CERTIFIED.run_seed,
    )
    found = {}
    for finding in report.blocking:
        found.setdefault(finding.feature, set()).add(finding.mechanism)
    assert found == {
        "boat": {"skill", "missingness", "record_purity"},
        "body": {"record_purity"},
    }


def test_an_entry_point_adapter_is_registered_only_if_it_complies(
    monkeypatch,
):
    """Enforces README.md:220."""
    good = custom.rebuild(
        custom.Registration("plugin_nb", naive_bayes.GaussianNB, None, cost=1)
    )
    found = [entry("good", good), entry("bad", Taskless())]
    monkeypatch.setattr(registry.metadata, "entry_points", lambda group: found)
    catalog = registry.Catalog.stock()
    assert catalog.load_plugins() == ["bad"]
    assert catalog.get("plugin_nb") is good
    assert "taskless" not in catalog.adapters
    assert "declares no supported tasks" in catalog.plugin_failures["bad"]
