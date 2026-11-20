"""Guarantees of website/src/pages/docs/modeling.astro."""

import types

from scmle.contracts import experiments
from scmle.contracts import tasks
from scmle.models.deeplearning import networks
from scmle.pipeline import rules


def situation(budget):
    """Planning facts for a small binary table under a time budget."""
    spec = tasks.Config(
        data=tasks.Dataset(uri="x.csv", target="y"),
        resources=tasks.Resources(time_budget_seconds=budget),
    )
    profile = types.SimpleNamespace(
        manifest=types.SimpleNamespace(rows=569),
        plan=types.SimpleNamespace(features=("a", "b"), cardinal=()),
        summary={"n_classes": 2},
    )
    return rules.Situation(spec, profile, tasks.Task.BINARY)


def test_under_300_seconds_the_plan_rejects_both_opt_ins_as_slow():
    """Enforces website/src/pages/docs/modeling.astro:66."""
    opted = [networks.FT_TRANSFORMER, networks.Jepa()]
    reason = f"time budget < {tasks.FAST_BUDGET_SECONDS}s: fast models only"

    short = []
    kept = rules.Budget().apply(situation(299), opted, short)
    enough = []
    allowed = rules.Budget().apply(situation(300), opted, enough)

    assert tasks.FAST_BUDGET_SECONDS == 300
    assert kept == []
    rejected = [d for d in short if d.action is experiments.Edict.REJECT]
    assert [(d.technique, d.reason) for d in rejected] == [
        ("torch_ft_transformer", reason),
        ("torch_jepa", reason),
    ]
    assert allowed == opted
    assert enough == []
