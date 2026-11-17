"""Model selection: the rule's description and its tie-breaking order."""

import pytest

from scmle.contracts import experiments
from scmle.contracts import tasks
from scmle.evaluation import metrics
from scmle.evaluation import selection
import scmle.models.registry


@pytest.fixture(scope="module")
def registry():
    return scmle.models.registry.Catalog.stock()


def trial(name, score, strategy=experiments.DEFAULT_STRATEGY):
    return experiments.Trial(
        f"t_{name}_{strategy}",
        name,
        experiments.Termination.OK,
        "m",
        score_mean=score,
        strategy=strategy,
    )


def test_the_default_rule_is_described_exactly() -> None:
    assert selection.Averager().description == (
        "best mean CV score; exact ties -> simpler_model; dummy baseline "
        "excluded when other models ran"
    )


def test_a_tolerance_counts_for_a_lower_is_better_metric(registry) -> None:
    trials = [trial("random_forest", 0.100), trial("decision_tree", 0.103)]
    policy = selection.Averager(tasks.Selection(tie_tolerance=0.005))
    ranked = policy.rank(trials, metrics.CATALOG.metrics["rmse"], registry)
    assert [t.model_name for t in ranked] == ["decision_tree", "random_forest"]


def test_ties_prefer_the_default_strategy(registry) -> None:
    trials = [trial("ridge", 0.8, "aaa"), trial("ridge", 0.8)]
    ranked = selection.Averager().rank(
        trials, metrics.CATALOG.metrics["roc_auc"], registry
    )
    assert [t.strategy for t in ranked] == [
        experiments.DEFAULT_STRATEGY,
        "aaa",
    ]
