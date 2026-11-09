"""Selection among candidates of equal skill.

Three layers, all with a known truth:

* a pre-specified paired comparison (`comparison.compare_paired`, the
  correlated Bayesian t test) of two models with the same expected loss, scored
  on real cross-validation folds, must call one of them better with
  probability at most the stated 0.05;
* `MeanSelector` with the Bayesian rule declares a winner credibly better than
  every other candidate; with three equal candidates that can happen at most
  as often as the union of the three pairwise claims allows;
* a handful of full runs on pure-noise labels (the 200-run version is
  `benchmarks/statistics/null_selection.py`) must not claim skill often. Ten
  runs can only catch a gross failure, so the bound is deliberately loose.
"""

import itertools
import math

import numpy
import pandas
import pytest
import support_statistics

import scmle.api.project
from scmle.contracts import experiments
from scmle.contracts import tasks
from scmle.evaluation import comparison
from scmle.evaluation import metrics
from scmle.evaluation import selection
import scmle.models.registry

CONFIDENCE = 0.95
LEVEL = 1.0 - CONFIDENCE


def cross_validated_losses(rng, designs, target, folds):
    """Mean squared error per fold of ordinary least squares on each design."""
    order = rng.permutation(len(target))
    scores = [[] for ignored in designs]
    for held in numpy.array_split(order, folds):
        train = numpy.setdiff1d(order, held)
        for position, design in enumerate(designs):
            beta = numpy.linalg.lstsq(design[train], target[train], rcond=None)[
                0
            ]
            residual = target[held] - design[held] @ beta
            scores[position].append(float(numpy.mean(residual**2)))
    return scores


@pytest.mark.parametrize(("folds", "features"), [(5, 5), (10, 5)])
def test_a_prespecified_comparison_of_equal_models_stays_within_its_level(
    folds, features
):
    rng = numpy.random.default_rng(0)
    rows, runs = 100, 2000
    claims = 0
    for ignored in range(runs):
        target = rng.normal(size=rows)
        first, second = (rng.normal(size=(rows, features)) for _ in range(2))
        losses = cross_validated_losses(rng, [first, second], target, folds)
        verdict = comparison.compare_paired(
            [-loss for loss in losses[0]], [-loss for loss in losses[1]]
        )
        claims += verdict.probability_first_better >= CONFIDENCE
    support_statistics.assert_rate_at_most(
        claims / runs, LEVEL, runs, f"paired comparison, {folds} folds"
    )


def trial_of(name, losses):
    rmse = [math.sqrt(loss) for loss in losses]
    return experiments.Trial(
        f"t_{name}",
        name,
        experiments.TrialStatus.OK,
        "rmse",
        score_mean=float(numpy.mean(rmse)),
        score_std=float(numpy.std(rmse, ddof=1)),
        fold_scores=rmse,
    )


def test_the_bayesian_selector_rarely_names_a_credible_winner_among_equals():
    rng = numpy.random.default_rng(1)
    names = ["ridge", "decision_tree", "random_forest"]
    registry = scmle.models.registry.Catalog.with_builtins()
    metric = metrics.CATALOG.metrics["rmse"]
    policy = selection.MeanSelector(
        tasks.Selection(uncertainty="bayesian", bayesian_confidence=CONFIDENCE)
    )
    rows, folds, features, runs = 100, 5, 5, 1000
    sign = -1
    claims = 0
    for ignored in range(runs):
        target = rng.normal(size=rows)
        designs = [rng.normal(size=(rows, features)) for name in names]
        losses = cross_validated_losses(rng, designs, target, folds)
        trials = [
            trial_of(n, loss) for n, loss in zip(names, losses, strict=True)
        ]
        ranked = policy.rank(trials, metric, registry)
        best, others = ranked[0], ranked[1:]
        claims += all(
            not policy.probably_equivalent(best, other, sign)
            for other in others
        )
    pairs = len(list(itertools.combinations(names, 2)))
    support_statistics.assert_rate_at_most(
        claims / runs, pairs * LEVEL, runs, "credible winner among 3 equals"
    )


def noise_table(seed, rows=1000):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame({f"f{i}": rng.normal(size=rows) for i in range(6)})
    frame["cat"] = rng.choice(list("abc"), rows)
    frame["y"] = rng.integers(0, 2, rows)
    return frame


def test_a_run_on_pure_noise_does_not_claim_skill_over_the_baseline(tmp_path):
    keep = {"decision_tree", "knn", "logistic_regression", "dummy"}
    names = scmle.models.registry.Catalog.with_builtins().adapters
    excluded = sorted(set(names) - keep)
    runs = 10
    claims = 0
    for seed in range(runs):
        result = scmle.api.project.Project(
            noise_table(seed),
            "y",
            output_dir=tmp_path / f"run_{seed}",
            folds=5,
            seed=seed,
            time_budget=300,
            enforcement="cooperative",
            search=tasks.Search(top_models=0),
            strategies=tasks.Strategies(top_models=0),
            ensemble=tasks.Stacking(size=0),
            models=tasks.Candidates(exclude_models=excluded),
        ).run()
        low = result.diagnostics["metric_uncertainty"]["ci_95"][0]
        claims += low > 0.5
        assert abs(result.evaluation["test_metrics"]["roc_auc"] - 0.5) < 0.12
    # a correct 95 percent interval claims skill in 2.5 percent of runs; four
    # or more of ten would be a gross failure (probability below 1e-4)
    print(f"MEASURED noise runs claiming skill: {claims} of {runs}")
    assert claims <= 3, claims
