"""Wide tables get a model in a short budget; variants win only on evidence.

C14: the reserve's bootstrap is deadline-bounded, so it holds a share of the
budget, and the first honest trial's cost grows with the table's width; the
planner runs the cheaper learners first when the boosted one would outlast
the trial window. C4: a variant of a default pipeline is selected only when
the shared folds make it credibly better.
"""

import types

import pytest

from scmle.contracts import errors
from scmle.contracts import experiments
from scmle.models.sklearn import classical
from scmle.pipeline import journal
from scmle.pipeline import profiler
from scmle.pipeline import rules
from scmle.pipeline import strategy_search
from scmle.pipeline.stages import profiling
from scmle.pipeline.stages import selecting
from scmle.runtime import budget


def prepared(seconds):
    """A preparation cost model that always predicts `seconds`."""
    return lambda rows, columns, classes, limit: seconds


def test_a_wide_table_is_admitted_in_thirty_seconds():
    # gina (C14): 3,153 rows, 970 columns; the full bootstrap alone was
    # estimated at 18.3 s and refused the run.
    reserve = budget.Reserve(30)
    reserve.provision(3153, 970, 2, prepared(8.8))
    reserve.admit()
    tested = round(budget.TEST_SHARE * 3153)
    assert budget.bootstrap(tested, 970, 2) > 15
    held = reserve.parts["diagnostics"] - budget.OTHER_DIAGNOSTIC_SECONDS
    assert held - budget.EXPLAIN_SHARE * 30 == pytest.approx(
        budget.BOOTSTRAP_SHARE * 30
    )


def test_the_bootstrap_share_lies_between_its_least_and_full_cost():
    for rows, columns, seconds in [(1132, 111, 60), (3153, 970, 30)]:
        tested = round(budget.TEST_SHARE * rows)
        least = budget.bootstrap(tested, columns, 2, budget.RESAMPLES_MIN)
        full = budget.bootstrap(tested, columns, 2)
        part = (
            budget.reservation(rows, columns, 2, seconds)["diagnostics"]
            - budget.OTHER_DIAGNOSTIC_SECONDS
            - budget.EXPLAIN_SHARE * seconds
        )
        assert least <= part + 1e-9 and part <= full + 1e-9
    roomy = budget.reservation(1132, 111, 8, 1800)["diagnostics"]
    tested = round(budget.TEST_SHARE * 1132)
    assert roomy == pytest.approx(
        budget.bootstrap(tested, 111, 8)
        + budget.OTHER_DIAGNOSTIC_SECONDS
        + budget.EXPLAIN_SHARE * 1800
    )


def test_the_first_honest_trial_grows_with_the_width():
    narrow, wide = budget.cheapest(3153, 10), budget.cheapest(3153, 970)
    assert budget.START_SECONDS < narrow < wide
    assert budget.cheapest(3153) == pytest.approx(
        budget.START_SECONDS + budget.CHEAPEST_RATE * 3153
    )
    assert wide - narrow == pytest.approx(budget.LINEAR_CELL * 3153 * 960)


def test_a_minimum_too_large_is_still_refused():
    reserve = budget.Reserve(30)
    reserve.provision(3153, 970, 2, prepared(25.0))
    with pytest.raises(errors.BlockedError, match="smallest budget"):
        reserve.admit()


def test_the_boosted_cost_counts_folds_rounds_and_classes():
    binary = budget.boosted(1000, 10, 2)
    assert binary == pytest.approx(
        budget.BOOSTED_CELL * budget.BOOSTED_ROUNDS * 1000 * 4 * 10
    )
    assert budget.boosted(1000, 10, 5) == pytest.approx(5 * binary)
    shrunk = budget.BOOSTED_ROUNDS * classical.SHRINKAGE
    assert shrunk == pytest.approx(10.0)


def situation(rows, columns, seconds):
    """Planning facts for a binary table under a tight budget."""
    search = types.SimpleNamespace(top_models=1, configs_per_model=3)
    spec = types.SimpleNamespace(
        brief=True,
        budgeted=lambda: search,
        effective=lambda: search,
        resources=types.SimpleNamespace(time_budget_seconds=seconds),
    )
    profile = types.SimpleNamespace(
        manifest=types.SimpleNamespace(rows=rows),
        feature_plan=types.SimpleNamespace(
            features=tuple(f"c{i}" for i in range(columns))
        ),
        target_summary={"n_classes": 2},
    )
    return types.SimpleNamespace(
        spec=spec, profile=profile, cost=profiler.estimate
    )


CANDIDATES = [
    classical.Dummy(),
    classical.DECISION_TREE,
    classical.LOGISTIC_REGRESSION,
    classical.KNN,
    classical.HIST_GRADIENT_BOOSTING,
]


def test_a_narrow_table_runs_the_boosted_learner_first():
    decisions = []
    order = rules.Budget().apply(
        situation(1194, 25, 60), list(CANDIDATES), decisions
    )
    assert [a.name for a in order[:2]] == ["dummy", "hist_gradient_boosting"]
    assert all(d.technique != "cheapest_learner_first" for d in decisions)


def test_a_wide_table_runs_the_linear_learner_first():
    decisions = []
    order = rules.Budget().apply(
        situation(2522, 970, 30), list(CANDIDATES), decisions
    )
    assert [a.name for a in order[:2]] == ["dummy", "logistic_regression"]
    assert order[-1].name == "hist_gradient_boosting"
    noted = [d for d in decisions if d.technique == "cheapest_learner_first"]
    assert len(noted) == 1 and "970 columns" in noted[0].reason


def scored(name, folds, strategy="default", params=None, status=None):
    """An OK trial with the given fold scores."""
    return experiments.Trial(
        f"t-{name}-{strategy}",
        name,
        status or experiments.Termination.OK,
        "roc_auc",
        sum(folds) / len(folds),
        0.0,
        list(folds),
        1.0,
        params=params or {},
        strategy=strategy,
    )


BASE = [0.70, 0.74, 0.72, 0.76, 0.73]


def test_the_odds_need_shared_folds_and_reward_consistency():
    default = scored("m", BASE)
    steady = scored("m", [s + 0.02 for s in BASE], "log_skewed")
    noisy = scored("m", [0.75, 0.70, 0.77, 0.72, 0.76], "random_undersample")
    assert strategy_search.odds(steady, default, 1) >= (
        strategy_search.CONFIDENCE
    )
    assert strategy_search.odds(noisy, default, 1) < (
        strategy_search.CONFIDENCE
    )
    assert strategy_search.odds(scored("m", BASE[:2]), default, 1) == 0.0
    assert strategy_search.odds(steady, default, -1) < 0.5
    assert strategy_search.plain(default)
    assert not strategy_search.plain(steady)
    assert not strategy_search.plain(scored("m", BASE, params={"a": 1}))


def board(trials):
    """A board stand-in: a partition's metric, a journal and its trials."""
    spec = types.SimpleNamespace()
    record = journal.Journal(spec, None, None, None)
    record.trials.extend(trials)
    partition = types.SimpleNamespace(
        metric=types.SimpleNamespace(greater_is_better=True, name="roc_auc")
    )
    return types.SimpleNamespace(
        get={profiling.Partition: partition}.__getitem__, journal=record
    )


def test_selection_keeps_defaults_and_variants_earned_on_evidence():
    default = scored("m", BASE)
    other = scored("n", [s - 0.01 for s in BASE])
    steady = scored("m", [s + 0.02 for s in BASE], "log_skewed")
    noisy = scored("m", [0.75, 0.70, 0.77, 0.72, 0.76], "random_undersample")
    shifts = [0.01, -0.01, 0.0, 0.005, -0.004]
    blend = scored(
        "blending",
        [s + d for s, d in zip(BASE, shifts, strict=True)],
        params={"k": 1},
    )
    failed = scored(
        "m", BASE, "calibrated", status=experiments.Termination.FAILED
    )
    trials = [default, other, steady, noisy, blend, failed]
    context = board(trials)
    kept = selecting.earn(context, trials)
    assert kept == [default, other, steady, failed]
    rejected = [d.technique for d in context.journal.outcomes]
    assert rejected == ["m/random_undersample", "blending/default"]
    assert all(
        d.action is experiments.Edict.REJECT for d in context.journal.outcomes
    )


def test_selection_keeps_everything_without_a_default():
    trials = [scored("m", BASE, "log_skewed"), scored("n", BASE, "calibrated")]
    context = board(trials)
    assert selecting.earn(context, trials) == trials
    assert context.journal.outcomes == []
