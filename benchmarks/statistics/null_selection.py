"""Measures how often a run claims skill when every candidate has none.

The data have a binary target drawn independently of every feature, so each
candidate has a true ROC AUC of exactly 0.5 and the constant (`dummy`)
baseline has 0.5 as well. `--configs-per-model` adds hyperparameter trials to
each of the three models, so a run chooses among 3, 9, 30 or more candidates,
and `--signal` gives the target a real dependence on two features so that the
candidates differ in skill. Every run goes through the public `Project.run`
(in process, cooperative enforcement, so no subprocess per trial), and the
script counts how often the run's own outputs say the winner is better than
the baseline. A run has no single "the winner beats the baseline" statement,
so four operational readings are counted, each as the run would present it:

* `test_interval_excludes_baseline`: the 95 percent bootstrap interval of the
  winner's test ROC AUC (`diagnostics/metric_uncertainty`) lies wholly above
  0.5, the baseline's test score.
* `cv_interval_excludes_baseline`: the Nadeau-Bengio interval of the winner's
  cross-validation score (`cv_confidence`) lies wholly above the baseline's
  cross-validation score (`baseline_cv_score`).
* `lead_exceeds_spread`: the winner's cross-validation lead over the
  next-best trial is larger than the larger fold spread, the rule the trail
  comparison report (`scmle.trail.reports`) uses for `lead_exceeds_spread`.
* `fit_health_healthy`: the run labels the winner `healthy` (its training
  gain over the baseline carries over to validation).

It also records how often the two intervals contain the true value 0.5 and
how often the leakage screen names a suspect among the honest columns. With a
signal the true value of the test interval is the winner's own ROC AUC on
200,000 fresh rows from the same process, and the cross-validation interval is
not scored (its target depends on the training size). Each rate comes with
exact Clopper-Pearson and Wilson 95 percent intervals.

    python -m benchmarks.statistics.null_selection --runs 200 --output out.json
    python -m benchmarks.statistics.null_selection --configs-per-model 9 \
        --output out30.json
"""

import argparse
from collections.abc import Sequence
import json
import pathlib
import sys
import tempfile
import time
from typing import Any

import numpy
import pandas
from scipy import stats
from sklearn import metrics

import scmle.api.project
from scmle.contracts import experiments
from scmle.contracts import tasks
import scmle.models.registry

TRUE_SKILL = 0.5
CANDIDATES = ("decision_tree", "knn", "logistic_regression")
CLAIMS = (
    "test_interval_excludes_baseline",
    "cv_interval_excludes_baseline",
    "lead_exceeds_spread",
    "fit_health_healthy",
    "leakage_suspect",
)
COVERAGES = ("test_interval_covers_truth", "cv_interval_covers_truth")
FRESH_ROWS = 200_000
FRESH_SEED = 10**6
DEFAULT_OUTPUT = pathlib.Path(__file__).with_name("null_selection_result.json")


def draw_frame(rows: int, seed: int, signal: float) -> pandas.DataFrame:
    """Draws a table whose target depends on two features, or on none.

    Args:
        rows: Number of rows.
        seed: Seed of the draw.
        signal: Coefficient of `f0 + f1` in the log-odds of the target; 0
            makes the target a fair coin independent of every feature.

    Returns:
        Six standard normal columns, one categorical column and a binary
        target `y`.
    """
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame({f"f{i}": rng.normal(size=rows) for i in range(6)})
    frame["cat"] = rng.choice(list("abc"), rows)
    odds = signal * (frame["f0"] + frame["f1"])
    frame["y"] = (rng.random(rows) < 1 / (1 + numpy.exp(-odds))).astype(int)
    return frame


def excluded_models() -> list[str]:
    """Names every built-in model except the three candidates and `dummy`.

    Returns:
        The model names to exclude, so a run has three candidates and the
        constant baseline.
    """
    names = scmle.models.registry.Catalog.with_builtins().adapters
    return sorted(set(names) - set(CANDIDATES) - {"dummy"})


def candidate_count(run: Any) -> int:
    """Counts the candidates the winner was chosen among.

    Args:
        run: A completed `Run`.

    Returns:
        The number of trials that finished, not counting the constant
        baseline.
    """
    return sum(
        t.status is experiments.TrialStatus.OK and t.model_name != "dummy"
        for t in run.trials
    )


def lead_exceeds_spread(run: Any) -> bool:
    """Applies the trail report's rule to the winner and the next trial.

    Args:
        run: A completed `Run`.

    Returns:
        Whether the winner's mean score minus the runner-up's exceeds the
        larger of the two fold standard deviations. The runner-up may be the
        dummy baseline, as in the report.
    """
    done = [t for t in run.trials if t.status is experiments.TrialStatus.OK]
    done.sort(key=lambda t: (-t.score_mean, t.trial_id))
    selected = run.evaluation["selected_trial_id"]
    chosen = next(t for t in done if t.trial_id == selected)
    others = [t for t in done if t.trial_id != selected]
    if not others:
        return False
    runner = others[0]
    lead = chosen.score_mean - runner.score_mean
    return bool(lead > max(chosen.score_std, runner.score_std))


def one_run(
    index: int,
    rows: int,
    folds: int,
    seed: int,
    directory: pathlib.Path,
    configs_per_model: int,
    signal: float,
) -> dict[str, Any]:
    """Runs the full loop once on null data and reads the claims.

    Args:
        index: Position of the run, used in the data seed.
        rows: Rows of the table.
        folds: Cross-validation folds.
        seed: Base seed.
        directory: Folder for the run's artifacts.
        configs_per_model: Hyperparameter trials added to each model (0
            searches nothing).
        signal: Dependence of the target on the features, see `draw_frame`.

    Returns:
        One record: the claims as booleans plus the numbers behind them.
    """
    frame = draw_frame(rows, seed + index, signal)
    started = time.monotonic()
    run = scmle.api.project.Project(
        frame,
        "y",
        output_dir=directory,
        folds=folds,
        seed=seed + index,
        time_budget=300,
        enforcement="cooperative",
        search=tasks.Search(
            top_models=len(CANDIDATES) if configs_per_model else 0,
            configs_per_model=max(configs_per_model, 1),
            prune_fraction=0.0,
        ),
        strategies=tasks.Strategies(top_models=0),
        ensemble=tasks.Stacking(size=0),
        models=tasks.Candidates(exclude_models=excluded_models()),
    ).run()
    evaluation = run.evaluation
    test_low, test_high = run.diagnostics["metric_uncertainty"]["ci_95"]
    cv_low, cv_high = evaluation["cv_confidence"]["interval"]
    chosen = next(
        t for t in run.trials if t.trial_id == evaluation["selected_trial_id"]
    )
    truth = TRUE_SKILL
    if signal:
        fresh = draw_frame(FRESH_ROWS, FRESH_SEED + seed + index, signal)
        truth = float(
            metrics.roc_auc_score(
                fresh["y"], run.predict_proba(fresh.drop(columns="y"))[:, 1]
            )
        )
    record = {
        "run": index,
        "candidates": candidate_count(run),
        "selected_model": evaluation["selected_model"],
        "test_auc": evaluation["test_metrics"]["roc_auc"],
        "cv_score": evaluation["selected_cv_score"],
        "baseline_cv_score": evaluation["baseline_cv_score"],
        "test_interval": [test_low, test_high],
        "cv_interval": [cv_low, cv_high],
        "fit_health": run.health_label(chosen),
        "test_interval_excludes_baseline": bool(test_low > TRUE_SKILL),
        "cv_interval_excludes_baseline": bool(
            cv_low > evaluation["baseline_cv_score"]
        ),
        "lead_exceeds_spread": lead_exceeds_spread(run),
        "fit_health_healthy": run.health_label(chosen) == "healthy",
        "leakage_suspect": bool(run.profile.leakage.get("suspects")),
        "true_test_auc": truth,
        "test_interval_covers_truth": bool(test_low <= truth <= test_high),
        "test_interval_above_truth": bool(test_low > truth),
        "test_interval_below_truth": bool(test_high < truth),
        "selected_trial_id": evaluation["selected_trial_id"],
        "trial_fold_scores": {
            t.trial_id: t.fold_scores
            for t in run.trials
            if t.status is experiments.TrialStatus.OK
        },
        "seconds": time.monotonic() - started,
    }
    if not signal:
        record["cv_interval_covers_truth"] = bool(
            cv_low <= TRUE_SKILL <= cv_high
        )
    return record


def rate_summary(count: int, runs: int) -> dict[str, Any]:
    """Describes a rate with its exact and Wilson 95 percent intervals.

    Args:
        count: Number of runs in which the event happened.
        runs: Number of runs.

    Returns:
        The count, the runs, the rate and the two intervals.
    """
    test = stats.binomtest(count, runs)
    exact = test.proportion_ci(0.95, method="exact")
    wilson = test.proportion_ci(0.95, method="wilson")
    return {
        "count": count,
        "runs": runs,
        "rate": count / runs,
        "clopper_pearson_95": [float(exact.low), float(exact.high)],
        "wilson_95": [float(wilson.low), float(wilson.high)],
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates the per-run records.

    Args:
        records: One record per run, from `one_run`.

    Returns:
        Rate summaries for every claim and coverage, and the mean test AUC
        and mean run time.
    """
    runs = len(records)
    return {
        "claims": {
            name: rate_summary(sum(r[name] for r in records), runs)
            for name in CLAIMS
        },
        "coverage_of_true_skill": {
            name: rate_summary(sum(r[name] for r in records), runs)
            for name in COVERAGES
            if name in records[0]
        },
        "test_interval_misses": {
            name: rate_summary(sum(r[name] for r in records), runs)
            for name in (
                "test_interval_above_truth",
                "test_interval_below_truth",
            )
        },
        "mean_candidates": float(
            numpy.mean([r["candidates"] for r in records])
        ),
        "mean_true_test_auc": float(
            numpy.mean([r["true_test_auc"] for r in records])
        ),
        "mean_cv_score_minus_test_auc": float(
            numpy.mean([r["cv_score"] - r["test_auc"] for r in records])
        ),
        "mean_test_auc": float(numpy.mean([r["test_auc"] for r in records])),
        "mean_seconds_per_run": float(
            numpy.mean([r["seconds"] for r in records])
        ),
        "winners": {
            name: sum(r["selected_model"] == name for r in records)
            for name in sorted({r["selected_model"] for r in records})
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the null simulation and writes the JSON result.

    Args:
        argv: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        0 after the result is written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--configs-per-model", type=int, default=0)
    parser.add_argument("--signal", type=float, default=0.0)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as scratch:
        records = [
            one_run(
                index,
                args.rows,
                args.folds,
                args.seed,
                pathlib.Path(scratch) / f"run_{index}",
                args.configs_per_model,
                args.signal,
            )
            for index in range(args.runs)
        ]
    result = {
        "configuration": {
            "runs": args.runs,
            "rows": args.rows,
            "folds": args.folds,
            "seed": args.seed,
            "models": list(CANDIDATES),
            "configs_per_model": args.configs_per_model,
            "signal": args.signal,
            "true_roc_auc_of_every_candidate": None
            if args.signal
            else TRUE_SKILL,
            "nominal_level_of_the_intervals": 0.95,
        },
        "summary": summarize(records),
        "records": records,
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
