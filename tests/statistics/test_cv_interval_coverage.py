"""Coverage of the Nadeau-Bengio interval for a cross-validated score.

`confidence.cv_interval` claims an interval for the mean cross-validation
score that respects the dependence between folds. The truth is known in each
simulation below, so the coverage is measured, not assumed:

* iid rows, ordinary least squares on pure-noise features: a model trained
  on `m` rows has expected squared error `1 + p / (m - p - 1)` exactly;
* grouped rows (random group effect), whole groups per fold, the sample mean
  as the model: expected squared error `2 + 1.2 / (training groups)`;
* time-ordered rows (autoregressive), expanding windows: the truth is the
  long-run mean of the same fold average, from a large simulation.
"""

import math

import numpy
from scipy import stats
import support_statistics

from scmle.evaluation import confidence

SIMULATIONS = 1000
NOMINAL = 0.95


def ols_fold_scores(rng, rows, folds, features):
    design = rng.normal(size=(rows, features))
    target = rng.normal(size=rows)
    order = rng.permutation(rows)
    scores = []
    for held in numpy.array_split(order, folds):
        train = numpy.setdiff1d(order, held)
        beta = numpy.linalg.lstsq(design[train], target[train], rcond=None)[0]
        residual = target[held] - design[held] @ beta
        scores.append(float(numpy.mean(residual**2)))
    return scores, len(held)


def naive_interval(report, folds):
    margin = (
        float(stats.t.ppf(0.975, folds - 1)) * report["naive_standard_error"]
    )
    return report["mean"] - margin, report["mean"] + margin


def iid_coverage(rows, folds, features, seed):
    rng = numpy.random.default_rng(seed)
    training = rows - rows // folds
    truth = 1.0 + features / (training - features - 1)
    corrected = naive = 0
    for ignored in range(SIMULATIONS):
        scores, held = ols_fold_scores(rng, rows, folds, features)
        report = confidence.cv_interval(scores, training, held, "random")
        low, high = report["interval"]
        corrected += low <= truth <= high
        low, high = naive_interval(report, folds)
        naive += low <= truth <= high
    return corrected / SIMULATIONS, naive / SIMULATIONS


def test_the_interval_covers_the_expected_score_with_iid_rows():
    assert confidence.LEVEL == NOMINAL
    for rows, folds, features in [(100, 5, 20), (100, 10, 20), (60, 5, 10)]:
        corrected, naive = iid_coverage(rows, folds, features, seed=2)
        label = f"n={rows} k={folds} p={features}"
        support_statistics.assert_rate_at_least(
            corrected, NOMINAL, SIMULATIONS, f"corrected {label}"
        )
        # the naive error is why the correction exists: it is too narrow
        limit = support_statistics.Z_LIMIT * support_statistics.standard_error(
            NOMINAL, SIMULATIONS
        )
        support_statistics.show(f"naive {label}", naive, NOMINAL, SIMULATIONS)
        assert naive < NOMINAL - limit, (label, naive)


def group_scores(rng, groups, per_group, folds):
    effect = rng.normal(size=groups)
    values = effect[:, None] + rng.normal(size=(groups, per_group))
    order = rng.permutation(groups)
    scores = []
    for held in numpy.array_split(order, folds):
        train = numpy.setdiff1d(order, held)
        scores.append(
            float(numpy.mean((values[held] - values[train].mean()) ** 2))
        )
    return scores


def test_the_interval_covers_the_expected_score_with_grouped_rows():
    groups, per_group, folds = 30, 5, 5
    training_groups = groups - groups // folds
    truth = 2.0 + (1.0 + 1.0 / per_group) / training_groups
    rng = numpy.random.default_rng(3)
    hits = 0
    for ignored in range(SIMULATIONS):
        scores = group_scores(rng, groups, per_group, folds)
        report = confidence.cv_interval(
            scores,
            training_groups * per_group,
            (groups // folds) * per_group,
            "group",
        )
        low, high = report["interval"]
        hits += low <= truth <= high
    support_statistics.assert_rate_at_least(
        hits / SIMULATIONS, NOMINAL, SIMULATIONS, "grouped"
    )


def autoregressive(rng, rows, rho):
    noise = rng.normal(size=rows)
    series = numpy.empty(rows)
    series[0] = noise[0]
    scale = math.sqrt(1.0 - rho * rho)
    for step in range(1, rows):
        series[step] = rho * series[step - 1] + scale * noise[step]
    return series


def expanding_scores(series, folds):
    block = len(series) // (folds + 1)
    scores = []
    for fold in range(1, folds + 1):
        train = series[: block * fold]
        held = series[block * fold : block * (fold + 1)]
        scores.append(float(numpy.mean((held - train.mean()) ** 2)))
    return scores, block


def test_the_interval_covers_the_long_run_score_with_time_ordered_rows():
    rows, folds, rho = 120, 5, 0.7
    rng = numpy.random.default_rng(4)
    truth = float(
        numpy.mean(
            [
                numpy.mean(
                    expanding_scores(autoregressive(rng, rows, rho), folds)[0]
                )
                for ignored in range(6000)
            ]
        )
    )
    hits = 0
    for ignored in range(SIMULATIONS):
        scores, block = expanding_scores(autoregressive(rng, rows, rho), folds)
        report = confidence.cv_interval(
            scores, block * (folds + 1) / 2.0, block, "temporal"
        )
        low, high = report["interval"]
        hits += low <= truth <= high
    support_statistics.assert_rate_at_least(
        hits / SIMULATIONS, NOMINAL, SIMULATIONS, "temporal"
    )
