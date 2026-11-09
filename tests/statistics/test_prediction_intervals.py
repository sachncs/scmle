"""Coverage of conformal and resampling intervals against a known truth.

Each test simulates at least 1,000 datasets from a known generating process
and checks the share of intervals that hold the truth:

* split conformal (`finite_sample_quantile`, `ConformalCalibrator`): for a
  continuous score the coverage is exactly `ceil((n + 1) c) / (n + 1)`, so
  that is the target, two-sided;
* class-conditional sets: coverage inside each class, at least nominal;
* one row per entity: coverage for a new entity with a random effect;
* weighted conformal under an exactly known covariate shift: at least nominal;
* CV+ (Barber et al. 2021): near nominal and above its `1 - 2 alpha` bound;
* percentile bootstrap intervals, the cluster-robust normal interval, DeLong
  and prediction-powered intervals: near nominal.
"""

import math

import numpy
from scipy import stats
import support_statistics

from scmle.data.detectors import labels
from scmle.data.framing import costs
from scmle.data.robustness import shift
from scmle.decision import ope
from scmle.decision import threshold
from scmle.evaluation import conformal
from scmle.operations import continuous

SIMULATIONS = 1000
COVERAGE = 0.9


def exact_rate(rows, coverage=COVERAGE):
    return math.ceil((rows + 1) * coverage) / (rows + 1)


def test_split_conformal_hits_the_exact_finite_sample_rate():
    rng = numpy.random.default_rng(0)
    runs = 3000
    for rows in (50, 100):
        hits = 0
        for ignored in range(runs):
            scores = numpy.abs(rng.normal(size=rows))
            margin = conformal.finite_sample_quantile(scores, COVERAGE)
            hits += abs(rng.normal()) <= margin
        support_statistics.assert_rate_near(
            hits / runs, exact_rate(rows), runs, f"split conformal n={rows}"
        )


def test_regression_calibrator_covers_a_new_row_with_heavy_tails():
    rng = numpy.random.default_rng(1)
    rows, runs = 100, 2000
    hits = 0
    for ignored in range(runs):
        predicted = rng.uniform(0, 10, rows)
        observed = predicted + rng.standard_t(3, rows)
        calibrator = conformal.ConformalCalibrator.for_regression(
            observed, predicted
        )
        new_prediction = rng.uniform(0, 10)
        new_observed = new_prediction + rng.standard_t(3)
        low, high = calibrator.interval([new_prediction], COVERAGE)
        hits += low[0] <= new_observed <= high[0]
    support_statistics.assert_rate_near(
        hits / runs, exact_rate(rows), runs, "regression calibrator, t(3)"
    )


def test_class_conditional_sets_cover_inside_each_class():
    rng = numpy.random.default_rng(2)
    rows, runs = 200, 2000
    classes = [0, 1]
    seen = {0: [0, 0], 1: [0, 0]}  # class -> [covered, total]
    for ignored in range(runs):
        x = rng.normal(size=rows + 1)
        posterior = 1.0 / (1.0 + numpy.exp(-2.0 * x))
        label = (rng.random(rows + 1) < posterior).astype(int)
        probabilities = numpy.c_[1.0 - posterior, posterior]
        calibrator = conformal.ConformalCalibrator.for_classification(
            label[:rows].tolist(), probabilities[:rows], classes
        )
        chosen = calibrator.sets(
            probabilities[rows:], COVERAGE, conditional="class"
        )[0]
        seen[int(label[rows])][0] += int(label[rows]) in chosen
        seen[int(label[rows])][1] += 1
    for name, (covered, total) in seen.items():
        support_statistics.assert_rate_at_least(
            covered / total, COVERAGE, total, f"class {name} conditional"
        )


def test_one_row_per_entity_covers_a_new_entity_with_a_random_effect():
    rng = numpy.random.default_rng(3)
    entities, per_entity, runs = 40, 8, 2000
    labels_of_rows = numpy.repeat(numpy.arange(entities), per_entity)
    hits = 0
    for ignored in range(runs):
        effect = rng.normal(scale=2.0, size=entities)
        rows = effect[:, None] + rng.normal(size=(entities, per_entity))
        chosen = conformal.one_row_per_entity(labels_of_rows, rng)
        scores = numpy.abs(rows.ravel()[chosen])
        margin = conformal.finite_sample_quantile(scores, COVERAGE)
        new = rng.normal(scale=2.0) + rng.normal()
        hits += abs(new) <= margin
    support_statistics.assert_rate_near(
        hits / runs, exact_rate(entities), runs, "one row per entity"
    )


def test_weighted_conformal_covers_under_a_known_covariate_shift():
    rng = numpy.random.default_rng(4)
    rows, runs = 200, 2000
    hits = 0
    for ignored in range(runs):
        x = rng.normal(size=rows)
        noise = rng.normal(size=rows) * (0.5 + 0.5 * numpy.abs(x))
        scores = numpy.abs(noise)
        # training covariates N(0, 1), test covariates N(1, 1): the exact
        # likelihood ratio is exp(x - 1/2)
        weights = numpy.exp(x - 0.5)
        new_x = rng.normal(loc=1.0)
        new_score = abs(rng.normal() * (0.5 + 0.5 * abs(new_x)))
        margin = conformal.weighted_quantile(
            scores, weights, float(numpy.exp(new_x - 0.5)), COVERAGE
        )
        hits += new_score <= margin
    support_statistics.assert_rate_at_least(
        hits / runs, COVERAGE, runs, "weighted conformal, covariate shift"
    )


def test_cv_plus_is_near_nominal_and_above_its_theoretical_bound():
    rng = numpy.random.default_rng(5)
    rows, folds, features, runs = 100, 10, 3, 1500
    hits = 0
    for ignored in range(runs):
        design = rng.normal(size=(rows + 1, features))
        target = design @ numpy.ones(features) + rng.normal(size=rows + 1)
        held_out = numpy.arange(rows) % folds
        rng.shuffle(held_out)
        predictions = numpy.empty(folds)
        residuals = numpy.empty(rows)
        for fold in range(folds):
            train = held_out != fold
            basis = numpy.c_[numpy.ones(train.sum()), design[:rows][train]]
            beta = numpy.linalg.lstsq(basis, target[:rows][train], rcond=None)[
                0
            ]
            predictions[fold] = beta[0] + design[rows] @ beta[1:]
            rest = design[:rows][~train]
            fit = beta[0] + rest @ beta[1:]
            residuals[~train] = numpy.abs(target[:rows][~train] - fit)
        low, high = conformal.cv_plus_bounds(
            predictions[None, :], held_out, residuals, COVERAGE
        )
        hits += low[0] <= target[rows] <= high[0]
    observed = hits / runs
    support_statistics.assert_rate_near(
        observed, COVERAGE, runs, "cv+ near nominal"
    )
    theorem = 1.0 - 2.0 * (1.0 - COVERAGE) - math.sqrt(2.0 / rows)
    assert observed >= theorem


def test_bootstrap_difference_interval_covers_a_mean_difference():
    rng = numpy.random.default_rng(6)
    hits = 0
    for ignored in range(SIMULATIONS):
        before = rng.exponential(size=60)
        after = rng.exponential(size=60) + 0.5
        low, high = continuous.bootstrap_difference(
            before, after, 0.05, 400, int(rng.integers(1 << 30))
        )
        hits += low <= 0.5 <= high
    support_statistics.assert_rate_near(
        hits / SIMULATIONS, 0.95, SIMULATIONS, "bootstrap difference"
    )


def cost_model():
    return costs.DecisionCostModel(
        0.0,
        100.0,
        5.0,
        0.0,
        0.02,
        0.15,
        (costs.CostSource.HUMAN,) * 4,
        costs.CostSource.HUMAN,
    )


def test_cost_interval_covers_the_expected_cost_per_row():
    rng = numpy.random.default_rng(7)
    rows, rate, act = 200, 0.1, 0.3
    truth = act * (1 - rate) * 5.0 + (1 - act) * rate * 100.0
    hits = 0
    for ignored in range(SIMULATIONS):
        outcome = (rng.random(rows) < rate).astype(float)
        action = (rng.random(rows) < act).astype(float)
        low, high = threshold.cost_interval(
            outcome,
            action,
            cost_model(),
            None,
            300,
            int(rng.integers(1 << 30)),
        )
        hits += low <= truth <= high
    support_statistics.assert_rate_near(
        hits / SIMULATIONS, 0.95, SIMULATIONS, "cost interval, rows"
    )


def test_cost_interval_resamples_whole_clusters():
    rng = numpy.random.default_rng(8)
    groups, per_group, rate, act = 40, 5, 0.1, 0.3
    truth = act * (1 - rate) * 5.0 + (1 - act) * rate * 100.0
    clusters = numpy.repeat(numpy.arange(groups), per_group)
    hits = 0
    for ignored in range(SIMULATIONS):
        outcome = (rng.random(groups * per_group) < rate).astype(float)
        action = (rng.random(groups * per_group) < act).astype(float)
        low, high = threshold.cost_interval(
            outcome,
            action,
            cost_model(),
            clusters,
            300,
            int(rng.integers(1 << 30)),
        )
        hits += low <= truth <= high
    support_statistics.assert_rate_near(
        hits / SIMULATIONS, 0.95, SIMULATIONS, "cost interval, clusters"
    )


def test_off_policy_normal_interval_covers_with_and_without_clusters():
    rng = numpy.random.default_rng(9)
    groups, per_group, runs = 40, 5, 3000
    clusters = numpy.repeat(numpy.arange(groups), per_group)
    plain = clustered = 0
    for ignored in range(runs):
        values = rng.normal(size=groups * per_group)
        low, high = ope.mean_interval(values, None, 0.95)[1]
        plain += low <= 0.0 <= high
        effect = numpy.repeat(rng.normal(size=groups), per_group)
        values = effect + rng.normal(size=groups * per_group)
        low, high = ope.mean_interval(values, clusters, 0.95)[1]
        clustered += low <= 0.0 <= high
    support_statistics.assert_rate_near(
        plain / runs, 0.95, runs, "ope interval, independent rows"
    )
    support_statistics.assert_rate_near(
        clustered / runs, 0.95, runs, "ope interval, clustered rows"
    )


def test_delong_interval_covers_the_true_auc():
    rng = numpy.random.default_rng(10)
    positives, negatives, gap, runs = 80, 120, 1.0, 2000
    truth = float(stats.norm.cdf(gap / math.sqrt(2.0)))
    side = numpy.r_[numpy.ones(positives), numpy.zeros(negatives)]
    hits = 0
    for ignored in range(runs):
        scores = numpy.r_[
            rng.normal(gap, 1.0, positives), rng.normal(0.0, 1.0, negatives)
        ]
        low, high = shift.auc_delong(scores, side)[1]
        hits += low <= truth <= high
    support_statistics.assert_rate_near(
        hits / runs, 0.95, runs, "DeLong interval"
    )


def test_prediction_powered_interval_covers_the_population_mean():
    rng = numpy.random.default_rng(11)
    unlabeled, audited, runs = 2000, 100, 2000
    hits = 0
    for ignored in range(runs):
        truth = rng.binomial(1, 0.3, unlabeled + audited).astype(float)
        predicted = numpy.clip(
            0.8 * truth + 0.15 + rng.normal(0.0, 0.1, unlabeled + audited),
            0.0,
            1.0,
        )
        low, high = labels.ppi_mean(
            predicted[:unlabeled],
            predicted[unlabeled:],
            truth[unlabeled:],
        )[1]
        hits += low <= 0.3 <= high
    support_statistics.assert_rate_near(
        hits / runs, 0.95, runs, "prediction-powered interval"
    )
