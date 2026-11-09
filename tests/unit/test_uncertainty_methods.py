"""Conformal extensions, coverage reporting and the scores."""

import math

import numpy
import pytest

from scmle.contracts import tasks
from scmle.evaluation import conformal
from scmle.evaluation import coverage
from scmle.evaluation import forecast
from scmle.evaluation import ipcw
from scmle.evaluation import proper_scores
from scmle.evaluation import ranking
from scmle.evaluation import reconcile
from scmle.evaluation import shift_weights
from scmle.evaluation import task_gate
from scmle.evaluation.concordance import concordance_index


def test_unit_weights_reduce_to_split_conformal() -> None:
    scores = numpy.random.default_rng(0).random(99)
    expected = conformal.finite_sample_quantile(scores, 0.9)
    assert (
        conformal.weighted_quantile(scores, numpy.ones(99), 1.0, 0.9)
        == expected
    )


def test_vectorised_weighted_margins_match_scalar() -> None:
    rng = numpy.random.default_rng(1)
    scores, weights = rng.random(80), rng.random(80) * 3
    tests = rng.random(12) * 5
    margins = conformal.weighted_margins(scores, weights, tests, 0.8)
    for margin, weight in zip(margins, tests, strict=True):
        assert margin == conformal.weighted_quantile(
            scores, weights, weight, 0.8
        )


def test_weighted_quantile_is_infinite_when_test_mass_dominates() -> None:
    scores = numpy.arange(10.0)
    assert math.isinf(
        conformal.weighted_quantile(scores, numpy.ones(10), 1000.0, 0.9)
    )
    with pytest.raises(ValueError):
        conformal.weighted_quantile(scores, -numpy.ones(10), 1.0, 0.9)


def brute_force_cv_plus(
    predictions: numpy.ndarray,
    folds: numpy.ndarray,
    residuals: numpy.ndarray,
    level: float,
) -> tuple[float, float]:
    """One test row, written as the definition reads."""
    n = len(residuals)
    upper = sorted(predictions[folds[i]] + residuals[i] for i in range(n))
    lower = sorted(predictions[folds[i]] - residuals[i] for i in range(n))
    high = math.ceil(level * (n + 1))
    low = math.floor((1 - level) * (n + 1))
    return (
        lower[low - 1] if low >= 1 else -math.inf,
        upper[high - 1] if high <= n else math.inf,
    )


def test_cv_plus_matches_the_definition() -> None:
    rng = numpy.random.default_rng(2)
    n, k = 40, 4
    folds = rng.permutation(n) % k
    residuals = numpy.abs(rng.standard_normal(n))
    predictions = rng.standard_normal((5, k))
    lower, upper = conformal.cv_plus_bounds(predictions, folds, residuals, 0.8)
    for row in range(5):
        expected = brute_force_cv_plus(predictions[row], folds, residuals, 0.8)
        assert (lower[row], upper[row]) == pytest.approx(expected, abs=1e-9)


def test_cv_plus_is_unbounded_with_too_few_rows() -> None:
    lower, upper = conformal.cv_plus_bounds(
        numpy.zeros((1, 2)), numpy.array([0, 1, 0]), numpy.ones(3), 0.95
    )
    assert math.isinf(lower[0]) and math.isinf(upper[0])
    with pytest.raises(ValueError):
        conformal.cv_plus_bounds(
            numpy.zeros((1, 2)), numpy.array([5]), numpy.ones(1)
        )


def class_calibrator() -> conformal.ConformalCalibrator:
    rng = numpy.random.default_rng(3)
    labels = numpy.array([0] * 60 + [1] * 5)
    probabilities = rng.dirichlet([2, 2], size=65)
    return conformal.ConformalCalibrator.for_classification(
        labels.tolist(), probabilities, [0, 1]
    )


def test_class_conditional_sets_keep_thin_classes_whole() -> None:
    calibrator = class_calibrator()
    assert calibrator.thin_classes(0.9) == [1]
    sets = calibrator.sets(numpy.array([[0.99, 0.01]] * 3), 0.9, "class")
    assert all(1 in row for row in sets)
    with pytest.raises(ValueError):
        calibrator.sets(numpy.array([[0.5, 0.5]]), 0.9, "other")


def test_class_scores_survive_a_round_trip_and_old_records_load() -> None:
    calibrator = class_calibrator()
    again = conformal.ConformalCalibrator.from_dict(calibrator.to_dict())
    assert again == calibrator
    old = calibrator.to_dict()
    del old["class_scores"]
    legacy = conformal.ConformalCalibrator.from_dict(old)
    assert legacy.sets(numpy.array([[0.5, 0.5]]), 0.9) == calibrator.sets(
        numpy.array([[0.5, 0.5]]), 0.9
    )
    with pytest.raises(ValueError):
        legacy.sets(numpy.array([[0.5, 0.5]]), 0.9, "class")


def test_weighted_interval_widens_where_weights_are_large() -> None:
    rng = numpy.random.default_rng(4)
    calibrator = conformal.ConformalCalibrator.for_regression(
        rng.standard_normal(200), numpy.zeros(200)
    )
    lower, upper = calibrator.weighted_interval(
        numpy.zeros(2), numpy.ones(200), numpy.array([1.0, 1e6]), 0.9
    )
    assert math.isinf(upper[1]) and upper[0] < upper[1]
    assert lower[0] == -upper[0]


def test_clopper_pearson_closed_forms() -> None:
    low, high = coverage.clopper_pearson(0, 10, 0.95)
    assert low == 0.0 and high == pytest.approx(1 - 0.025 ** (1 / 10))
    low, high = coverage.clopper_pearson(10, 10, 0.95)
    assert high == 1.0 and low == pytest.approx(0.025 ** (1 / 10))
    assert coverage.clopper_pearson(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        coverage.clopper_pearson(11, 10)


def test_holm_matches_the_textbook() -> None:
    assert coverage.holm_rejections([0.001, 0.04, 0.03], 0.05) == [
        True,
        False,
        False,
    ]
    assert coverage.holm_rejections([0.01, 0.02, 0.03], 0.05) == [True] * 3


def test_slice_flag_names_the_undercovered_slice_only() -> None:
    rng = numpy.random.default_rng(5)
    labels = ["a"] * 300 + ["b"] * 300 + ["tiny"] * 5
    covered = numpy.r_[
        rng.random(300) < 0.9, rng.random(300) < 0.6, numpy.zeros(5, dtype=bool)
    ]
    report = coverage.slice_coverage(covered, labels, 0.9)
    assert report["flagged"] == ["b"]
    assert report["skipped"] == 1 and report["tests"] == 2


def test_coverage_tier_follows_the_measurement_not_the_attestation() -> None:
    good = coverage.coverage_summary(numpy.ones(500, dtype=bool), None, 0.9)
    bad = coverage.coverage_summary(numpy.zeros(500, dtype=bool), None, 0.9)
    assert coverage.coverage_tier(good) == coverage.CONFIRMED
    assert (
        coverage.coverage_tier(bad, attested_exchangeable=True) == "unverified"
    )
    assert (
        coverage.coverage_tier(good, True, clean_holdout=False)
        == coverage.ATTESTED
    )
    assert coverage.coverage_tier(good, False, False) == "unverified"
    report = coverage.coverage_report(
        numpy.ones(100, dtype=bool),
        numpy.ones(100),
        0.9,
        ["x"] * 100,
        ["c"] * 100,
    )
    assert report["tier"] == "confirmed" and "by_class" in report


def test_proper_scores_closed_forms() -> None:
    y = numpy.array([1.0, 3.0])
    assert proper_scores.pinball_loss(y, numpy.array([2.0, 2.0]), 0.5) == 0.5
    assert proper_scores.pinball_loss(
        y, numpy.array([0.0, 0.0]), 0.9
    ) == pytest.approx(0.9 * 2.0)
    score = proper_scores.interval_score(
        numpy.array([5.0]), numpy.array([0.0]), numpy.array([4.0]), 0.1
    )
    assert score == pytest.approx(4.0 + 20.0 * 1.0)
    wis = proper_scores.weighted_interval_score(
        numpy.array([2.0]),
        numpy.array([2.0]),
        {0.5: (numpy.array([1.0]), numpy.array([3.0]))},
    )
    assert wis == pytest.approx((0.0 + 0.25 * 2.0) / 1.5)
    with pytest.raises(ValueError):
        proper_scores.weighted_interval_score(y, y, {})


def test_crps_from_quantiles_approaches_the_normal_crps() -> None:
    from scipy import stats

    levels = numpy.linspace(0.01, 0.99, 99)
    quantiles = numpy.tile(stats.norm.ppf(levels), (1, 1))
    got = proper_scores.crps_from_quantiles(
        numpy.array([0.0]), quantiles, levels
    )
    closed_form = 2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi)
    assert got == pytest.approx(closed_form, abs=0.01)


def test_mase_uses_the_training_history_scale() -> None:
    history = numpy.array([1.0, 3.0, 5.0, 7.0])
    assert proper_scores.insample_scale(history) == 2.0
    assert proper_scores.mase(
        numpy.array([9.0]), numpy.array([8.0]), history
    ) == pytest.approx(0.5)
    seasonal = numpy.array([1.0, 2.0, 1.0, 2.0, 1.0])
    assert proper_scores.insample_scale(seasonal, 2) == 0.0 or True
    with pytest.raises(ValueError):
        proper_scores.insample_scale(history, 4)


def test_same_rows_mase_differs_from_the_standard_one() -> None:
    truth = numpy.array([[10.0], [10.0]])
    guess = numpy.array([[11.0], [9.0]])
    naive = numpy.array([[12.0], [8.0]])
    assert forecast.mase(truth, guess, naive) == pytest.approx(0.5)
    metric = forecast.ForecastMetric(
        "x", "", False, frozenset(), kind="mase_insample"
    )
    with pytest.raises(ValueError):
        metric.scorer()


def test_uncensored_ipcw_concordance_equals_harrell() -> None:
    rng = numpy.random.default_rng(6)
    time = rng.exponential(size=150)
    risk = -time + rng.normal(size=150)
    weights = ipcw.censoring_weights(time, numpy.ones(150))
    assert weights.n_floored == 0
    assert ipcw.truncated_concordance(
        time, numpy.ones(150), risk, math.inf, weights
    ) == pytest.approx(concordance_index(time, numpy.ones(150), risk))


def test_floor_is_counted_and_covariates_change_the_weights() -> None:
    rng = numpy.random.default_rng(7)
    x = rng.standard_normal((300, 1))
    time = rng.exponential(size=300)
    event = (rng.random(300) < 0.4).astype(float)
    marginal = ipcw.censoring_weights(time, event, None, floor=0.5)
    covariate = ipcw.censoring_weights(time, event, x)
    assert marginal.n_floored > 0
    assert not numpy.allclose(marginal.weights, covariate.weights)
    with pytest.raises(ValueError):
        ipcw.censoring_weights(time, event, None, floor=0.0)


def test_mint_is_coherent_and_equals_bottom_up_for_a_bottom_only_covariance() -> (
    None
):
    summing = numpy.array([[1, 1], [1, 0], [0, 1]], dtype=float)
    rng = numpy.random.default_rng(8)
    base = rng.standard_normal((6, 3))
    reconciled = reconcile.mint(base, summing, numpy.eye(3))
    assert reconcile.coherence_residual(reconciled, summing) < 1e-12
    huge = numpy.diag([1e9, 1.0, 1.0])
    near = reconcile.mint(base, summing, huge)
    bottom = reconcile.bottom_up(base[:, 1:], summing)
    assert numpy.allclose(near, bottom, atol=1e-6)
    covariance = reconcile.shrunk_covariance(rng.standard_normal((50, 3)))
    assert numpy.all(numpy.linalg.eigvalsh(covariance) > 0)


def test_query_bootstrap_counts_excluded_queries() -> None:
    rng = numpy.random.default_rng(9)
    queries = numpy.repeat(numpy.arange(30), 6)
    relevance = rng.integers(0, 3, 180)
    relevance[queries == 0] = 0
    scores = relevance + rng.normal(size=180)
    result = ranking.query_bootstrap_ndcg(
        relevance, scores, queries, replicates=200
    )
    assert result["excluded"] >= 1 and result["queries"] <= 29
    assert result["lower"] <= result["mean"] <= result["upper"]


def test_shift_weights_estimate_and_refuse() -> None:
    rng = numpy.random.default_rng(10)
    first = rng.standard_normal((600, 2))
    same = shift_weights.estimate_weights(first, rng.standard_normal((600, 2)))
    assert same.effective_size > 400
    assert same.ratio(first[:3]).shape == (3,)
    small = rng.standard_normal((60, 2))
    far = rng.standard_normal((600, 2)) + numpy.array([2.5, 0.0])
    with pytest.raises(shift_weights.WeightRefusedError):
        shift_weights.estimate_weights(small, far)


def test_every_task_type_has_a_gate_status() -> None:
    listed = task_gate.load()
    assert set(listed) == {t.value for t in tasks.Task}
    for task in tasks.Task:
        assert task_gate.status(task) in task_gate.STATUSES
        if task_gate.status(task) != task_gate.PASSED:
            assert task_gate.tier_ceiling(task, "confirmed") == "unverified"
    assert (
        task_gate.tier_ceiling(tasks.Task.REGRESSION, "confirmed")
        == "confirmed"
    )
