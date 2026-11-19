"""Calibration tests and the claim of each structured task."""

import dataclasses
import json

import numpy
import pytest
from scipy import special
from scipy import stats

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.evaluation import calibration
from scmle.evaluation import ranking
from scmle.models.survival import estimators


def levels(rows=4000, seed=0, shift=0.0):
    """Ordinal truth drawn from known probabilities; `shift` biases them."""
    rng = numpy.random.default_rng(seed)
    latent = rng.normal(size=rows)
    edges = numpy.array([-0.5, 0.7])
    below = special.expit(edges[None, :] - latent[:, None])
    cumulative = numpy.column_stack([below, numpy.ones(rows)])
    probabilities = numpy.diff(
        numpy.column_stack([numpy.zeros(rows), cumulative]), axis=1
    )
    draw = rng.random(rows)
    truth = (draw[:, None] > cumulative).sum(axis=1)
    biased = special.expit(edges[None, :] + shift - latent[:, None])
    stated = numpy.diff(
        numpy.column_stack([numpy.zeros(rows), biased, numpy.ones(rows)]),
        axis=1,
    )
    return truth, stated if shift else probabilities


def test_group_statistics_follow_the_chi_square_formula():
    groups = (
        calibration.Group(count=10, predicted=0.2, observed=0.3, variance=0.01),
        calibration.Group(count=30, predicted=0.5, observed=0.4, variance=0.04),
    )

    check = calibration.Check("demo", groups)

    assert check.tested is True
    assert check.statistic == pytest.approx(1.0 + 0.25)
    assert check.freedom == 2
    assert check.pvalue == pytest.approx(stats.chi2.sf(1.25, 2))
    assert check.error == pytest.approx((10 * 0.1 + 30 * 0.1) / 40)


def test_a_check_without_groups_is_untested_and_never_supports_a_claim():
    check = calibration.Check("empty", ())
    claim = calibration.Claim(tasks.Task.ORDINAL, "statement", (check,))

    assert check.tested is False
    assert numpy.isnan(check.pvalue)
    assert claim.supported is False


def test_partition_merges_short_groups_into_their_neighbour():
    order = numpy.arange(10)

    found = calibration.partition(order, lambda members: len(members) >= 3, 4)

    assert [list(group) for group in found] == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8, 9],
    ]


def test_partition_finds_nothing_when_the_whole_sample_is_too_small():
    found = calibration.partition(numpy.arange(4), lambda m: len(m) >= 5, 2)

    assert found == []


def test_bernoulli_groups_use_the_exact_null_variance():
    probability = numpy.repeat([0.6, 0.2], 5)
    outcome = numpy.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1])

    check = calibration.bernoulli(
        "pair", probability, outcome, expected=1.0, groups=2
    )

    assert check.groups == (
        calibration.Group(5, pytest.approx(0.2), 0.2, pytest.approx(0.032)),
        calibration.Group(5, pytest.approx(0.6), 0.8, pytest.approx(0.048)),
    )
    assert check.statistic == pytest.approx(0.04 / 0.048)
    assert check.freedom == 2


def test_calibrated_ordinal_probabilities_support_the_claim():
    truth, probabilities = levels()

    claim = calibration.ordinal(truth, probabilities)

    assert claim.task is tasks.Task.ORDINAL
    assert [check.subject for check in claim.checks] == [
        "level <= 0",
        "level <= 1",
    ]
    assert all(check.tested for check in claim.checks)
    assert claim.threshold == pytest.approx(calibration.LEVEL / 2)
    assert claim.supported is True
    assert claim.statement == calibration.STATEMENTS[tasks.Task.ORDINAL]


def test_shifted_ordinal_probabilities_are_rejected():
    truth, probabilities = levels(shift=0.6)

    claim = calibration.ordinal(truth, probabilities)

    assert claim.supported is False
    assert min(check.pvalue for check in claim.checks) < claim.threshold


def test_ordinal_rejects_rows_that_do_not_sum_to_one():
    with pytest.raises(errors.InputError, match="sum to one"):
        calibration.ordinal(numpy.array([0, 1]), numpy.full((2, 3), 0.5))


def test_ordinal_rejects_a_level_outside_the_columns():
    with pytest.raises(errors.InputError, match="level"):
        calibration.ordinal(numpy.array([0, 3]), numpy.full((2, 3), 1 / 3))


def test_multilabel_checks_each_label_and_spots_the_biased_one():
    rng = numpy.random.default_rng(3)
    probabilities = rng.uniform(0.05, 0.95, size=(5000, 2))
    truth = (rng.random((5000, 2)) < probabilities).astype(int)
    stated = probabilities.copy()
    stated[:, 1] = numpy.clip(stated[:, 1] + 0.15, 0, 1)

    fair = calibration.multilabel(truth, probabilities, ("red", "blue"))
    biased = calibration.multilabel(truth, stated, ("red", "blue"))

    assert [check.subject for check in fair.checks] == [
        "label red",
        "label blue",
    ]
    assert fair.supported is True
    assert biased.supported is False
    assert biased.checks[0].pvalue >= biased.threshold
    assert biased.checks[1].pvalue < biased.threshold


def test_probabilities_outside_the_unit_interval_are_refused():
    with pytest.raises(errors.InputError, match="between 0 and 1"):
        calibration.multilabel(numpy.array([[0, 1]]), numpy.array([[1.2, 0.3]]))


def test_multilabel_needs_one_name_per_column():
    with pytest.raises(errors.InputError, match="names"):
        calibration.multilabel(
            numpy.zeros((3, 2)), numpy.full((3, 2), 0.5), ("only",)
        )


def cohort(rows=3000, seed=0):
    """Exponential event times with rate `exp(x)` and uniform censoring."""
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=rows)
    rate = numpy.exp(x)
    event_time = rng.exponential(1 / rate)
    censor_time = rng.uniform(0, 3, rows)
    duration = numpy.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)
    return x, rate, duration, event


def test_the_kaplan_meier_point_and_greenwood_variance_match_by_hand():
    duration = numpy.array([1.0, 2.0, 2.0, 3.0, 4.0])
    event = numpy.array([1, 1, 0, 1, 0])

    estimate, variance = calibration.meier(duration, event, 3.0)
    moments, curve = estimators.kaplan_meier(duration, event)

    # At-risk 5, 4, 2 with one event each: S = 4/5 * 3/4 * 1/2.
    assert estimate == pytest.approx(0.3)
    assert estimate == pytest.approx(curve[moments == 3.0][0])
    greenwood = 0.3**2 * (1 / (5 * 4) + 1 / (4 * 3) + 1 / (2 * 1))
    assert variance == pytest.approx(greenwood)
    assert calibration.meier(duration, event, 0.5) == (1.0, 0.0)


def test_true_survival_curves_support_the_claim_at_each_horizon():
    rate, duration, event = cohort()[1:]
    times = numpy.array([0.3, 0.7, 1.2])
    curve = numpy.exp(-numpy.outer(rate, times))

    claim = calibration.survival(duration, event, curve, times)

    assert [check.subject for check in claim.checks] == [
        "horizon 0.3",
        "horizon 0.7",
        "horizon 1.2",
    ]
    assert claim.supported is True


def test_survival_curves_that_halve_the_hazard_are_rejected():
    rate, duration, event = cohort()[1:]
    times = numpy.array([0.3, 0.7, 1.2])
    curve = numpy.exp(-numpy.outer(rate / 2, times))

    claim = calibration.survival(duration, event, curve, times)

    assert claim.supported is False


def test_survival_defaults_to_the_event_time_quartiles():
    duration, event = cohort()[2:]
    times = calibration.quartiles(duration, event)
    expected = numpy.quantile(duration[event == 1], [0.25, 0.5, 0.75])

    assert numpy.allclose(times, expected)


def test_a_fitted_cox_model_is_calibrated_on_fresh_rows():
    x = cohort(rows=6000, seed=1)[0]
    duration, event = cohort(rows=6000, seed=1)[2:]
    model = estimators.Hazards(penalty=1e-3).fit(
        x[:3000, None], duration[:3000], event[:3000]
    )
    times = calibration.quartiles(duration[3000:], event[3000:])
    curve = model.predict_survival(x[3000:, None], times)

    claim = calibration.survival(duration[3000:], event[3000:], curve, times)

    assert claim.supported is True


def test_survival_refuses_a_curve_of_the_wrong_width():
    duration, event = cohort(rows=50)[2:]
    with pytest.raises(errors.InputError, match="column per horizon"):
        calibration.survival(
            duration, event, numpy.ones((50, 2)), numpy.array([1.0])
        )


def queries(count=300, size=12, seed=0, link=special.expit):
    """Queries whose relevance depends on the score through `link`."""
    rng = numpy.random.default_rng(seed)
    scores = rng.normal(scale=2.0, size=count * size)
    relevant = (rng.random(count * size) < link(scores)).astype(int)
    labels = numpy.repeat(numpy.arange(count), size)
    return scores, relevant, labels


def test_logistic_relevance_scores_are_calibrated_after_platt_scaling():
    scores, relevant, labels = queries()

    claim = calibration.ranking(relevant, scores, labels)

    assert [check.subject for check in claim.checks] == ["relevance >= 1"]
    assert claim.supported is True


def test_scores_with_a_non_monotone_link_cannot_be_calibrated():
    scores, relevant, labels = queries(
        link=lambda s: numpy.where(numpy.abs(s) > 1.5, 0.9, 0.1)
    )

    claim = calibration.ranking(relevant, scores, labels)

    assert claim.supported is False


def test_platt_maps_scores_monotonically_to_probabilities():
    scores, relevant = queries()[:2]

    platt = calibration.Platt().fit(scores, relevant)
    mapped = platt.predict(numpy.array([-3.0, 0.0, 3.0]))

    assert numpy.all(numpy.diff(mapped) > 0)
    assert mapped[1] == pytest.approx(0.5, abs=0.05)


def test_platt_with_one_class_predicts_that_class_share():
    platt = calibration.Platt().fit(numpy.arange(4.0), numpy.ones(4))

    assert numpy.array_equal(platt.predict(numpy.zeros(2)), numpy.ones(2))


def test_reversed_scores_do_not_support_the_ranking_claim():
    scores, relevant, labels = queries()

    claim = calibration.ranking(relevant, -scores, labels)

    assert claim.checks[0].tested is False
    assert claim.intervals[0].name == "gain"
    assert claim.intervals[0].upper < 0.0
    assert claim.supported is False


def test_informative_scores_gain_ndcg_over_a_random_order():
    scores, relevant, labels = queries()

    claim = calibration.ranking(relevant, scores, labels, draws=400)

    interval = claim.intervals[0]
    assert interval.null == 0.0
    assert 0.0 < interval.lower <= interval.estimate <= interval.upper


def test_a_decreasing_platt_map_is_not_increasing():
    scores, relevant = queries()[:2]

    assert calibration.Platt().fit(scores, relevant).increasing is True
    assert calibration.Platt().fit(-scores, relevant).increasing is False
    assert (
        calibration.Platt().fit(scores, numpy.ones(len(scores))).increasing
        is False
    )


def test_random_order_ndcg_by_hand():
    grades = numpy.array([2.0, 0.0, 1.0])
    discount = 1 / numpy.log2(numpy.arange(3) + 2.0)

    expected = 1.0 * discount.sum() / (2 * discount[0] + 1 * discount[1])

    assert calibration.chance(grades) == pytest.approx(expected)


def test_the_random_order_expectation_matches_shuffled_orders():
    rng = numpy.random.default_rng(4)
    grades = numpy.array([3.0, 0.0, 1.0, 0.0, 2.0, 1.0])
    labels = numpy.zeros(len(grades))
    shuffled = [
        ranking.mean_ndcg(grades, rng.permutation(len(grades)), labels)
        for draw in range(5000)
    ]

    # 5,000 shuffles put the mean within 0.01 at about five standard errors.
    assert numpy.mean(shuffled) == pytest.approx(
        calibration.chance(grades), abs=0.01
    )


def test_a_gain_without_scorable_queries_never_supports_the_claim():
    labels = numpy.repeat(numpy.arange(20), 5)

    claim = calibration.ranking(numpy.zeros(100), numpy.arange(100.0), labels)

    assert numpy.isnan(claim.intervals[0].lower)
    assert claim.supported is False


def test_negative_relevance_grades_are_refused():
    with pytest.raises(errors.InputError, match="negative"):
        calibration.ranking(numpy.array([-1.0, 1.0]), numpy.ones(2), [0, 1])


def test_ranking_with_one_query_cannot_be_tested():
    scores, relevant = queries(count=1, size=200)[:2]

    claim = calibration.ranking(relevant, scores, numpy.zeros(200))

    assert claim.checks[0].tested is False
    assert claim.supported is False


def trial(rows=6000, seed=0):
    """A randomized trial whose effect depends on the first covariate."""
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=rows)
    arm = rng.integers(0, 2, rows)
    effect = 0.3 * (x > 0)
    outcome = (rng.random(rows) < 0.25 + arm * effect).astype(int)
    return outcome, arm, effect


def test_the_true_effect_supports_the_uplift_claim_with_intervals():
    outcome, arm, effect = trial()

    claim = calibration.uplift(outcome, arm, effect, draws=400)

    names = [interval.name for interval in claim.intervals]
    assert names == ["qini", "auuc"]
    qini = claim.intervals[0]
    assert qini.null == 0.0
    assert qini.lower > 0.0
    assert qini.lower <= qini.estimate <= qini.upper
    assert claim.intervals[1].null is None
    assert claim.checks[0].subject == "effect"
    assert claim.supported is True


def test_a_random_ranking_does_not_support_the_uplift_claim():
    outcome, arm = trial()[:2]
    noise = numpy.random.default_rng(9).normal(size=len(arm))

    claim = calibration.uplift(outcome, arm, noise * 0.01, draws=400)

    assert claim.intervals[0].lower <= 0.0
    assert claim.supported is False


def test_an_effect_three_times_too_large_fails_effect_calibration():
    outcome, arm, effect = trial()

    claim = calibration.uplift(outcome, arm, effect * 3 + 0.01, draws=200)

    assert claim.intervals[0].lower > 0.0
    assert claim.checks[0].pvalue < claim.threshold
    assert claim.supported is False


def test_the_area_under_the_uplift_curve_by_hand():
    outcome = numpy.array([1.0, 0.0, 1.0, 0.0])
    arm = numpy.array([1, 0, 0, 1])
    score = numpy.array([4.0, 3.0, 2.0, 1.0])

    # Walked 1: no control yet (0). Walked 2: 1 - 0 = 1, times 2/4.
    # Walked 3: 1 - 0.5 = 0.5, times 3/4. Walked 4: 0.5 - 0.5 = 0.
    assert calibration.area(outcome, arm, score) == pytest.approx(
        (0 + 0.5 + 0.375 + 0) / 4
    )


def test_uplift_needs_both_arms():
    with pytest.raises(errors.InputError, match="both arms"):
        calibration.uplift(numpy.ones(10), numpy.ones(10), numpy.ones(10))


def test_every_task_states_its_claim_and_the_claim_is_json_ready():
    truth, probabilities = levels(rows=200)

    claim = calibration.ordinal(truth, probabilities)
    record = json.loads(json.dumps(dataclasses.asdict(claim)))

    assert set(calibration.STATEMENTS) == {
        tasks.Task.ORDINAL,
        tasks.Task.MULTILABEL,
        tasks.Task.SURVIVAL,
        tasks.Task.RANKING,
        tasks.Task.UPLIFT,
    }
    assert record["task"] == "ordinal"
    assert record["supported"] == claim.supported
    assert record["level"] == calibration.LEVEL


REPLICATES = 1000
"""Seeded calibrated simulations per task in the size test."""

MARGIN = stats.norm.ppf(0.999) * numpy.sqrt(
    calibration.LEVEL * (1 - calibration.LEVEL) / REPLICATES
)
"""Monte Carlo allowance on the measured size: a one-sided 99.9% binomial
(normal) margin, so a test of exact size `LEVEL` fails this check once in a
thousand seeds; with 1,000 replicates it is about 2.1 percentage points."""


def calibrated(task, seed):
    """One held-out claim of a calibrated model of `task`, drawn at `seed`."""
    if task is tasks.Task.ORDINAL:
        return calibration.ordinal(*levels(seed=seed))
    if task is tasks.Task.MULTILABEL:
        rng = numpy.random.default_rng(seed)
        probabilities = rng.uniform(0.05, 0.95, size=(3000, 2))
        truth = (rng.random((3000, 2)) < probabilities).astype(int)
        return calibration.multilabel(truth, probabilities)
    if task is tasks.Task.SURVIVAL:
        rate, duration, event = cohort(seed=seed)[1:]
        times = numpy.array([0.3, 0.7, 1.2])
        curve = numpy.exp(-numpy.outer(rate, times))
        return calibration.survival(duration, event, curve, times)
    if task is tasks.Task.RANKING:
        scores, relevant, labels = queries(seed=seed)
        check = calibration.crossfit(relevant, scores, labels, "relevant")
        return calibration.Claim(task, "", (check,))
    outcome, arm, effect = trial(rows=3000, seed=seed)
    return calibration.uplift(outcome, arm, effect, draws=1)


def size(task, replicates=REPLICATES):
    """Share of calibrated simulations in which some check rejects."""
    rejected = 0
    for seed in range(replicates):
        claim = calibrated(task, seed)
        rejected += any(
            not check.tested or check.pvalue < claim.threshold
            for check in claim.checks
        )
    return rejected / replicates


@pytest.mark.parametrize("task", sorted(calibration.STATEMENTS, key=str))
def test_calibrated_predictions_are_rejected_at_most_at_the_level(task):
    assert size(task) <= calibration.LEVEL + MARGIN
