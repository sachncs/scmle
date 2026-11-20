"""The linear ranking and linear uplift families."""

import numpy
import pytest

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.evaluation import calibration
from scmle.evaluation import ranking
from scmle.models import conformance
import scmle.models.ranking.adapters
import scmle.models.ranking.estimators
import scmle.models.uplift.adapters
import scmle.models.uplift.estimators


def queries(count=200, size=10, seed=0):
    """Queries whose relevance grade rises with the first feature only."""
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=(count * size, 3))
    grade = numpy.clip(numpy.round(x[:, 0] + rng.normal(0, 0.5, len(x))), 0, 2)
    return x, grade, numpy.full(count, size)


def test_the_pairwise_ranker_learns_the_relevant_direction():
    x, grade, sizes = queries()

    model = scmle.models.ranking.estimators.Pairwise(random_state=0).fit(
        x, grade, group=sizes
    )

    weights = model.coefficients_
    assert weights[0] > 0
    assert abs(weights[1]) < weights[0] / 10
    assert abs(weights[2]) < weights[0] / 10


def test_the_pairwise_ranker_beats_a_constant_order_on_fresh_queries():
    x, grade, sizes = queries(seed=0)
    fresh, truth = queries(seed=1)[:2]
    labels = numpy.repeat(numpy.arange(200), 10)

    model = scmle.models.ranking.estimators.Pairwise().fit(
        x, grade, group=sizes
    )
    learnt = ranking.mean_ndcg(truth, model.predict(fresh), labels)
    flat = ranking.mean_ndcg(
        truth,
        scmle.models.ranking.estimators.Constant().predict(fresh),
        labels,
    )

    assert learnt > flat + 0.1


def test_the_pair_cap_bounds_the_pairs_drawn():
    grade, sizes = queries(count=50, size=40)[1:]

    model = scmle.models.ranking.estimators.Pairwise(pairs=500, random_state=3)
    first, second = model.couple(grade, sizes)

    assert len(first) == len(second)
    assert len(first) <= 500 + len(sizes)
    assert numpy.all(grade[first] != grade[second])
    starts = numpy.repeat(numpy.arange(50) * 40, 40)
    assert numpy.array_equal(starts[first], starts[second])


def test_small_queries_use_every_pair_with_different_grades():
    grade = numpy.array([0.0, 1.0, 1.0, 2.0, 0.0])

    first, second = scmle.models.ranking.estimators.Pairwise().couple(
        grade, numpy.array([3, 2])
    )

    assert sorted(zip(first.tolist(), second.tolist(), strict=True)) == [
        (0, 1),
        (0, 2),
        (3, 4),
    ]


def test_queries_without_any_preference_give_zero_weights():
    x = numpy.ones((6, 2))

    model = scmle.models.ranking.estimators.Pairwise().fit(
        x, numpy.ones(6), group=[3, 3]
    )

    assert numpy.array_equal(model.coefficients_, numpy.zeros(2))
    assert numpy.array_equal(model.predict(x), numpy.zeros(6))


def test_group_sizes_must_cover_the_rows():
    with pytest.raises(errors.InputError, match="group sizes"):
        scmle.models.ranking.estimators.Pairwise().fit(
            numpy.ones((4, 1)), numpy.arange(4.0), [3]
        )


def test_the_linear_ranker_adapter_is_registered_and_conforms():
    adapter = scmle.models.ranking.adapters.PAIRWISE

    conformance.comply(adapter)

    assert adapter.name == "pairwise_linear_ranker"
    assert adapter.family == "linear"
    assert adapter.capabilities.needs_scaling is True
    assert adapter.capabilities.handles_missing is False
    assert list(adapter.search_space.names()) == ["penalty"]
    built = adapter.build(tasks.Task.RANKING, 7, {"penalty": 2.0})
    assert (built.penalty, built.random_state) == (2.0, 7)
    assert scmle.models.ranking.adapters.RANKING_ADAPTERS[-1] is adapter


def trial(rows=4000, seed=0):
    """A trial whose effect on a numeric outcome is linear in `x0`."""
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=(rows, 2))
    arm = rng.integers(0, 2, rows)
    effect = 1.0 + 0.8 * x[:, 0]
    outcome = x[:, 1] + arm * effect + rng.normal(0, 0.5, rows)
    return x, arm, outcome, effect


def test_a_linear_t_learner_recovers_a_linear_effect():
    x, arm, outcome, effect = trial()

    model = scmle.models.uplift.estimators.Pair(
        classify=False, kind="linear"
    ).fit(x, outcome, arm)

    assert numpy.abs(model.predict(x) - effect).mean() < 0.05


def test_a_linear_t_learner_is_calibrated_on_fresh_rows():
    x, arm, outcome = trial(seed=0)[:3]
    fresh, fresh_arm, fresh_outcome = trial(seed=1)[:3]

    model = scmle.models.uplift.estimators.Pair(
        classify=False, kind="linear"
    ).fit(x, outcome, arm)
    claim = calibration.uplift(
        fresh_outcome, fresh_arm, model.predict(fresh), draws=200
    )

    assert claim.supported is True


def test_a_linear_learner_with_one_class_in_an_arm_predicts_a_constant():
    x = numpy.random.default_rng(0).normal(size=(40, 2))
    arm = numpy.repeat([0, 1], 20)
    outcome = numpy.where(arm == 1, 1, numpy.arange(40) % 2)

    model = scmle.models.uplift.estimators.Pair(kind="linear").fit(
        x, outcome, arm
    )
    treated = scmle.models.uplift.estimators.expected(model.treated_, x, True)

    assert numpy.array_equal(treated, numpy.ones(40))


def test_an_unknown_outcome_family_is_a_config_error():
    with pytest.raises(errors.ConfigError, match="kind"):
        scmle.models.uplift.estimators.Pair(kind="forest").fit(
            numpy.ones((4, 1)), numpy.array([0, 1, 0, 1]), [0, 0, 1, 1]
        )


def test_every_learner_accepts_the_linear_family():
    x, arm, outcome, effect = trial(rows=2000)

    for learner in (
        scmle.models.uplift.estimators.Pair,
        scmle.models.uplift.estimators.Cross,
    ):
        model = learner(classify=False, kind="linear").fit(x, outcome, arm)
        assert numpy.abs(model.predict(x) - effect).mean() < 0.1


def test_a_linear_s_learner_predicts_the_mean_effect_for_every_row():
    x, arm, outcome, effect = trial(rows=2000)

    model = scmle.models.uplift.estimators.Pool(
        classify=False, kind="linear"
    ).fit(x, outcome, arm)
    predicted = model.predict(x)

    # A linear outcome model shifts every row by one treatment coefficient.
    assert numpy.ptp(predicted) < 1e-9
    assert abs(predicted.mean() - effect.mean()) < 0.1


def test_the_linear_uplift_adapter_is_registered_and_conforms():
    adapter = scmle.models.uplift.adapters.LINEAR

    conformance.comply(adapter)

    assert adapter.name == "uplift_linear_t_learner"
    assert adapter.family == "linear"
    assert adapter.capabilities.handles_missing is False
    assert adapter.capabilities.needs_scaling is True
    built = adapter.build(tasks.Task.UPLIFT, 5, {"classify": False})
    assert (built.kind, built.classify, built.random_state) == (
        "linear",
        False,
        5,
    )
    assert scmle.models.uplift.adapters.UPLIFT_ADAPTERS[-1] is adapter
    assert (
        scmle.models.uplift.adapters.T_LEARNER.build(tasks.Task.UPLIFT, 0).kind
        == "boosting"
    )


class Legacy:
    """A learner class written before outcome kinds existed."""

    def __init__(self, classify=True, random_state=0):
        self.classify = classify
        self.random_state = random_state


def test_a_learner_without_kinds_still_builds_under_the_default_kind():
    adapter = scmle.models.uplift.adapters.Uplift(name="legacy", learner=Legacy)

    built = adapter.build(tasks.Task.UPLIFT, 4)

    assert (type(built), built.random_state) == (Legacy, 4)


def test_a_learner_without_kinds_refuses_a_linear_kind():
    adapter = scmle.models.uplift.adapters.Uplift(
        name="legacy", learner=Legacy, kind="linear"
    )

    with pytest.raises(errors.ConfigError, match="takes no outcome kind"):
        adapter.build(tasks.Task.UPLIFT, 0)
