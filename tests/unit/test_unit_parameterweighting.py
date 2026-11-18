"""Influence variance, Hajek weighting and outcome models for causal ATEs."""

import numpy
import pytest

from scmle.contracts import errors
from scmle.evaluation.parameter_estimation import weighting


def test_a_centering_subtracts_the_mean():
    assert weighting.center(numpy.array([1.0, 2.0, 6.0])).tolist() == [
        -2.0,
        -1.0,
        3.0,
    ]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (numpy.zeros((2, 2)), r"influence scores must have shape \(rows,\)"),
        (numpy.array([1.0]), "needs at least two rows"),
        (numpy.array([1.0, numpy.nan]), "only finite values"),
    ],
)
def test_a_centering_rejects_bad_scores(values, message):
    with pytest.raises(errors.InputError, match=message):
        weighting.center(values)


def test_an_iid_dispersion_is_the_variance_of_the_mean():
    variance, about = weighting.dispersion(
        numpy.array([1.0, 2.0, 6.0]), "iid", None
    )
    assert variance == pytest.approx(14.0 / 2 / 3)
    assert about == {
        "method": "iid_centered_influence_function",
        "centered": True,
    }


def test_a_clustered_dispersion_sums_within_clusters():
    influence = numpy.array([1.0, 2.0, 6.0, 3.0])
    clusters = numpy.array(["a", "a", "b", "b"], dtype=object)
    variance, about = weighting.dispersion(influence, "clustered", clusters)
    centered = influence - 3.0
    sums = numpy.array([centered[:2].sum(), centered[2:].sum()])
    expected = (2 / 1) * (3 / 4) * (sums**2).sum() / 16
    assert variance == pytest.approx(expected)
    assert about["method"] == "clustered_centered_influence_function"
    assert about["cluster_diagnostics"] == {
        "cluster_count": 2,
        "cluster_sizes": [2, 2],
        "uncertainty_method": "clustered_centered_influence_function",
    }


def test_a_dispersion_rejects_hc3():
    with pytest.raises(
        errors.InputError, match="supports only iid or clustered"
    ):
        weighting.dispersion(numpy.array([1.0, 2.0]), "hc3", None)


def test_a_tally_needs_two_hashable_labelled_clusters():
    centered = numpy.array([1.0, -1.0])
    with pytest.raises(errors.InputError, match="needs cluster labels"):
        weighting.tally(centered, None)
    with pytest.raises(errors.InputError, match="at least two clusters"):
        weighting.tally(centered, numpy.array([1, 1]))
    with pytest.raises(errors.InputError, match="same row count"):
        weighting.tally(centered, numpy.array([1, 2, 3]))
    variance, count, sizes = weighting.tally(
        numpy.array([1.0, -1.0, 2.0]), numpy.array([9, 9, 4])
    )
    assert (count, sizes) == (2, [2, 1])
    assert variance == pytest.approx(2 * (2 / 3) * (0 + 4) / 9)


def test_a_hajek_estimate_is_the_normalised_contrast():
    t = numpy.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    y = numpy.array([3.0, 5.0, 1.0, 2.0, 4.0, 0.0])
    p = numpy.array([0.5, 0.25, 0.5, 0.75, 0.5, 0.2])
    ate, variance, about = weighting.hajek(t, y, p)
    wt = t / p
    wc = (1 - t) / (1 - p)
    mt = (wt * y).sum() / wt.sum()
    mc = (wc * y).sum() / wc.sum()
    assert ate == pytest.approx(mt - mc)
    influence = wt * (y - mt) - wc * (y - mc)
    centered = influence - influence.mean()
    assert variance == pytest.approx((centered**2).sum() / 5 / 6)
    assert about["treatment_means"] == {
        "treated": pytest.approx(mt),
        "control": pytest.approx(mc),
    }
    assert about["effective_sample_size"]["treated"] == pytest.approx(
        wt.sum() ** 2 / (wt**2).sum()
    )
    assert about["uncertainty_method"] == "iid_centered_influence_function"


def test_a_hajek_estimate_needs_both_arms():
    with pytest.raises(errors.InputError, match="must have positive support"):
        weighting.hajek(numpy.ones(3), numpy.ones(3), numpy.full(3, 0.5))
    with pytest.raises(errors.InputError, match="must have positive support"):
        weighting.hajek(numpy.zeros(3), numpy.ones(3), numpy.full(3, 0.5))


def test_outcome_models_are_fitted_per_arm():
    design = numpy.column_stack([numpy.ones(8), numpy.arange(8.0)])
    treated = numpy.array([True] * 4 + [False] * 4)
    y = numpy.where(
        treated, 1.0 + 2.0 * numpy.arange(8.0), 3.0 - numpy.arange(8.0)
    )
    control, effect = weighting.outcomes(
        design, y, (treated, ~treated), numpy.array([[1.0, 10.0]]), "rank"
    )
    assert control.tolist() == pytest.approx([3.0 - 10.0])
    assert effect.tolist() == pytest.approx([21.0])


def test_outcome_models_reject_a_rank_deficient_arm():
    design = numpy.column_stack([numpy.ones(4), [1.0, 1.0, 2.0, 3.0]])
    treated = numpy.array([True, True, False, False])
    with pytest.raises(errors.InputError, match=r"^custom message$"):
        weighting.outcomes(
            design,
            numpy.zeros(4),
            (treated, ~treated),
            design,
            "custom message",
        )


def test_an_aipw_score_adds_weighted_residuals():
    score = weighting.aipw(
        numpy.array([5.0, 5.0]),
        numpy.array([2.0, 2.0]),
        numpy.array([1.0, 0.0]),
        numpy.array([6.0, 1.0]),
        numpy.array([0.5, 0.75]),
    )
    assert score.tolist() == pytest.approx(
        [3.0 + 1.0 / 0.5, 3.0 - (-1.0) / 0.25]
    )
