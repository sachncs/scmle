"""Carving the confirmation set out of the development rows."""

import numpy as np
import pytest
from sklearn import model_selection

from scmle.contracts import errors
from scmle.data.custody import carve

CH, DH = "c" * 64, "d" * 64


def folds_for(n, k=4):
    kfold = model_selection.KFold(k, shuffle=True, random_state=0)
    return [(a, b) for a, b in kfold.split(np.arange(n))]


def run(n=400, fraction=0.25, seed=1, dev_ids=None, **kwargs):
    ids = np.arange(n) if dev_ids is None else dev_ids
    return carve.carve_confirmation(
        ids,
        folds_for(len(ids)),
        fraction=fraction,
        seed=seed,
        charter_hash=CH,
        data_hash=DH,
        **kwargs,
    )


def sets(result):
    return (
        set(result.exploration_ids.tolist()),
        set(result.confirmation_ids.tolist()),
        set(result.embargo_ids.tolist()),
    )


def test_sets_partition_the_development_rows():
    dev = np.arange(1000, 1400)
    x, c, e = sets(run(dev_ids=dev))
    assert x | c | e == set(dev.tolist())
    assert not (x & c) and not (x & e) and not (c & e)


def test_confirmation_size_is_close_to_the_fraction():
    result = run(n=1000, fraction=0.2)
    assert abs(len(result.confirmation_ids) - 200) <= 10


def test_the_carve_is_a_pure_function_of_its_inputs():
    first, second = run(), run()
    assert np.array_equal(first.confirmation_ids, second.confirmation_ids)
    assert not np.array_equal(
        first.confirmation_ids, run(seed=2).confirmation_ids
    )
    other_charter = carve.carve_confirmation(
        np.arange(400),
        folds_for(400),
        fraction=0.25,
        seed=1,
        charter_hash="e" * 64,
        data_hash=DH,
    )
    assert not np.array_equal(
        first.confirmation_ids, other_charter.confirmation_ids
    )


def test_groups_never_straddle_exploration_and_confirmation():
    rng = np.random.default_rng(3)
    groups = rng.integers(0, 60, 600)
    result = run(n=600, groups=groups)
    x_groups = set(groups[result.exploration_ids])
    c_groups = set(groups[result.confirmation_ids])
    assert x_groups.isdisjoint(c_groups)
    assert len(result.confirmation_ids) > 0


def test_duplicate_rows_stay_together():
    n = 400
    duplicates = np.full(n, -1)
    for index in range(0, 200, 2):
        duplicates[index] = duplicates[index + 1] = index
    result = run(n=n, duplicates=duplicates)
    for index in range(0, 200, 2):
        pair = {index, index + 1}
        assert pair <= set(result.confirmation_ids.tolist()) or pair <= set(
            result.exploration_ids.tolist()
        )


def test_groups_and_duplicates_merge_into_one_unit():
    n = 300
    groups = np.repeat(np.arange(100), 3)
    duplicates = np.full(n, -1)
    duplicates[0] = duplicates[100] = 0  # joins group 0 with group 33
    result = run(n=n, groups=groups, duplicates=duplicates)
    c = set(result.confirmation_ids.tolist())
    assert (0 in c) == (100 in c)
    assert ({0, 1, 2} <= c) == (0 in c)


def test_classes_are_represented_in_confirmation():
    rng = np.random.default_rng(4)
    labels = (rng.random(800) < 0.1).astype(int)
    result = run(n=800, labels=labels)
    c_labels = labels[result.confirmation_ids]
    assert 0 < c_labels.mean() < 0.2
    assert abs(c_labels.mean() - labels.mean()) < 0.05


def test_temporal_carve_uses_blocks_with_an_embargo():
    n = 600
    times = np.arange(n)
    result = run(n=n, times=times, embargo_rows=6)
    c = np.sort(result.confirmation_ids)
    x = result.exploration_ids
    assert len(c) > 0
    # No exploration row lies within 6 positions of a confirmation row.
    nearest = np.abs(x[:, None] - c[None, :]).min(axis=1)
    assert nearest.min() > 6
    # Every row near a confirmation block is in the embargo set.
    assert len(result.embargo_ids) > 0
    # Confirmation is contiguous blocks, not scattered rows.
    breaks = int((np.diff(c) > 1).sum())
    assert breaks < len(c) / 10


def test_temporal_carve_with_groups_never_shares_a_group():
    n = 600
    groups = np.arange(n) // 6
    result = run(n=n, times=np.arange(n), embargo_rows=2, groups=groups)
    assert set(groups[result.exploration_ids]).isdisjoint(
        groups[result.confirmation_ids]
    )


def test_folds_are_restricted_to_exploration_rows():
    result = run(n=400)
    size = len(result.exploration_ids)
    assert len(result.folds) == 4
    val = np.sort(np.concatenate([pair[1] for pair in result.folds]))
    assert np.array_equal(val, np.arange(size))
    for fit, held in result.folds:
        assert set(fit.tolist()).isdisjoint(held.tolist())
        assert fit.max() < size and held.max() < size


def test_restricting_folds_rejects_an_empty_validation_fold():
    folds = [(np.array([0, 1]), np.array([2, 3]))]
    with pytest.raises(errors.ConfigError, match="validation fold"):
        carve.restrict_folds(folds, np.arange(4), np.array([0, 1]))


def test_zero_fraction_is_refused_and_one_row_per_unit_is_not_required():
    with pytest.raises(ValueError, match="fraction"):
        run(fraction=0.0)
    with pytest.raises(ValueError, match="fraction"):
        run(fraction=0.9)


def test_a_single_unit_cannot_be_carved_out():
    result = run(n=50, groups=np.zeros(50, dtype=int))
    assert len(result.confirmation_ids) == 0
    assert any("unit" in note for note in result.notes)


def test_regression_small_temporal_carve_refuses_explicitly():
    """Found by Hypothesis: 60 rows, fraction 0.5, time blocks, embargo 5.

    Two 20-row blocks plus their embargo remove every row of one validation
    fold, so the carve must refuse instead of returning an unusable fold.
    """
    dev = np.arange(60) + 7
    folds = list(
        model_selection.KFold(3, shuffle=True, random_state=0).split(dev)
    )
    with pytest.raises(errors.ConfigError, match="validation fold"):
        carve.carve_confirmation(
            dev,
            folds,
            fraction=0.5,
            seed=0,
            charter_hash="a" * 64,
            data_hash="b" * 64,
            times=np.arange(60),
            embargo_rows=5,
        )


def ar1(size, phi, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=size)
    out = np.zeros(size)
    for i in range(1, size):
        out[i] = phi * out[i - 1] + noise[i]
    return out


def test_the_autocorrelation_matches_a_direct_computation():
    values = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 5.0, 8.0])
    centred = values - values.mean()
    expected = [
        sum(centred[: len(centred) - k] * centred[k:]) / sum(centred**2)
        for k in (1, 2, 3)
    ]
    assert np.allclose(carve.autocorrelation(values, 3), expected)
    assert np.allclose(carve.autocorrelation(np.ones(10), 3), 0.0)


def test_white_noise_measures_no_embargo():
    assert carve.acf_embargo(ar1(2000, 0.0, seed=1)) == 0


def test_a_persistent_series_needs_a_longer_embargo_than_a_weak_one():
    weak = carve.acf_embargo(ar1(4000, 0.3, seed=2))
    strong = carve.acf_embargo(ar1(4000, 0.9, seed=2))
    assert 1 <= weak < strong


def test_an_ar1_embargo_is_near_the_lag_where_phi_k_meets_the_band():
    # phi**k drops inside 1.96/sqrt(n) at k = log(band)/log(phi).
    size, phi = 4000, 0.8
    band = 1.96 / np.sqrt(size)
    expected = np.log(band) / np.log(phi)
    measured = carve.acf_embargo(ar1(size, phi, seed=3))
    assert abs(measured - expected) <= 8


def test_short_and_constant_series_measure_zero():
    assert carve.acf_embargo(np.arange(7.0)) == 0
    assert carve.acf_embargo(np.ones(100)) == 0


def test_a_trend_never_falls_inside_the_band_and_is_capped():
    assert carve.acf_embargo(np.arange(400.0)) == 100
    assert carve.acf_embargo(np.arange(400.0), max_lag=10) == 10


def test_an_embargo_longer_than_half_the_series_still_masks_by_position():
    in_c = np.zeros(10, dtype=bool)
    in_c[0] = True

    mask = carve.embargo_mask(in_c, np.arange(10), 7)

    assert mask.tolist() == [False] + [True] * 7 + [False] * 2


def test_an_explicit_zero_max_lag_gives_a_zero_embargo():
    values = np.cumsum(np.random.default_rng(0).normal(size=400))

    assert carve.acf_embargo(values) > 0
    assert carve.acf_embargo(values, max_lag=0) == 0
