"""The defect injector plants known defects and reports exactly where."""

import numpy as np
import pandas as pd
import pytest

from benchmarks.qualification import inject


def table(rows: int = 400, classes: int = 2, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x1": rng.normal(size=rows),
            "x2": rng.lognormal(size=rows),
            "n1": rng.integers(0, 50, size=rows),
            "s": rng.choice(["a", "b", "c"], size=rows),
            "y": rng.integers(0, classes, size=rows),
        }
    )


def assert_exact(original: pd.DataFrame, planted: inject.Injection) -> None:
    """The mask equals an independent before/after diff, cell by cell."""
    diff = inject.changed_cells(original, planted.frame)
    assert list(planted.mask.columns) == list(planted.frame.columns)
    assert planted.mask.index.equals(planted.frame.index)
    mismatched = int((planted.mask.to_numpy() != diff.to_numpy()).sum())
    assert mismatched == 0


def same(first: inject.Injection, second: inject.Injection) -> bool:
    return first.frame.equals(second.frame) and first.mask.equals(second.mask)


CASES = {
    "flip_symmetric": lambda f, s: inject.flip_labels(f, "y", 0.1, s),
    "flip_conditional": lambda f, s: inject.flip_labels(
        f, "y", 0.0, s, mode="class_conditional", class_rates={0: 0.2, 1: 0.05}
    ),
    "mcar": lambda f, s: inject.inject_missing(f, ["x1", "s"], 0.15, s),
    "mar": lambda f, s: inject.inject_missing(
        f, ["x2"], 0.2, s, mechanism="mar", driver="x1"
    ),
    "mnar": lambda f, s: inject.inject_missing(
        f, ["x1", "n1"], 0.2, s, mechanism="mnar"
    ),
    "sentinel": lambda f, s: inject.inject_sentinels(
        f, ["x1", "n1", "s"], 0.1, s
    ),
    "duplicates": lambda f, s: inject.inject_duplicates(f, 0.1, s),
    "near_duplicates": lambda f, s: inject.inject_duplicates(
        f, 0.1, s, near=True
    ),
    "outliers": lambda f, s: inject.inject_outliers(f, ["x1", "n1"], 0.05, s),
    "leak_copy": lambda f, s: inject.inject_leak(f, "y", "copy", s),
    "leak_noisy": lambda f, s: inject.inject_leak(f, "y", "noisy_copy", s),
    "leak_missing": lambda f, s: inject.inject_leak(
        f, "y", "missing_for_class", s
    ),
    "shift": lambda f, s: inject.shift_covariates(f, ["x1", "x2"], 0.3, 1.5, s),
}

# A verbatim copy of the target draws nothing, so the seed cannot matter.
NO_RANDOMNESS = {"leak_copy"}


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_mask_matches_the_actual_mutation_cell_by_cell(name):
    original = table()
    planted = CASES[name](original, 5)
    assert planted.mask.to_numpy().any()
    assert_exact(original, planted)


@pytest.mark.parametrize("name", sorted(CASES))
def test_equal_seeds_give_equal_output_and_other_seeds_do_not(name):
    original = table()
    assert same(CASES[name](original, 3), CASES[name](original, 3))
    if name in NO_RANDOMNESS:
        assert same(CASES[name](original, 3), CASES[name](original, 4))
    else:
        assert not same(CASES[name](original, 3), CASES[name](original, 4))


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_input_frame_is_never_modified(name):
    original = table()
    snapshot = original.copy(deep=True)
    CASES[name](original, 9)
    pd.testing.assert_frame_equal(original, snapshot)


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_record_names_its_defect_and_random_stream(name):
    planted = CASES[name](table(), 21)
    assert planted.defect_type
    assert planted.seed_path[0] == 21
    assert planted.row_mask.dtype == bool
    assert len(planted.row_mask) == len(planted.frame)


def test_symmetric_flips_change_exactly_the_requested_share():
    original = table(classes=3)
    planted = inject.flip_labels(original, "y", 0.1, seed=1)
    flipped = planted.mask["y"]
    assert int(flipped.sum()) == round(0.1 * len(original))
    assert (planted.frame.loc[flipped, "y"] != original.loc[flipped, "y"]).all()
    assert (
        planted.frame.loc[~flipped, "y"] == original.loc[~flipped, "y"]
    ).all()
    assert set(planted.frame["y"]) <= {0, 1, 2}
    assert planted.mask.drop(columns="y").to_numpy().sum() == 0


def test_class_conditional_flips_follow_each_class_rate():
    original = table(rows=1000)
    rates = {0: 0.3, 1: 0.05}
    planted = inject.flip_labels(
        original, "y", 0.0, seed=2, mode="class_conditional", class_rates=rates
    )
    for label, rate in rates.items():
        members = original["y"] == label
        flipped = int(planted.mask.loc[members, "y"].sum())
        assert flipped == round(rate * int(members.sum()))


def test_label_flips_keep_a_categorical_dtype():
    original = table()
    original["y"] = original["y"].astype("category")
    planted = inject.flip_labels(original, "y", 0.1, seed=1)
    assert isinstance(planted.frame["y"].dtype, pd.CategoricalDtype)
    assert_exact(original, planted)


def test_mcar_missingness_ignores_the_values():
    original = table(rows=4000)
    planted = inject.inject_missing(original, ["x1"], 0.2, seed=4)
    gone = planted.mask["x1"]
    assert int(gone.sum()) == round(0.2 * len(original))
    assert planted.frame.loc[gone, "x1"].isna().all()
    assert planted.info["mechanism"] == "mcar"
    gap = original.loc[gone, "x1"].mean() - original.loc[~gone, "x1"].mean()
    assert abs(gap) < 0.15


def test_mar_depends_on_the_driver_and_mnar_on_the_value_itself():
    original = table(rows=4000)
    mar = inject.inject_missing(
        original,
        ["x2"],
        0.2,
        seed=4,
        mechanism="mar",
        driver="x1",
        strength=2.0,
    )
    gone = mar.mask["x2"]
    assert (
        original.loc[gone, "x1"].mean() > original.loc[~gone, "x1"].mean() + 0.5
    )
    assert mar.mask.drop(columns="x2").to_numpy().sum() == 0
    mnar = inject.inject_missing(
        original, ["x1"], 0.2, seed=4, mechanism="mnar", strength=2.0
    )
    lost = mnar.mask["x1"]
    assert (
        original.loc[lost, "x1"].mean() > original.loc[~lost, "x1"].mean() + 0.5
    )
    assert abs(float(lost.mean()) - 0.2) < 0.03


def test_missingness_only_hits_observed_cells():
    original = table()
    original.loc[:50, "x1"] = np.nan
    planted = inject.inject_missing(original, ["x1"], 0.2, seed=6)
    assert not planted.mask.loc[:50, "x1"].any()
    assert_exact(original, planted)


def test_sentinels_replace_values_with_the_code_and_skip_existing_codes():
    original = table()
    original.loc[:10, "n1"] = -999
    planted = inject.inject_sentinels(original, ["n1"], 0.1, seed=3)
    hit = planted.mask["n1"]
    assert (planted.frame.loc[hit, "n1"] == -999).all()
    assert not planted.mask.loc[:10, "n1"].any()
    text = inject.inject_sentinels(original, ["s"], 0.1, seed=3)
    assert (text.frame.loc[text.mask["s"], "s"] == "N/A").all()


def test_exact_duplicates_copy_source_rows_and_flag_only_the_copies():
    original = table()
    planted = inject.inject_duplicates(original, 0.1, seed=7)
    added = planted.row_mask
    assert int(added.sum()) == round(0.1 * len(original))
    assert len(planted.frame) == len(original) + int(added.sum())
    assert not added.iloc[: len(original)].any()
    sources = planted.info["source_rows"]
    copies = planted.frame.loc[added].reset_index(drop=True)
    originals = original.loc[list(sources)].reset_index(drop=True)
    pd.testing.assert_frame_equal(copies, originals)
    assert planted.frame.iloc[: len(original)].equals(original)


def test_near_duplicates_jitter_floats_only_and_stay_close():
    original = table()
    planted = inject.inject_duplicates(original, 0.1, seed=7, near=True)
    added = planted.row_mask
    copies = planted.frame.loc[added].reset_index(drop=True)
    sources = original.loc[list(planted.info["source_rows"])].reset_index(
        drop=True
    )
    assert (copies["x1"] != sources["x1"]).all()
    assert (copies["n1"] == sources["n1"]).all()
    assert (copies["s"] == sources["s"]).all()
    assert ((copies["x1"] - sources["x1"]).abs() < 0.05).all()
    assert not planted.frame.duplicated().any()


def test_duplicates_need_a_default_index_and_floats_for_near_copies():
    shifted = table().set_index(pd.Index(range(5, 405)))
    with pytest.raises(ValueError, match="RangeIndex"):
        inject.inject_duplicates(shifted, 0.1, seed=1)
    ints = table()[["n1", "y"]]
    with pytest.raises(ValueError, match="float"):
        inject.inject_duplicates(ints, 0.1, seed=1, near=True)


def test_outliers_move_cells_far_from_the_robust_centre():
    original = table(rows=2000)
    planted = inject.inject_outliers(
        original, ["x1"], 0.02, seed=8, magnitude=10.0
    )
    hit = planted.mask["x1"]
    assert int(hit.sum()) == round(0.02 * len(original))
    robust = 1.4826 * float(
        (original["x1"] - original["x1"].median()).abs().median()
    )
    moved = (planted.frame.loc[hit, "x1"] - original.loc[hit, "x1"]).abs()
    assert np.allclose(moved, 10.0 * robust)


def test_outliers_reject_a_constant_column():
    flat = table()
    flat["x1"] = 1.0
    with pytest.raises(ValueError, match="spread"):
        inject.inject_outliers(flat, ["x1"], 0.05, seed=1)


def test_leaks_add_one_flagged_column_with_the_documented_relation():
    original = table()
    copy = inject.inject_leak(original, "y", "copy", seed=1, name="leaked")
    assert copy.info["column"] == "leaked"
    assert (copy.frame["leaked"] == original["y"]).all()
    assert copy.mask["leaked"].all()
    assert not copy.mask.drop(columns="leaked").to_numpy().any()

    noisy = inject.inject_leak(original, "y", "noisy_copy", seed=1, noise=0.2)
    agreement = float(
        (noisy.frame[noisy.info["column"]] == original["y"]).mean()
    )
    assert 0.7 < agreement < 0.9

    missing = inject.inject_leak(
        original, "y", "missing_for_class", seed=1, missing_class=1
    )
    column = missing.frame[missing.info["column"]]
    assert column[original["y"] == 1].isna().all()
    assert column[original["y"] == 0].notna().all()


def test_a_numeric_target_gets_gaussian_noise_in_a_noisy_leak():
    original = table()
    original["y"] = original["x1"] * 2 + 1
    noisy = inject.inject_leak(original, "y", "noisy_copy", seed=2, noise=0.1)
    gap = noisy.frame[noisy.info["column"]] - original["y"]
    assert 0.05 * original["y"].std() < gap.std() < 0.2 * original["y"].std()


def test_covariate_shift_moves_a_chosen_share_of_rows_by_delta_sds():
    original = table(rows=2000)
    planted = inject.shift_covariates(original, ["x1"], 0.25, 2.0, seed=5)
    rows = planted.mask["x1"]
    assert int(rows.sum()) == round(0.25 * len(original))
    moved = planted.frame.loc[rows, "x1"] - original.loc[rows, "x1"]
    assert np.allclose(moved, 2.0 * original["x1"].std())
    assert not planted.mask.drop(columns="x1").to_numpy().any()


@pytest.mark.parametrize(
    "call",
    [
        lambda f: inject.flip_labels(f, "y", 0.0, 1),
        lambda f: inject.flip_labels(f, "y", 1.5, 1),
        lambda f: inject.flip_labels(f, "missing", 0.1, 1),
        lambda f: inject.flip_labels(f, "y", 0.1, 1, mode="other"),
        lambda f: inject.flip_labels(f, "y", 0.1, 1, mode="class_conditional"),
        lambda f: inject.flip_labels(f.assign(y=0), "y", 0.1, 1),
        lambda f: inject.flip_labels(f.head(5), "y", 0.01, 1),
        lambda f: inject.inject_missing(f, ["x1"], 0.1, 1, mechanism="mar"),
        lambda f: inject.inject_missing(
            f, ["x1"], 0.1, 1, mechanism="mar", driver="x1"
        ),
        lambda f: inject.inject_missing(f, ["s"], 0.1, 1, mechanism="mnar"),
        lambda f: inject.inject_missing(f, [], 0.1, 1),
        lambda f: inject.inject_missing(f, ["nope"], 0.1, 1),
        lambda f: inject.inject_missing(f, ["x1"], 0.1, 1, mechanism="x"),
        lambda f: inject.inject_sentinels(f.assign(b=True), ["b"], 0.1, 1),
        lambda f: inject.inject_leak(f, "y", "other", 1),
        lambda f: inject.inject_leak(f, "y", "copy", 1, name="x1"),
        lambda f: inject.shift_covariates(f, ["x1"], 0.3, 0.0, 1),
        lambda f: inject.shift_covariates(f, ["s"], 0.3, 1.0, 1),
        lambda f: inject.inject_outliers(f, ["x1"], 0.05, 1, magnitude=0.0),
        lambda f: inject.inject_duplicates(f, 0.0, 1),
    ],
)
def test_invalid_requests_are_refused(call):
    with pytest.raises(ValueError):
        call(table())


def test_the_catalogue_lists_every_injector():
    assert set(inject.CATALOGUE) == {
        "label_flip",
        "missing",
        "sentinel",
        "duplicate",
        "outlier",
        "leak",
        "covariate_shift",
    }
