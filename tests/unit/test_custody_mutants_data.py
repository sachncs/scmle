"""Tests added to kill surviving mutants in the manifest, carve and pipeline.

Messages are compared whole: a mutant that wraps a message in extra
characters still contains the original as a substring.
"""

from functools import partial

import numpy as np
import pandas as pd
import pytest

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.custody import carve
from scmle.data.custody import manifest as m
from scmle.data.custody import pipeline
from scmle.data.custody import sealed
from scmle.data.splitting import splitters
from tests.unit.test_charter_contracts import make_charter
from tests.unit.test_custody_manifest import make
from tests.unit.test_custody_pipeline import BINARY
from tests.unit.test_custody_pipeline import DATA_HASH
from tests.unit.test_custody_pipeline import make_frame
from tests.unit.test_custody_pipeline import make_spec
from tests.unit.test_custody_pipeline import run


def message(call):
    with pytest.raises((ValueError, errors.ArtifactError)) as caught:
        call()
    return str(caught.value)


@pytest.mark.parametrize("field", ["strategy", "charter_hash", "data_hash"])
def test_an_empty_text_field_is_named_in_the_error(field):
    assert message(lambda: make(**{field: " "})) == f"{field} must not be empty"


@pytest.mark.parametrize("field", ["seed", "embargo_rows"])
def test_a_bad_count_is_named_in_the_error(field):
    assert message(lambda: make(**{field: -1})) == (
        f"{field} must be non-negative"
    )
    assert message(lambda: make(**{field: True})) == (
        f"{field} must be an integer"
    )


@pytest.mark.parametrize(
    "field",
    [
        "exploration_ids",
        "confirmation_ids",
        "embargo_ids",
        "final_test_ids",
        "folds",
        "isolation",
        "notes",
        "row_roots",
    ],
)
def test_a_list_instead_of_a_tuple_is_named_in_the_error(field):
    assert message(lambda: make(**{field: []})) == f"{field} must be a tuple"


@pytest.mark.parametrize(
    "field",
    ["exploration_ids", "confirmation_ids", "embargo_ids", "final_test_ids"],
)
def test_non_integer_ids_are_refused_by_name(field):
    for bad in ((1.5,), (True,), ("a",)):
        assert message(partial(make, **{field: bad})) == (
            f"{field} must hold integers"
        )


@pytest.mark.parametrize(
    "field,other",
    [
        ("exploration_ids", (9, 9)),
        ("confirmation_ids", (9, 9)),
        ("embargo_ids", (9, 9)),
        ("final_test_ids", (9, 9)),
    ],
)
def test_a_repeated_row_is_named_by_its_set(field, other):
    assert message(lambda: make(**{field: other})) == f"{field} repeats a row"


def test_an_overlap_is_named_by_the_later_set():
    assert message(lambda: make(confirmation_ids=(3, 4))) == (
        "confirmation_ids overlap another set"
    )
    assert message(lambda: make(embargo_ids=(0,))) == (
        "embargo_ids overlap another set"
    )
    assert message(lambda: make(final_test_ids=(1,))) == (
        "final_test_ids overlap another set"
    )


def test_scope_error_lists_the_allowed_scopes():
    assert message(lambda: make(scope="x")) == (
        f"scope must be one of {m.SCOPES}"
    )


def test_verify_rows_messages_are_exact():
    frame = pd.DataFrame({"a": [float(i) for i in range(8)]})
    assert message(lambda: make().verify_rows(frame)) == (
        "manifest holds no row content roots"
    )


def test_hash_and_version_errors_are_exact_and_a_missing_hash_is_an_error():
    record = make().to_dict()
    record["strategy"] = "other"
    assert message(lambda: m.CustodyManifest.from_dict(record)) == (
        "split manifest hash mismatch: the manifest was edited"
    )
    del record["sha256"]
    assert message(lambda: m.CustodyManifest.from_dict(record)) == (
        "split manifest hash mismatch: the manifest was edited"
    )
    assert message(lambda: m.CustodyManifest.from_dict({})) == (
        "unsupported split manifest version"
    )


def test_a_record_without_a_version_or_ids_is_unsupported_not_a_key_error():
    assert message(lambda: m.load_manifest({})) == (
        "unsupported split manifest version: None"
    )
    assert message(lambda: m.load_manifest({"manifest_version": 9})) == (
        "unsupported split manifest version: 9"
    )


def test_a_v1_record_without_notes_loads_and_a_bad_fingerprint_is_exact():
    v1 = splitters.SplitManifest("random", 3, (0, 1), (2,), (((0,), (1,)),))
    record = v1.to_dict()
    record.pop("notes", None)
    assert m.load_manifest(record).notes == ()
    record["fingerprint"] = "0" * 16
    assert message(lambda: m.load_manifest(record)) == (
        "split manifest fingerprint mismatch: the manifest was edited"
    )


def test_a_manifest_without_roots_loads_with_empty_roots():
    record = make().to_dict()
    assert "row_roots" not in record
    assert m.CustodyManifest.from_dict(record).row_roots == ()


def test_pairs_of_a_root_are_stored_as_tuples_after_a_round_trip():
    manifest = make(row_roots=(("exploration", "a" * 64),))
    loaded = m.CustodyManifest.from_dict(manifest.to_dict())
    assert loaded.row_roots == (("exploration", "a" * 64),)
    assert loaded == manifest


# carve ---------------------------------------------------------------------


def test_pick_units_takes_the_target_and_never_every_unit():
    rng = np.random.default_rng(0)
    units = np.arange(10)
    strata = np.zeros(10, dtype=str)
    assert carve.pick_units(units, strata, 0.3, rng).sum() == 3
    assert carve.pick_units(units, strata, 0.5, rng).sum() == 5
    assert carve.pick_units(units, strata, 1.0, rng).sum() == 9


def test_pick_blocks_sizes_and_cap_are_exact():
    times = np.arange(40)
    rng = np.random.default_rng(1)
    mask, order = carve.pick_blocks(times, 0.5, 0, rng)
    assert mask.sum() == 20 and list(order) == list(range(40))
    mask, _ = carve.pick_blocks(times, 0.425, 0, np.random.default_rng(2))
    assert mask.sum() == 18
    mask, _ = carve.pick_blocks(times, 1.0, 0, np.random.default_rng(3))
    assert mask.sum() == 38
    mask, _ = carve.pick_blocks(
        np.arange(100), 0.3, 3, np.random.default_rng(4)
    )
    assert mask.sum() == 36


def test_pick_blocks_has_a_block_of_at_least_one_row():
    mask, order = carve.pick_blocks(
        np.array([]), 0.3, 0, np.random.default_rng(0)
    )
    assert len(mask) == 0 and len(order) == 0
    mask, _ = carve.pick_blocks(np.arange(10), 0.3, 0, np.random.default_rng(0))
    assert mask.sum() == 3


def test_pick_blocks_orders_ties_by_position():
    times = np.array([5, 5, 5, 5, 1, 1, 1, 1] * 4)
    _, order = carve.pick_blocks(times, 0.3, 0, np.random.default_rng(0))
    assert list(order) == [int(i) for i in np.argsort(times, kind="stable")]


def test_a_unit_split_between_c_and_the_rest_leaves_c_entirely():
    in_c = np.array([True, False, True, True])
    units = np.array([0, 0, 2, 2])
    assert carve.close_over_units(in_c, units).tolist() == [
        False,
        False,
        True,
        True,
    ]


def test_embargo_mask_window_is_exactly_rows_on_each_side():
    in_c = np.zeros(11, dtype=bool)
    in_c[5] = True
    order = np.arange(11)
    one = carve.embargo_mask(in_c, order, 1)
    two = carve.embargo_mask(in_c, order, 2)
    assert np.flatnonzero(one).tolist() == [4, 6]
    assert np.flatnonzero(two).tolist() == [3, 4, 6, 7]
    assert not carve.embargo_mask(in_c, order, 0).any()


def test_acf_embargo_boundaries():
    alternating = np.array([1.0, -1.0] * 4)
    assert carve.acf_embargo(alternating) == 2
    assert carve.acf_embargo(np.array([1.0, -1.0] * 3 + [1.0])) == 0
    flip = np.array([1.0, -1.0] * 20)
    assert carve.acf_embargo(flip, max_lag=1000) == 20
    assert carve.acf_embargo(np.array([1.0, -1.0] * 2200), max_lag=5000) == 1000


def test_carve_defaults_and_input_messages():
    ids = np.arange(40)
    folds = [(np.arange(20), np.arange(20, 40))]
    result = carve.carve_confirmation(
        ids,
        folds,
        fraction=0.3,
        seed=1,
        charter_hash="c",
        data_hash="d",
        times=np.arange(40),
    )
    assert len(result.embargo_ids) == 0
    for name in ("labels", "groups", "times", "duplicates"):
        with pytest.raises(ValueError) as caught:
            carve.carve_confirmation(
                ids,
                folds,
                fraction=0.3,
                seed=1,
                charter_hash="c",
                data_hash="d",
                **{name: np.arange(3)},
            )
        assert str(caught.value) == (
            f"{name} must have one entry per development row"
        )


def test_string_labels_stratify_and_two_units_can_still_be_split():
    ids = np.arange(40)
    folds = [(np.arange(1, 40, 2), np.arange(0, 40, 2))]
    labels = np.array(["a", "b"] * 20)
    result = carve.carve_confirmation(
        ids,
        folds,
        fraction=0.3,
        seed=1,
        charter_hash="c",
        data_hash="d",
        labels=labels,
    )
    assert len(result.confirmation_ids) > 0
    groups = np.repeat([0, 1], 20)
    two = carve.carve_confirmation(
        ids,
        folds,
        fraction=0.5,
        seed=1,
        charter_hash="c",
        data_hash="d",
        groups=groups,
    )
    assert len(two.confirmation_ids) == 20 and len(two.exploration_ids) == 20


def test_an_empty_draw_is_reported_with_its_exact_note():
    ids = np.arange(40)
    folds = [(np.arange(20), np.arange(20, 40))]
    result = carve.carve_confirmation(
        ids,
        folds,
        fraction=0.01,
        seed=1,
        charter_hash="c",
        data_hash="d",
    )
    assert len(result.confirmation_ids) == 0
    assert result.notes == (
        "no confirmation set carved: too few units to separate",
    )


def test_a_validation_fold_left_empty_is_refused_in_full():
    with pytest.raises(errors.ConfigError) as caught:
        carve.restrict_folds(
            [(np.array([0, 1]), np.array([2, 3]))],
            np.arange(4),
            np.array([0, 1]),
        )
    assert str(caught.value) == (
        "carving the confirmation set left a validation fold empty"
    )


# pipeline ------------------------------------------------------------------


def test_minority_rows_counts_the_smaller_class_share():
    labels = np.array([0] * 8 + [1] * 2)
    assert pipeline.minority_rows(labels, 0.5) == 1
    assert pipeline.minority_rows(labels, 0.25) == 0
    assert pipeline.minority_rows(None, 0.5) is None
    assert pipeline.minority_rows(np.array([0] * 6 + [1] * 4), 0.5) == 2


def test_an_infeasible_set_reports_the_reason_in_the_manifest_notes():
    frame = make_frame(rows=120)
    ran = run(frame)
    assert ran.outcome.manifest.notes == (
        "24 confirmation rows is below the 30 needed for a normal approximation",
    )
    ran.vault.close()


def test_the_view_and_split_carry_what_the_manifest_says():
    frame = make_frame()
    ran = run(frame)
    outcome, original = ran.outcome, ran.original
    manifest, view, split = outcome.manifest, outcome.view, outcome.split
    assert split.manifest.strategy == original.manifest.strategy
    assert split.manifest.seed == original.manifest.seed
    assert split.manifest.test_ids == original.manifest.test_ids
    assert split.manifest.notes == manifest.notes
    assert split.locked_test is original.locked_test
    assert isinstance(split.dev_features.index, pd.RangeIndex)
    assert isinstance(split.dev_target.index, pd.RangeIndex)
    assert list(split.dev_features.columns) == ["x", "z"]
    assert isinstance(split.dev_target, pd.Series)
    assert view.target.equals(split.dev_target)
    assert view.folds == manifest.folds
    assert view.manifest_sha256 == manifest.sha256
    ran.vault.close()


def test_sealed_rows_have_a_fresh_index_and_their_own_labels():
    frame = make_frame()
    ran = run(frame)
    held = ran.vault.sets["C"]
    features, target = held.open(held.issue_token(), actor="test")
    assert isinstance(features.index, pd.RangeIndex)
    assert isinstance(target.index, pd.RangeIndex)
    assert list(features.columns) == ["x", "z"]
    ids = list(held.ids)
    assert np.allclose(target.to_numpy(), frame["y"].iloc[ids].to_numpy())
    ran.vault.close()


def test_a_time_column_does_not_make_a_stratified_split_temporal():
    frame = make_frame()
    frame["t"] = np.arange(len(frame))
    spec = make_spec(time_column="t")
    ran = run(frame, spec=spec)
    assert not any("time blocks" in note for note in ran.outcome.manifest.notes)
    ran.vault.close()


def test_copies_of_a_record_land_on_one_side():
    frame = make_frame(rows=1000)
    frame.loc[500:549, ["x", "z", "y"]] = frame.loc[
        0:49, ["x", "z", "y"]
    ].to_numpy()
    ran = run(frame)
    manifest = ran.outcome.manifest
    c = set(manifest.confirmation_ids)
    x = set(manifest.exploration_ids)
    for i in range(50):
        pair = {i, 500 + i}
        assert pair <= c | set(manifest.final_test_ids) | x
        assert not (pair & c and pair & x)
    ran.vault.close()


def test_a_missing_class_in_c_means_no_confirmation_set():
    frame = make_frame(rows=1600)
    frame["g"] = np.arange(len(frame)) // 100
    frame["y"] = 0
    frame.loc[frame["g"] == 3, "y"] = 1
    spec = make_spec(group_column="g")
    split = splitters.SplitManager(spec, BINARY).split(frame)
    vault = sealed.CustodyVault(sealed.OpenLog(), process=False)
    outcome = pipeline.apply_custody(
        charter=make_charter(min_effect=0.3, k_max=5),
        split=split,
        frame=frame,
        spec=spec,
        task=BINARY,
        data_hash=DATA_HASH,
        vault=vault,
    )
    if not outcome.manifest.confirmation_ids:
        assert any("not carved" in n for n in outcome.manifest.notes)
    else:
        labels = frame["y"].to_numpy()
        in_c = set(labels[list(outcome.manifest.confirmation_ids)])
        in_x = set(labels[list(outcome.manifest.exploration_ids)])
        assert in_x <= in_c


def test_a_classification_target_is_measured_through_its_codes():
    rng = np.random.default_rng(0)
    runs = np.repeat(rng.integers(0, 2, size=60), 25)
    frame = pd.DataFrame(
        {"t": np.arange(len(runs)), "x": rng.normal(size=len(runs))}
    )
    frame["y"] = np.where(runs == 1, "yes", "no")
    dev = np.arange(len(frame))
    data = tasks.DataSpec(uri="m", target="y", time_column="t")
    assert (
        pipeline.measured_embargo(
            frame, dev, data, frame["t"].to_numpy(), BINARY
        )
        > 5
    )
    assert pipeline.measured_embargo(frame, dev, data, None, BINARY) == 0


def test_a_text_target_in_a_regression_gives_no_measured_embargo():
    frame = pd.DataFrame(
        {"t": np.arange(40), "y": ["a", "b"] * 20, "x": np.arange(40.0)}
    )
    data = tasks.DataSpec(uri="m", target="y", time_column="t")
    got = pipeline.measured_embargo(
        frame,
        np.arange(40),
        data,
        frame["t"].to_numpy(),
        tasks.TaskType.REGRESSION,
    )
    assert got == 0
