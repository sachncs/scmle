"""Tests for split strategies, manifest and the locked test set."""

import numpy as np
import pandas as pd
import pytest

from mml.contracts import errors
from mml.contracts import tasks
from mml.data.splitting import splitters

BINARY = tasks.TaskType.BINARY


def make_spec(strategy="auto", **data_overrides):
    return tasks.ProjectSpec(
        data=tasks.DataSpec(uri="m", target="y", **data_overrides),
        validation=tasks.ValidationSpec(strategy=strategy),
    )


def split(frame, spec=None, task=BINARY):
    return splitters.SplitManager(spec or make_spec(), task).split(frame)


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    rows = 400
    return pd.DataFrame(
        {
            "x": rng.normal(size=rows),
            "g": rng.integers(0, 40, rows),
            "t": pd.date_range("2024-01-01", periods=rows, freq="h"),
            "y": (rng.random(rows) < 0.2).astype(int),
        }
    )


def test_stratified_preserves_ratio_and_is_disjoint(frame):
    result = split(frame)
    manifest = result.manifest
    assert manifest.strategy == "stratified"
    assert set(manifest.train_ids).isdisjoint(manifest.test_ids)
    assert len(manifest.train_ids) + len(manifest.test_ids) == len(frame)
    assert abs(result.dev_target.mean() - frame.y.mean()) < 0.03
    for train, val in manifest.folds:
        assert set(train).isdisjoint(val)
        assert (
            abs(result.dev_target.iloc[list(val)].mean() - frame.y.mean())
            < 0.08
        )
    assert "y" not in result.dev_features.columns


def test_group_split_has_no_group_overlap(frame):
    result = split(frame, make_spec(group_column="g"))
    manifest = result.manifest
    assert manifest.strategy == "group"
    train_groups = set(frame.g.iloc[list(manifest.train_ids)])
    test_groups = set(frame.g.iloc[list(manifest.test_ids)])
    assert train_groups.isdisjoint(test_groups)
    dev_groups = frame.g.iloc[list(manifest.train_ids)].to_numpy()
    for train, val in manifest.folds:
        assert set(dev_groups[list(train)]).isdisjoint(dev_groups[list(val)])


def test_temporal_split_respects_order(frame):
    shuffled = frame.sample(frac=1, random_state=1).reset_index(drop=True)
    manifest = split(shuffled, make_spec(time_column="t")).manifest
    times = shuffled.t
    assert manifest.strategy == "temporal"
    assert (
        times.iloc[list(manifest.train_ids)].max()
        <= times.iloc[list(manifest.test_ids)].min()
    )
    dev_times = times.iloc[list(manifest.train_ids)].to_numpy()
    for train, val in manifest.folds:
        assert dev_times[list(train)].max() <= dev_times[list(val)].min()


def test_temporal_with_unparseable_time_blocks(frame):
    broken = frame.assign(t=frame.t.astype(str))
    broken.loc[3, "t"] = "not a date"
    with pytest.raises(errors.BlockedError, match="ambiguous"):
        split(broken, make_spec(time_column="t"))


def test_group_split_blocks_with_few_groups(frame):
    few = frame.assign(g=frame.index % 3)
    with pytest.raises(errors.BlockedError, match="groups"):
        split(few, make_spec(group_column="g"))


def test_manifest_is_deterministic(frame):
    first = split(frame).manifest
    second = split(frame.copy()).manifest
    assert first.fingerprint == second.fingerprint
    assert first.to_dict()["n_test"] == len(first.test_ids)


def test_tiny_minority_class_blocks(frame):
    rare = frame.assign(y=0)
    rare.loc[0, "y"] = 1
    with pytest.raises(errors.BlockedError):
        split(rare)


def test_folds_shrink_to_minority_size(frame):
    thin = frame.assign(y=0)
    thin.loc[:3, "y"] = 1
    manifest = split(thin).manifest
    assert len(manifest.folds) < 5
    assert any("folds reduced" in note for note in manifest.notes)


def test_regression_defaults_to_random_and_rejects_stratified(frame):
    regression = frame.assign(y=frame.x * 2)
    result = split(regression, task=tasks.TaskType.REGRESSION)
    assert result.manifest.strategy == "random"
    with pytest.raises(errors.BlockedError):
        splitters.StratifiedSplit(make_spec(), tasks.TaskType.REGRESSION)


def test_strategy_registry_is_polymorphic():
    assert set(splitters.STRATEGIES) == {
        "random",
        "stratified",
        "stratified_events",
        "multilabel_stratified",
        "treatment_stratified",
        "group",
        "temporal",
        "rolling",
    }
    assert issubclass(splitters.StratifiedSplit, splitters.RandomSplit)
    assert issubclass(splitters.EventStratifiedSplit, splitters.RandomSplit)


def test_locked_test_set_opens_once(frame):
    locked = split(frame).locked_test
    assert locked.opened is False and "features" not in repr(locked)
    with pytest.raises(errors.LeakageError, match="invalid token"):
        locked.open("guess")
    features, target = locked.open(locked.issue_token())
    assert len(features) == len(locked) == len(target)
    with pytest.raises(errors.LeakageError, match="already used"):
        locked.open(locked.issue_token())


def temporal_spec(embargo):
    return tasks.ProjectSpec(
        data=tasks.DataSpec(uri="m", target="y", time_column="t"),
        validation=tasks.ValidationSpec(embargo_rows=embargo),
    )


def test_embargo_keeps_a_gap_between_training_and_validation_windows(frame):
    manifest = (
        splitters.SplitManager(temporal_spec(10), BINARY).split(frame).manifest
    )
    dev_times = frame.t.iloc[list(manifest.train_ids)].to_numpy()
    for fit, validate in manifest.folds:
        assert min(validate) - max(fit) > 10  # at least `embargo` rows apart
        assert dev_times[list(fit)].max() < dev_times[list(validate)].min()
    assert any("embargo of 10 rows" in note for note in manifest.notes)


def test_embargo_rows_before_the_test_set_belong_to_neither_partition(frame):
    plain = (
        splitters.SplitManager(temporal_spec(0), BINARY).split(frame).manifest
    )
    embargoed = (
        splitters.SplitManager(temporal_spec(10), BINARY).split(frame).manifest
    )
    assert embargoed.test_ids == plain.test_ids
    assert len(embargoed.train_ids) == len(plain.train_ids) - 10
    unused = (
        set(range(len(frame)))
        - set(embargoed.train_ids)
        - set(embargoed.test_ids)
    )
    assert len(unused) == 10
    ordered = frame.t.sort_values().index.to_list()
    assert (
        set(
            ordered[
                len(ordered) - len(embargoed.test_ids) - 10 : len(ordered)
                - len(embargoed.test_ids)
            ]
        )
        == unused
    )


def test_embargo_requires_a_time_column_and_enough_rows(frame):
    with pytest.raises(Exception, match="time_column"):
        tasks.ProjectSpec(
            data=tasks.DataSpec(uri="m", target="y"),
            validation=tasks.ValidationSpec(embargo_rows=5),
        )
    with pytest.raises(errors.BlockedError, match="too few"):
        splitters.SplitManager(temporal_spec(400), BINARY).split(frame)


def group_classification_frame(groups=30, per_group=10, seed=0):
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(groups), per_group)
    positive_groups = rng.choice(groups, groups // 3, replace=False)
    label = np.isin(unit, positive_groups).astype(int)
    flip = rng.random(len(label)) < 0.1
    frame = pd.DataFrame(
        {"unit": unit, "a": rng.normal(size=len(unit)), "y": label ^ flip}
    )
    return frame


def test_group_folds_keep_groups_apart_and_every_class_in_every_fold():
    frame = group_classification_frame()
    spec = tasks.ProjectSpec(
        data=tasks.DataSpec(uri="x", target="y", group_column="unit"),
        validation=tasks.ValidationSpec(strategy="group"),
    )
    part = splitters.SplitManager(spec, tasks.TaskType.BINARY).split(frame)
    dev_groups = frame.unit.to_numpy()[list(part.manifest.train_ids)]
    labels = frame.y.to_numpy()[list(part.manifest.train_ids)]
    for fit, held in part.manifest.folds:
        assert not set(dev_groups[list(fit)]) & set(dev_groups[list(held)])
        assert set(labels[list(fit)]) == {0, 1}
        assert set(labels[list(held)]) == {0, 1}


def test_group_validation_blocks_when_classes_hide_in_one_group():
    frame = group_classification_frame(groups=8, per_group=10)
    frame["y"] = (frame.unit == 0).astype(int)  # one group holds the class
    spec = tasks.ProjectSpec(
        data=tasks.DataSpec(uri="x", target="y", group_column="unit"),
        validation=tasks.ValidationSpec(strategy="group", folds=3),
    )
    with pytest.raises(errors.BlockedError):
        splitters.SplitManager(spec, tasks.TaskType.BINARY).split(frame)


def test_temporal_folds_are_listed_latest_window_first():
    rows = 200
    frame = pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01", periods=rows),
            "a": np.arange(rows, dtype=float),
            "y": np.arange(rows) % 2,
        }
    )
    spec = tasks.ProjectSpec(
        data=tasks.DataSpec(uri="x", target="y", time_column="t"),
        validation=tasks.ValidationSpec(strategy="temporal", folds=4),
    )
    part = splitters.SplitManager(spec, tasks.TaskType.BINARY).split(frame)
    sizes = [len(fit) for fit, held in part.manifest.folds]
    assert sizes == sorted(sizes, reverse=True)  # largest training first
    assert any("latest window first" in n for n in part.manifest.notes)


def rolling_spec(**validation):
    return tasks.ProjectSpec(
        data=tasks.DataSpec(uri="m", target="y", time_column="t"),
        validation=tasks.ValidationSpec(strategy="rolling", **validation),
    )


def test_rolling_windows_slide_forward_and_always_validate_on_the_future(frame):
    result = split(
        frame,
        rolling_spec(folds=4, horizon_rows=30, window_rows=100),
    )
    manifest = result.manifest
    times = pd.to_datetime(frame["t"]).iloc[list(manifest.train_ids)].to_numpy()
    order = np.argsort(times, kind="stable")

    assert manifest.strategy == "rolling" and len(manifest.folds) == 4
    for train, held in manifest.folds:
        assert max(order.tolist().index(i) for i in train) < min(
            order.tolist().index(i) for i in held
        )
        assert len(held) == 30 and len(train) <= 100


def test_rolling_embargo_leaves_a_gap_before_each_validation_window(frame):
    result = split(
        frame, rolling_spec(folds=3, horizon_rows=20, embargo_rows=5)
    )

    positions = {
        row: rank for rank, row in enumerate(sorted(result.manifest.train_ids))
    }
    ordered = frame["t"].iloc[sorted(result.manifest.train_ids)]
    assert ordered.is_monotonic_increasing
    for train, held in result.manifest.folds:
        ranks_train = sorted(positions[i] for i in train)
        ranks_held = sorted(positions[i] for i in held)
        assert ranks_held[0] - ranks_train[-1] > 5


def test_rolling_blocks_when_no_window_fits(frame):
    with pytest.raises(errors.BlockedError, match="rolling window"):
        split(frame, rolling_spec(folds=4, horizon_rows=200))


def test_the_rolling_strategy_needs_a_time_column_and_its_options_need_rolling():
    with pytest.raises(ValueError, match=r"requires data\.time_column"):
        tasks.ProjectSpec(
            data=tasks.DataSpec(uri="m", target="y"),
            validation=tasks.ValidationSpec(strategy="rolling"),
        )
    with pytest.raises(ValueError, match="rolling strategy"):
        tasks.ProjectSpec(
            data=tasks.DataSpec(uri="m", target="y", time_column="t"),
            validation=tasks.ValidationSpec(
                strategy="stratified", horizon_rows=10
            ),
        )


def test_feature_delays_become_lagged_nodes_and_exclusions_idempotently():
    data = tasks.DataSpec(
        uri="m", target="y", time_column="t", feature_delays={"report": 3}
    )
    spec = tasks.ProjectSpec(data=data)

    node = spec.feature_graph.nodes[0]
    assert (node.name, node.op, node.params) == (
        "report__available",
        "lag",
        {"k": 3},
    )
    assert "report" in spec.data.excluded_columns
    again = tasks.ProjectSpec.model_validate_json(spec.canonical_json())
    assert again == spec and len(again.feature_graph.nodes) == 1


def test_feature_delays_need_a_time_column_and_forbid_reading_the_raw_column():
    with pytest.raises(ValueError, match=r"requires data\.time_column"):
        tasks.ProjectSpec(
            data=tasks.DataSpec(
                uri="m", target="y", feature_delays={"report": 2}
            )
        )
    with pytest.raises(ValueError, match="read a delayed column"):
        tasks.ProjectSpec(
            data=tasks.DataSpec(
                uri="m", target="y", time_column="t", feature_delays={"r": 2}
            ),
            feature_graph=tasks.FeatureGraphSpec(
                nodes=[
                    tasks.FeatureNode(name="doubled", op="log1p", inputs=["r"])
                ]
            ),
        )
