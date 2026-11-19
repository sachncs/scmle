"""Tests for split strategies, manifest and the locked test set."""

import math

import numpy
import pandas
import pytest

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.splitting import splitters
from scmle.data.splitting.splitters import lockbox
from scmle.data.splitting.splitters import partitions
from scmle.data.splitting.splitters import splits
from scmle.data.splitting.splitters import temporal


def make_spec(strategy="auto", **data_overrides):
    return tasks.Config(
        data=tasks.Dataset(uri="m", target="y", **data_overrides),
        validation=tasks.Validation(strategy=strategy),
    )


def split(frame, spec=None, task=tasks.Task.BINARY):
    return splitters.Splitter(spec or make_spec(), task).split(frame)


@pytest.fixture
def frame():
    rng = numpy.random.default_rng(0)
    rows = 400
    return pandas.DataFrame(
        {
            "x": rng.normal(size=rows),
            "g": rng.integers(0, 40, rows),
            "t": pandas.date_range("2024-01-01", periods=rows, freq="h"),
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
    # Enforces docs/GLOSSARY.md:238.
    # Enforces website/src/learn/glossary.html:125.
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
    # Enforces docs/GLOSSARY.md:252.
    # Enforces website/src/learn/glossary.html:133.
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
    # Enforces README.md:151.
    # Enforces docs/WHEN_THINGS_GO_WRONG.md:168.
    # Enforces website/src/learn/when-things-go-wrong.html:69.
    broken = frame.assign(t=frame.t.astype(str))
    broken.loc[3, "t"] = "not a date"
    with pytest.raises(errors.BlockedError, match="ambiguous"):
        split(broken, make_spec(time_column="t"))


def test_group_split_blocks_with_few_groups(frame):
    # Enforces README.md:151.
    few = frame.assign(g=frame.index % 3)
    with pytest.raises(errors.BlockedError, match="groups"):
        split(few, make_spec(group_column="g"))


def test_manifest_is_deterministic(frame):
    first = split(frame).manifest
    second = split(frame.copy()).manifest
    assert first.fingerprint == second.fingerprint
    assert first.to_dict()["n_test"] == len(first.test_ids)


def test_tiny_minority_class_blocks(frame):
    # Enforces README.md:151.
    rare = frame.assign(y=0)
    rare.loc[0, "y"] = 1
    with pytest.raises(errors.BlockedError):
        split(rare)


def test_folds_shrink_to_minority_size(frame):
    thin = frame.assign(y=0)
    thin.loc[:7, "y"] = 1
    manifest = split(thin).manifest
    assert len(manifest.folds) < 5
    assert any("folds reduced" in note for note in manifest.notes)


def test_the_test_minimum_is_where_a_perfect_class_beats_a_coin():
    least = partitions.MIN_TEST_CLASS_ROWS
    alpha = partitions.EVIDENCE_ALPHA
    assert least == math.ceil(math.log(alpha) / math.log(0.5)) == 5
    # all `m` rows of a class right: the exact upper bound of its error rate
    # is 1 - alpha ** (1 / m); below one half only from the minimum on
    assert 1 - alpha ** (1 / least) < 0.5 <= 1 - alpha ** (1 / (least - 1))


def test_a_class_absent_from_the_test_rows_blocks(frame):
    """Two minority rows both stay in development (the C1 measurement)."""
    rare = frame.assign(y=0)
    rare.loc[:1, "y"] = 1
    with pytest.raises(errors.BlockedError, match="test rows") as raised:
        split(rare)
    assert raised.value.details["short"] == {"1": 0}


def test_a_class_with_enough_rows_reaches_the_test_minimum(frame):
    rare = frame.assign(y=0)
    least = partitions.MIN_TEST_CLASS_ROWS
    rare.loc[: least + partitions.MIN_STRATUM_ROWS - 1, "y"] = 1
    result = split(rare)
    held = rare["y"].iloc[list(result.manifest.test_ids)]
    kept = rare["y"].iloc[list(result.manifest.train_ids)]
    assert (held == 1).sum() == least
    assert (kept == 1).sum() == partitions.MIN_STRATUM_ROWS
    assert not set(result.manifest.test_ids) & set(result.manifest.train_ids)
    assert result.manifest == split(rare).manifest


def test_a_common_class_keeps_its_proportional_test_rows(frame):
    result = split(frame)
    plain = splits.Random(make_spec(), tasks.Task.BINARY)
    labels = frame["y"]
    reference = plain.partition(frame)
    assert len(result.manifest.test_ids) == len(reference.test_ids)
    assert (labels.iloc[list(result.manifest.test_ids)] == 1).sum() >= (
        partitions.MIN_TEST_CLASS_ROWS
    )


def test_an_unstratified_split_with_few_test_rows_of_a_class_blocks(frame):
    rare = frame.assign(y=0)
    rare.loc[:3, "y"] = 1
    with pytest.raises(errors.BlockedError, match="test rows"):
        split(rare, make_spec("random"))


def test_a_regression_split_is_not_held_to_class_counts(frame):
    result = split(
        frame.assign(y=frame["x"]), make_spec("random"), tasks.Task.REGRESSION
    )
    assert len(result.manifest.test_ids) == 80


def test_regression_defaults_to_random_and_rejects_stratified(frame):
    regression = frame.assign(y=frame.x * 2)
    result = split(regression, task=tasks.Task.REGRESSION)
    assert result.manifest.strategy == "random"
    with pytest.raises(errors.BlockedError):
        splits.Stratified(make_spec(), tasks.Task.REGRESSION)


def test_strategy_registry_is_polymorphic():
    assert set(splitters.STRATEGIES) == {
        "random",
        "stratified",
        "stratified_events",
        "multilabel_stratified",
        "treatment_stratified",
        "preset",
        "group",
        "temporal",
        "rolling",
    }
    assert issubclass(splits.Stratified, splits.Random)
    assert issubclass(splits.Events, splits.Random)


def test_strategies_are_one_family_of_single_noun_classes():
    family = {
        "random": splits.Random,
        "stratified": splits.Stratified,
        "stratified_events": splits.Events,
        "multilabel_stratified": splits.Multilabel,
        "treatment_stratified": splits.Treatment,
        "preset": splits.Preset,
        "group": splits.Group,
        "temporal": temporal.Chronology,
        "rolling": temporal.Rolling,
    }
    assert family == splitters.STRATEGIES
    timed = {temporal.Chronology, temporal.Rolling}
    assert all(
        kind.__module__ == (temporal if kind in timed else splits).__name__
        for kind in family.values()
    )


def test_lockbox_module_holds_the_sealed_test_set_family():
    assert issubclass(lockbox.Resident, lockbox.Lockbox)
    assert splitters.Lockbox is lockbox.Lockbox
    assert lockbox.Resident.__module__ == lockbox.__name__


def test_locked_test_set_opens_once(frame):
    locked = split(frame).locked_test
    assert locked.opened is False and "features" not in repr(locked)
    with pytest.raises(errors.LeakageError, match="invalid token"):
        locked.open("guess")
    with pytest.raises(errors.LeakageError, match="bind it"):
        locked.open(locked.mint())
    locked.bind("b")
    with pytest.raises(errors.LeakageError, match="binding mismatch"):
        locked.open(locked.mint(), "other")
    features, target = locked.open(locked.mint(), "b")
    assert len(features) == len(locked) == len(target)
    with pytest.raises(errors.LeakageError, match="already used"):
        locked.open(locked.mint(), "b")


def temporal_spec(embargo):
    return tasks.Config(
        data=tasks.Dataset(uri="m", target="y", time_column="t"),
        validation=tasks.Validation(embargo_rows=embargo),
    )


def test_embargo_keeps_a_gap_between_training_and_validation_windows(frame):
    manifest = (
        splitters.Splitter(temporal_spec(10), tasks.Task.BINARY)
        .split(frame)
        .manifest
    )
    dev_times = frame.t.iloc[list(manifest.train_ids)].to_numpy()
    for fit, validate in manifest.folds:
        assert min(validate) - max(fit) > 10  # at least `embargo` rows apart
        assert dev_times[list(fit)].max() < dev_times[list(validate)].min()
    assert any("embargo of 10 rows" in note for note in manifest.notes)


def test_embargo_rows_before_the_test_set_belong_to_neither_partition(frame):
    plain = (
        splitters.Splitter(temporal_spec(0), tasks.Task.BINARY)
        .split(frame)
        .manifest
    )
    embargoed = (
        splitters.Splitter(temporal_spec(10), tasks.Task.BINARY)
        .split(frame)
        .manifest
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
        tasks.Config(
            data=tasks.Dataset(uri="m", target="y"),
            validation=tasks.Validation(embargo_rows=5),
        )
    with pytest.raises(errors.BlockedError, match="too few"):
        splitters.Splitter(temporal_spec(400), tasks.Task.BINARY).split(frame)


def group_classification_frame(groups=30, per_group=10, seed=0):
    rng = numpy.random.default_rng(seed)
    unit = numpy.repeat(numpy.arange(groups), per_group)
    positive_groups = rng.choice(groups, groups // 3, replace=False)
    label = numpy.isin(unit, positive_groups).astype(int)
    flip = rng.random(len(label)) < 0.1
    return pandas.DataFrame(
        {"unit": unit, "a": rng.normal(size=len(unit)), "y": label ^ flip}
    )


def test_group_folds_keep_groups_apart_and_every_class_in_every_fold():
    frame = group_classification_frame()
    spec = tasks.Config(
        data=tasks.Dataset(uri="x", target="y", group_column="unit"),
        validation=tasks.Validation(strategy="group"),
    )
    part = splitters.Splitter(spec, tasks.Task.BINARY).split(frame)
    dev_groups = frame.unit.to_numpy()[list(part.manifest.train_ids)]
    labels = frame.y.to_numpy()[list(part.manifest.train_ids)]
    for fit, held in part.manifest.folds:
        assert not set(dev_groups[list(fit)]) & set(dev_groups[list(held)])
        assert set(labels[list(fit)]) == {0, 1}
        assert set(labels[list(held)]) == {0, 1}


def test_group_validation_blocks_when_classes_hide_in_one_group():
    frame = group_classification_frame(groups=8, per_group=10)
    frame["y"] = (frame.unit == 0).astype(int)  # one group holds the class
    spec = tasks.Config(
        data=tasks.Dataset(uri="x", target="y", group_column="unit"),
        validation=tasks.Validation(strategy="group", folds=3),
    )
    with pytest.raises(errors.BlockedError):
        splitters.Splitter(spec, tasks.Task.BINARY).split(frame)


def test_temporal_folds_are_listed_latest_window_first():
    rows = 200
    frame = pandas.DataFrame(
        {
            "t": pandas.date_range("2024-01-01", periods=rows),
            "a": numpy.arange(rows, dtype=float),
            "y": numpy.arange(rows) % 2,
        }
    )
    spec = tasks.Config(
        data=tasks.Dataset(uri="x", target="y", time_column="t"),
        validation=tasks.Validation(strategy="temporal", folds=4),
    )
    part = splitters.Splitter(spec, tasks.Task.BINARY).split(frame)
    sizes = [len(fit) for fit, held in part.manifest.folds]
    assert sizes == sorted(sizes, reverse=True)  # largest training first
    assert any("latest window first" in n for n in part.manifest.notes)


def rolling_spec(**validation):
    return tasks.Config(
        data=tasks.Dataset(uri="m", target="y", time_column="t"),
        validation=tasks.Validation(strategy="rolling", **validation),
    )


def test_rolling_windows_slide_forward_and_always_validate_on_the_future(frame):
    result = split(
        frame,
        rolling_spec(folds=4, horizon_rows=30, window_rows=100),
    )
    manifest = result.manifest
    times = (
        pandas.to_datetime(frame["t"]).iloc[list(manifest.train_ids)].to_numpy()
    )
    order = numpy.argsort(times, kind="stable")

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
        tasks.Config(
            data=tasks.Dataset(uri="m", target="y"),
            validation=tasks.Validation(strategy="rolling"),
        )
    with pytest.raises(ValueError, match="rolling strategy"):
        tasks.Config(
            data=tasks.Dataset(uri="m", target="y", time_column="t"),
            validation=tasks.Validation(strategy="stratified", horizon_rows=10),
        )


def test_feature_delays_become_lagged_nodes_and_exclusions_idempotently():
    data = tasks.Dataset(
        uri="m", target="y", time_column="t", feature_delays={"report": 3}
    )
    spec = tasks.Config(data=data)

    node = spec.feature_graph.nodes[0]
    assert (node.name, node.op, node.params) == (
        "report__available",
        "lag",
        {"k": 3},
    )
    assert "report" in spec.data.excluded_columns
    again = tasks.Config.model_validate_json(spec.canonize())
    assert again == spec and len(again.feature_graph.nodes) == 1


def test_feature_delays_need_a_time_column_and_forbid_reading_the_raw_column():
    with pytest.raises(ValueError, match=r"requires data\.time_column"):
        tasks.Config(
            data=tasks.Dataset(
                uri="m", target="y", feature_delays={"report": 2}
            )
        )
    with pytest.raises(ValueError, match="read a delayed column"):
        tasks.Config(
            data=tasks.Dataset(
                uri="m", target="y", time_column="t", feature_delays={"r": 2}
            ),
            feature_graph=tasks.Derivation(
                nodes=[tasks.Node(name="doubled", op="log1p", inputs=["r"])]
            ),
        )


def test_a_table_too_small_for_its_classes_gets_advice():
    tiny = pandas.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]})
    with pytest.raises(errors.DataError, match="Provide at least"):
        split(tiny)


def test_a_table_just_large_enough_still_splits():
    rng = numpy.random.default_rng(1)
    enough = pandas.DataFrame({"x": rng.normal(size=60), "y": [0, 1] * 30})
    result = split(enough)
    assert len(result.dev_features) < 60
    assert result.dev_target.nunique() == 2


def tied_frame(rows=400, tie=7):
    """Rows whose timestamps repeat in runs of `tie` (shuffled order)."""
    rng = numpy.random.default_rng(1)
    order = rng.permutation(rows)
    return pandas.DataFrame(
        {
            "x": rng.normal(size=rows),
            "t": pandas.Timestamp("2024-01-01")
            + pandas.to_timedelta(order // tie, unit="D"),
            "y": (rng.random(rows) < 0.3).astype(int),
        }
    )


def straddles(times, left, right):
    """Timestamps on both sides of one split."""
    return set(times[list(left)]) & set(times[list(right)])


@pytest.mark.parametrize("strategy", ["temporal", "rolling"])
@pytest.mark.parametrize("embargo", [0, 3])
def test_no_tied_timestamp_sits_on_both_sides_of_a_time_split(
    strategy, embargo
):
    frame = tied_frame()
    spec = tasks.Config(
        data=tasks.Dataset(uri="m", target="y", time_column="t"),
        validation=tasks.Validation(
            strategy=strategy, folds=4, embargo_rows=embargo
        ),
    )
    manifest = split(frame, spec).manifest
    times = frame["t"].to_numpy()
    assert not straddles(times, manifest.train_ids, manifest.test_ids)
    dev = numpy.asarray(manifest.train_ids)
    assert len(manifest.folds) >= 2
    for fit, held in manifest.folds:
        assert len(fit) and len(held)
        assert not straddles(times, dev[list(fit)], dev[list(held)])
        assert times[dev[list(fit)]].max() < times[dev[list(held)]].min()
    # every tie block is whole: the test set starts at a new timestamp
    assert times[list(manifest.test_ids)].min() > times[dev].max()


def test_a_time_split_moves_a_tie_at_the_cut_wholly_into_the_test_set():
    frame = tied_frame(rows=100, tie=10)
    spec = tasks.Config(
        data=tasks.Dataset(uri="m", target="y", time_column="t"),
        validation=tasks.Validation(strategy="temporal", folds=3),
    )
    manifest = split(frame, spec).manifest
    # 20 percent of 100 rows is 20, a whole number of ties of 10.
    assert len(manifest.test_ids) == 20
    tilted = tied_frame(rows=95, tie=10)
    manifest = split(tilted, spec).manifest
    # 19 rows would cut a tie of 10; the tie goes to the test set whole.
    counts = tilted["t"].value_counts()
    held = tilted["t"].iloc[list(manifest.test_ids)]
    assert all(counts[t] == n for t, n in held.value_counts().items())


def test_a_time_split_with_one_timestamp_is_blocked():
    frame = pandas.DataFrame(
        {"t": ["2024-01-01"] * 50, "x": range(50), "y": [0, 1] * 25}
    )
    spec = tasks.Config(
        data=tasks.Dataset(uri="m", target="y", time_column="t"),
        validation=tasks.Validation(strategy="temporal"),
    )
    with pytest.raises(errors.BlockedError, match="timestamp"):
        split(frame, spec)


def panel(entities=30, steps=20):
    """Every entity observed at every time step."""
    rng = numpy.random.default_rng(3)
    rows = entities * steps
    return pandas.DataFrame(
        {
            "who": numpy.repeat(numpy.arange(entities), steps),
            "t": pandas.Timestamp("2024-01-01")
            + pandas.to_timedelta(
                numpy.tile(numpy.arange(steps), entities), "D"
            ),
            "x": rng.normal(size=rows),
            "y": rng.normal(size=rows),
        }
    )


def entity_spec(strategy, entities=None, **validation):
    declared = {} if entities is None else {"entities": entities}
    return tasks.Config(
        data=tasks.Dataset(
            uri="m", target="y", time_column="t", group_column="who"
        ),
        validation=tasks.Validation(
            strategy=strategy, folds=3, **declared, **validation
        ),
    )


def sides(manifest):
    """Every pair of sides of a split: the holdout, then each fold."""
    dev = numpy.asarray(manifest.train_ids)
    pairs = [(dev, numpy.asarray(manifest.test_ids))]
    pairs += [(dev[list(fit)], dev[list(held)]) for fit, held in manifest.folds]
    return pairs


def test_known_entities_is_the_default_and_keeps_todays_crossing():
    frame = panel()
    spec = entity_spec("temporal")
    assert spec.validation.entities == "known"
    manifest = split(frame, spec, tasks.Task.REGRESSION).manifest
    who = frame["who"].to_numpy()
    assert manifest.entities == "known"
    assert all(set(who[a]) & set(who[b]) for a, b in sides(manifest))


@pytest.mark.parametrize("strategy", ["temporal", "rolling"])
def test_unseen_entities_keep_each_entity_on_one_side_of_every_split(strategy):
    frame = panel()
    manifest = split(
        frame, entity_spec(strategy, "unseen"), tasks.Task.REGRESSION
    ).manifest
    who, times = frame["who"].to_numpy(), frame["t"].to_numpy()
    assert manifest.entities == "unseen"
    assert manifest.to_dict()["entities"] == "unseen"
    assert len(manifest.folds) >= 2
    for left, right in sides(manifest):
        assert len(left) and len(right)
        assert not set(who[left]) & set(who[right])
        assert times[left].max() < times[right].min()
    assert any("unseen" in note for note in manifest.notes)


def test_a_split_without_a_time_strategy_records_no_entity_declaration(frame):
    manifest = split(frame).manifest
    assert manifest.entities is None
    assert "entities" not in manifest.to_dict()


def test_unseen_entities_need_an_entity_column():
    with pytest.raises(ValueError, match="group_column"):
        tasks.Config(
            data=tasks.Dataset(uri="m", target="y", time_column="t"),
            validation=tasks.Validation(strategy="temporal", entities="unseen"),
        )


def test_unseen_entities_with_a_single_entity_are_refused():
    frame = panel(entities=1, steps=200)
    with pytest.raises(errors.BlockedError, match="entit"):
        split(frame, entity_spec("temporal", "unseen"), tasks.Task.REGRESSION)


def test_the_declaration_keeps_the_hash_of_an_existing_configuration():
    stored = entity_spec("temporal").model_dump(mode="json")
    stored["validation"].pop("entities")
    old = tasks.Config.model_validate(stored)
    assert old.validation.entities == "known"
    assert old.config_hash() == entity_spec("temporal").config_hash()
    assert "entities" not in old.anchor()["validation"]
    unseen = entity_spec("temporal", "unseen")
    assert unseen.config_hash() != old.config_hash()
    assert unseen.anchor()["validation"]["entities"] == "unseen"
