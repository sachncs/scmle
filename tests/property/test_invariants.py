"""Property-based tests over randomized shapes, missingness and structure."""

from hypothesis import given
from hypothesis import HealthCheck
from hypothesis import settings
from hypothesis import strategies
import numpy
import pandas
import pytest

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.curation import preprocessing
from scmle.data.modalities import document
from scmle.data.profiling import profiler
from scmle.data.readers import tabular
from scmle.data.splitting import splitters
import scmle.data.validation.target
from scmle.governance import attestation
from scmle.models import registry
from scmle.models import thresholds
from scmle.monitoring import reference
from scmle.sampling import resampling

SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
REGISTRY = registry.ModelRegistry.with_builtins()


@strategies.composite
def frames(draw, min_rows=40, max_rows=240):
    """Random classification frames with missingness and mixed types."""
    rows = draw(strategies.integers(min_rows, max_rows))
    seed = draw(strategies.integers(0, 10_000))
    rng = numpy.random.default_rng(seed)
    numeric = draw(strategies.integers(0, 4))
    categorical = draw(strategies.integers(0, 3))
    frame = pandas.DataFrame(index=range(rows))
    for index in range(numeric):
        frame[f"n{index}"] = rng.normal(size=rows)
        rate = draw(strategies.sampled_from([0.0, 0.1, 0.5, 0.9]))
        frame.loc[rng.random(rows) < rate, f"n{index}"] = numpy.nan
    for index in range(categorical):
        levels = draw(strategies.integers(1, 40))
        frame[f"c{index}"] = rng.choice([f"v{k}" for k in range(levels)], rows)
        rate = draw(strategies.sampled_from([0.0, 0.2]))
        frame.loc[rng.random(rows) < rate, f"c{index}"] = None
    if numeric + categorical == 0:
        frame["n0"] = rng.normal(size=rows)
    positive = draw(strategies.sampled_from([0.5, 0.2, 0.08]))
    frame["y"] = (rng.random(rows) < positive).astype(int)
    if draw(strategies.booleans()):
        frame = pandas.concat([frame, frame.head(rows // 4)], ignore_index=True)
    return frame


def spec_for(**kwargs):
    return tasks.ProjectSpec(data=tasks.DataSpec(uri="m", target="y", **kwargs))


JSON_SCALARS = (
    strategies.none()
    | strategies.booleans()
    | strategies.integers()
    | strategies.text(max_size=20)
)
JSON_VALUES = strategies.recursive(
    JSON_SCALARS,
    lambda children: (
        strategies.lists(children, max_size=5)
        | strategies.dictionaries(
            strategies.text(max_size=12), children, max_size=8
        )
    ),
    max_leaves=30,
)


@SETTINGS
@given(
    payload=strategies.dictionaries(
        strategies.text(max_size=12), JSON_VALUES, max_size=12
    )
)
def test_attestation_parser_fails_closed_on_arbitrary_json(payload):
    """Malformed persisted evidence never escapes as an incidental error."""
    try:
        parsed = attestation.ReleaseAttestation.from_dict(payload)
    except (TypeError, ValueError):
        return
    assert parsed.to_dict()["run_id"] == parsed.run_id


@SETTINGS
@given(payload=strategies.binary(max_size=512))
def test_document_reader_fails_closed_on_arbitrary_bytes(payload):
    """Arbitrary document bytes become structured outcomes or valid samples."""
    reader = document.DocumentReader(max_bytes=1024)
    try:
        sample = reader.load(payload, document_id="fuzzed")
    except document.DocumentValidationError as error:
        assert error.category == "document_validation"
        return
    assert sample.document_id == "fuzzed"
    assert sample.content


@SETTINGS
@given(frame=frames())
def test_profiling_yields_a_consistent_plan_or_a_clear_refusal(frame):
    spec = spec_for()
    manifest = tabular.build_manifest(frame, "y")
    try:
        task = scmle.data.validation.target.TaskResolver(spec).resolve(
            frame["y"]
        )
        profile = profiler.DatasetProfiler(spec).profile(frame, manifest, task)
    except errors.ScmleError:
        return
    plan = profile.feature_plan
    assert "y" not in plan.features
    assert set(plan.features) <= set(frame.columns)
    assert not set(plan.features) & set(plan.dropped)
    assert set(plan.features) | set(plan.dropped) == set(frame.columns) - {"y"}
    assert all(
        0.0 <= s["missing_rate"] <= 1.0
        for s in profile.column_statistics.values()
    )


@SETTINGS
@given(frame=frames(min_rows=60), seed=strategies.integers(0, 100))
def test_pipelines_fit_and_predict_on_arbitrary_clean_frames(frame, seed):
    spec = spec_for()
    manifest = tabular.build_manifest(frame, "y")
    try:
        task = scmle.data.validation.target.TaskResolver(spec).resolve(
            frame["y"]
        )
        plan = (
            profiler.DatasetProfiler(spec)
            .profile(frame, manifest, task)
            .feature_plan
        )
    except errors.ScmleError:
        return
    features = frame.drop(columns="y")
    for name in (
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    ):
        pipe = preprocessing.build_pipeline(
            plan, REGISTRY.get(name), task, seed
        )
        pipe.fit(features, frame["y"])
        proba = pipe.predict_proba(features)
        assert proba.shape == (len(frame), 2) and numpy.isfinite(proba).all()
        numpy.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@SETTINGS
@given(
    rows=strategies.integers(60, 400),
    positive=strategies.sampled_from([0.5, 0.25, 0.1]),
    folds=strategies.integers(2, 6),
    test_fraction=strategies.sampled_from([0.1, 0.2, 0.3]),
    seed=strategies.integers(0, 1000),
)
def test_stratified_splits_partition_rows_without_overlap(
    rows, positive, folds, test_fraction, seed
):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {
            "x": rng.normal(size=rows),
            "y": (rng.random(rows) < positive).astype(int),
        }
    )
    spec = tasks.ProjectSpec(
        data=tasks.DataSpec(uri="m", target="y"),
        validation=tasks.ValidationSpec(
            folds=folds, test_fraction=test_fraction, seed=seed
        ),
    )
    try:
        result = splitters.SplitManager(spec, tasks.TaskType.BINARY).split(
            frame
        )
    except errors.BlockedError:
        return
    manifest = result.manifest
    train, test = set(manifest.train_ids), set(manifest.test_ids)
    assert not train & test and train | test == set(range(rows))
    covered = set()
    for fit, validate in manifest.folds:
        assert not set(fit) & set(validate)
        assert set(fit) | set(validate) <= set(range(len(train)))
        covered |= set(validate)
    assert covered == set(range(len(train)))
    assert result.locked_test.ids == manifest.test_ids


@SETTINGS
@given(
    groups=strategies.integers(8, 60),
    rows=strategies.integers(80, 400),
    seed=strategies.integers(0, 1000),
)
def test_group_splits_never_share_a_group(groups, rows, seed):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {
            "g": rng.integers(0, groups, rows),
            "x": rng.normal(size=rows),
            "y": rng.integers(0, 2, rows),
        }
    )
    spec = spec_for(group_column="g")
    try:
        result = splitters.SplitManager(spec, tasks.TaskType.BINARY).split(
            frame
        )
    except errors.BlockedError:
        return
    manifest = result.manifest
    train_ids = numpy.asarray(manifest.train_ids, dtype=numpy.int64)
    test_ids = numpy.asarray(manifest.test_ids, dtype=numpy.int64)
    dev_groups = frame.g.iloc[train_ids].to_numpy()
    assert set(dev_groups).isdisjoint(frame.g.iloc[test_ids])
    for fit, validate in manifest.folds:
        fit_ids = numpy.asarray(fit, dtype=numpy.int64)
        validate_ids = numpy.asarray(validate, dtype=numpy.int64)
        assert set(dev_groups[fit_ids]).isdisjoint(dev_groups[validate_ids])


@SETTINGS
@given(rows=strategies.integers(60, 300), seed=strategies.integers(0, 1000))
def test_temporal_splits_never_train_on_the_future(rows, seed):
    rng = numpy.random.default_rng(seed)
    times = pandas.Series(
        pandas.date_range("2024-01-01", periods=rows, freq="h")
    ).sample(frac=1, random_state=seed)
    frame = pandas.DataFrame(
        {
            "t": times.to_numpy(),
            "x": rng.normal(size=rows),
            "y": rng.integers(0, 2, rows),
        }
    )
    spec = spec_for(time_column="t")
    try:
        result = splitters.SplitManager(spec, tasks.TaskType.BINARY).split(
            frame
        )
    except errors.BlockedError:
        return
    manifest = result.manifest
    t = frame.t
    train_ids = numpy.asarray(manifest.train_ids, dtype=numpy.int64)
    test_ids = numpy.asarray(manifest.test_ids, dtype=numpy.int64)
    assert t.iloc[train_ids].max() <= t.iloc[test_ids].min()
    dev = t.iloc[train_ids].to_numpy()
    for fit, validate in manifest.folds:
        fit_ids = numpy.asarray(fit, dtype=numpy.int64)
        validate_ids = numpy.asarray(validate, dtype=numpy.int64)
        assert dev[fit_ids].max() <= dev[validate_ids].min()


@SETTINGS
@given(
    counts=strategies.lists(strategies.integers(2, 80), min_size=2, max_size=5),
    seed=strategies.integers(0, 1000),
)
def test_resamplers_balance_classes_using_only_existing_rows(counts, seed):
    labels = pandas.Series(numpy.repeat(numpy.arange(len(counts)), counts))
    features = pandas.DataFrame({"row": numpy.arange(len(labels))})
    over = resampling.RandomOverSampler().resample(features, labels, seed)
    under = resampling.RandomUnderSampler().resample(features, labels, seed)
    assert set(over[1].value_counts()) == {max(counts)}
    assert set(under[1].value_counts()) == {min(counts)}
    for sampled, target in (over, under):
        assert set(sampled["row"]) <= set(features["row"])
        assert (
            labels.iloc[sampled["row"].to_numpy()].to_numpy()
            == target.to_numpy()
        ).all()


@SETTINGS
@given(
    values=strategies.lists(
        strategies.floats(0.0, 1.0), min_size=20, max_size=200
    ),
    seed=strategies.integers(0, 1000),
)
def test_optimized_threshold_never_loses_to_the_default(values, seed):
    probabilities = numpy.array(values)
    rng = numpy.random.default_rng(seed)
    actual = rng.random(len(values)) < probabilities
    for objective in (
        thresholds.F1Objective(),
        thresholds.AccuracyObjective(),
        thresholds.BalancedAccuracyObjective(),
    ):
        best, score = thresholds.optimize_threshold(
            probabilities, actual, objective
        )
        assert best in thresholds.GRID
        assert score >= objective.score(actual, probabilities >= 0.5) - 1e-12


@SETTINGS
@given(
    left=strategies.lists(
        strategies.floats(0.001, 1.0), min_size=3, max_size=12
    ),
    right=strategies.lists(
        strategies.floats(0.001, 1.0), min_size=3, max_size=12
    ),
)
def test_psi_is_non_negative_and_symmetric(left, right):
    size = min(len(left), len(right))
    a = numpy.array(left[:size]) / sum(left[:size])
    b = numpy.array(right[:size]) / sum(right[:size])
    forward = reference.population_stability_index(a, b)
    backward = reference.population_stability_index(b, a)
    assert forward >= 0.0 and forward == pytest.approx(
        backward, rel=1e-6, abs=1e-9
    )
    assert reference.population_stability_index(a, a) == pytest.approx(
        0.0, abs=1e-12
    )
