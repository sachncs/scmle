"""Adversarial leakage checks."""

import json

import numpy
import pandas
import pytest
import support

import scmle.api.project
from scmle.contracts import errors
from scmle.contracts import tasks


def test_pure_noise_is_not_inflated_by_the_pipeline(tmp_path):
    rng = numpy.random.default_rng(5)
    rows = 600
    frame = pandas.DataFrame(
        {f"f{i}": rng.normal(size=rows) for i in range(10)}
    )
    frame["cat"] = rng.choice(list("abcdefgh"), rows)
    frame["y"] = rng.integers(0, 2, rows)
    result = support.make_project(
        frame, "y", directory=tmp_path, budget=120
    ).run()
    assert abs(result.evaluation["metrics"]["roc_auc"] - 0.5) < 0.12
    assert abs(result.evaluation["selected"]["score"] - 0.5) < 0.12


def test_leaked_feature_is_flagged_and_can_block(binary_frame, tmp_path):
    # Enforces README.md:112 and README.md:199.
    # Enforces docs/BENCHMARKS.md:16 and docs/LEAKAGE.md:233.
    # Enforces docs/WHEN_THINGS_GO_WRONG.md:110.
    # Enforces website/src/learn/leakage.html:69-70.
    # Enforces website/src/learn/when-things-go-wrong.html:59.
    # Enforces website/src/pages/docs/benchmarks.astro:709-711.
    # Enforces website/src/pages/docs/benchmarks.astro:773.
    # Enforces website/src/pages/index.astro:149.
    # Enforces website/src/pages/docs/architecture.astro:48.
    # Enforces website/src/pages/docs/operating-loop.astro:18.
    leaky = binary_frame.assign(leak=binary_frame.y * 3.0 + 1)
    result = support.make_project(
        leaky, "y", directory=tmp_path, leakage="warn"
    ).run()
    assert any("'leak'" in w for w in result.profile.risk_findings)
    assert result.profile.blockers == ("leak",)
    with pytest.raises(errors.BlockedError, match="leakage") as raised:
        support.make_project(
            leaky, "y", directory=tmp_path, leakage="block"
        ).run()
    assert "leak" in str(raised.value)
    assert raised.value.details["leakage"]["blocking"] == ["leak"]


def test_a_noisy_copy_and_a_value_recorded_for_one_class_also_block(
    binary_frame, tmp_path
):
    # Enforces docs/API_STABILITY.md:54.
    rng = numpy.random.default_rng(8)
    flip = rng.random(len(binary_frame)) < 0.02
    noisy = binary_frame.assign(
        noisy=numpy.where(flip, 1 - binary_frame.y, binary_frame.y)
    )
    recorded = binary_frame.assign(
        boat=numpy.where(binary_frame.y == 1, rng.normal(size=500), numpy.nan)
    )
    for frame in (noisy, recorded):
        with pytest.raises(errors.BlockedError, match="leakage"):
            support.make_project(
                frame, "y", directory=tmp_path, leakage="block"
            ).run()


def test_strong_legitimate_signal_is_not_blocked(binary_frame, tmp_path):
    """The fixture's own features are strong predictors; none is a leak."""
    result = support.make_project(
        binary_frame, "y", directory=tmp_path, leakage="block"
    ).run()
    assert result.profile.blockers == ()
    assert not result.profile.risk_findings


def regression_leak():
    rng = numpy.random.default_rng(2)
    rows = 400
    frame = pandas.DataFrame({"a": rng.normal(size=rows)})
    frame["y"] = rng.normal(size=rows)
    frame["leak"] = frame.y * 2.0
    return frame


def test_a_regression_leak_blocks_under_block(tmp_path):
    # ERR-1: the screen is certified for regression, so the copied target
    # blocks by name; before, regression was uncertified and trained.
    # Enforces docs/API_STABILITY.md:54.
    # Enforces website/src/pages/docs/api-stability.astro:64.
    with pytest.raises(
        errors.BlockedError, match="unresolved leakage risk"
    ) as raised:
        support.make_project(
            regression_leak(),
            "y",
            task="regression",
            directory=tmp_path,
            leakage="block",
        ).run()
    assert raised.value.details["leakage"]["blocking"] == ["leak"]


def test_a_regression_leak_warns_under_warn(tmp_path):
    # Enforces docs/LEAKAGE.md:228.
    # Enforces docs/WHEN_THINGS_GO_WRONG.md:110.
    # Enforces website/src/learn/leakage.html:68-69.
    # Enforces website/src/learn/when-things-go-wrong.html:59.
    result = support.make_project(
        regression_leak(),
        "y",
        task="regression",
        directory=tmp_path,
        leakage="warn",
    ).run()
    assert any("'leak'" in w for w in result.profile.risk_findings)
    assert result.profile.blockers == ("leak",)


def test_groups_never_straddle_the_test_boundary(binary_frame, tmp_path):
    # Enforces website/src/pages/docs/architecture.astro:48.
    frame = binary_frame.assign(g=numpy.arange(len(binary_frame)) % 40)
    result = support.make_project(
        frame, "y", group="g", directory=tmp_path
    ).run()
    manifest = json.loads(
        (result.directory / "validation_manifest.json").read_text()
    )
    train_groups = set(frame.g.iloc[manifest["training"]])
    test_groups = set(frame.g.iloc[manifest["test"]])
    assert train_groups.isdisjoint(test_groups)
    assert "g" not in result.bundle.features


def test_test_set_is_opened_exactly_once_per_run(binary_frame, tmp_path):
    # Enforces docs/WHAT_SCMLE_CANNOT_DO.md:61.
    # Enforces website/src/learn/what-scmle-cannot-do.html:16-17.
    result = support.make_project(binary_frame, "y", directory=tmp_path).run()
    assert result.evaluation["opened"] is True
    events = (result.directory / "events.jsonl").read_text().splitlines()
    testing = [e for e in events if json.loads(e)["kind"] == "state.TESTING"]
    assert len(testing) == 1


def run_default(frame, directory):
    return scmle.api.project.Project(
        frame,
        "y",
        budget=600,
        directory=directory,
        validation="random",  # Stratified splits legitimately use labels.
        search=tasks.Search(top_models=1, configs_per_model=4),
        strategies=tasks.Strategies(top_models=1),
        ensemble=tasks.Stacking(size=2),
    ).run()


def test_search_is_invariant_to_test_labels_and_test_features(
    imbalanced_frame, tmp_path
):
    """Nothing the search sees may depend on the locked test partition.

    Also enforces website/src/learn/what-scmle-cannot-do.html:16-17.
    """
    # Enforces docs/WHAT_SCMLE_CANNOT_DO.md:61.
    first = run_default(imbalanced_frame, tmp_path / "a")
    manifest = json.loads(
        (first.directory / "validation_manifest.json").read_text()
    )
    test_ids = numpy.array(manifest["test"])

    altered = imbalanced_frame.copy()
    rng = numpy.random.default_rng(99)
    altered.loc[test_ids, "y"] = rng.permutation(
        altered.loc[test_ids, "y"].to_numpy()
    )
    for column in ("spend", "age", "visits"):
        altered.loc[test_ids, column] = rng.normal(500, 100, len(test_ids))
    second = run_default(altered, tmp_path / "b")

    assert (
        second.evaluation["digests"]["split"]
        == first.evaluation["digests"]["split"]
    )
    assert [
        (t.identifier, t.model, t.strategy, t.params) for t in second.trials
    ] == [(t.identifier, t.model, t.strategy, t.params) for t in first.trials]
    for a, b in zip(first.trials, second.trials, strict=True):
        assert a.folds == pytest.approx(b.folds)
    assert second.model == first.model
    assert (
        second.evaluation["selected"]["params"]
        == first.evaluation["selected"]["params"]
    )
    assert second.evaluation["selected"]["score"] == pytest.approx(
        first.evaluation["selected"]["score"]
    )
    # Sanity: the test partition really was different.
    assert second.evaluation["metrics"] != first.evaluation["metrics"]


def without_timings(statistics):
    """Removes wall-clock measurements, which differ between any two runs."""
    cleaned = json.loads(json.dumps(statistics, default=str))
    cleaned["operational"].pop("read_seconds")
    cleaned["ml_difficulty"].pop("seconds")
    return cleaned


def test_profile_and_feature_plan_never_see_the_locked_test_rows(
    binary_frame, tmp_path
):
    base = support.make_project(binary_frame, "y", directory=tmp_path / "a")
    base.plan()
    from scmle.data.splitting import splitters

    split = splitters.Splitter(base.spec, tasks.Task.BINARY).split(binary_frame)
    test_positions = list(split.manifest.test)

    altered = binary_frame.copy()
    altered.loc[altered.index[test_positions], "a"] = numpy.nan  # all missing
    altered.loc[altered.index[test_positions], "b"] = 1e9  # absurd values
    altered.loc[altered.index[test_positions], "c"] = "never-seen"
    other = support.make_project(altered, "y", directory=tmp_path / "b")
    other.plan()

    before, after = base.profile, other.profile
    assert before.statistics == after.statistics
    assert before.plan == after.plan
    assert without_timings(before.overview) == without_timings(after.overview)
    rows = before.overview["shape"]
    assert rows["rows"] == len(split.manifest.training)
    assert rows["total_rows"] == len(binary_frame)
    assert rows["rows"] + len(test_positions) == rows["total_rows"]


def test_a_leak_present_only_in_the_test_rows_is_not_a_profile_finding(
    binary_frame, tmp_path
):
    from scmle.data.splitting import splitters

    project = support.make_project(binary_frame, "y", directory=tmp_path)
    split = splitters.Splitter(project.spec, tasks.Task.BINARY).split(
        binary_frame
    )
    test_positions = list(split.manifest.test)
    leaky = binary_frame.assign(
        late=numpy.random.default_rng(1).normal(size=500)
    )
    leaky.loc[leaky.index[test_positions], "late"] = (
        leaky.loc[leaky.index[test_positions], "y"] * 5.0
    )
    probe = support.make_project(leaky, "y", directory=tmp_path / "p")
    probe.plan()
    assert not any("'late'" in r for r in probe.profile.risk_findings)


def test_hard_enforcement_moves_the_test_rows_out_of_reach(
    tmp_path, monkeypatch
):
    from scmle.runtime import limits

    rng = numpy.random.default_rng(21)
    frame = pandas.DataFrame(
        {
            "sentinel": numpy.arange(400, dtype=float),
            "x": rng.normal(size=400),
        }
    )
    frame["y"] = (frame.x + rng.normal(0, 0.5, 400) > 0).astype(int)
    seen = []
    original = limits.Supervisor.run

    def record(self, function, argument):
        seen.append(set(argument.features["sentinel"]))
        return original(self, function, argument)

    monkeypatch.setattr(limits.Supervisor, "run", record)
    result = support.make_project(
        frame,
        "y",
        budget=300,
        directory=tmp_path,
        enforcement="hard",
    ).run()

    manifest = json.loads(
        (result.directory / "validation_manifest.json").read_text()
    )
    test_ids = set(manifest["test"])
    test_values = {float(frame.loc[i, "sentinel"]) for i in test_ids}
    events = [
        json.loads(line)
        for line in (result.directory / "events.jsonl").read_text().splitlines()
    ]
    created = next(e for e in events if e["kind"] == "split.created")

    assert created["data"]["test_set_isolation"] == "separate vault process"
    assert seen and all(values.isdisjoint(test_values) for values in seen)
    assert result.evaluation["rows"]["test"] == len(test_ids)
    assert result.evaluation["opened"] is True


def test_cooperative_mode_says_it_relies_on_the_in_process_protocol(
    binary_frame, tmp_path
):
    result = support.make_project(
        binary_frame, "y", budget=300, directory=tmp_path
    ).run()

    events = [
        json.loads(line)
        for line in (result.directory / "events.jsonl").read_text().splitlines()
    ]
    created = next(e for e in events if e["kind"] == "split.created")
    assert created["data"]["test_set_isolation"] == "in-process protocol"
