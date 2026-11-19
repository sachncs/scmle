"""Claim C3 evidence: skill claimed on null tasks."""

import collections
from concurrent import futures
import json

import numpy
import pandas
import pytest

from benchmarks import acceptance
from benchmarks.evidence import nulls
from benchmarks.evidence import quality
import scmle.api.project
from scmle.contracts import errors
from scmle.contracts import tasks


def outcome(population: str, status: str) -> dict:
    """One task's outcome with the given population and status."""
    return {"population": population, "status": status}


def document(hits: int = 0, failed: int = 0) -> dict:
    """A current document of the planned size with `hits` claims."""
    planned = nulls.required()[1]
    statuses = ["claimed"] * hits + ["failed"] * failed
    statuses += ["silent"] * (planned - len(statuses))
    outcomes = [
        outcome(nulls.POPULATIONS[i % 2], s) for i, s in enumerate(statuses)
    ]
    variables = acceptance.load()
    return {
        "bound": variables[nulls.VARIABLE].value,
        "alpha": variables["alpha"].value,
        "power": variables["power"].value,
        "margin": variables[nulls.MARGIN].value,
        "data": quality.files(),
        "code": nulls.digest(),
        "populations": [
            nulls.tally(p, [o for o in outcomes if o["population"] == p])
            for p in nulls.POPULATIONS
        ],
        "pooled": nulls.tally("pooled", outcomes),
    }


def test_the_task_count_comes_from_the_ceiling_and_the_size_margin():
    variables = acceptance.load()
    alpha = variables["alpha"].value
    power = variables["power"].value
    rule = acceptance.ceiling(nulls.VARIABLE, variables)
    truth = rule.bound / (1 + variables[nulls.MARGIN].value)
    least, planned = nulls.required()
    assert least == 59
    assert planned == 577
    assert planned >= variables[nulls.COUNT].value
    assert rule.chance(truth, planned, alpha) >= power
    assert rule.chance(truth, planned - 1, alpha) < power


def test_the_budget_is_the_c4_benchmark_budget():
    assert nulls.BUDGET == quality.BUDGET == 60


def test_the_assumed_rate_is_the_bound_shrunk_by_the_size_margin():
    assert nulls.assumed() == pytest.approx(0.05 / 1.69, abs=1e-15)


def test_the_catalog_has_the_planned_count_split_between_populations():
    found = nulls.catalog()
    planned = nulls.required()[1]
    tables = sum(len(ids) for ids in quality.TASKS.values())
    assert len(found) == planned
    real = [t for t in found if t.population == "real"]
    simulated = [t for t in found if t.population == "simulated"]
    assert len(real) == tables * 9 == 315
    assert len(simulated) == planned - 315 == 262
    assert len({t.name for t in found}) == planned
    assert len({t.seed for t in found}) == planned
    counts = collections.Counter(t.source for t in real)
    assert set(counts.values()) == {9}
    assert len(counts) == tables


def test_simulated_tasks_cover_every_world_kind_and_size():
    simulated = [t for t in nulls.catalog() if t.population == "simulated"]
    cells = {(t.source, t.kind, t.rows) for t in simulated}
    assert len(cells) == len(nulls.WORLDS) * len(nulls.KINDS) * len(nulls.ROWS)


def test_a_seed_comes_from_the_registered_validation_stream():
    assert nulls.STREAM == "coverage.null.validation"
    assert nulls.seed("simulated", 0) == nulls.seed("simulated", 0)
    assert nulls.seed("simulated", 0) != nulls.seed("simulated", 1)
    assert nulls.seed("simulated", 0) != nulls.seed("real", 0)
    assert 0 <= nulls.seed("real", 7) < 2**31


@pytest.mark.parametrize("kind", nulls.KINDS)
@pytest.mark.parametrize("world", nulls.WORLDS)
def test_a_simulated_target_is_drawn_apart_from_the_features(world, kind):
    task = nulls.Simulated("s", "simulated", world, kind, 11, 300)
    frame = task.draw()
    again = task.draw()
    pandas.testing.assert_frame_equal(frame, again)
    assert len(frame) == 300
    assert nulls.TARGET in frame
    labels = frame[nulls.TARGET]
    if kind == "binary":
        assert set(labels) == {"a", "b"}
    elif kind == "multiclass":
        assert set(labels) == {"a", "b", "c", "d"}
    else:
        assert labels.dtype == float
    # The target is a function of its own generator only: the same seed
    # with other features (another world) gives the same target.
    other = nulls.Simulated(
        "s", "simulated", nulls.WORLDS[0], kind, 11, 300
    ).draw()
    pandas.testing.assert_series_equal(labels, other[nulls.TARGET])


def test_a_mixed_world_has_categories_and_missing_values():
    frame = nulls.Simulated("s", "simulated", "mixed", "binary", 3, 1000).draw()
    features = frame.drop(columns=[nulls.TARGET])
    text = [
        c for c in features if pandas.api.types.is_string_dtype(features[c])
    ]
    assert text == ["c3", "c12"]
    assert features[["c3", "c12"]].nunique().tolist() == [3, 12]
    assert features.isna().all(axis=0).sum() == 0
    assert features.isna().any(axis=0).all()


def test_a_permuted_table_keeps_its_features_and_its_labels():
    table = quality.TASKS["binary"][0]
    task = nulls.Permuted("r", "real", table, "binary", 5)
    frame = task.draw()
    original = nulls.Permuted.original(table)
    target = original.attrs["target"]
    assert len(frame) == len(original)
    pandas.testing.assert_frame_equal(
        frame.drop(columns=[target]), original.drop(columns=[target])
    )
    assert sorted(frame[target]) == sorted(original[target])
    assert not frame[target].equals(original[target])
    pandas.testing.assert_frame_equal(frame, task.draw())


def test_a_permuted_table_keeps_a_class_named_none():
    # In analcatdata_dmft "None" is a prevention method, not a missing label.
    original = nulls.Permuted.original(3560)
    target = original.attrs["target"]
    assert len(original) == 797
    assert (original[target] == "None").sum() == 136
    assert original[target].notna().all()


@pytest.mark.parametrize(
    ("interval", "anchor", "higher", "expected"),
    [
        ([0.51, 0.6], 0.5, True, True),
        ([0.5, 0.6], 0.5, True, False),
        ([0.4, 0.6], 0.5, True, False),
        ([0.8, 0.9], 1.0, False, True),
        ([0.8, 1.0], 1.0, False, False),
        ([0.8, 1.1], 1.0, False, False),
        (None, 0.5, True, None),
        ([0.51, 0.6], None, True, None),
        ([0.51, 0.6], float("nan"), True, None),
        ([float("nan"), 0.6], 0.5, True, None),
    ],
)
def test_an_interval_claims_skill_only_wholly_on_the_better_side(
    interval, anchor, higher, expected
):
    assert nulls.better(interval, anchor, higher) is expected


def test_a_run_claims_skill_when_either_interval_clears_the_baseline():
    evaluation = {
        "direction": "higher",
        "baseline": 0.5,
        "confidence": {"available": True, "interval": [0.52, 0.6]},
        "health": {"verdict": "underfit"},
    }
    uncertainty = {"ci_95": [0.45, 0.55]}
    assert nulls.judge(evaluation, uncertainty, 0.5) == {
        "cv": True,
        "test": False,
        "health": False,
    }
    evaluation["confidence"] = {"available": False, "reason": "one fold"}
    assert nulls.judge(evaluation, uncertainty, 0.5) == {
        "cv": None,
        "test": False,
        "health": False,
    }
    loss = {
        "direction": "lower",
        "baseline": 1.0,
        "confidence": {"available": True, "interval": [0.9, 1.2]},
    }
    assert nulls.judge(loss, {"ci_95": [0.7, 0.95]}, 1.0) == {
        "cv": False,
        "test": True,
        "health": None,
    }
    assert nulls.judge(loss, {}, 1.0) == {
        "cv": False,
        "test": None,
        "health": None,
    }


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("healthy", True),
        ("overfit", False),
        ("unstable", False),
        ("underfit", False),
    ],
)
def test_only_a_healthy_fit_verdict_claims_skill(verdict, expected):
    evaluation = {
        "direction": "higher",
        "baseline": 0.5,
        "confidence": {"available": True, "interval": [0.4, 0.6]},
        "health": {"verdict": verdict},
    }
    found = nulls.judge(evaluation, {"ci_95": [0.45, 0.55]}, 0.5)
    assert found == {"cv": False, "test": False, "health": expected}
    assert any(found.values()) is expected


def test_the_direction_decides_which_side_is_better():
    evaluation = {
        "direction": "higher",
        "baseline": 0.5,
        "confidence": {"available": True, "interval": [0.52, 0.6]},
    }
    uncertainty = {"ci_95": [0.52, 0.6]}
    assert nulls.judge(evaluation, uncertainty, 0.5) == {
        "cv": True,
        "test": True,
        "health": None,
    }
    evaluation["direction"] = "lower"
    assert nulls.judge(evaluation, uncertainty, 0.5) == {
        "cv": False,
        "test": False,
        "health": None,
    }


@pytest.mark.parametrize("direction", [True, None, "higer"])
def test_an_unknown_direction_is_refused(direction):
    evaluation = {"baseline": 0.5}
    if direction is not None:
        evaluation["direction"] = direction
    with pytest.raises(errors.InputError, match="unknown direction"):
        nulls.judge(evaluation, {"ci_95": [0.52, 0.6]}, 0.5)


def test_the_anchor_is_the_product_baseline_on_the_product_test_rows():
    frame = pandas.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 3.0, 5.0, 0.0, 10.0, 2.0],
        }
    )
    split = {"training": [0, 1, 2], "test": [3, 4, 5]}
    value = nulls.anchor(frame, "y", split, "rmse", tasks.Task.REGRESSION)
    # The training mean is 3; the test errors are 3, 7 and 1.
    assert value == pytest.approx(numpy.sqrt((9 + 49 + 1) / 3))
    labels = frame.assign(y=["a", "b", "a", "b", "a", "b"])
    assert nulls.anchor(labels, "y", split, "roc_auc", tasks.Task.BINARY) == 0.5


def test_the_anchor_refuses_a_split_that_does_not_cover_the_table():
    frame = pandas.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(errors.InputError, match="split"):
        nulls.anchor(
            frame,
            "y",
            {"training": [0], "test": [1]},
            "rmse",
            tasks.Task.REGRESSION,
        )
    with pytest.raises(errors.InputError, match="split"):
        nulls.anchor(
            frame,
            "y",
            {"training": [0, 1], "test": [5]},
            "rmse",
            tasks.Task.REGRESSION,
        )


def test_an_unknown_kind_or_world_is_refused():
    with pytest.raises(errors.InputError, match="kind"):
        nulls.Simulated("s", "simulated", "gaussian", "ranking", 1, 300).draw()
    with pytest.raises(errors.InputError, match="world"):
        nulls.Simulated("s", "simulated", "spiral", "binary", 1, 300).draw()


def test_failed_runs_count_as_claims_and_refusals_do_not():
    outcomes = [
        outcome("real", "claimed"),
        outcome("real", "failed"),
        outcome("real", "refused"),
        outcome("real", "silent"),
    ]
    found = nulls.tally("real", outcomes)
    alpha = acceptance.load()["alpha"].value
    assert found["hits"] == 2
    assert found["trials"] == 4
    assert found["claimed"] == 1
    assert found["failed"] == 1
    assert found["refused"] == 1
    assert found["upper"] == pytest.approx(
        acceptance.Ceiling(0.05).limit(2, 4, alpha)
    )
    assert found["passed"] is False


def test_unknown_statuses_are_refused():
    with pytest.raises(errors.InputError, match="status"):
        nulls.tally("real", [outcome("real", "maybe")])


def test_a_current_document_with_few_claims_passes():
    found = document(hits=6)
    assert nulls.problems(found) == []
    text, verdict = nulls.summary(found)
    assert verdict == "pass"
    assert text.startswith("6/577 null tasks claimed skill")


def test_too_many_claims_fail():
    # 20 of 577 is the most that passes: its bound is below 0.05.
    assert nulls.summary(document(hits=20))[1] == "pass"
    text, verdict = nulls.summary(document(hits=21))
    assert verdict == "fail"
    assert "21/577" in text


def test_failures_push_the_rate_up():
    assert nulls.summary(document(hits=11, failed=10))[1] == "fail"
    assert nulls.summary(document(hits=10, failed=10))[1] == "pass"


def test_a_short_or_stale_document_is_not_measured():
    short = document()
    short["pooled"] = nulls.tally("pooled", [outcome("real", "silent")] * 200)
    assert any("fewer than the 577 planned" in p for p in nulls.problems(short))
    assert nulls.summary(short)[1] == "not measured"
    stale = document()
    stale["code"] = "other"
    assert nulls.problems(stale) == ["nulls: measured with other code"]
    assert nulls.summary(stale)[1] == "not measured"
    loose = document()
    loose["bound"] = 0.1
    assert nulls.problems(loose) == ["nulls: bound 0.1 is not 0.05"]
    moved = document()
    moved["data"] = {"3560": "other"}
    assert nulls.problems(moved) == ["nulls: measured on other data"]
    assert nulls.summary(moved)[1] == "not measured"
    variables = acceptance.load()
    for key, name in (
        ("alpha", "alpha"),
        ("power", "power"),
        ("margin", nulls.MARGIN),
    ):
        changed = document()
        changed[key] = 0.5
        value = variables[name].value
        assert nulls.problems(changed) == [f"nulls: {key} 0.5 is not {value}"]
        assert nulls.summary(changed)[1] == "not measured"


def test_the_digest_covers_the_catalog_and_the_thresholds(monkeypatch):
    before = nulls.digest()
    calls = []
    real = nulls.certificates.code_hash

    def spy(*objects):
        calls.append(objects)
        return real(*objects)

    monkeypatch.setattr(nulls.certificates, "code_hash", spy)
    assert nulls.digest() == before
    assert calls == [(nulls, quality, acceptance)]


def test_nothing_stored_is_not_measured(tmp_path, monkeypatch):
    monkeypatch.setattr(nulls, "PATH", tmp_path / "nulls.json")
    assert nulls.summary() == ("no measurement", "not measured")
    assert nulls.problems() == ["nulls.json is missing"]
    assert nulls.main(["--check"]) == 1


def test_the_check_reads_the_stored_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(nulls, "PATH", tmp_path / "nulls.json")
    nulls.PATH.write_text(json.dumps(document(hits=3)), encoding="utf-8")
    assert nulls.main(["--check"]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "problems": []}


def test_a_crash_is_recorded_as_failed(monkeypatch):
    def crash(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(scmle.api.project.Project, "run", crash)
    found = nulls.run(
        nulls.Simulated("s", "simulated", "gaussian", "binary", 1, 300)
    )
    assert found["status"] == "failed"
    assert "boom" in found["error"]
    assert found["readings"] == {"cv": None, "test": None, "health": None}
    assert nulls.tally("s", [found])["hits"] == 1


def test_a_refusal_is_recorded_as_refused(monkeypatch):
    def refuse(self):
        raise errors.BlockedError("no")

    monkeypatch.setattr(scmle.api.project.Project, "run", refuse)
    found = nulls.run(
        nulls.Simulated("s", "simulated", "gaussian", "binary", 1, 300)
    )
    assert found["status"] == "refused"
    assert "no" in found["error"]
    assert found["readings"] == {"cv": None, "test": None, "health": None}
    assert nulls.tally("s", [found])["hits"] == 0


def test_write_stores_a_checkable_document(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(nulls, "PATH", tmp_path / "nulls.json")
    small = [
        nulls.Simulated("a", "simulated", "gaussian", "binary", 1, 300),
        nulls.Simulated("b", "simulated", "mixed", "regression", 2, 300),
    ]
    monkeypatch.setattr(nulls, "catalog", lambda: small)

    def silent(task, budget=nulls.BUDGET):
        return {
            "name": task.name,
            "population": "simulated",
            "status": "silent",
        }

    monkeypatch.setattr(nulls, "run", silent)
    monkeypatch.setattr(quality, "fetch", lambda table: None)
    monkeypatch.setattr(
        futures, "ProcessPoolExecutor", futures.ThreadPoolExecutor
    )
    assert nulls.main(["--write", "--workers", "2"]) == 0
    stored = json.loads(nulls.PATH.read_text(encoding="utf-8"))
    assert stored["pooled"]["trials"] == 2
    assert stored["pooled"]["hits"] == 0
    assert [t["name"] for t in stored["tasks"]] == ["a", "b"]
    assert set(stored["design"]) == {
        "budget_seconds",
        "stream",
        "share_real",
        "worlds",
        "kinds",
        "rows",
        "claim",
        "failed",
        "refused",
    }
    assert stored["design"]["claim"] == (
        "cv or test interval wholly better than the dummy, "
        "or a healthy fit verdict"
    )
    assert stored["code"] == nulls.digest()
    assert stored["data"] == quality.files()
    capsys.readouterr()
    # Two tasks are fewer than the planned count, so the check refuses it.
    assert nulls.main(["--check"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["problems"] == ["nulls: 2 tasks; none can pass below 59"]


def test_a_null_task_runs_the_product_end_to_end(tmp_path):
    task = nulls.Simulated(
        "simulated/smoke", "simulated", "gaussian", "binary", 1, 300
    )
    found = nulls.run(task, budget=60)
    assert found["status"] in {"claimed", "silent"}, found["error"]
    assert found["name"] == "simulated/smoke"
    assert found["metric"] == "roc_auc"
    assert found["anchor"] == 0.5
    assert set(found["readings"]) == {"cv", "test", "health"}
    assert found["readings"]["test"] is not None
    assert (found["status"] == "claimed") == any(found["readings"].values())
